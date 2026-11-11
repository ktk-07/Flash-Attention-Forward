#!/usr/bin/env python3
"""Build, validate, freeze bounded tuning, measure and export dense FP32 attention."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import shlex
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workflow import Workflow,architecture,device_default,device_info,dump,sha
from export_results import export,config_name

class PartBWorkflow(Workflow):
    def run(self,label,argv):
        argv=list(argv)
        # CUDA's Windows PATH entries are .bat shims, not executable images.
        # Resolve the installed binaries directly, avoiding shell interpolation.
        if os.name=='nt' and str(argv[0]) in ['compute-sanitizer','ncu']:
            shim=Path(shutil.which(str(argv[0])))
            binary=(shim.parent/'../compute-sanitizer/compute-sanitizer.exe' if argv[0]=='compute-sanitizer'
                    else shim.parent/'target/windows-desktop-win7-x64/ncu.exe').resolve()
            if not binary.is_file():raise ValueError('Missing executable behind CUDA tool shim: '+str(binary))
            argv[0]=binary
        if not (label.startswith('candidate_') or label.startswith('frozen_')):
            return super().run(label,argv)
        argv=list(map(str,argv));print(label+': '+shlex.join(argv),flush=True)
        code=None
        try:
            with (self.out/'logs'/f'{label}.txt').open('w') as stream:
                r=subprocess.run(argv,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,timeout=45)
                code=r.returncode
        finally:
            with (self.out/'commands.jsonl').open('a') as stream:stream.write(json.dumps({'label':label,'argv':argv,'exit_code':code,'timeout_seconds':45,'CUDA_VISIBLE_DEVICES':self.env['CUDA_VISIBLE_DEVICES']})+'\n')
        if code:raise RuntimeError(label+' failed; inspect its log')

# Bounded list shared by every d; matched tiled/vector pair isolates loading.
CANDIDATES=[{'variant':'streaming','warps':4,'key_tile':32},
            {'variant':'tiled','warps':4,'key_tile':32},
            {'variant':'vector','warps':4,'key_tile':32},
            {'variant':'tiled','warps':8,'key_tile':32},
            {'variant':'vector','warps':4,'key_tile':64}]

def runner_args(exe,c,b,h,n,d,pattern='random',sampled=False,timing=True,path=None,profile=False):
    result=[exe,'--B',b,'--H',h,'--N',n,'--d',d,'--variant',c['variant'],'--warps',c['warps'],'--key-tile',c['key_tile'],
            '--pattern',pattern,'--sampled',int(sampled),'--timing',int(timing),'--profile',int(profile)]
    if path is not None:result+=['--output',path]
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('part_b_%Y%m%dT%H%M%SZ'))
    p.add_argument('--build',type=Path,default=Path(tempfile.gettempdir())/'part_b-build')
    p.add_argument('--arch',default='auto');p.add_argument('--device',default=device_default())
    p.add_argument('--grid',choices=['full','smoke'],default='full');p.add_argument('--no-profile',action='store_true')
    p.add_argument('--no-sanitize',action='store_true',help='diagnostic measurements only; audit remains incomplete and root rejects the bundle')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze-only',type=Path)
    a=p.parse_args()
    if a.analyze_only:
        result=export(a.analyze_only.resolve());print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
    if not a.device or ',' in a.device:p.error('Select one GPU index or UUID')
    arch=architecture(a.arch,a.device,a.dry_run);out=a.output.resolve();build=a.build.resolve()
    profile=not a.no_profile and a.grid!='smoke'
    # Plan mode must not require dependencies, a GPU, or create any directories.
    if a.dry_run:
        print(json.dumps({'part':'part_b','output':str(out),'build':str(build),'arch':arch,'grid':a.grid,
            'candidates':CANDIDATES,'phases':['build','shared_oracle','full_correctness','candidate_safety','tuning','freeze','frozen_safety','main','profiles' if profile else 'profiles_explicitly_omitted','export'],
            'profile_captures':6 if profile else 0},indent=2));return 0
    w=PartBWorkflow(out,a.device,profile=profile)
    native=out/'raw/part_b';native.mkdir(parents=True);(out/'configs').mkdir()
    index={'cases':[],'profiles':[],'sanitizers':[],'evidence_hashes':{}}
    dump(out/'run_status.json',{'status':'running','grid':a.grid,'profiling_requested':profile})
    try:
        device=device_info(a.device);dump(out/'device.json',device)
        w.run('gpu',['nvidia-smi','-q']);w.run('nvcc',['nvcc','--version'])
        w.run('configure',['cmake','-S',ROOT/'part_b','-B',build,'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}'])
        w.run('build',['cmake','--build',build,'--config','Release','--parallel','2'])
        suffix='.exe' if os.name=='nt' else ''
        def executable(name,sub=''):
            options=[build/sub/(name+suffix),build/sub/'Release'/(name+suffix)]
            return next((x for x in options if x.is_file()),options[0])
        exe=executable('part_b_runner');w.run('shared_reference',[executable('attention_reference_test','shared_reference')])
        counter=0
        def case(c,b,h,n,d,pattern='random',experiment='correctness',public=None,timing=False):
            nonlocal counter
            label=f'case_{counter:04d}';counter+=1;relative=f'raw/part_b/{label}.json';path=out/relative
            # Exhaustive small and candidate validation. Sample large timing cases.
            sampled=timing and b*h*n*n*d>32*1024*1024
            index['cases'].append({'path':relative,'experiment':experiment,'public_variant':public or c['variant']})
            try:w.run(label,runner_args(exe,c,b,h,n,d,pattern,sampled,timing,path))
            finally:
                if path.exists():index['evidence_hashes'][relative]=sha(path)
                dump(native/'index.json',index)
            return json.loads(path.read_text()),relative
        # Boundary/tail/stability cases for every proposed configuration.
        shapes=[(1,1,1,1),(1,1,17,32),(2,3,31,64),(1,2,32,128),(2,1,33,37),(1,1,63,64),(1,1,64,128),(1,1,65,32),(1,1,9,256)]
        for c in CANDIDATES:
            for b,h,n,d in shapes:
                if c['variant']=='vector' and d%4:continue
                if c['key_tile']==64 and d==256:continue
                case(c,b,h,n,d)
            for pattern in ['uniform','constant_v','rising','stress']:case(c,2,2,65,64,pattern)
        def safety(c,label):
            for tool in ['memcheck','racecheck','synccheck','initcheck']:
                log=label+'_'+tool;item={'config':config_name(c),'tool':tool,'exit_code':None,'log':'logs/'+log+'.txt'}
                index['sanitizers'].append(item)
                try:
                    w.run(log,['compute-sanitizer','--tool',tool,'--error-exitcode','3']+runner_args(exe,c,2,2,65,64,timing=False))
                    item['exit_code']=0
                finally:
                    log_path=out/item['log']
                    if log_path.exists():index['evidence_hashes'][item['log']]=sha(log_path)
                    dump(native/'index.json',index)
        if not a.no_sanitize:
            for i,c in enumerate(CANDIDATES):safety(c,f'candidate_{i}')
        ds=[64] if a.grid=='smoke' else [32,64,128];b,h=(1,1) if a.grid=='smoke' else (4,8)
        tuning_ns=[17,33] if a.grid=='smoke' else [256,1024]
        report_ns=[33,65] if a.grid=='smoke' else [256,512,1024,2048,4096]
        selected={};scores={}
        for d in ds:
            ranked=[]
            for c in CANDIDATES:
                # Full reference checks on tails for each candidate at this d.
                case(c,2,2,33,d,'stress')
                medians=[]
                for n in tuning_ns:
                    r,_=case(c,b,h,n,d,experiment='tuning',timing=True)
                    medians.append(statistics.median(r['samples_ms']))
                score=math.exp(sum(math.log(x) for x in medians)/len(medians));ranked.append((score,config_name(c),c))
            ranked.sort(key=lambda x:(x[0],x[1]));selected[str(d)]=ranked[0][2]
            scores[str(d)]=[{'config':name,'geomean_ms':value} for value,name,_ in ranked]
        frozen={'candidates':CANDIDATES,'selected':selected,'scores':scores,'tuning_lengths':tuning_ns,
            'reporting_lengths':report_ns,'selection':'minimum geometric mean forward median, lexical config-name tie break',
            'frozen_utc':datetime.now(timezone.utc).isoformat()}
        dump(out/'configs/part_b.json',frozen);index['evidence_hashes']['configs/part_b.json']=sha(out/'configs/part_b.json')
        if not a.no_sanitize:
            for i,c in enumerate({config_name(c):c for c in selected.values()}.values()):safety(c,f'frozen_{i}')
        main_cases={}
        for d in ds:
            # Include controlled progression on the same grid, plus exactly one selected series.
            variants=[('selected',selected[str(d)]),('streaming',CANDIDATES[0]),('tiled',CANDIDATES[1]),('vector',CANDIDATES[2])]
            for public,c in variants:
                for n in report_ns:
                    _,relative=case(c,b,h,n,d,experiment='proposal',public=public,timing=True)
                    if public=='selected':main_cases[(n,d)]=relative
        if profile:
            for n in [256,4096]:
                c=selected['64'];folder=native/'profiles'/f'n{n}';folder.mkdir(parents=True)
                for repeat in range(3):
                    prefix=folder/f'repeat_{repeat}';csv_path=prefix.with_suffix('.csv')
                    metrics='dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,gpu__time_duration.sum'
                    w.run(f'profile_n{n}_{repeat}',['ncu','--metrics',metrics,'--launch-count','1','--export',prefix]+runner_args(exe,c,b,h,n,64,timing=False,profile=True))
                    # Redirect only exported CSV, retaining normal profiler logs separately.
                    argv=['ncu','--import',str(prefix)+'.ncu-rep','--page','raw','--csv','--print-units','base','--log-file',str(csv_path)]
                    w.run(f'profile_export_n{n}_{repeat}',argv)
                    relative=str(csv_path.relative_to(out)).replace('\\','/')
                    index['profiles'].append({'case':main_cases[(n,64)],'repeat':repeat,'csv':relative})
                    index['evidence_hashes'][relative]=sha(csv_path)
                    rep=str(prefix.relative_to(out)).replace('\\','/')+'.ncu-rep';index['evidence_hashes'][rep]=sha(out/rep)
        dump(native/'index.json',index)
        sources={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in sorted((ROOT/'part_b').rglob('*')) if path.is_file() and not {'__pycache__','results'}.intersection(path.relative_to(ROOT/'part_b').parts)}
        # Save the exact B sources used by this run, independent of later edits.
        snapshot=native/'source';snapshot.mkdir()
        for relative in sources:
            path=Path(relative);target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,target)
        shared={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for folder in ['part_a/common','part_a/cpu_reference','scripts'] for path in sorted((ROOT/folder).glob('*')) if path.is_file()}
        manifest={'schema_version':1,'part':'part_b','grid':a.grid,'run_id':out.name,'device':device,'profile_requested':profile,
            'shared_cpu_reference_linked':True,'sanitizers_requested':not a.no_sanitize,'source_hashes':sources,'shared_source_hashes':shared,'binary_sha256':sha(exe),
            'layout':'BHNd_contiguous_row_major','generator':'splitmix64-fp32-v1','seed':5206,
            'registry':{'FP32':{'input':'FP32','accumulator':'FP32','output':'FP32','selected':selected,
                'kernels':{'streaming':'streaming_dense<W>','tiled':'tiled_dense<W,BK,false>','vector':'tiled_dense<W,BK,true>'}}}}
        dump(out/'manifest.json',manifest);audit=export(out)
        if not audit['passed'] and not a.no_sanitize:raise RuntimeError('Part B evidence audit failed')
        if a.no_sanitize:
            dump(out/'run_status.json',{'status':'diagnostic_only_missing_sanitizers','profiling_requested':profile,'grid':a.grid})
            print(json.dumps(audit,indent=2));return 0
        dump(out/'run_status.json',{'status':'completed_part_b' if profile else 'completed_without_fresh_profile_evidence','profiling_requested':profile,'grid':a.grid})
        print(json.dumps(audit,indent=2));return 0
    except Exception as error:
        dump(native/'index.json',index);dump(out/'audit.json',{'passed':False,'error':str(error),'profile_captures':len(index['profiles'])})
        dump(out/'run_status.json',{'status':'failed','error':str(error),'grid':a.grid});raise

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:print('ERROR: '+str(error),file=sys.stderr);sys.exit(1)
