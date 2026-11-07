#!/usr/bin/env python3
"""Standalone Part A evaluation; shared correctness, frozen tuning and full raw evidence."""
import argparse,json,math,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from workflow import Workflow,architecture,device_default,device_info,dump,sha
import results_schema as schema
from export_results import publish,identity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('part_a_%Y%m%dT%H%M%SZ'))
    p.add_argument('--build',type=Path,default=Path('/tmp/part_a-standard-build'));p.add_argument('--arch',default='auto');p.add_argument('--device',default=device_default())
    p.add_argument('--grid',choices=['full','smoke'],default='full');p.add_argument('--no-profile',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze-only',type=Path)
    a=p.parse_args()
    if a.analyze_only:
        publish(a.analyze_only);subprocess.run([sys.executable,ROOT/'scripts/plot_combined.py',a.analyze_only],check=True);return
    if not a.device or ',' in a.device:p.error('Select one device index or UUID')
    if a.grid=='smoke':a.no_profile=True
    arch=architecture(a.arch,a.device,a.dry_run);out=a.output.resolve();build=a.build.resolve();binary=build/'attention';native=out/'raw/part_a'
    w=Workflow(out,a.device,a.dry_run,not a.no_profile)
    plan={'run_id':out.name,'grid':a.grid,'outputs':[],'selection':{},'profile_requested':not a.no_profile,'profiling_policy':{'cache_control':'all','clock_control':'none','repetitions':3,'separate_instrumented_runs':True}}
    if not a.dry_run:
        native.mkdir(parents=True);plan['device']=device_info(a.device)
    def call(label,args,phase=None):
        target=native/label
        w.run(label,[binary,*args,'--output',target])
        if phase:plan['outputs'].append({'path':label,'phase':phase})
        return [] if a.dry_run else schema.read(target/'summary.csv') if (target/'summary.csv').exists() else []
    try:
        w.run('gpu',['nvidia-smi','-q']);w.run('compiler',['nvcc','--version'])
        w.run('configure',['cmake','-S',ROOT/'part_a','-B',build,'-DPART_A_ENABLE_CUDA=ON','-DPART_A_PROFILE_INFO=ON','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}'])
        w.run('build',['cmake','--build',build,'--parallel','2'])
        w.run('shared_reference_contract',[build/'cpu_reference/attention_reference_test'])
        for variant in ['all','tensorcore']:call('correctness_'+variant,['--grid','correctness','--variants',variant,'--mode','both'],'correctness')
        for pattern in ['uniform','constant_v','rising','stress']:
            call('pattern_'+pattern,['--shape',2,3,33,64,'--variants','all','--mode','both','--pattern',pattern],'correctness_pattern')
        for variant in ['optimized','tensorcore']:
            for tool in ['memcheck','racecheck','synccheck','initcheck']:
                w.run('safety_'+variant+'_'+tool,['compute-sanitizer','--tool',tool,'--error-exitcode','3',binary,'--shape',1,1,33,64,'--variants',variant,'--mode','both','--output',native/'safety'/f'{variant}_{tool}'])
        bh=[1,1] if a.grid=='smoke' else [4,8];ds=[64] if a.grid=='smoke' else [32,64,128]
        tuning_ns=[17,33] if a.grid=='smoke' else [256,1024];ns=[33,65] if a.grid=='smoke' else [256,512,1024,2048,4096]
        plan.update(B=bh[0],H=bh[1],d=ds,main_N=ns,tuning_N=tuning_ns)
        tuning=[]
        for variant in ['optimized','tensorcore']:
            for d in ds:
                for n in tuning_ns:tuning+=call(f'tuning_{variant}_{d}_{n}',['--grid','tuning','--shape',*bh,n,d,'--variants',variant,'--mode','both'],'tuning')
                for mode in ['dense','causal']:
                    rows=[r for r in tuning if r['variant']==variant and int(r['d'])==d and r['mode']==mode];groups={}
                    for r in rows:groups.setdefault((int(r['tile']),int(r['softmax_threads'])),[]).append(float(r['forward_median_ms']))
                    if a.dry_run:choice=(16,256)
                    else:
                        expected={(16,128),(16,256)} if variant=='tensorcore' else {(8,256),(16,256),(32,256),(16,128)}
                        if set(groups)!=expected or any(len(v)!=len(tuning_ns) for v in groups.values()):raise ValueError('Incomplete tuning grid')
                        choice=min(groups,key=lambda c:(sum(map(math.log,groups[c]))/len(groups[c]),c))
                    plan['selection'][f'{variant}:{mode}:d{d}']={'tile':choice[0],'softmax_threads':choice[1],'selection':'minimum_geometric_mean_forward_latency_across_tuning_N'}
        if not a.dry_run:dump(native/'frozen_selection.json',{'selection':plan['selection'],'tuning_N':tuning_ns,'frozen_utc':datetime.now(timezone.utc).isoformat()})
        main_rows=[]
        for variant in ['optimized','tensorcore']:
            for d in ds:
                for mode in ['dense','causal']:
                    c=plan['selection'][f'{variant}:{mode}:d{d}'];flags=['--tile',c['tile'],'--softmax-threads',c['softmax_threads']]
                    for n in ns:main_rows+=call(f'main_{variant}_{mode}_{d}_{n}',['--grid','proposal','--shape',*bh,n,d,'--variants','all' if variant=='optimized' else variant,'--mode',mode,*flags],'proposal')
        if not a.dry_run:
            plan['binary_sha256']=sha(binary)
            files=[p for p in (ROOT/'part_a').rglob('*') if p.is_file() and (p.suffix in ['.cpp','.cu','.hpp','.py','.sh'] or p.name=='CMakeLists.txt') and not any(x in p.parts for x in ['build','results','__pycache__'])]
            plan['sources']={str(path.relative_to(ROOT)):sha(path) for path in files}
            snapshot=native/'source_snapshot'
            for path in files:
                dst=snapshot/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
        if not a.no_profile:
            if a.dry_run:print('profiles: 24 reports, QK/softmax/PV, full metrics, 3 repetitions; separate from timing samples')
            else:
                for row in main_rows:
                    if row['variant'] not in ['optimized','tensorcore'] or row['d']!='64' or int(row['N']) not in [256,4096]:continue
                    for rep in range(3):
                        label=f"{row['variant']}_{row['mode']}_{row['N']}_{rep}";folder=native/'profiles'/label;folder.mkdir(parents=True)
                        w.run('profile_'+label,['ncu','--set','full','--force-overwrite','--launch-skip','18','--launch-count','3','--cache-control','all','--clock-control','none','--export',folder/'capture',binary,'--profile-only','--shape',row['B'],row['H'],row['N'],row['d'],'--mode',row['mode'],'--variants',row['variant'],'--tile',row['tile'],'--softmax-threads',row['softmax_threads'],'--output',folder/'program'])
                        w.run('profile_csv_'+label,['ncu','--import',folder/'capture.ncu-rep','--page','raw','--csv','--print-units','base','--log-file',folder/'raw.csv'])
                        dump(folder/'identity.json',{'identity':identity(row),'rep':rep,'binary_sha256':plan['binary_sha256']})
        if not a.dry_run:
            dump(native/'plan.json',plan);publish(out)
            w.run('plots',[sys.executable,ROOT/'scripts/plot_combined.py',out])
            dump(out/'run_status.json',{'status':'completed','grid':a.grid,'device':plan['device'],'profiling_requested':not a.no_profile})
        else:print('Dry run: no files written or commands executed. Configurations shown are placeholders until tuning runs.')
    except Exception as error:
        if not a.dry_run:dump(out/'run_status.json',{'status':'failed','error':str(error)})
        raise
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(1)
