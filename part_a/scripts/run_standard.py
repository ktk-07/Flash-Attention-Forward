#!/usr/bin/env python3
"""Standalone Part A evaluation; shared correctness, frozen tuning and full raw evidence."""
import argparse,json,math,os,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from workflow import Workflow,architecture,device_default,device_info,dump,sha
import results_schema as schema
from export_results import publish,identity,save_report_results

def collect_profiles(w,native,binary,plan,main_rows):
    if sha(binary)!=plan['binary_sha256']:raise ValueError('Profile binary changed since timing')
    for row in main_rows:
        if row['variant'] not in ['optimized','wmma','ptx'] or row['d']!='64' or int(row['N']) not in [256,4096]:continue
        for rep in range(3):
            label=f"{row['variant']}_{row['mode']}_{row['N']}_{rep}";folder=native/'profiles'/label;folder.mkdir(parents=True,exist_ok=True)
            metadata={'identity':identity(row),'rep':rep,'binary_sha256':plan['binary_sha256']}
            if (folder/'identity.json').exists():
                if json.loads((folder/'identity.json').read_text())!=json.loads(json.dumps(metadata)):raise ValueError('Existing profile identity changed')
                if (folder/'raw.csv').exists():continue
            # Compressed or plain Nsight reports
            reports=[p for p in [folder/'capture.ncu-rep',folder/'capture.ncu-repz'] if p.exists()]
            if not reports:
                w.run('profile_'+label,['ncu','--set','full','--force-overwrite','--launch-skip','18','--launch-count','3','--cache-control','all','--clock-control','none','--export',folder/'capture',binary,'--profile-only','--shape',row['B'],row['H'],row['N'],row['d'],'--mode',row['mode'],'--variants',row['variant'],'--tile',row['tile'],'--softmax-threads',row['softmax_threads'],'--output',folder/'program'])
                reports=[p for p in [folder/'capture.ncu-rep',folder/'capture.ncu-repz'] if p.exists()]
            if len(reports)!=1:raise ValueError('Expected one Nsight report for '+label)
            w.run('profile_csv_'+label,['ncu','--import',reports[0],'--page','raw','--csv','--print-units','base','--log-file',folder/'raw.csv'])
            dump(folder/'identity.json',metadata)

def resume_profiles(out,build,device,target):
    out=out.resolve();native=out/'raw/part_a';plan=json.loads((native/'plan.json').read_text())
    if not plan['profile_requested']:raise ValueError('Original run did not request profiling')
    if device_info(device)['uuid']!=plan['device']['uuid']:raise ValueError('Resume requires the original GPU')
    binary=build.resolve()/('Release/attention.exe' if os.name=='nt' else 'attention')
    w=Workflow(out,device,resume=True)
    rows=[r for item in plan['outputs'] if item['phase']=='proposal' for r in schema.read(native/item['path']/'summary.csv')]
    collect_profiles(w,native,binary,plan,rows)
    publish(out);w.run('plots',[sys.executable,ROOT/'scripts/plot_combined.py',out])
    dump(out/'run_status.json',{'status':'completed','grid':plan['grid'],'device':plan['device'],'profiling_requested':True})
    if target:save_report_results(out,target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('part_a_%Y%m%dT%H%M%SZ'))
    p.add_argument('--build',type=Path,default=Path(tempfile.gettempdir())/'part_a-standard-build');p.add_argument('--arch',default='auto');p.add_argument('--device',default=device_default())
    p.add_argument('--publish',type=Path,help='Copy report tables and figures to this directory')
    p.add_argument('--grid',choices=['full','smoke'],default='full');p.add_argument('--no-profile',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze-only',type=Path)
    p.add_argument('--resume-profiles',type=Path,help='Resume profiling/export of a saved run without repeating timings')
    a=p.parse_args()
    if a.resume_profiles:
        resume_profiles(a.resume_profiles,a.build,a.device,a.publish);return
    if a.analyze_only:
        publish(a.analyze_only);subprocess.run([sys.executable,ROOT/'scripts/plot_combined.py',a.analyze_only],check=True)
        if a.publish:save_report_results(a.analyze_only,a.publish)
        return
    if not a.device or ',' in a.device:p.error('Select one device index or UUID')
    if a.grid=='smoke':a.no_profile=True
    arch=architecture(a.arch,a.device,a.dry_run);out=a.output.resolve();build=a.build.resolve();binary=build/('Release/attention.exe' if os.name=='nt' else 'attention');native=out/'raw/part_a'
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
        w.run('gpu_settings',['nvidia-smi','-i',a.device,'--query-gpu=power.limit,clocks.current.sm,clocks.current.memory,pstate,temperature.gpu','--format=csv'])
        w.run('configure',['cmake','-S',ROOT/'part_a','-B',build,*(['-G','Visual Studio 17 2022','-A','x64'] if os.name=='nt' else []),'-DPART_A_ENABLE_CUDA=ON','-DPART_A_PROFILE_INFO=ON','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}'])
        w.run('build',['cmake','--build',build,'--config','Release','--parallel','2'])
        w.run('shared_reference_contract',[build/('cpu_reference/Release/attention_reference_test.exe' if os.name=='nt' else 'cpu_reference/attention_reference_test')])
        for variant in ['all','wmma','ptx']:call('correctness_'+variant,['--grid','correctness','--variants',variant,'--mode','both'],'correctness')
        for pattern in ['uniform','constant_v','rising','stress']:
            call('pattern_'+pattern,['--shape',2,3,33,64,'--variants','all','--mode','both','--pattern',pattern],'correctness_pattern')
        for variant in ['optimized','wmma','ptx']:
            for tool in ['memcheck','racecheck','synccheck','initcheck']:
                w.run('safety_'+variant+'_'+tool,['compute-sanitizer','--tool',tool,'--error-exitcode','3',binary,'--shape',1,1,33,64,'--variants',variant,'--mode','both','--output',native/'safety'/f'{variant}_{tool}'])
        bh=[1,1] if a.grid=='smoke' else [4,8];ds=[64] if a.grid=='smoke' else [32,64,128]
        tuning_ns=[17,33] if a.grid=='smoke' else [256,1024];ns=[33,65] if a.grid=='smoke' else [256,512,1024,2048,4096]
        plan.update(B=bh[0],H=bh[1],d=ds,main_N=ns,tuning_N=tuning_ns)
        tuning=[]
        for variant in ['optimized','wmma','ptx']:
            for d in ds:
                for n in tuning_ns:tuning+=call(f'tuning_{variant}_{d}_{n}',['--grid','tuning','--shape',*bh,n,d,'--variants',variant,'--mode','both'],'tuning')
                for mode in ['dense','causal']:
                    rows=[r for r in tuning if r['variant']==variant and int(r['d'])==d and r['mode']==mode];groups={}
                    for r in rows:groups.setdefault((int(r['tile']),int(r['softmax_threads'])),[]).append(float(r['forward_median_ms']))
                    if a.dry_run:choice=(16,256)
                    else:
                        expected={(16,128),(16,256)} if variant in ['wmma','ptx'] else {(8,256),(16,256),(32,256),(16,128)}
                        if set(groups)!=expected or any(len(v)!=len(tuning_ns) for v in groups.values()):raise ValueError('Incomplete tuning grid')
                        choice=min(groups,key=lambda c:(sum(map(math.log,groups[c]))/len(groups[c]),c))
                    plan['selection'][f'{variant}:{mode}:d{d}']={'tile':choice[0],'softmax_threads':choice[1],'selection':'minimum_geometric_mean_forward_latency_across_tuning_N'}
        if not a.dry_run:dump(native/'frozen_selection.json',{'selection':plan['selection'],'tuning_N':tuning_ns,'frozen_utc':datetime.now(timezone.utc).isoformat()})
        main_rows=[]
        for variant in ['optimized','wmma','ptx']:
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
            if a.dry_run:print('profiles: 36 reports, QK/softmax/PV, full metrics, 3 repetitions; separate from timing samples')
            else:
                dump(native/'plan.json',plan)
                collect_profiles(w,native,binary,plan,main_rows)
        if not a.dry_run:
            dump(native/'plan.json',plan);publish(out)
            w.run('plots',[sys.executable,ROOT/'scripts/plot_combined.py',out])
            dump(out/'run_status.json',{'status':'completed','grid':a.grid,'device':plan['device'],'profiling_requested':not a.no_profile})
            if a.publish:save_report_results(out,a.publish)
        else:print('Dry run: no files written or commands executed. Configurations shown are placeholders until tuning runs.')
    except Exception as error:
        if not a.dry_run:dump(out/'run_status.json',{'status':'failed','error':str(error)})
        raise
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(1)
