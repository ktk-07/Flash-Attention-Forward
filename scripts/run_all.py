#!/usr/bin/env python3
"""Sequential one-machine evaluation with strict canonical aggregation and offline plots."""
import argparse,csv,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import results_schema as schema
from workflow import Workflow,architecture,device_default,device_info,dump,sha
from plot_combined import generate
ROOT=Path(__file__).resolve().parents[1]
PARTS=['part_a','part_b','part_c']

def load_tables(folder):
    result={}
    for table,header in schema.HEADERS.items():
        with (folder/f'{table}.csv').open(newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=header:raise ValueError(f'{folder}: incompatible {table}.csv header')
            result[table]=list(reader)
    schema.validate(result);return result

def aggregate(out,requested,grid):
    tables={k:[] for k in schema.HEADERS};devices={};inputs={};effective={};source_hashes={};counts={};profile_requested={};profile_captures={}
    for part in requested:
        bundle=out/'parts'/part;native=load_tables(bundle);manifest=json.loads((bundle/'manifest.json').read_text());audit=json.loads((bundle/'audit.json').read_text())
        if manifest.get('schema_version')!=schema.VERSION:raise ValueError(part+' schema version mismatch')
        if not audit.get('passed'):raise ValueError(part+' audit did not pass')
        linked=manifest.get('shared_cpu_reference_linked',audit.get('shared_cpu_reference_linked',False))
        if not linked:raise ValueError(part+' has not linked the shared FP64 oracle')
        device=json.loads((bundle/'device.json').read_text()) if (bundle/'device.json').exists() else manifest.get('device',{})
        if not device.get('uuid'):raise ValueError(part+' lacks a physical GPU UUID')
        status=json.loads((bundle/'run_status.json').read_text()) if (bundle/'run_status.json').exists() else {}
        profile_requested[part]=manifest.get('profile_requested',status.get('profiling_requested',False))
        profile_captures[part]=audit.get('profile_captures',0)
        if profile_requested[part] and profile_captures[part] < {'part_a':108,'part_b':6,'part_c':12}[part]:raise ValueError(part+' lacks requested fresh profiling captures')
        devices[part]=device
        if len({d['uuid'] for d in devices.values()})!=1:raise ValueError('Cannot combine measurements from different physical GPUs')
        main=[r for r in native['summary'] if r['experiment']=='proposal'];counts[part]=len(main)
        if not main:raise ValueError(part+' lacks main-grid measurements')
        ns=[33,65] if grid=='smoke' else [256,512,1024,2048,4096];ds=[64] if grid=='smoke' else [32,64,128];b,h=(1,1) if grid=='smoke' else (4,8)
        modes=['dense','causal'] if part=='part_a' else ['dense'] if part=='part_b' else ['causal']
        expected={(str(b),str(h),str(n),str(d),mode) for n in ns for d in ds for mode in modes}
        for precision in ['FP32','FP16'] if part!='part_b' else ['FP32']:
            found={tuple(r[k] for k in ['B','H','N','d','mode']) for r in main if r['precision']==precision}
            if found!=expected:raise ValueError(f'{part}: incomplete {precision} {grid} grid')
            required=(['naive','tiled','parallel','optimized'] if precision=='FP32' else ['wmma','ptx']) if part=='part_a' else (['skip','vector'] if precision=='FP32' else ['wmma','ptx']) if part=='part_c' else ['selected']
            for variant in required:
                covered={tuple(r[k] for k in ['B','H','N','d','mode']) for r in main if r['precision']==precision and r['variant']==variant}
                if covered!=expected:raise ValueError(f'{part}: incomplete required variant {variant}')
        for row in main:
            key=tuple(row[k] for k in ['B','H','N','d','pattern','generator','seed'])
            if key in inputs and inputs[key]!=row['input_digest']:raise ValueError('Shared input fingerprint mismatch')
            inputs[key]=row['input_digest']
            if row['precision']=='FP16':
                if key in effective and effective[key]!=row['rounded_input_digest']:raise ValueError('Rounded input fingerprint mismatch')
                effective[key]=row['rounded_input_digest']
        for name,rows in native.items():
            source_hashes[f'parts/{part}/{name}.csv']=sha(bundle/f'{name}.csv')
            for old in rows:
                row=dict(old);row['run_id']=out.name;row['case_id']=schema.case_id(row)
                if name=='profiles':
                    path=Path(row['source_report'])
                    if path.is_absolute() or '..' in path.parts or not (bundle/path).is_file():raise ValueError('Invalid profile evidence path')
                    row['source_report']=str(Path('parts')/part/path)
                tables[name].append(row)
        configs=bundle/'configs';(out/'configs').mkdir(exist_ok=True)
        for path in configs.glob('*.json'):shutil.copy2(path,out/'configs'/(part+'.json' if path.name==part+'.json' else part+'_'+path.name))
        for name in ['plots','supplementary']:
            source=bundle/name/part
            if source.exists():shutil.copytree(source,out/name/part,dirs_exist_ok=True)
    audit=schema.validate(tables)
    audit.update(requested_parts=requested,requested_parts_complete=True,all_three_parts_present=set(requested)==set(PARTS),grid=grid,final_report_grid=grid=='full',main_rows=counts,profiling_requested_by_part=profile_requested,profile_captures=profile_captures,shared_cpu_reference_linked=True,physical_gpu_uuid=next(iter(devices.values()))['uuid'],shared_input_fingerprints_verified=True,profiling_parts_with_counters=sorted({r['part'] for r in tables['profiles'] if r['availability']=='available'}))
    for name,rows in tables.items():schema.write(out/f'{name}.csv',name,rows)
    dump(out/'aggregation_audit.json',audit);dump(out/'audit.json',audit)
    dump(out/'manifest.json',{'schema_version':schema.VERSION,'requested_parts':requested,'grid':grid,'devices':devices,'source_tables':source_hashes,'missing_parts':[p for p in PARTS if p not in requested],'processing_sources':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},'generated_utc':datetime.now(timezone.utc).isoformat()})
    generate(out)
    (out/'ANALYSIS.md').write_text('# Combined evaluation\n\nRequested parts: '+', '.join(requested)+'. Grid: '+grid+'.\n\nAll accepted timing rows have output validation against the shared Part A FP64 oracle. Benchmark checks use three query rows per head plus all-output finiteness checks; Part A additionally checks full output for small shapes and N=256,d=64. Small correctness suites compare full outputs.\n\nInput fingerprints and physical GPU UUIDs agree. FP32 A/C comparisons match numerical precision. FP16 comparisons use identical rounded inputs and FP32 accumulation/output in A and C. A rounds normalized P; C rounds tile probabilities before online normalization. Configuration selection precedes the final grid; full-grid N=512/2048/4096 are held out from tuning.\n\nTwenty repetitions describe within-case variation, not independent-session uncertainty. Explicit allocation ledgers are not total process peaks. Fused stage timings are unavailable. Nsight instrumented timings are excluded. Missing parts and missing counters are declared in aggregation_audit.json; an A/C or smoke run is not the final A/B/C experiment. Native data, commands and additional insights remain in parts/.\n')
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parts',default='all',help='all or comma-separated a,b,c')
    p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ'))
    p.add_argument('--build-root',type=Path,default=Path('/tmp/attention-builds'));p.add_argument('--arch',default='auto');p.add_argument('--device',default=device_default())
    p.add_argument('--grid',choices=['full','smoke'],default='full');p.add_argument('--no-profile',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze-only',type=Path)
    a=p.parse_args()
    if a.analyze_only:
        out=a.analyze_only.resolve();manifest=json.loads((out/'manifest.json').read_text())
        for name,value in manifest['source_tables'].items():
            if sha(out/name)!=value:raise ValueError('Source table changed: '+name)
        for part in manifest['requested_parts']:
            subprocess.run(['bash',ROOT/part/'run.sh','--analyze-only',out/'parts'/part],check=True)
        for name,value in manifest['source_tables'].items():
            if sha(out/name)!=value:raise ValueError('Re-extraction changed source table: '+name)
        print(json.dumps(aggregate(out,manifest['requested_parts'],manifest['grid']),indent=2));return
    selected=PARTS if a.parts=='all' else ['part_'+x.strip().removeprefix('part_') for x in a.parts.split(',')]
    if not selected or len(set(selected))!=len(selected) or set(selected)-set(PARTS):p.error('Use --parts all or e.g. --parts a,c')
    selected=[part for part in PARTS if part in selected]
    for part in selected:
        if not (ROOT/part/'run.sh').is_file():p.error(f'{part}/run.sh is missing. Part B must implement the README contract. Use --parts a,c for an explicit A/C run.')
    if not a.device or ',' in a.device:p.error('Select exactly one GPU index or UUID')
    arch=architecture(a.arch,a.device,a.dry_run);out=a.output.resolve();w=Workflow(out,a.device,a.dry_run,not a.no_profile and a.grid!='smoke')
    try:
        w.run('gpu',['nvidia-smi','-q'])
        # Build both existing parts before measurement. Child runners reuse these builds.
        for part in selected:
            args=['cmake','-S',ROOT/part,'-B',a.build_root.resolve()/part,'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}']
            if part=='part_a':args+=['-DPART_A_ENABLE_CUDA=ON','-DPART_A_PROFILE_INFO=ON']
            w.run('configure_'+part,args);w.run('build_'+part,['cmake','--build',a.build_root.resolve()/part,'--parallel','2'])
        for part in selected:
            args=['bash',ROOT/part/'run.sh','--output',out/'parts'/part,'--build',a.build_root.resolve()/part,'--arch',arch,'--device',a.device,'--grid',a.grid]
            if a.no_profile:args+=['--no-profile']
            w.run(part,args)
        if not a.dry_run:print(json.dumps(aggregate(out,selected,a.grid),indent=2));dump(out/'run_status.json',{'status':'completed_requested_parts','parts':selected,'grid':a.grid})
        else:print('Dry run: no files created or commands executed. Default all requires Part B; explicit subsets are declared partial.')
    except Exception as e:
        if not a.dry_run:dump(out/'run_status.json',{'status':'failed','error':str(e)})
        raise
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(1)
