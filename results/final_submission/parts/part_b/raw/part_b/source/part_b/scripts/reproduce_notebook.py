#!/usr/bin/env python3
"""Replay the recorded B case order, inputs and launch configs on another GPU.

Diagnostic cross-check: kernels and the independent FP64 oracle run locally.
No tuning winner is reselected and no reference GPU identity/timings are reused
as local measurements. Root safety/profiling acceptance remains independent.
"""
import argparse,copy,json,math,os,shlex,shutil,statistics,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import results_schema as schema
from workflow import architecture,device_default,device_info,dump,sha
from export_results import export,config_name
from run_standard import runner_args

PIN_FILE=ROOT/'part_b/EXECUTION_REFERENCE.json'

def load_reference(grid):
    pin=json.loads(PIN_FILE.read_text())
    ref=ROOT/pin['bundles'][grid]
    manifest=json.loads((ref/'manifest.json').read_text())
    index=json.loads((ref/'raw/part_b/index.json').read_text())
    for name,digest in pin['measured_core_sources'].items():
        canonical=__import__('hashlib').sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if sha(ROOT/name)!=digest and canonical!=pin['canonical_core_sources'][name]:raise ValueError('Measured B source changed: '+name)
    for name,digest in pin['shared_sources'].items():
        canonical=__import__('hashlib').sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if sha(ROOT/name)!=digest and canonical!=pin['canonical_shared_sources'][name]:raise ValueError('Shared reference/input/schema dependency changed: '+name)
    if sha(ref/'manifest.json')!=pin['bundle_manifest_sha256'][grid]:raise ValueError('Reference manifest changed')
    for name,digest in index['evidence_hashes'].items():
        if sha(ref/name)!=digest:raise ValueError('Reference native evidence changed: '+name)
    tables={name:schema.read(ref/(name+'.csv')) for name in schema.HEADERS}
    schema.validate(tables)
    for row in tables['correctness']:
        if row['status']!='PASS':raise ValueError('Reference has a numerical failure')
    expected=pin['expected_counts'][grid]
    if len(index['cases'])!=expected['correctness'] or len(tables['samples'])!=expected['samples']:
        raise ValueError('Reference case/sample counts changed')
    return pin,ref,manifest,index,tables

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',choices=['smoke','full'],default='smoke')
    parser.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('b_replay_%Y%m%dT%H%M%SZ'))
    parser.add_argument('--build',type=Path,default=Path(tempfile.gettempdir())/'part-b-replay-build')
    parser.add_argument('--device',default=device_default());parser.add_argument('--arch',default='auto')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--check-only',action='store_true',help='Verify pinned source and recorded data without GPU/build')
    args=parser.parse_args()
    pin,ref,old_manifest,old_index,tables=load_reference(args.grid)
    print('Reference:',ref,flush=True)
    print('Pinned source, shared dependencies and recorded data: PASS',flush=True)
    print('Exact recorded cases:',len(old_index['cases']),'; timing samples:',len(tables['samples']),flush=True)
    if args.check_only:return 0
    if not args.device or ',' in args.device:parser.error('Select exactly one physical GPU index or UUID')
    out=args.output.resolve();build=args.build.resolve()
    if out.exists():raise ValueError('Choose a fresh output directory')
    if args.dry_run:
        print('Dry run: build the pinned implementation, run the shared oracle test, replay cases in reference order, validate local output, export seven tables and compare fingerprints. No files created or GPU accessed.')
        return 0
    arch=architecture(args.arch,args.device)
    for name in ['cmake','nvcc','nvidia-smi']:
        if not shutil.which(name):raise ValueError('Missing tool: '+name)
    for package in ['numpy','matplotlib']:
        if __import__('importlib.util',fromlist=['find_spec']).find_spec(package) is None:raise ValueError('Missing Python package: '+package)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.device)
    (out/'logs').mkdir(parents=True)
    native=out/'raw/part_b';native.mkdir(parents=True);(out/'configs').mkdir()
    local_index={'cases':[],'profiles':[],'sanitizers':[],'evidence_hashes':{}}
    comparison=[]
    dump(out/'run_status.json',{'status':'running_exact_replay','grid':args.grid,'profiling_requested':False})
    def run(label,argv):
        argv=list(map(str,argv));print(label+': '+shlex.join(argv),flush=True)
        started=datetime.now(timezone.utc).isoformat()
        with (out/'logs'/(label+'.txt')).open('w') as stream:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'label':label,'argv':argv,'cwd':str(ROOT),'started_utc':started,'exit_code':result.returncode,'CUDA_VISIBLE_DEVICES':args.device})+'\n')
        if result.returncode:raise RuntimeError(label+' failed; inspect its log')
    def persist():dump(native/'index.json',local_index)
    try:
        device=device_info(args.device);dump(out/'device.json',device)
        run('gpu',['nvidia-smi','-q']);run('nvcc',['nvcc','--version'])
        run('configure',['cmake','-S',ROOT/'part_b','-B',build,'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={arch}'])
        run('build',['cmake','--build',build,'--config','Release','--parallel','2'])
        suffix='.exe' if os.name=='nt' else ''
        def executable(name,sub=''):
            paths=[build/sub/(name+suffix),build/sub/'Release'/(name+suffix)]
            return next((p for p in paths if p.is_file()),paths[0])
        executable_path=executable('part_b_runner')
        run('shared_reference',[executable('attention_reference_test','shared_reference')])
        frozen=json.loads((ref/'configs/part_b.json').read_text())
        frozen['reference_tuning_scores']=frozen.pop('scores',{})
        frozen.update(selection='replayed fixed selection from the recorded notebook; no local winner reselection',
                      reference_run_id=old_manifest['run_id'],reference_gpu=old_manifest['device'],frozen_configuration_replay=True)
        dump(out/'configs/part_b.json',frozen)
        local_index['evidence_hashes']['configs/part_b.json']=sha(out/'configs/part_b.json')
        for number,entry in enumerate(old_index['cases']):
            original=json.loads((ref/entry['path']).read_text())
            path=out/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
            config={key:original[key] for key in ['variant','warps','key_tile']}
            timing=bool(original['samples_ms']);sampled=original['reference_scope']=='sampled'
            local_index['cases'].append(copy.deepcopy(entry))
            try:
                run(f'case_{number:04d}',runner_args(executable_path,config,original['B'],original['H'],original['N'],original['d'],original['pattern'],sampled,timing,path))
            finally:
                if path.exists():local_index['evidence_hashes'][entry['path']]=sha(path)
                persist()
            local=json.loads(path.read_text())
            required=['B','H','N','d','variant','warps','key_tile','pattern','input_digest','reference_scope','checked_query_rows','checked_elements']
            for key in required:
                if local[key]!=original[key]:raise ValueError(f'Exact replay mismatch in case {number}: {key}')
            if local['status']!='PASS' or local['failures'] or local['nonfinite'] or local.get('materialised_failures',0):raise ValueError(f'Numerical failure in case {number}')
            if len(local['samples_ms'])!=len(original['samples_ms']):raise ValueError('Timing repetition count changed')
            comparison.append({'case':entry['path'],'experiment':entry['experiment'],'variant':entry['public_variant'],
                'B':local['B'],'H':local['H'],'N':local['N'],'d':local['d'],'config':config_name(config),
                'input_digest_match':True,'local_status':local['status'],'reference_max_absolute':original['max_absolute'],'local_max_absolute':local['max_absolute'],
                'reference_forward_median_ms':statistics.median(original['samples_ms']) if timing else '',
                'local_forward_median_ms':statistics.median(local['samples_ms']) if timing else ''})
        snapshot=native/'source'
        for name in pin['measured_core_sources']:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        new_manifest=copy.deepcopy(old_manifest)
        new_manifest.update(run_id=out.name,device=device,profile_requested=False,sanitizers_requested=False,
            binary_sha256=sha(executable_path),source_hashes={name:sha(ROOT/name) for name in pin['measured_core_sources']},shared_source_hashes={name:sha(ROOT/name) for name in pin['shared_sources']},
            reference_run_id=old_manifest['run_id'],reference_gpu=old_manifest['device'],frozen_configuration_replay=True,
            replay_script_sha256=sha(__file__),reference_manifest_sha256=sha(ref/'manifest.json'))
        dump(out/'manifest.json',new_manifest);persist()
        audit=export(out)
        import csv
        with (out/'comparison_to_reference.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
        crosscheck={'numerical_replay_passed':True,'pinned_source_verified':True,'shared_dependencies_verified':True,
            'all_recorded_cases_replayed':len(comparison)==len(old_index['cases']),'input_fingerprints_matched':True,
            'cases':len(comparison),'samples':sum(len(json.loads((out/e['path']).read_text())['samples_ms']) for e in local_index['cases']),
            'reference_run':old_manifest['run_id'],'reference_gpu':old_manifest['device'],'local_gpu':device,
            'selection':'fixed reference configurations; no local retuning','overall_evidence_audit_passed':audit['passed'],
            'sanitizers_and_profiles':'not performed by exact-replay workflow',
            'timing_interpretation':'reference and local timings are separate observations; cross-GPU differences are not optimisation speedups'}
        dump(out/'crosscheck.json',crosscheck)
        note='# Exact notebook replay\n\nThis local run replayed the recorded case order, shapes, patterns, input fingerprints, precision, reference scopes and launch configurations. No local tuning winner was selected. Reference tuning scores are retained separately and are not local measurements.\n\nThe seven canonical tables are local measurements. comparison_to_reference.csv keeps original and local timings in separate columns; differences across systems are not an optimisation speedup. Numerical outputs are checked against the local linked FP64 oracle with the unchanged tolerance; bitwise output/timing/error equality across compilers is not required.\n\nThis is a diagnostic cross-check. Sanitizers and hardware profiles were not run; audit.passed remains false and root aggregation must reject it for final acceptance. Use the normal part_b/run.sh and root all-parts workflow for fresh accepted safety/profiling/comparison evidence.\n'
        (out/'REPLAY.md').write_text(note)
        for name in ['REPORT.md','ANALYSIS.md']:
            p=out/name;p.write_text(note+'\n---\n\n'+p.read_text())
        dump(out/'run_status.json',{'status':'completed_exact_numerical_replay','grid':args.grid,'profiling_requested':False,'overall_evidence_accepted':False})
        print(json.dumps(crosscheck,indent=2));return 0
    except Exception as error:
        persist();dump(out/'crosscheck.json',{'numerical_replay_passed':False,'error':str(error),'completed_cases':len(comparison)})
        dump(out/'audit.json',{'passed':False,'error':str(error)})
        dump(out/'run_status.json',{'status':'failed_exact_replay','error':str(error)})
        raise

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError) as error:print('ERROR: '+str(error),file=sys.stderr);sys.exit(1)
