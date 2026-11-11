#!/usr/bin/env python3
"""Standalone Part C workflow with Part A-based public tables and full raw evidence."""
import argparse
import importlib.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workflow import architecture, device_info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results')/datetime.now(timezone.utc).strftime('part_c_%Y%m%dT%H%M%SZ'))
    p.add_argument('--build',type=Path,default=Path('/tmp/part_c-standard-build'))
    p.add_argument('--arch',default='auto');p.add_argument('--device',default=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0],help='physical CUDA index or UUID; selected through CUDA_VISIBLE_DEVICES')
    p.add_argument('--grid',choices=['full','smoke'],default='full')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-profile',action='store_true',help='explicit partial run without fresh hardware counters')
    p.add_argument('--supplementary-profile',type=Path,action='append',default=[])
    p.add_argument('--analyze-only',type=Path,metavar='BUNDLE')
    a=p.parse_args()
    exporter=ROOT/'part_c/scripts/export_results.py'
    if a.analyze_only:
        subprocess.run([sys.executable,str(exporter),'--analyze-only',str(a.analyze_only.resolve())],check=True);return
    if a.grid=='smoke':a.no_profile=True
    a.arch=architecture(a.arch,a.device,a.dry_run)
    if not a.device or ',' in a.device:p.error('select one CUDA device index or UUID')
    out=a.output.resolve();build=a.build.resolve()
    if out.exists():p.error('choose a fresh output directory')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.device,PART_C_BUILD_DIR=str(build),PART_C_CUDA_ARCHITECTURES=a.arch,ATTENTION_GRID=a.grid)
    native=out/'raw/part_c/study'
    def run(label,argv):
        argv=list(map(str,argv));print(f'{label}: {shlex.join(argv)}',flush=True)
        if a.dry_run:return
        with (out/'logs'/f'{label}.txt').open('w') as stream:
            r=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        with (out/'commands.jsonl').open('a') as stream:stream.write(json.dumps({'label':label,'argv':argv,'exit_code':r.returncode,'CUDA_VISIBLE_DEVICES':a.device})+'\n')
        if r.returncode:raise RuntimeError(f'{label} failed; see logs/{label}.txt')
    state={'part':'part_c','status':'running','device':a.device,'architecture':a.arch,'profiling_requested':not a.no_profile,'shared_oracle_integration_pending':False,'grid':a.grid}
    if not a.dry_run:
        for package in ['numpy','matplotlib']:
            if importlib.util.find_spec(package) is None:raise RuntimeError('missing Python dependency: '+package)
        tools=['cmake','nvcc','nvidia-smi','compute-sanitizer']+([] if a.no_profile else ['ncu'])
        for tool in tools:
            if not shutil.which(tool):raise RuntimeError(f'missing dependency: {tool}')
        (out/'logs').mkdir(parents=True)
        (out/'device.json').write_text(json.dumps(device_info(a.device),indent=2)+'\n')
    try:
        run('gpu',['nvidia-smi','-q'])
        run('configure',['cmake','-S',ROOT/'part_c','-B',build,'-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_CUDA_ARCHITECTURES={a.arch}'])
        run('build',['cmake','--build',build,'--parallel','2'])
        run('shared_reference_contract',[build/'shared_reference/attention_reference_test'])
        run('fa1_invariants',[build/'part_c_fa1_test','--output',out/'raw/part_c/fa1_invariants'])
        for config in range(4):
            run(f'fa1_ptx_c{config}_invariants',[build/'part_c_fa1_test','--ptx','--config',str(config),'--output',out/f'raw/part_c/fa1_ptx_c{config}_invariants'])
        run('input_contract',[build/'part_c_input_contract'])
        run('default_correctness',[build/'part_c_runner','--suite','correctness','--output',out/'raw/part_c/default_correctness'])
        for tool in ['memcheck','racecheck','synccheck','initcheck']:
            run(tool,['compute-sanitizer','--tool',tool,'--error-exitcode','3',build/'part_c_runner','--suite','smoke','--output',out/'raw/part_c/safety'/tool])
        run('study',['bash',ROOT/'part_c/evaluate.sh',native])
        run('frozen_safety',['bash',ROOT/'part_c/evaluate.sh','--verify',native])
        if not a.no_profile:run('frozen_profiles',['bash',ROOT/'part_c/profile.sh','--frozen',native,'--build',build])
        command=[sys.executable,exporter,'--study',native,'--output',out,'--in-place']
        for folder in a.supplementary_profile:command+=['--supplementary-profile',folder.resolve()]
        run('standardize',command)
        if not a.dry_run:
            state['status']='completed_part_c' if not a.no_profile else 'completed_without_fresh_profile_evidence'
            (out/'run_status.json').write_text(json.dumps(state,indent=2)+'\n')
        else:print('Dry run only: no commands executed, results created or GPU benchmarks run.')
    except Exception as error:
        state.update(status='failed',error=str(error))
        if not a.dry_run:(out/'run_status.json').write_text(json.dumps(state,indent=2)+'\n')
        raise


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}',file=sys.stderr);sys.exit(1)
