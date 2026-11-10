"""Reproducible same-GPU Nsight profiling, isolated from latency benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('part_c/results/profile_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
p.add_argument('--build',type=Path,default=Path('/tmp/part_c-profile-build'))
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--resume',action='store_true',help='Resume only if source hashes and plan match')
a=p.parse_args()
if a.repeats<1: p.error('--repeats must be positive')
os.chdir(ROOT)
out=a.output.resolve();build=a.build.resolve()
if out.exists() and not a.resume: p.error('output already exists; use a fresh directory or --resume')
out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
sources=[Path('part_c/CMakeLists.txt'),Path('part_c/tests/kernel_runner.cpp'),Path('part_c/tests/test_support.cpp'),Path('part_c/tests/test_support.h'),Path('part_c/profile.sh')]
sources+=sorted(Path('part_c/include/flash_attention').glob('*.h'))
sources+=[Path('part_c/src')/f for f in ['flash_attention_1.cu','flash_attention_1_tensor_core.cu','flash_attention_2_cuda_core.cu','flash_attention_2_tensor_core.cu']]
sources += [Path('part_c/scripts')/f for f in ['profile_kernels.py','analyze_profiles.py']]
sources += [Path('part_a')/name for name in ['common/input_generation.hpp','common/attention_types.hpp','cpu_reference/attention_reference.cpp','cpu_reference/sampled_reference.cpp','cpu_reference/CMakeLists.txt']]
hashes={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
plan={'fa1_implementation':'kv_outer_shared_fp32_v2','fa1_grid':'one_CTA_per_batch_head','fa1_wmma_implementation':'kv_outer_wmma_v1','fa1_baselines':['fa1','fa1_wmma'],'kernels':['fa1','fa1_wmma','scalar','warp','tiled','skip','vector','wmma','ptx'],'shapes':[[4,8,256,64],[4,8,4096,64]],'repetitions':a.repeats,'seed':5206,'warmups':5,'validation_launches':1,'profiled_launches':1,'launch_skip':6,'sections':'full','cache_control':'all','clock_control':'none','architecture':86,'source_hashes':hashes}
manifest=out/'plan.json'
if a.resume:
    if not manifest.exists() or json.loads(manifest.read_text())!=plan: p.error('resume plan/source mismatch')
else:
    manifest.write_text(json.dumps(plan,indent=2)+'\n')
    for s in sources:
        dest=out/'source_snapshot'/s;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,dest)
    (out/'invocation.txt').write_text(shlex.join(['bash','part_c/profile.sh',*sys.argv[1:]])+'\n')

def run(label,cmd,stdout_path=None):
    cmd=list(map(str,cmd))
    print(label,flush=True)
    with (out/'commands.jsonl').open('a') as f:
        f.write(json.dumps({'label':label,'argv':cmd,'cwd':str(ROOT),'started_utc':datetime.now(timezone.utc).isoformat()})+'\n')
    if stdout_path:
        with stdout_path.open('w') as stdout, (out/'logs'/f'{label}.log').open('w') as log:
            status=subprocess.run(cmd,stdout=stdout,stderr=log).returncode
    else:
        with (out/'logs'/f'{label}.log').open('w') as log: status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'label':label,'exit_code':status,'finished_utc':datetime.now(timezone.utc).isoformat()})+'\n')
    if status: raise SystemExit(f'{label} failed with exit {status}; see logs (no result substituted)')

for label,cmd in [('gpu',['nvidia-smi','-q']),('nvcc',['nvcc','--version']),('host',['c++','--version']),('ncu',['ncu','--version']),('cmake',['cmake','--version'])]:run(label,cmd)
run('configure',['cmake','-S','part_c','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86','-DBUILD_TESTING=ON'])
run('build',['cmake','--build',build,'--parallel','2'])
binary=build/'part_c_runner'
binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
(out/'binary.json').write_text(json.dumps({'path':str(binary),'sha256':binary_hash},indent=2)+'\n')
# Do not mix new launch-mode bugs with a hardware-counter interpretation.
if not (out/'correctness_done.json').exists():
    run('correctness',[binary,'--suite','correctness','--output',out/'correctness'])
    (out/'correctness_done.json').write_text(json.dumps({'binary_sha256':binary_hash})+'\n')
extra_metrics=','.join('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum' for op in ['ffma','fadd','fmul'])
for b,h,n,d in plan['shapes']:
    for kernel in plan['kernels']:
        for rep in range(a.repeats):
            label=f'{kernel}_n{n}_d{d}_r{rep}'
            case=out/'cases'/label;case.mkdir(parents=True,exist_ok=True)
            if (case/'complete.json').exists(): continue
            prefix=case/'profile'
            cmd=['ncu','--set','full','--metrics',extra_metrics,'--target-processes','all','--launch-skip','6','--launch-count','1','--cache-control','all','--clock-control','none','--export',prefix,'--force-overwrite',binary,'--suite','profile','--kernel',kernel,'--shape',b,h,n,d,'--output',case/'validation']
            run(label,cmd)
            run(label+'_raw',['ncu','--import',str(prefix)+'.ncu-rep','--page','raw','--csv','--print-units','base'],case/'raw.csv')
            run(label+'_details',['ncu','--import',str(prefix)+'.ncu-rep','--page','details','--csv'],case/'details.csv')
            (case/'complete.json').write_text(json.dumps({'kernel':kernel,'B':b,'H':h,'N':n,'d':d,'rep':rep,'binary_sha256':binary_hash},indent=2)+'\n')
run('analysis',['python3','part_c/scripts/analyze_profiles.py',out])
print(f'Complete: {out}/ANALYSIS.md',flush=True)
