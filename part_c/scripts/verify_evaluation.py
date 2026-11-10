#!/usr/bin/env python3
"""Regression plus sanitizer checks of both FA1 baselines and frozen d=64 FA2 winners."""
import argparse,csv,hashlib,json,subprocess
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--build',type=Path,default=Path('/tmp/part_c-evaluation-build'));a=p.parse_args();out=a.study/'verification'
if out.exists():raise SystemExit('Verification directory exists; keep historical checks intact')
out.mkdir(parents=True);selection=json.loads((a.study/'frozen_selection.json').read_text())
def run(cmd,label):
    with (out/f'{label}.txt').open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':list(map(str,cmd)),'exit_code':r.returncode})+'\n')
    if r.returncode:raise RuntimeError(f'{label} failed: see {out}')
run([a.build/'part_c_runner','--suite','correctness','--output',out/'regression'],'regression')
plan=json.loads((a.study/'plan.json').read_text())
targets=[(k,selection['variants'][k]['64']['config'] if k in selection['variants'] else 0) for k in plan.get('fa1_baselines',['fa1'])]+[(selection['families'][family]['64'],selection['variants'][selection['families'][family]['64']]['64']['config']) for family in ['cuda_core','tensor_core']]
if 'fa1_ptx' in plan.get('fa1_baselines',[]):targets[-1]=('ptx',selection['variants']['ptx']['64']['config'])
for k,c in targets:
    for tool in ['memcheck','racecheck','synccheck','initcheck']:
        label=f'{k}_c{c}_{tool}'
        run(['compute-sanitizer','--tool',tool,'--error-exitcode','3',a.build/'part_c_evaluation',k,c,1,1,33,64,'random','validate',out/f'{label}.csv'],label)
with (out/'regression/correctness.csv').open() as f:rows=list(csv.DictReader(f))
declared=json.loads((out/'regression/run_manifest.json').read_text())['kernels']
assert set(declared)=={'fa1','fa1_ptx','scalar','warp','tiled','skip','vector','wmma','ptx'}
workloads={tuple(r[key] for key in ['B','H','N','d','pattern']) for r in rows}
assert len(rows)==len(declared)*len(workloads)
for kernel in declared:
    checks=[r for r in rows if r['kernel']==kernel]
    assert len(checks)==len(workloads)
    assert {tuple(r[key] for key in ['B','H','N','d','pattern']) for r in checks}==workloads
    for r in checks:
        unsupported=kernel in ['ptx','fa1_ptx'] and (int(r['d'])<16 or int(r['d'])%16!=0)
        assert r['status']==('UNSUPPORTED' if unsupported else 'PASS')
counts=Counter(r['status'] for r in rows)
sources=json.loads((Path(__file__).resolve().parents[1]/'tests/kernel_bodies.json').read_text())
for name,expected in sources.items():
    body=(Path(__file__).resolve().parents[1]/'src'/name).read_text().split('\nnamespace part_c {',1)[0]
    assert hashlib.sha256(body.encode()).hexdigest()==expected,'Kernel differs from declared current implementation'
(out/'audit.json').write_text(json.dumps({'default_regression_passes':counts['PASS'],'explicit_unsupported':counts['UNSUPPORTED'],'kernels':declared,'sanitizer_runs':4*len(targets),'sanitizer_failures':0,'kernel_bodies_match_current_manifest':True,'fa1_implementation':'kv_outer_shared_fp32_v2','fa2_kernel_bodies_preserved':True,'fa1_ptx_implementation':'kv_outer_ptx_v1','scope':'B1 H1 N33 d64 FA1 baselines and selected FA2 core/PTX; representative, not exhaustive sanitizer coverage'},indent=2)+'\n')
print(out/'audit.json')
