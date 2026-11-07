#!/usr/bin/env python3
"""Regression plus sanitizer checks of the frozen d=64 family winners."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--build',type=Path,default=Path('/tmp/part_c-evaluation-build'));a=p.parse_args();out=a.study/'extra/verification'
if out.exists():raise SystemExit('Verification directory exists; keep historical checks intact')
out.mkdir(parents=True);selection=json.loads((a.study/'frozen_selection.json').read_text())
def run(cmd,label):
    with (out/f'{label}.txt').open('w') as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
    with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'argv':list(map(str,cmd)),'exit_code':r.returncode})+'\n')
    if r.returncode:raise RuntimeError(f'{label} failed: see {out}')
run([a.build/'part_c_runner','--suite','correctness','--output',out/'regression'],'regression')
for family in ['cuda_core','tensor_core']:
    k=selection['families'][family]['64'];c=selection['variants'][k]['64']['config']
    for tool in ['memcheck','racecheck','synccheck','initcheck']:
        label=f'{k}_c{c}_{tool}'
        run(['compute-sanitizer','--tool',tool,'--error-exitcode','3',a.build/'part_c_evaluation',k,c,1,1,33,64,'random','validate',out/f'{label}.csv'],label)
with (out/'regression/correctness.csv').open() as f:rows=list(csv.DictReader(f))
assert sum(r['status']=='PASS' for r in rows)==380
assert sum(r['status']=='UNSUPPORTED' for r in rows)==4
sources=json.loads((Path(__file__).resolve().parents[1]/'docs/kernel_provenance/evaluation_kernel_bodies.json').read_text())
for name,expected in sources.items():
    body=(Path(__file__).resolve().parents[1]/'src'/name).read_text().split('\nnamespace part_c {',1)[0]
    assert hashlib.sha256(body.encode()).hexdigest()==expected,'Kernel algorithm changed'
(out/'audit.json').write_text(json.dumps({'default_regression_passes':380,'explicit_unsupported':4,'sanitizer_runs':8,'sanitizer_failures':0,'kernel_bodies_unchanged':True,'scope':'B1 H1 N33 d64 frozen family winners; representative, not exhaustive sanitizer coverage'},indent=2)+'\n')
print(out/'audit.json')
