"""Check measured CSV completeness, statistics, and maintained kernel bodies."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args()
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
checks=read(a.run/"correctness/correctness.csv")
summary=read(a.run/"benchmark/summary.csv")
samples=read(a.run/"benchmark/samples.csv")
bench_checks=read(a.run/"benchmark/correctness.csv")
manifest_file=a.run/"benchmark/run_manifest.json"
if manifest_file.exists():declared=json.loads(manifest_file.read_text())["kernels"]
else:declared=["fa1","scalar","warp","tiled","skip","vector","wmma","ptx"] # Historical workflow.
expected={(k,"4","8",str(n),str(d)) for k in declared for n in [256,512,1024,2048,4096] for d in [32,64,128]}
actual={(r["kernel"],r["B"],r["H"],r["N"],r["d"]) for r in summary}
assert len(summary)==len(expected) and actual==expected,"Incomplete benchmark grid"
assert len(samples)==20*len(expected)
assert all(r["status"] in ("PASS","UNSUPPORTED") for r in checks)
assert all(r["status"]=="PASS" for r in bench_checks)
identity=["kernel","B","H","N","d","pattern"]
def key(r):return tuple(r[k] for k in identity)
grouped=defaultdict(list)
for r in samples:grouped[key(r)].append(r)
assert len(grouped)==len(summary)
def quantile(v,p):
    z=(len(v)-1)*p;l=int(z);h=min(l+1,len(v)-1)
    return v[l]+(z-l)*(v[h]-v[l])
for r in summary:
    group=grouped[key(r)]
    assert len(group)==20 and {int(x["repeat"]) for x in group}==set(range(20))
    values=sorted(float(x["forward_ms"]) for x in group)
    assert all(math.isfinite(x) and x>0 for x in values)
    for column,pct in [("median_ms",.5),("q1_ms",.25),("q3_ms",.75)]:
        assert math.isclose(float(r[column]),quantile(values,pct),rel_tol=1e-12)
    assert math.isclose(float(r["iqr_ms"]),float(r["q3_ms"])-float(r["q1_ms"]),abs_tol=1e-12)
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/"tests/kernel_bodies.json").read_text())
for name,expected in manifest.items():
    body=(root/"src"/name).read_text().split("\nnamespace part_c {",1)[0]
    assert hashlib.sha256(body.encode()).hexdigest()==expected,"Kernel body differs from the recorded baseline"
report={"correctness":dict(Counter(r["status"] for r in checks)),"benchmark_configurations":len(summary),"benchmark_samples":len(samples),"benchmark_validation":"three CPU-oracle query rows per batch/head; all outputs finite","kernel_bodies_match_current_manifest":True,"statistics_recomputed":True}
(a.run/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
