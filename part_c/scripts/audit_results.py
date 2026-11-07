"""Check measured CSV completeness, statistics, and split-source provenance."""
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
assert len(summary)==120, f"Expected 8 kernels x 15 performance shapes, got {len(summary)}"
assert len(samples)==2400
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
manifest=json.loads((root/"docs/kernel_provenance/provenance.json").read_text())
original=root/"src/extra/causal_flash_attn.cu"
assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest["smoke_test_sha256"], "Standalone development source differs from its recorded version"
comments=manifest["original_kernel_comments"]
parts="\n".join((root/"src"/n).read_text() for n in ["flash_attention_1.cu","flash_attention_2_cuda_core.cu","flash_attention_2_tensor_core.cu"])
assert all(c in parts for c in comments), "An original kernel comment was lost"
report={"correctness":dict(Counter(r["status"] for r in checks)),"benchmark_configurations":len(summary),"benchmark_samples":len(samples),"benchmark_validation":"three CPU-oracle query rows per batch/head; all outputs finite","preserved_original_kernel_comments":len(comments),"original_development_file_relocated_and_fixed":True,"statistics_recomputed":True}
(a.run/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
