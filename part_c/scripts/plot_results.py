"""Plot measured standalone Part C CSVs; never substitute synthetic timings."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/part-c-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run
with (run / "summary.csv").open() as f:
    rows = list(csv.DictReader(f))
with (run / "correctness.csv").open() as f:
    checks = list(csv.DictReader(f))
if not rows:
    raise SystemExit("No benchmark results: plots require measured summary.csv rows")
if any(r["status"] != "PASS" for r in rows):
    raise SystemExit("Nonpassing timing rows are not eligible for plots")
out = run / "figures"
out.mkdir(exist_ok=True)
labels = {"fa1": "FA1 FP32 (see run version)", "fa1_wmma": "FA1 WMMA", "scalar": "FA2 scalar", "warp": "Warp/query", "tiled": "Register tiled + skip", "skip": "Skip snapshot", "vector": "Vector Q/K/V", "wmma": "Fused WMMA", "ptx": "Fused PTX MMA"}
colors = dict(zip(labels, plt.get_cmap("tab10").colors))

def save(fig, name):
    fig.savefig(out / (name + ".png"), dpi=180)
    fig.savefig(out / (name + ".pdf"))
    plt.close(fig)

def select(kernel, n=None, d=64):
    return [r for r in rows if r["kernel"] == kernel and int(r["d"]) == d and (n is None or int(r["N"]) == n)]

def progression(ids, name, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, n in zip(axes, [256, 1024, 4096]):
        data = [select(k, n)[0] for k in ids if select(k, n)]
        if not data:
            ax.text(.5,.5,"No measurements",ha="center")
            continue
        med = [float(r["median_ms"]) for r in data]
        err = [[m-float(r["q1_ms"]) for m,r in zip(med,data)], [float(r["q3_ms"])-m for m,r in zip(med,data)]]
        ax.bar(range(len(data)), med, yerr=err, capsize=3, color=[colors[r["kernel"]] for r in data])
        ax.set_xticks(range(len(data)), [labels[r["kernel"]] for r in data], rotation=40, ha="right")
        ax.set(title=f"N={n}, d=64", ylabel="Median forward latency (ms)", yscale="log")
    fig.suptitle(title + "\nB=4, H=8; fixed configurations; 5 warmups, 20 repeats; bars show Q1–Q3")
    save(fig, name)

progression(["fa1","scalar","warp","tiled","skip","vector"], "c1_cuda_core_progression", "CUDA-core progression — FP32 inputs / accumulation / output")
progression(["fa1_wmma","wmma","ptx"], "c2_tensor_core_progression", "Tensor Core progression — FP16 inputs; FP32 accumulation / output")

# Exploratory winner selection on the specified small tuning subset. This is
# explicitly not a frozen, separately tuned performance claim.
scores = {}
for k in labels:
    subset = [r for r in select(k) if int(r["N"]) in (256,1024)]
    if len(subset)==2:
        scores[k] = math.exp(sum(math.log(float(r["median_ms"])) for r in subset)/2)
cuda_ids=[k for k in scores if k not in ("fa1_wmma","wmma","ptx")]
tensor_ids=[k for k in scores if k in ("fa1_wmma","wmma","ptx")]
if not cuda_ids or not tensor_ids:
    raise SystemExit("Need measured CUDA and Tensor Core tuning-subset rows")
best_cuda=min(cuda_ids,key=lambda k:(scores[k],k))
best_tensor=min(tensor_ids,key=lambda k:(scores[k],k))
(run/"selection.json").write_text(json.dumps({"policy":"exploratory geometric mean on B4 H8 d64 N256/1024, fixed configurations; not independent tuning", "scores_ms":scores,"best_cuda":best_cuda,"best_tensor":best_tensor},indent=2)+"\n")
fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
for ax,d in zip(axes,[32,64,128]):
    for k in (best_cuda,best_tensor):
        data=sorted(select(k,d=d),key=lambda r:int(r["N"]))
        if not data:continue
        y=[float(r["median_ms"]) for r in data]
        ax.errorbar([int(r["N"]) for r in data],y,yerr=[[v-float(r["q1_ms"]) for v,r in zip(y,data)],[float(r["q3_ms"])-v for v,r in zip(y,data)]],marker="o",capsize=3,label=labels[k],color=colors[k])
    ax.set(title=f"d={d}",xlabel="Sequence length N",ylabel="Median forward latency (ms)")
    ax.set_xscale("log",base=2);ax.set_yscale("log");ax.legend()
fig.suptitle("Best observed CUDA-core vs Tensor Core (exploratory subset selection)\nB=4 H=8; FP32 vs FP16 inputs; both FP32 accumulation/output; precision and hardware change")
save(fig,"c3_best_cuda_vs_tensor")

fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
for ax,d in zip(axes,[32,64,128]):
    for k in labels:
        data=sorted(select(k,d=d),key=lambda r:int(r["N"]))
        if data:ax.plot([int(r["N"]) for r in data],[float(r["median_ms"]) for r in data],marker=".",label=labels[k],color=colors[k])
    ax.set(title=f"d={d}",xlabel="Sequence length N",ylabel="Median forward latency (ms)");ax.set_yscale("log");ax.legend(fontsize=7)
fig.suptitle("All Part C variants — causal attention, B=4 H=8; fixed launch configurations")
save(fig,"all_kernel_scaling")

fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
for k in labels:
    data=sorted([r for r in checks if r["kernel"]==k and r["status"]=="PASS" and r["d"]=="64"],key=lambda r:int(r["N"]))
    if data:ax.plot([int(r["N"]) for r in data],[max(float(r["max_absolute"]),1e-12) for r in data],marker="o",label=labels[k],color=colors[k])
ax.set(xlabel="Sequence length N",ylabel="Max absolute error on sampled query rows",yscale="log",title="Benchmark accuracy — CPU FP64 reference, d=64; all outputs checked finite")
ax.legend(fontsize=8)
save(fig,"sampled_numerical_error")
print(f"Saved five measured figures (PDF + PNG) to {out}; selected {best_cuda} and {best_tensor}")
