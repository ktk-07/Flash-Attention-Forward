import argparse
import csv
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Plot Part A performance results")
parser.add_argument("runs", type=Path, nargs="?", default=Path("results"))
parser.add_argument("--output", type=Path)
parser.add_argument("--select-config", action="store_true")
args = parser.parse_args()
args.output = args.output or args.runs / "figures"
args.output.mkdir(parents=True, exist_ok=True)
with (args.runs / "summary.csv").open(newline="") as stream:
    summary = list(csv.DictReader(stream))

# choose the configuration
if args.select_config:
    tuning = [r for r in summary if r["experiment"] == "tuning" and r["precision"] == "FP32"]
    scores = {}
    for row in tuning:
        key = (int(row["tile"]), int(row["softmax_threads"]))
        scores.setdefault(key, []).append(float(row["forward_median_ms"]))
    scores = {key: math.exp(sum(map(math.log, values))/len(values)) for key, values in scores.items()}
    best = min(scores, key=lambda key: (scores[key], key))
    print(*best)
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator

# read benchmark results
rows = {grid: [r for r in summary if r["experiment"] == grid and r["precision"] == "FP32"]
        for grid in ["proposal", "tuning", "ablation"]}
half_rows = [r for r in summary if r["experiment"] == "proposal" and r["precision"] == "FP16"]

main_config = {(r["tile"], r["softmax_threads"]) for r in rows["proposal"]}
if len(main_config) != 1:
    raise SystemExit("Main grid must use one frozen configuration")
tile, threads = next(iter(main_config))

colors = {"naive": "#3477ad", "optimized": "#d28b39"}

# Sequence length scaling
fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
for row, mode in enumerate(["dense", "causal"]):
    for col, dim in enumerate([32, 64, 128]):
        ax = axes[row, col]
        for variant in ["naive", "optimized"]:
            data = sorted((r for r in rows["proposal"] if r["mode"] == mode and int(r["d"]) == dim and r["variant"] == variant), key=lambda r: int(r["N"]))
            x = [int(r["N"]) for r in data]
            y = [float(r["forward_median_ms"]) for r in data]
            low = [v-float(r["forward_q1_ms"]) for v, r in zip(y, data)]
            high = [float(r["forward_q3_ms"])-v for v, r in zip(y, data)]
            ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=variant, color=colors[variant])
        ax.set(xscale="log", yscale="log", title=f"{mode.capitalize()}, d={dim}", xlabel="Sequence length N", ylabel="GPU latency (ms)")
        ax.xaxis.set_major_locator(FixedLocator([256, 512, 1024, 2048, 4096]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=8)
        ax.legend()
fig.suptitle(f"B=4, H=8; median and Q1-Q3; tile={tile}, softmax threads={threads}")
fig.savefig(args.output / "latency_scaling.svg")
plt.close(fig)

# Throughput and workspace
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
for mode in ["dense", "causal"]:
    for variant in ["naive", "optimized"]:
        data = sorted((r for r in rows["proposal"] if r["mode"] == mode and r["d"] == "64" and r["variant"] == variant), key=lambda r: int(r["N"]))
        axes[0].plot([int(r["N"]) for r in data], [float(r["sequences_per_second"]) for r in data], marker="o", label=f"{mode}, {variant}")
data = sorted((r for r in rows["proposal"] if r["mode"] == "dense" and r["variant"] == "naive" and r["d"] == "64"), key=lambda r: int(r["N"]))
axes[1].plot([int(r["N"]) for r in data], [float(r["peak_workspace_allocated_bytes"])/2**20 for r in data], marker="o")
axes[0].set(xlabel="Sequence length N", ylabel="Sequences/s", title="GPU throughput, d=64")
axes[0].legend()
axes[1].set(xlabel="Sequence length N", ylabel="Allocated workspace (MiB)", title="Q/K/V/S/P/O storage, d=64")
fig.suptitle("B=4, H=8; allocation and transfers excluded from GPU timing")
fig.savefig(args.output / "throughput_workspace.svg")
plt.close(fig)

# tuning and ablation
for grid in ["tuning", "ablation"]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, mode in enumerate(["dense", "causal"]):
        for col, length in enumerate([256, 1024]):
            data = [r for r in rows[grid] if r["mode"] == mode and int(r["N"]) == length]
            labels = [f"({r['tile']},{r['softmax_threads']})" if grid == "tuning" else r["variant"] for r in data]
            ax = axes[row, col]
            for key, label in [("qk_median_ms", "QK"), ("softmax_median_ms", "Softmax"), ("pv_median_ms", "PV")]:
                ax.plot(labels, [float(r[key]) for r in data], marker="o", label=label)
            medians = [float(r["forward_median_ms"]) for r in data]
            low = [v-float(r["forward_q1_ms"]) for v, r in zip(medians, data)]
            high = [float(r["forward_q3_ms"])-v for v, r in zip(medians, data)]
            ax.errorbar(labels, medians, yerr=[low, high], marker="o", capsize=3, label="Forward", linewidth=2)
            ax.set(title=f"{mode.capitalize()}, N={length}", ylabel="Median GPU latency (ms)")
            ax.legend()
    fig.suptitle(f"{grid.capitalize()}: B=4, H=8, d=64; forward and stages measured separately")
    fig.savefig(args.output / f"{grid}.svg")
    plt.close(fig)
# Speedup against matched naive cases
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, mode in zip(axes, ["dense", "causal"]):
    for dim in [32, 64, 128]:
        data = sorted((r for r in rows["proposal"] if r["mode"] == mode and int(r["d"]) == dim and r["variant"] == "optimized"), key=lambda r: int(r["N"]))
        ax.plot([int(r["N"]) for r in data], [float(r["speedup_vs_naive"]) for r in data], marker="o", label=f"d={dim}")
    ax.set(title=mode.capitalize(), xlabel="Sequence length N", ylabel="Speedup over naive")
    ax.legend()
fig.savefig(args.output / "speedup.svg")
plt.close(fig)


# FP16 results
if half_rows:
    baseline_rows = [r for r in rows["proposal"] if r["variant"] == "optimized"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, mode in enumerate(["dense", "causal"]):
        for col, dim in enumerate([32, 64, 128]):
            ax = axes[row, col]
            for data, label in [(baseline_rows, "FP32 optimized (saved)"), (half_rows, "FP16 Tensor Core")]:
                selected = sorted((r for r in data if r["mode"] == mode and int(r["d"]) == dim), key=lambda r: int(r["N"]))
                y = [float(r["forward_median_ms"]) for r in selected]
                low = [v-float(r["forward_q1_ms"]) for v, r in zip(y, selected)]
                high = [float(r["forward_q3_ms"])-v for v, r in zip(y, selected)]
                ax.errorbar([int(r["N"]) for r in selected], y, yerr=[low, high], marker="o", capsize=3, label=label)
            ax.set(xscale="log", yscale="log", title=f"{mode.capitalize()}, d={dim}", xlabel="Sequence length N", ylabel="GPU latency (ms)")
            ax.xaxis.set_major_locator(FixedLocator([256, 512, 1024, 2048, 4096]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(fontsize=8)
    fig.suptitle("B=4, H=8; precision and matrix hardware both change; median and Q1-Q3")
    fig.savefig(args.output / "fp16_latency.svg")
    plt.close(fig)
print("Saved performance figures")
