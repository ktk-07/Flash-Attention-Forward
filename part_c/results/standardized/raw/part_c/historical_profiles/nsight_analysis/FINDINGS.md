# Measured findings and next optimisation experiments

These findings use B=4,H=8,N=4096,d=64 unless stated otherwise, and medians over three Nsight captures. See ANALYSIS.md for the complete small/large tables, metric definitions, ranges, and capture conditions. Warp-state percentages are fractions of sampled warp states, not fractions of wall time. The candidates below have not been implemented or established as speedups.

## 1. The selected CUDA-core kernel is not simply reducing total DRAM traffic

The FA1-style baseline uses 154 registers/thread, with 25.00% theoretical and 23.41% achieved occupancy. It has only 0.08 eligible warps/scheduler and 66.68% long-scoreboard samples.

The selected `skip` version achieves 32.57% occupancy out of a 33.33% ceiling and 0.63 eligible warps/scheduler. This supports improved cooperative scheduling/latency hiding. However, measured total DRAM read+write traffic is 12.04 GiB versus 8.62 GiB for the baseline under this cold-cache profiling protocol. We must not claim that this implementation's speedup is demonstrated to come from lower total DRAM traffic.

`skip` generates 128.10 GiB of local L1 sector traffic versus 2.02 GiB for the baseline. These are repeated local-memory transactions, not that much allocated memory, and not all reach DRAM. Dynamically indexed per-thread state/accumulator arrays are a concrete optimisation target; inspect the exported LDL/STL instructions before calling every access register spilling.

## 2. The busy SM statistic is not FP32 arithmetic utilisation

For `skip`, the LSU instruction pipeline reaches 84.29% of its active-cycle peak, while the FMA pipeline is at 21.17%. DRAM throughput is 21.62% of the profiler's elapsed-cycle peak metric. A high aggregate SM-throughput percentage must not be interpreted as saturated FP32 arithmetic.

The useful QK+PV contract is 68.74 GFLOPs, excluding softmax. Measured scalar FP32 FADD/FMUL/FFMA work totals 271.68 GFLOPs versus 72.95 GFLOPs in the baseline. This includes reductions/bookkeeping and masked/padded work; it is not all avoidable overhead. More executed FLOPs can artificially increase an arithmetic-intensity numerator without improving useful efficiency.

## 3. Why float4 did not automatically win

`vector` reduces registers/thread from 69 to 55, but both allocate 24 KiB of static shared memory and have the same 33.33% theoretical occupancy ceiling. Achieved occupancy is 32.57% versus 32.53%; local sector traffic remains 128.10 GiB.

MIO-throttle samples fall from 2.60% to 0.99%, but short-scoreboard samples rise from 30.49% to 32.32%. Eligible warps/scheduler change from 0.63 to 0.61. The loader changed without removing the common thread-local traffic and reduction/dependency structure. This is a supported explanation for limited benefit, not proof that vector loads themselves caused every timing difference.

The source loader gives four adjacent features to each lane. At d=64, only 16 lanes per warp have a complete float4 to load for each row. Testing a flat cooperative loader across multiple rows is a possible experiment; it does not guarantee improvement. Keep the existing scalar and vector variants as controls.

## 4. Higher occupancy alone does not select the fastest kernel

The warp-per-query kernel reaches 41.62% occupancy, higher than `skip` at 32.57%, but it has fewer eligible warps/scheduler (0.36 versus 0.63), 62.85% long-scoreboard samples, and 56.99 GiB of DRAM traffic. The working data/reuse pattern and dependency chain matter alongside resident warp count.

## 5. WMMA versus PTX has a specific shared-memory/resource trade-off

WMMA uses 25.38 KiB shared memory per block versus PTX's 13.00 KiB. Their theoretical occupancy ceilings are 12.50% and 29.17%; achieved occupancy is 11.90% and 27.03%. PTX uses more registers/thread (64 versus 40) while avoiding some shared output/state round trips.

WMMA has 31.74% short-scoreboard samples versus PTX's 11.58%, and 3.21 times as many shared-load wavefronts. Tensor-pipeline active-cycle utilisation is 2.12% versus 6.25%. Both still have substantial headroom; none of these individual ratios is a measured causal speedup.

Source-level actual/ideal shared wavefront expansion is 7.418x for WMMA, 4.067x for PTX, and 2.749x for warp-per-query. These have positive excessive wavefronts and motivate controlled layout experiments. Tiled, skip and vector have 0 excessive wavefronts: there is no source-proven bank conflict in these CUDA-core layouts. Aggregate L1 conflict counters include arbitration and do not justify swizzling them. None of these ratios is a percentage of runtime.

## 6. Source locations support targeted experiments

The table below gives the highest sampled PC per selected category from representative repetition 0. Optimised/inlined line attribution can be approximate. Duplicate source mappings to the same PC were deduplicated and checked; inspect surrounding instructions and dependencies in the full exports.

| Kernel | Category | % within this category | CUDA location | SASS |
|---|---|---:|---|---|
| fa1 | stall_long_sb | 4.78 | flash_attention_1.cu:121 | `FSETP.NEU.AND P0, PT, R25, -INF , PT` |
| warp | stall_long_sb | 11.83 | flash_attention_2_cuda_core.cu:268 | `STS [R58+0x1c00], R37` |
| skip | stall_short_sb | 1.72 | flash_attention_2_cuda_core.cu:845 | `FMUL R30, R53, c[0x0][0x1d0]` |
| skip | stall_long_sb | 9.04 | flash_attention_2_cuda_core.cu:882 | `FMNMX R37, R9, R10, !PT` |
| vector | stall_long_sb | 14.55 | flash_attention_2_cuda_core.cu:1088 | `STS.128 [R11], R12` |
| wmma | stall_short_sb | 9.62 | flash_attention_2_tensor_core.cu:281 | `FFMA R7, R8, R7, R13` |
| ptx | stall_long_sb | 92.88 | flash_attention_2_tensor_core.cu:393 | `STS.U16 [R43+0x2000], R39` |

In the CUDA-core kernels, exported source includes short-scoreboard samples at shuffle-result arithmetic and scale/reduction instructions, as well as local state traffic. In WMMA, the shared `out_acc` update is prominent. For PTX, long-scoreboard source samples concentrate at the K/V load-to-shared path: a store may be waiting on a preceding global load. This makes input-load overlap a candidate for PTX, not a blanket justification for double buffering every kernel.

## 7. Divergence is measured, not assumed

At the large shape, branch uniformity is 99.51% in the baseline, 100.00% in `skip`, and 98.52% in `vector`. Average predicated-on threads/warp are 31.59, 30.63, and 30.22 respectively. Thus even fully uniform branches do not imply all lanes perform useful work. The evidence does not justify calling branch divergence the dominant large-shape bottleneck.

## Recommended order of experiments

1. **CUDA-core local state and reductions:** specialise/unroll small array indexing where useful, inspect LDL/STL and register allocation, and reduce repeated warp reductions or row-statistic work where mathematically valid. Monitor shared/register occupancy limits after every change.
2. **Tensor Core shared layouts:** try one controlled padding or swizzle variant, verifying both writes and reads plus fragment operand mapping. Recheck bank conflicts and ordinary timings.
3. **PTX input pipeline:** evaluate asynchronous copies/double buffering only with a concrete load/compute schedule. Account for extra shared memory and its occupancy cost.
4. **Bounded launch/tile tuning:** vary warps/query/KV tile sizes under a predefined search. These historical profiles use fixed educational launch configurations; the separate bounded evaluation studies other configurations and must not inherit these counters.
5. **Alternative vector loader:** test cooperative loading across rows rather than silently replacing the existing baseline.

Each experiment needs numerical validation, sanitizer checks where memory/synchronisation changes, ordinary repeated CUDA-event timing, and a fresh profiling directory. The current workflow analyses kernels; it does not implement these optimisations. No new speedup is claimed here.

## Reproduce these findings

```bash
python3 part_c/scripts/interpret_profiles.py part_c/results/nsight_analysis
```

This report is generated from `metric_summary.csv`, `all_metrics.csv`, and the deduplicated `source_analysis/hotspots.csv`. These observations are limited to the two profiled shapes on the recorded RTX 3070 Ti Laptop GPU, with unmanaged clocks, cache flushing, instrumentation/replay, and the sampling limitations documented in ANALYSIS.md. Ordinary benchmark timing remains separate.
