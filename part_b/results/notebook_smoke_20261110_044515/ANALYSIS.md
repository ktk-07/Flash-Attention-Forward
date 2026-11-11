# Part B dense FlashAttention evaluation

FP32 contiguous BHNd inputs, FP32 accumulation/output. The shared C++ FP64 oracle validates every accepted timing case outside timing. Full small-case checks and sampled large-case checks are distinguished in correctness.csv. All outputs are checked for finiteness.

| N | d | Frozen configuration | Median ms | IQR ms | Explicit bytes |
|---|---|---|---|---|---|
| 33 | 64 | streaming_w4_k32 | 0.011264 | 0.000176 | 33792 |
| 65 | 64 | streaming_w4_k32 | 0.019440 | 0.000368 | 66560 |

The allocation ledger is 4 × B × H × N × d × sizeof(float), covering Q/K/V/O, with zero auxiliary global allocation. It is linear in N. Shared K/V/probability tiles and registers are per-block resources. Device free-memory deltas are observational and may be unreliable under Windows WDDM.

## Controlled loading comparison

Same W4/BK32 scalar and float4 kernels on the same generated tensors. Ratios below compare median latencies from separate sequential cases; close ratios are not statistically conclusive.

| N | d | Scalar ms | float4 ms | Scalar / float4 |
|---|---|---|---|---|
| 33 | 64 | 0.013888 | 0.012288 | 1.130 |
| 65 | 64 | 0.019456 | 0.017408 | 1.118 |

Validation records: 83 full and 0 sampled; 607,393 oracle-compared elements. Failed records: 0. All timed cases have matching validation records and input fingerprints.

The streaming and tiled kernels share the warp-per-query mapping. Tiling changes both K/V reuse and online update granularity; this comparison does not isolate one change. Tiled versus float4 with identical warps/key tile is the loading ablation. Candidate choices are frozen using geometric mean latency on tuning lengths only, with lexical ties. Main lengths include held-out lengths.

## Tuning and bottleneck interpretation

| d | Frozen winner | Tuning geometric mean ms | Slowest candidate | Tuning geometric mean ms |
|---|---|---|---|---|
| 64 | streaming_w4_k32 | 0.008986 | vector_w4_k64 | 0.012288 |

Larger shared tiles need not win: they increase per-block storage and can reduce occupancy. Cooperative loading adds barriers, probability-tile communication and warp reductions. Global-cache reuse can make the streaming baseline competitive. These are source/resource-based explanations to investigate; without fresh counters, the measurements do not establish DRAM bandwidth, bank conflicts or a definitive hardware bottleneck.

Five warmups and twenty CUDA-event samples measure complete fused forward only. Allocation, transfers, oracle, resource queries and file I/O are excluded. Fixed order, display-GPU contention, unmanaged clocks and within-case variation limit close-winner conclusions. No same-machine A/B/C performance comparison is claimed by this standalone run.

Fresh profiling requested: False. See audit.json for actual captures and safety checks. Missing counters must not be interpreted as zero traffic. AI-assisted implementation and analysis require member review and disclosure.
