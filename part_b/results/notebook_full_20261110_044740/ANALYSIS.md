# Part B dense FlashAttention evaluation

FP32 contiguous BHNd inputs, FP32 accumulation/output. The shared C++ FP64 oracle validates every accepted timing case outside timing. Full small-case checks and sampled large-case checks are distinguished in correctness.csv. All outputs are checked for finiteness.

| N | d | Frozen configuration | Median ms | IQR ms | Explicit bytes |
|---|---|---|---|---|---|
| 256 | 32 | vector_w4_k64 | 0.445440 | 0.007704 | 4194304 |
| 512 | 32 | vector_w4_k64 | 1.730528 | 0.003088 | 8388608 |
| 1024 | 32 | vector_w4_k64 | 7.080736 | 0.258560 | 16777216 |
| 2048 | 32 | vector_w4_k64 | 28.125600 | 0.265872 | 33554432 |
| 4096 | 32 | vector_w4_k64 | 112.053246 | 0.774670 | 67108864 |
| 256 | 64 | vector_w4_k32 | 0.497664 | 0.000784 | 8388608 |
| 512 | 64 | vector_w4_k32 | 1.932288 | 0.019712 | 16777216 |
| 1024 | 64 | vector_w4_k32 | 7.869408 | 0.207864 | 33554432 |
| 2048 | 64 | vector_w4_k32 | 31.204256 | 0.433033 | 67108864 |
| 4096 | 64 | vector_w4_k32 | 125.168049 | 0.688728 | 134217728 |
| 256 | 128 | streaming_w4_k32 | 0.599792 | 0.093080 | 16777216 |
| 512 | 128 | streaming_w4_k32 | 2.471392 | 0.531232 | 33554432 |
| 1024 | 128 | streaming_w4_k32 | 10.250080 | 0.572584 | 67108864 |
| 2048 | 128 | streaming_w4_k32 | 41.121681 | 1.467150 | 134217728 |
| 4096 | 128 | streaming_w4_k32 | 163.779892 | 3.734146 | 268435456 |

The allocation ledger is 4 × B × H × N × d × sizeof(float), covering Q/K/V/O, with zero auxiliary global allocation. It is linear in N. Shared K/V/probability tiles and registers are per-block resources. Device free-memory deltas are observational and may be unreliable under Windows WDDM.

## Controlled loading comparison

Same W4/BK32 scalar and float4 kernels on the same generated tensors. Ratios below compare median latencies from separate sequential cases; close ratios are not statistically conclusive.

| N | d | Scalar ms | float4 ms | Scalar / float4 |
|---|---|---|---|---|
| 256 | 32 | 0.468656 | 0.450272 | 1.041 |
| 512 | 32 | 1.818976 | 1.751744 | 1.038 |
| 1024 | 32 | 7.467088 | 7.135024 | 1.047 |
| 2048 | 32 | 29.559376 | 28.363472 | 1.042 |
| 4096 | 32 | 118.381950 | 113.783615 | 1.040 |
| 256 | 64 | 0.543744 | 0.497664 | 1.093 |
| 512 | 64 | 2.123152 | 1.940224 | 1.094 |
| 1024 | 64 | 9.177600 | 8.359936 | 1.098 |
| 2048 | 64 | 36.596865 | 33.735647 | 1.085 |
| 4096 | 64 | 147.226685 | 135.006195 | 1.091 |
| 256 | 128 | 1.113104 | 0.917056 | 1.214 |
| 512 | 128 | 4.408672 | 3.543040 | 1.244 |
| 1024 | 128 | 17.863680 | 14.790544 | 1.208 |
| 2048 | 128 | 70.918434 | 60.363199 | 1.175 |
| 4096 | 128 | 283.188568 | 225.536507 | 1.256 |

Validation records: 75 full and 90 sampled; 1,317,025 oracle-compared elements. Failed records: 0. All timed cases have matching validation records and input fingerprints.

The streaming and tiled kernels share the warp-per-query mapping. Tiling changes both K/V reuse and online update granularity; this comparison does not isolate one change. Tiled versus float4 with identical warps/key tile is the loading ablation. Candidate choices are frozen using geometric mean latency on tuning lengths only, with lexical ties. Main lengths include held-out lengths.

## Tuning and bottleneck interpretation

| d | Frozen winner | Tuning geometric mean ms | Slowest candidate | Tuning geometric mean ms |
|---|---|---|---|---|
| 32 | vector_w4_k64 | 1.766675 | streaming_w4_k32 | 2.360889 |
| 64 | vector_w4_k32 | 1.978168 | vector_w4_k64 | 2.963114 |
| 128 | streaming_w4_k32 | 2.363710 | vector_w4_k64 | 6.112063 |

Larger shared tiles need not win: they increase per-block storage and can reduce occupancy. Cooperative loading adds barriers, probability-tile communication and warp reductions. Global-cache reuse can make the streaming baseline competitive. These are source/resource-based explanations to investigate; without fresh counters, the measurements do not establish DRAM bandwidth, bank conflicts or a definitive hardware bottleneck.

Five warmups and twenty CUDA-event samples measure complete fused forward only. Allocation, transfers, oracle, resource queries and file I/O are excluded. Fixed order, display-GPU contention, unmanaged clocks and within-case variation limit close-winner conclusions. No same-machine A/B/C performance comparison is claimed by this standalone run.

Fresh profiling requested: False. See audit.json for actual captures and safety checks. Missing counters must not be interpreted as zero traffic. AI-assisted implementation and analysis require member review and disclosure.
