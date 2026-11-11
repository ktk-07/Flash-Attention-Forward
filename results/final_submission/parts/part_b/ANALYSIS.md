# Part B dense FlashAttention evaluation

FP32 contiguous BHNd inputs, FP32 accumulation/output. The shared C++ FP64 oracle validates every accepted timing case outside timing. Full small-case checks and sampled large-case checks are distinguished in correctness.csv. All outputs are checked for finiteness.

| N | d | Frozen configuration | Median ms | IQR ms | Explicit bytes |
|---|---|---|---|---|---|
| 256 | 32 | vector_w4_k64 | 1.139712 | 0.006144 | 4194304 |
| 512 | 32 | vector_w4_k64 | 4.487168 | 0.001024 | 8388608 |
| 1024 | 32 | vector_w4_k64 | 14.028720 | 0.010832 | 16777216 |
| 2048 | 32 | vector_w4_k64 | 55.810383 | 0.020991 | 33554432 |
| 4096 | 32 | vector_w4_k64 | 222.590462 | 0.007156 | 67108864 |
| 256 | 64 | vector_w4_k32 | 1.004544 | 0.001872 | 8388608 |
| 512 | 64 | vector_w4_k32 | 3.966976 | 0.004608 | 16777216 |
| 1024 | 64 | vector_w4_k32 | 15.814336 | 0.002472 | 33554432 |
| 2048 | 64 | vector_w4_k32 | 63.435087 | 1.197826 | 67108864 |
| 4096 | 64 | vector_w4_k32 | 254.082497 | 1.465927 | 134217728 |
| 256 | 128 | streaming_w4_k32 | 1.231872 | 0.018520 | 16777216 |
| 512 | 128 | streaming_w4_k32 | 4.984832 | 0.032768 | 33554432 |
| 1024 | 128 | streaming_w4_k32 | 20.704241 | 0.171775 | 67108864 |
| 2048 | 128 | streaming_w4_k32 | 85.647873 | 0.642363 | 134217728 |
| 4096 | 128 | streaming_w4_k32 | 336.906754 | 1.693665 | 268435456 |

The allocation ledger is 4 × B × H × N × d × sizeof(float), covering Q/K/V/O, with zero auxiliary global allocation. It is linear in N. Shared K/V/probability tiles and registers are per-block resources. Device free-memory deltas are observational and may be unreliable under Windows WDDM.

## Controlled loading comparison

Same W4/BK32 scalar and float4 kernels on the same generated tensors. Ratios below compare median latencies from separate sequential cases; close ratios are not statistically conclusive.

| N | d | Scalar ms | float4 ms | Scalar / float4 |
|---|---|---|---|---|
| 256 | 32 | 0.947712 | 0.913872 | 1.037 |
| 512 | 32 | 3.717120 | 3.573760 | 1.040 |
| 1024 | 32 | 14.678448 | 14.123008 | 1.039 |
| 2048 | 32 | 58.889168 | 56.176609 | 1.048 |
| 4096 | 32 | 234.964478 | 224.167938 | 1.048 |
| 256 | 64 | 1.097728 | 1.004544 | 1.093 |
| 512 | 64 | 4.443520 | 3.967440 | 1.120 |
| 1024 | 64 | 17.692000 | 15.944704 | 1.110 |
| 2048 | 64 | 71.348225 | 63.969791 | 1.115 |
| 4096 | 64 | 283.954117 | 255.563263 | 1.111 |
| 256 | 128 | 2.277888 | 1.758608 | 1.295 |
| 512 | 128 | 8.719312 | 6.806016 | 1.281 |
| 1024 | 128 | 35.732992 | 26.998624 | 1.324 |
| 2048 | 128 | 142.513527 | 107.435394 | 1.327 |
| 4096 | 128 | 636.480835 | 480.199677 | 1.325 |

Validation records: 75 full and 90 sampled; 1,317,025 oracle-compared elements. Failed records: 0. All timed cases have matching validation records and input fingerprints.

The streaming and tiled kernels share the warp-per-query mapping. Tiling changes both K/V reuse and online update granularity; this comparison does not isolate one change. Tiled versus float4 with identical warps/key tile is the loading ablation. Candidate choices are frozen using geometric mean latency on tuning lengths only, with lexical ties. Main lengths include held-out lengths.

## Tuning and bottleneck interpretation

| d | Frozen winner | Tuning geometric mean ms | Slowest candidate | Tuning geometric mean ms |
|---|---|---|---|---|
| 32 | vector_w4_k64 | 3.564628 | streaming_w4_k32 | 5.164126 |
| 64 | vector_w4_k32 | 3.979651 | vector_w4_k64 | 5.603803 |
| 128 | streaming_w4_k32 | 4.987981 | vector_w4_k64 | 11.927511 |

Larger shared tiles need not win: they increase per-block storage and can reduce occupancy. Cooperative loading adds barriers, probability-tile communication and warp reductions. Global-cache reuse can make the streaming baseline competitive. These are source/resource-based explanations to investigate; without fresh counters, the measurements do not establish DRAM bandwidth, bank conflicts or a definitive hardware bottleneck.

Five warmups and twenty CUDA-event samples measure complete fused forward only. Allocation, transfers, oracle, resource queries and file I/O are excluded. Fixed order, display-GPU contention, unmanaged clocks and within-case variation limit close-winner conclusions. No same-machine A/B/C performance comparison is claimed by this standalone run.

Fresh profiling requested: True. See audit.json for actual captures and safety checks. Missing counters must not be interpreted as zero traffic. AI-assisted implementation and analysis require member review and disclosure.
