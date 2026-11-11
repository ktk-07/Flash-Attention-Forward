# Part C bounded evaluation

Correctness: 352 full small/tail/pattern checks; 352 sampled benchmark checks; 60 sampled memory checks. All outputs checked finite.

Selection was frozen after tuning and before the main grid. Family winners are selected only among skip/vector and WMMA/PTX; FP32 FA1 uses fixed config 0; PTX FA1 and FA2 each use four candidates. Tuning lengths: [256, 1024]; reporting lengths: [256, 512, 1024, 2048, 4096]; held-out reporting lengths: [512, 2048, 4096].

## Frozen configurations

| d | CUDA core | Tensor core |
|---|---|---|
| 32 | skip, config 2 | ptx, config 2 |
| 64 | vector, config 2 | ptx, config 2 |
| 128 | vector, config 4 | ptx, config 2 |

## Largest reporting N=4096

| d | CUDA ms | Tensor ms | Core / tensor |
|---|---|---|---|
| 32 | 65.289 | 5.067 | 12.88x |
| 64 | 115.085 | 12.912 | 8.91x |
| 128 | 148.998 | 36.108 | 4.13x |

## Tuning effect at N=4096

| d | Family | Kernel | Default ms | Frozen ms | Default / frozen | Reg/thread default → frozen | Local bytes/thread default → frozen |
|---|---|---|---|---|---|---|---|
| 32 | cuda_core | skip | 98.153 | 65.289 | 1.50x | 40 → 56 | 96 → 0 |
| 32 | tensor_core | ptx | 5.551 | 5.067 | 1.10x | 54 → 47 | 0 → 0 |
| 64 | cuda_core | vector | 134.428 | 115.085 | 1.17x | 55 → 40 | 128 → 64 |
| 64 | tensor_core | ptx | 13.806 | 12.912 | 1.07x | 64 → 63 | 0 → 0 |
| 128 | cuda_core | vector | 201.968 | 148.998 | 1.36x | 74 → 43 | 192 → 96 |
| 128 | tensor_core | ptx | 55.037 | 36.108 | 1.52x | 111 → 111 | 0 → 0 |

## Memory at N=4096

| d | Kernel | Explicit live buffers MiB | Observed device delta MiB |
|---|---|---|---|
| 32 | fa1 | 65.0 | 66.0 |
| 32 | fa1_ptx | 41.0 | 42.0 |
| 32 | skip | 64.0 | 64.0 |
| 32 | ptx | 40.0 | 40.0 |
| 64 | fa1 | 129.0 | 130.0 |
| 64 | fa1_ptx | 81.0 | 82.0 |
| 64 | vector | 128.0 | 128.0 |
| 64 | ptx | 80.0 | 80.0 |
| 128 | fa1 | 257.0 | 258.0 |
| 128 | fa1_ptx | 161.0 | 162.0 |
| 128 | vector | 256.0 | 256.0 |
| 128 | ptx | 160.0 | 160.0 |

FA1 in this run is the CUDA-core Algorithm-1 schedule: K/V outer, Q inner, one CTA/head, shared Q/K/V/S/O, global m/l. Its live allocation adds 8*B*H*N bytes; it is not the historical query-parallel scalar baseline. The Tensor Core FA1 baseline uses the K/V-outer state schedule, half Q/K/V/P and FP32 accumulators/softmax/O; its implementation is identified in plan.json. Both baselines add global m/l; neither is the official FA1 library. FA2-style kernels are unchanged. Do not reuse old FA1 progression ratios.

## Interpretation limits

The matched-load study changes scalar versus float4 loading with the same launch configuration. Other progression steps combine changes and are not isolated causal ablations. Config tuning changes resource use without changing kernel algorithms.

Resource queries are available for tiled/skip/vector/WMMA/PTX. Raw scalar/warp zero defaults are unavailable placeholders; summary.csv leaves those fields blank. FA1 resource queries include dynamic shared storage.

Memory plots show live explicit Q/K/V/O allocations plus FA1 m/l where recorded, not DRAM traffic or total process memory. FP32 core uses 16 bytes/element; FP16 inputs with FP32 output use 10. The S+P line is an analytical materialized-storage model, not a measured Part A result. cudaMemGetInfo deltas include allocation granularity and concurrent device activity; only three observations are made. Resource occupancy is an API estimate, not achieved Nsight occupancy.

Tensor Core paths use FP16 Q/K/V and round probabilities for PV; CUDA Core uses FP32. Error is checked against the corresponding rounded-input FP64 oracle. These are precision-labelled performance comparisons.

Cases run in a fixed sequential order on one laptop GPU with unmanaged clocks. Twenty repetitions characterize within-case spread, not independent-session uncertainty; close winners may change on a fresh run. No confidence interval or universal winner is claimed.

See plan.json, commands.jsonl, cases/, measurements.csv, summary.csv, comparison.csv, matched_ablation.csv, manifest.json, frozen_selection.json and audit.json. Historical profiling remains a separate fixed-default study; its counters do not describe tuned configurations.
