# Part C bounded evaluation

Correctness: 288 full small/tail/pattern checks; 318 sampled benchmark checks; 30 sampled memory checks. All outputs checked finite.

Selection was frozen after tuning and before the final grid. Family winners are selected only among skip/vector and WMMA/PTX; fixed-default progression includes the earlier variants. Tuning lengths 256 and 1024 overlap the reporting grid; 512, 2048 and 4096 are held out.

## Frozen configurations

| d | CUDA core | Tensor core |
|---|---|---|
| 32 | skip, config 2 | ptx, config 2 |
| 64 | skip, config 4 | ptx, config 2 |
| 128 | vector, config 4 | ptx, config 2 |

## Held-out N=4096

| d | CUDA ms | Tensor ms | Core / tensor |
|---|---|---|---|
| 32 | 65.716 | 5.042 | 13.03x |
| 64 | 107.663 | 12.153 | 8.86x |
| 128 | 150.936 | 36.106 | 4.18x |

## Tuning effect at held-out N=4096

| d | Family | Kernel | Default ms | Frozen ms | Default / frozen | Reg/thread default → frozen | Local bytes/thread default → frozen |
|---|---|---|---|---|---|---|---|
| 32 | cuda_core | skip | 101.673 | 65.716 | 1.55x | 40 → 56 | 96 → 0 |
| 32 | tensor_core | ptx | 5.543 | 5.042 | 1.10x | 54 → 47 | 0 → 0 |
| 64 | cuda_core | skip | 132.246 | 107.663 | 1.23x | 69 → 40 | 128 → 64 |
| 64 | tensor_core | ptx | 14.077 | 12.153 | 1.16x | 64 → 63 | 0 → 0 |
| 128 | cuda_core | vector | 204.705 | 150.936 | 1.36x | 74 → 43 | 192 → 96 |
| 128 | tensor_core | ptx | 55.020 | 36.106 | 1.52x | 111 → 111 | 0 → 0 |

## Memory at N=4096

| d | Kernel | Explicit live buffers MiB | Observed device delta MiB |
|---|---|---|---|
| 32 | skip | 64.0 | 64.0 |
| 32 | ptx | 40.0 | 40.0 |
| 64 | skip | 128.0 | 128.0 |
| 64 | ptx | 80.0 | 80.0 |
| 128 | vector | 256.0 | 256.0 |
| 128 | ptx | 160.0 | 160.0 |

## Interpretation limits

The matched-load study changes scalar versus float4 loading with the same launch configuration. Other progression steps combine changes and are not isolated causal ablations. Config tuning changes resource use without changing kernel algorithms.

Resource queries are available for tiled/skip/vector/WMMA/PTX. Raw scalar/warp zero defaults are unavailable placeholders; summary.csv leaves those fields blank. FA1 resources are also unavailable in this study.

Memory plots show live explicit Q/K/V/O allocations, not DRAM traffic or total process memory. FP32 core uses 16 bytes/element; FP16 inputs with FP32 output use 10. The S+P line is an analytical materialized-storage model, not a measured Part A result. cudaMemGetInfo deltas include allocation granularity and concurrent device activity; only three observations are made. Resource occupancy is an API estimate, not achieved Nsight occupancy.

Tensor Core paths use FP16 Q/K/V and round probabilities for PV; CUDA Core uses FP32. Error is checked against the corresponding rounded-input FP64 oracle. These are precision-labelled performance comparisons.

Cases run in a fixed sequential order on one laptop GPU with unmanaged clocks. Twenty repetitions characterize within-case spread, not independent-session uncertainty; close winners may change on a fresh run. No confidence interval or universal winner is claimed.

See plan.json, commands.jsonl, cases/, measurements.csv, summary.csv, comparison.csv, matched_ablation.csv, manifest.json, frozen_selection.json and audit.json. Historical profiling remains a separate fixed-default study; its counters do not describe tuned configurations.
