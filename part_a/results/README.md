# Measured results

**Recorded GPU run passed**: measured source `e2487da` compiled, all 18 CTest groups passed, all four experiments completed, three sanitizer tools passed, and both README profiling runs completed. The full Part A grid measured **60/60 rows with 1200 timing samples**, using one frozen configuration `(tile=16, softmax_threads=128)`. No rows were skipped for memory or runtime failures.


The [Release validation](validation/2026-10-15T052608Z-fa1f796/README.md) of `fa1f796` passed 18 CTest groups, 7 analysis tests and the representative benchmark at (16,128). Full-grid and profile measurements remain from `e2487da`.

## Source and environment

Measured source: `e2487da7c7de5572316f2bf2a1a3d926d19d0517`. Each run records source/build identity and hardware settings in `metadata.txt`.

Windows 11 Pro 25H2, RTX 5060 Ti 8151 MiB, compute capability 12.0, driver 610.88, CUDA Toolkit 13.4, MSVC 19.44 and CMake 4.4.4. Builds use Release and `sm_120`, without fast math. The run notes report no manual power or clock changes. Saved before/after CSVs contain pstate and temperature, but omit power and clocks. Desktop activity, thermal changes and unlocked clocks limit reproducibility.

## Commands and configuration

See [experiment commands](../benchmarks/README.md). Test evidence: [CTest log](gpu-benchmarks/2026-10-14T154318Z-review/ctest.log), [JUnit](gpu-benchmarks/2026-10-14T154318Z-review/ctest.xml) and [exit codes](gpu-benchmarks/2026-10-14T154318Z-review/exit-codes.txt).

The bounded candidate set is (8,256), (16,256), (32,256), (16,128). The last candidate had the lowest median in all four tuning shape/mode combinations and was frozen before proposal/ablation. N=1024 differences versus (16,256) are small relative to some observed IQRs. Selection does not establish a global optimum, and the library default remains (16,256). The freeze record is [configuration-frozen-utc.txt](gpu-benchmarks/2026-10-14T154318Z-review/configuration-frozen-utc.txt): `2026-10-14T15:45:11.9262703Z`, before the main measurement started at `15:45:17Z`.

Tuning medians for `B=4,H=8,d=64`, in milliseconds:

| N | Mode | (8,256) | (16,256) | (32,256) | (16,128) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Dense | 0.429312 | 0.359072 | 0.430112 | 0.320976 |
| 256 | Causal | 0.355216 | 0.305440 | 0.366912 | 0.266080 |
| 1024 | Dense | 6.301680 | 5.032240 | 6.371376 | 5.000592 |
| 1024 | Causal | 4.841664 | 3.972944 | 5.273920 | 3.911792 |

The selection rule compares forward medians across the four equally weighted cases. One configuration is used for the main grid and ablation; it is not selected separately for each main-grid shape.

## Coverage and validation

| Test category | Executed | Passed | Failed | Not run |
| --- | ---: | ---: | ---: | ---: |
| CPU oracle | 8 | 8 | 0 | 0 |
| CPU comparator | 4 | 4 | 0 | 0 |
| CPU benchmark harness | 1 | 1 | 0 | 0 |
| GPU variant groups | 4 | 4 | 0 | 0 |
| GPU configuration-validation group | 1 | 1 | 0 | 0 |
| Total | 18 | 18 | 0 | 0 |

The four GPU groups each contain eight numerical/boundary selectors. Materialized S/P/O use identical FP32 inputs and the independent FP64 oracle, with unchanged atol=rtol=1e-4. Representative gates apply to every selected mode/variant/configuration. N<=257 timed cases have exact-shape comparisons before and after timing; larger shapes have representative gates and finite-O checks, not a full exact-shape FP64 comparison.

| Experiment | Planned/completed rows | Samples | Passing S/P/O stage reports | Data and analysis |
| --- | ---: | ---: | ---: | --- |
| representative | 8/8 | 160 | 96 | [CSV](gpu-benchmarks/2026-10-14T154423Z-0f768e/benchmark/summary.csv), [audit/charts](gpu-benchmarks/2026-10-14T154423Z-0f768e/analysis/README.md) |
| tuning | 16/16 | 320 | 144 | [CSV](gpu-benchmarks/2026-10-14T154428Z-18c60f/benchmark/summary.csv), [audit/charts](gpu-benchmarks/2026-10-14T154428Z-18c60f/analysis/README.md) |
| proposal | 60/60 | 1200 | 120 | [CSV](gpu-benchmarks/2026-10-14T154516Z-a73070/benchmark/summary.csv), [audit/charts](gpu-benchmarks/2026-10-14T154516Z-a73070/analysis/README.md) |
| ablation | 16/16 | 320 | 144 | [CSV](gpu-benchmarks/2026-10-14T154841Z-f210f0/benchmark/summary.csv), [audit/charts](gpu-benchmarks/2026-10-14T154841Z-f210f0/analysis/README.md) |

The analysis command verified sample indices/counts, identical fingerprints, medians, quartiles/IQR, throughput, workspace bytes and matched naive speedups. PNG and SVG plots were generated from actual data, with English labels. All three sanitizer tools exited 0 on shape (1,1,33,33), both modes and the four candidate configurations. [memcheck](gpu-benchmarks/2026-10-14T154318Z-review/memcheck.log), [racecheck](gpu-benchmarks/2026-10-14T154318Z-review/racecheck.log), [synccheck](gpu-benchmarks/2026-10-14T154318Z-review/synccheck.log) reported zero errors/hazards/warnings. Their instrumented runs are not timing evidence.

## Full-grid results

Main grid: B=4, H=8, N=256/512/1024/2048/4096, d=32/64/128, dense and causal, naive and optimized. Five warmups and 20 repeats, seed 5206, generator splitmix64-fp32-v1. Resident preallocated buffers, default-stream CUDA events, and independent forward/stage passes exclude allocation, copies, input generation, CPU validation and file I/O. Speedups compare matched naive GPU medians. Stage medians need not sum to forward median.

| Mode | Minimum optimized speedup | Maximum optimized speedup |
| --- | ---: | ---: |
| dense | 4.848x | 6.932x |
| causal | 3.177x | 4.776x |

Largest sequence-length cases:

| N,d | Mode | Naive median ms | Optimized median ms | Optimized IQR ms | Matched speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096,32 | dense | 322.991287 | 48.957008 | 1.222222 | 6.597x |
| 4096,32 | causal | 162.615738 | 38.301792 | 0.136785 | 4.246x |
| 4096,64 | dense | 523.843384 | 85.741951 | 0.832872 | 6.110x |
| 4096,64 | causal | 272.618546 | 64.501057 | 0.286123 | 4.227x |
| 4096,128 | dense | 922.806335 | 154.259377 | 0.613548 | 5.982x |
| 4096,128 | causal | 497.233734 | 116.231567 | 0.757023 | 4.278x |

Maximum requested six-buffer workspace: 4563402752 bytes (4352 MiB). This is allocated Q/K/V/S/P/O storage, not total process peak memory or DRAM traffic. Setup time remains separate in CSV.

## Current-configuration profiling

Profiling uses tile 16 and 128 softmax threads. The three kernels were captured after validation and warmup. Full counter CSVs are committed; binary reports remain locally under ignored `results/local/profiles`. Profile timings are separate from normal measurements.

| N | Kernel | Registers/thread | Static shared memory | Dynamic shared memory | Theoretical occupancy % | Achieved occupancy % | DRAM throughput |
| ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| 256 | QK | 48 | 2.112000 Kbyte/block | 0 byte/block | 83.333333 | 80.088211 | 32.080580 Gbyte/s |
| 256 | softmax | 26 | 0.512000 Kbyte/block | 0 byte/block | 100.000000 | 93.595585 | 169.372903 Gbyte/s |
| 256 | PV | 36 | 2.048000 Kbyte/block | 0 byte/block | 100.000000 | 95.512830 | 70.011945 Gbyte/s |
| 2048 | QK | 48 | 2.112000 Kbyte/block | 0 byte/block | 83.333333 | 82.874149 | 62.943578 Gbyte/s |
| 2048 | softmax | 26 | 0.512000 Kbyte/block | 0 byte/block | 100.000000 | 98.847587 | 382.174307 Gbyte/s |
| 2048 | PV | 36 | 2.048000 Kbyte/block | 0 byte/block | 100.000000 | 99.425519 | 64.976935 Gbyte/s |

Full counter exports: [N=256](profiles/2026-10-14T154925Z-74efb1/metrics.csv), [N=2048](profiles/2026-10-14T154957Z-553569/metrics.csv).

[Selected metrics with units](gpu-benchmarks/2026-10-14T154318Z-review/profile-selected-metrics.csv) also include DRAM read/write traffic and the profiled launch dimensions. Registers/shared memory/theoretical occupancy are also saved per configuration in each benchmark's resources.csv. Resource values alone do not prove a bottleneck or the cause of a slowdown. Nsight byte units are decimal; workspace MiB is binary.

## Interpretation

### Individual optimizations

The [ablation run](gpu-benchmarks/2026-10-14T154841Z-f210f0/analysis/README.md) uses the same inputs and (16,128) configuration for all four variants. Dense forward medians are:

| N,d | Naive ms | Tiled ms | Parallel ms | Optimized ms |
| --- | ---: | ---: | ---: | ---: |
| 256,64 | 1.725664 | 0.455168 | 1.594208 | 0.323792 |
| 1024,64 | 33.099392 | 13.255504 | 26.390512 | 5.176800 |

Tiling provides the larger individual forward improvement in these dense cases. Threads cooperatively load Q/K or P/V tiles, then reuse shared values during multiplication. This reduces repeated global-memory loads. At N=1024, dense QK falls from 22.482080 ms to 2.433280 ms with tiling. PV falls from 2.708496 ms to 2.253792 ms. The measured benefit is therefore much larger for QK in this case.

Parallel softmax assigns a block to each row. Threads split the maximum, exponential sum and normalization work. At N=1024, its dense stage falls from 7.867904 ms to 0.698368 ms. The parallel-only forward still spends most time in naive QK. With tiled multiplication, serial softmax becomes the largest stage at 8.289632 ms. Combining both optimizations removes this new hotspot. Optimized stage medians are QK=2.397152 ms, softmax=0.681296 ms and PV=2.108928 ms. QK and PV then account for most of the measured stage time.

These stage medians come from separate breakdown passes. They need not add to the forward median, and individual speedups do not multiply. The N=1024 dense optimized forward IQR is 0.901712 ms, so small differences between separate experiments need caution.

### Configuration choice

Tile 16 is faster than tiles 8 and 32 in all four tuning cases. Larger tiles offer more reuse, but also use more threads and shared storage per block. Smaller tiles provide less reuse and require more tiles. These are possible trade-offs; the saved data does not isolate the cause of each slower candidate. Choosing the largest tile or highest occupancy alone is insufficient.

Softmax 128 beats 256 in all four observed medians. At N=256, dense latency decreases by 10.61% and causal latency by 12.89%. At N=1024, decreases are only 0.63% and 1.54%. Fewer threads may reduce reduction and synchronization work, but the current counters do not prove this explanation. One tuning run supports this bounded selection, without establishing a universal optimum.

### Causal attention and memory

At N=4096,d=128, optimized causal latency is 116.231567 ms, compared with 154.259377 ms for dense attention. Causal QK skips fully masked tiles. Softmax still handles rows, and PV still traverses the full sequence dimension, including zero probabilities. S/P storage also remains square. Therefore, halving valid attention pairs does not halve total work or storage.

The largest workspace is 4352 MiB, including 4096 MiB for S/P alone. Tiling and parallel reduction improve execution time while retaining these quadratic intermediates. This gives Parts B/C a clear materialized baseline for evaluating fusion and memory savings. It does not predict their eventual measured speedup.

### Profiler interpretation

The optimized dense profiles show high achieved occupancy and no local-memory allocation in the saved resource reports. High occupancy describes active warps; it does not establish efficient arithmetic or a saturated memory system. The stage timings identify matrix multiplication as the remaining runtime hotspot in the N=1024 ablation case.

At N=2048, softmax DRAM throughput is 382.17 GB/s, compared with 62.94 GB/s for QK and 64.98 GB/s for PV. The softmax pass performs repeated streaming reads/writes of S/P, so these counters are consistent with substantial memory activity. They do not prove a bandwidth limit without a sustained-bandwidth reference and stall analysis. The profiles cover optimized dense N=256/2048 only; they cannot establish the causal bottleneck or explain every naive-to-optimized difference.

Profiler replay, cache control and clock control can change execution conditions. Normal benchmarks keep buffers resident without explicit cache flushing. Profile durations and DRAM counters must therefore stay separate from normal latency and logical tensor byte counts. In particular, low or zero DRAM writes in a small profile do not imply that the kernel wrote no output.

The mechanisms above follow [NVIDIA's matrix-multiplication guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab). Profiling limits follow the [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control).

## Evidence limits

This completes the requested Part A materialized main grid and supporting representative/tuning/ablation/profile runs. It does not implement or measure fused FlashAttention, backward attention, mixed precision, or team Parts B/C. Large-shape exact FP64 agreement is not claimed. Representative sanitizer checks are not exhaustive safety proofs. All project text and chart labels use English. Localized diagnostics are English transcriptions; unrelated process names are omitted. Diagnostic codes, locations, measurements and exit codes are retained. Original hashes are recorded in `log_sources.csv` and the corresponding transcription records.
