# Measured results

Measurements use RTX 5060 Ti 8151 MiB, compute capability 12.0, driver 610.88, CUDA Toolkit 13.4, MSVC 19.44 and Release builds for sm_120, without fast math.

## Data

| Experiment | Configurations | Timing samples | Data |
| --- | ---: | ---: | --- |
| Representative | 8 | 160 | [Summary](validation/2026-10-15T052608Z-fa1f796/benchmark/summary.csv) |
| Tuning | 16 | 320 | [Summary](gpu-benchmarks/2026-10-14T154428Z-18c60f/benchmark/summary.csv) |
| Main grid | 60 | 1200 | [Summary](gpu-benchmarks/2026-10-14T154516Z-a73070/benchmark/summary.csv) |
| Ablation | 16 | 320 | [Summary](gpu-benchmarks/2026-10-14T154841Z-f210f0/benchmark/summary.csv) |

Each data directory retains its plan, completion status, metadata, numerical comparisons, raw timing samples and kernel resource records. Representative and main-grid measurements are separate runs. Profiling is recorded separately from normal timing.

Inputs use seed 5206 and deterministic FP32 values in [-1,1). GPU S/P/O are compared against FP64 arithmetic with atol=rtol=1e-4. Small timed shapes receive exact comparisons before and after timing. Large timed shapes use representative comparisons and finite-output checks; full large-shape FP64 agreement is not claimed.

## Configuration

Four candidates were compared at B=4,H=8,d=64 in dense and causal modes:

| N | Mode | (8,256) ms | (16,256) ms | (32,256) ms | (16,128) ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Dense | 0.429312 | 0.359072 | 0.430112 | 0.320976 |
| 256 | Causal | 0.355216 | 0.305440 | 0.366912 | 0.266080 |
| 1024 | Dense | 6.301680 | 5.032240 | 6.371376 | 5.000592 |
| 1024 | Causal | 4.841664 | 3.972944 | 5.273920 | 3.911792 |

(16,128) had the lowest median in all four cases and was [frozen before the main run](gpu-benchmarks/2026-10-14T154318Z-review/configuration-frozen-utc.txt). The N=1024 advantage is small relative to some observed IQRs. This supports the bounded selection, not a universal optimum. The library default remains (16,256).

## Performance

The main grid uses B=4,H=8,N={256,512,1024,2048,4096},d={32,64,128}, FP32 and both modes. All 60 planned rows completed. Five warmups and 20 repeats use CUDA events with resident preallocated buffers. Allocation, transfers, CPU comparisons and file I/O are excluded. Forward and stage timings use separate passes.

Optimized speedup over matched naive medians ranges from 4.848x to 6.932x for dense attention and 3.177x to 4.776x for causal attention.

At N=4096,d=128:

| Mode | Naive median ms | Optimized median ms | Optimized IQR ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Dense | 922.806335 | 154.259377 | 0.613548 | 5.982x |
| Causal | 497.233734 | 116.231567 | 0.757023 | 4.278x |

[Main-grid figures](gpu-benchmarks/2026-10-14T154516Z-a73070/analysis/plots) show latency, speedup and allocated workspace for both modes.

## Optimization effects

Dense ablation medians at d=64:

| N | Naive ms | Tiled ms | Parallel ms | Optimized ms |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.725664 | 0.455168 | 1.594208 | 0.323792 |
| 1024 | 33.099392 | 13.255504 | 26.390512 | 5.176800 |

Tiling reuses shared-memory Q/K and P/V values. At N=1024, QK falls from 22.482080 ms to 2.433280 ms in the tiled-only variant, while PV falls from 2.708496 ms to 2.253792 ms. QK shows the larger benefit.

Parallel softmax splits each row's maximum, sum and normalization across a block. Its N=1024 stage falls from 7.867904 ms to 0.698368 ms in the parallel-only variant. After tiling, serial softmax becomes the largest stage at 8.289632 ms. Combining both optimizations reduces this hotspot; optimized QK/softmax/PV medians are 2.397152/0.681296/2.108928 ms.

Stage medians need not sum to the separately measured forward median. The N=1024 dense optimized ablation IQR is 0.901712 ms, so small differences between separate runs need caution.

Causal QK skips fully masked tiles. PV still traverses the full sequence dimension and S/P remain square, so causal attention does not halve total latency or storage. The largest workspace is 4352 MiB, including 4096 MiB for S/P. Allocated bytes exclude context overhead and must not be interpreted as DRAM traffic.

## Profiling and limits

Optimized dense profiles cover N=256 and N=2048 at B=4,H=8,d=64 with (16,128): [small counter export](profiles/2026-10-14T154925Z-74efb1/metrics.csv), [large counter export](profiles/2026-10-14T154957Z-553569/metrics.csv). Their metadata records the profiled shapes and build settings.

QK uses 48 registers/thread and 2112 shared bytes/block; softmax uses 26 and 512; PV uses 36 and 2048. Theoretical occupancy is 83.33%, 100% and 100%. High achieved occupancy alone does not establish efficient execution or identify a bottleneck.

At N=2048, measured softmax DRAM throughput is 382.17 GB/s, versus 62.94 GB/s for QK and 64.98 GB/s for PV. These counters show memory activity; they do not establish bandwidth saturation without a sustained-bandwidth reference and stall analysis. Profiler replay and cache/clock controls differ from normal timing conditions, so profiler durations remain separate.

The main-run state capture contains pstate and temperature but omits power and clocks. Desktop activity and unlocked clocks limit reproducibility. No fused B/C performance is included.
