# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-13T105851Z-645406/benchmark`

CSV checks passed for 12 configurations and 240 samples.
All 108 stage reports passed the fixed tolerance.
This run covers 4/60 main-grid rows with tile 16 and softmax 256.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4,8,256,64 | dense | optimized | 8/128 | 0.392096 | 0.001320 |  |
| 4,8,256,64 | dense | optimized | 16/256 | 0.359712 | 0.001312 |  |
| 4,8,256,64 | dense | optimized | 32/256 | 0.430432 | 0.001160 |  |
| 4,8,256,64 | causal | optimized | 8/128 | 0.315856 | 0.001344 |  |
| 4,8,256,64 | causal | optimized | 16/256 | 0.305808 | 0.001224 |  |
| 4,8,256,64 | causal | optimized | 32/256 | 0.367488 | 0.001264 |  |
| 4,8,1024,64 | dense | optimized | 8/128 | 6.274592 | 0.285944 |  |
| 4,8,1024,64 | dense | optimized | 16/256 | 5.027856 | 0.017040 |  |
| 4,8,1024,64 | dense | optimized | 32/256 | 6.370368 | 0.237264 |  |
| 4,8,1024,64 | causal | optimized | 8/128 | 4.808288 | 0.193680 |  |
| 4,8,1024,64 | causal | optimized | 16/256 | 3.954464 | 0.162712 |  |
| 4,8,1024,64 | causal | optimized | 32/256 | 5.114128 | 0.167640 |  |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
Missing main-grid rows remain NOT_RUN. This run alone does not establish complete proposal coverage.
