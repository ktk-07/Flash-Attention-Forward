# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-14T154428Z-18c60f/benchmark`

CSV checks passed for 16 configurations and 320 samples.
All 144 stage reports passed the fixed tolerance.
This run covers 4/60 main-grid rows with tile 16 and softmax 256.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4,8,256,64 | dense | optimized | 8/256 | 0.429312 | 0.001688 |  |
| 4,8,256,64 | dense | optimized | 16/256 | 0.359072 | 0.001936 |  |
| 4,8,256,64 | dense | optimized | 32/256 | 0.430112 | 0.000904 |  |
| 4,8,256,64 | dense | optimized | 16/128 | 0.320976 | 0.002552 |  |
| 4,8,256,64 | causal | optimized | 8/256 | 0.355216 | 0.000976 |  |
| 4,8,256,64 | causal | optimized | 16/256 | 0.305440 | 0.001088 |  |
| 4,8,256,64 | causal | optimized | 32/256 | 0.366912 | 0.000464 |  |
| 4,8,256,64 | causal | optimized | 16/128 | 0.266080 | 0.000520 |  |
| 4,8,1024,64 | dense | optimized | 8/256 | 6.301680 | 0.023840 |  |
| 4,8,1024,64 | dense | optimized | 16/256 | 5.032240 | 0.070792 |  |
| 4,8,1024,64 | dense | optimized | 32/256 | 6.371376 | 0.323768 |  |
| 4,8,1024,64 | dense | optimized | 16/128 | 5.000592 | 0.026064 |  |
| 4,8,1024,64 | causal | optimized | 8/256 | 4.841664 | 0.024136 |  |
| 4,8,1024,64 | causal | optimized | 16/256 | 3.972944 | 0.262840 |  |
| 4,8,1024,64 | causal | optimized | 32/256 | 5.273920 | 0.402904 |  |
| 4,8,1024,64 | causal | optimized | 16/128 | 3.911792 | 0.184432 |  |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
Missing main-grid rows remain NOT_RUN.
Fused FlashAttention and Parts B/C remain outside this Part A measurement scope.
