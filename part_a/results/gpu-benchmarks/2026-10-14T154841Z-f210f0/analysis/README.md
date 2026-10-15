# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-14T154841Z-f210f0/benchmark`

CSV checks passed for 16 configurations and 320 samples.
All 144 stage reports passed the fixed tolerance.
This run covers 8/60 main-grid rows with tile 16 and softmax 128.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4,8,256,64 | dense | naive | 16/128 | 1.725664 | 0.003744 | 1.000 |
| 4,8,256,64 | dense | tiled | 16/128 | 0.455168 | 0.004168 | 3.791 |
| 4,8,256,64 | dense | parallel | 16/128 | 1.594208 | 0.001056 | 1.082 |
| 4,8,256,64 | dense | optimized | 16/128 | 0.323792 | 0.001432 | 5.330 |
| 4,8,256,64 | causal | naive | 16/128 | 0.987312 | 0.004712 | 1.000 |
| 4,8,256,64 | causal | tiled | 16/128 | 0.333136 | 0.004968 | 2.964 |
| 4,8,256,64 | causal | parallel | 16/128 | 0.919744 | 0.004128 | 1.073 |
| 4,8,256,64 | causal | optimized | 16/128 | 0.266736 | 0.000576 | 3.701 |
| 4,8,1024,64 | dense | naive | 16/128 | 33.099392 | 0.183254 | 1.000 |
| 4,8,1024,64 | dense | tiled | 16/128 | 13.255504 | 1.784024 | 2.497 |
| 4,8,1024,64 | dense | parallel | 16/128 | 26.390512 | 1.350647 | 1.254 |
| 4,8,1024,64 | dense | optimized | 16/128 | 5.176800 | 0.901712 | 6.394 |
| 4,8,1024,64 | causal | naive | 16/128 | 17.900768 | 2.722904 | 1.000 |
| 4,8,1024,64 | causal | tiled | 16/128 | 6.805824 | 0.270000 | 2.630 |
| 4,8,1024,64 | causal | parallel | 16/128 | 14.356448 | 0.419824 | 1.247 |
| 4,8,1024,64 | causal | optimized | 16/128 | 3.906528 | 0.023016 | 4.582 |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
Missing main-grid rows remain NOT_RUN.
Fused FlashAttention and Parts B/C remain outside this Part A measurement scope.
