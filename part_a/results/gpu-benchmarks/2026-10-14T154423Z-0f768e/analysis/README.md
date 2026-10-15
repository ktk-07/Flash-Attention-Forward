# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-14T154423Z-0f768e/benchmark`

CSV checks passed for 8 configurations and 160 samples.
All 96 stage reports passed the fixed tolerance.
This run covers 4/60 main-grid rows with tile 16 and softmax 256.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1,1,33,32 | dense | naive | 16/256 | 0.032224 | 0.005568 | 1.000 |
| 1,1,33,32 | dense | optimized | 16/256 | 0.028608 | 0.002088 | 1.126 |
| 1,1,33,32 | causal | naive | 16/256 | 0.035072 | 0.004712 | 1.000 |
| 1,1,33,32 | causal | optimized | 16/256 | 0.028736 | 0.002712 | 1.220 |
| 4,8,256,64 | dense | naive | 16/256 | 1.720240 | 0.011080 | 1.000 |
| 4,8,256,64 | dense | optimized | 16/256 | 0.359776 | 0.001368 | 4.781 |
| 4,8,256,64 | causal | naive | 16/256 | 0.980816 | 0.003528 | 1.000 |
| 4,8,256,64 | causal | optimized | 16/256 | 0.305520 | 0.001304 | 3.210 |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
Missing main-grid rows remain NOT_RUN.
Fused FlashAttention and Parts B/C remain outside this Part A measurement scope.
