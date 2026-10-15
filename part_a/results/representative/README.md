# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-13T105709Z-ec50cd/benchmark`

CSV checks passed for 16 configurations and 320 samples.
All 192 stage reports passed the fixed tolerance.
This run covers 4/60 main-grid rows with tile 16 and softmax 256.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1,1,33,32 | dense | naive | 16/256 | 0.032048 | 0.004240 | 1.000 |
| 1,1,33,32 | dense | tiled | 16/256 | 0.029840 | 0.002848 | 1.074 |
| 1,1,33,32 | dense | parallel | 16/256 | 0.029392 | 0.003552 | 1.090 |
| 1,1,33,32 | dense | optimized | 16/256 | 0.028464 | 0.003464 | 1.126 |
| 1,1,33,32 | causal | naive | 16/256 | 0.032208 | 0.005008 | 1.000 |
| 1,1,33,32 | causal | tiled | 16/256 | 0.031952 | 0.003216 | 1.008 |
| 1,1,33,32 | causal | parallel | 16/256 | 0.026656 | 0.001232 | 1.208 |
| 1,1,33,32 | causal | optimized | 16/256 | 0.027088 | 0.002656 | 1.189 |
| 4,8,256,64 | dense | naive | 16/256 | 1.719680 | 0.023384 | 1.000 |
| 4,8,256,64 | dense | tiled | 16/256 | 0.455232 | 0.048192 | 3.778 |
| 4,8,256,64 | dense | parallel | 16/256 | 1.625360 | 0.175200 | 1.058 |
| 4,8,256,64 | dense | optimized | 16/256 | 0.359584 | 0.006000 | 4.782 |
| 4,8,256,64 | causal | naive | 16/256 | 0.985104 | 0.012232 | 1.000 |
| 4,8,256,64 | causal | tiled | 16/256 | 0.334128 | 0.004384 | 2.948 |
| 4,8,256,64 | causal | parallel | 16/256 | 0.959616 | 0.169304 | 1.027 |
| 4,8,256,64 | causal | optimized | 16/256 | 0.305824 | 0.001712 | 3.221 |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
Missing main-grid rows remain NOT_RUN. This run alone does not establish complete proposal coverage.
