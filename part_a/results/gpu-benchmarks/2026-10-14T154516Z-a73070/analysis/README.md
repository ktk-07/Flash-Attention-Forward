# Measured attention results

Source data: `results/gpu-benchmarks/2026-10-14T154516Z-a73070/benchmark`

CSV checks passed for 60 configurations and 1200 samples.
All 120 stage reports passed the fixed tolerance.
This run covers 60/60 main-grid rows with tile 16 and softmax 128.

| B,H,N,d | Mode | Variant | Tile/threads | Median ms | IQR ms | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4,8,256,32 | dense | naive | 16/128 | 0.958688 | 0.001592 | 1.000 |
| 4,8,256,32 | dense | optimized | 16/128 | 0.197760 | 0.001120 | 4.848 |
| 4,8,256,32 | causal | naive | 16/128 | 0.554096 | 0.001472 | 1.000 |
| 4,8,256,32 | causal | optimized | 16/128 | 0.174416 | 0.000664 | 3.177 |
| 4,8,256,64 | dense | naive | 16/128 | 1.719616 | 0.002856 | 1.000 |
| 4,8,256,64 | dense | optimized | 16/128 | 0.322336 | 0.002960 | 5.335 |
| 4,8,256,64 | causal | naive | 16/128 | 0.979680 | 0.005496 | 1.000 |
| 4,8,256,64 | causal | optimized | 16/128 | 0.267072 | 0.000880 | 3.668 |
| 4,8,256,128 | dense | naive | 16/128 | 3.245936 | 0.009448 | 1.000 |
| 4,8,256,128 | dense | optimized | 16/128 | 0.575936 | 0.002888 | 5.636 |
| 4,8,256,128 | causal | naive | 16/128 | 1.838192 | 0.004304 | 1.000 |
| 4,8,256,128 | causal | optimized | 16/128 | 0.459872 | 0.001144 | 3.997 |
| 4,8,512,32 | dense | naive | 16/128 | 5.321840 | 0.304080 | 1.000 |
| 4,8,512,32 | dense | optimized | 16/128 | 0.771200 | 0.002568 | 6.901 |
| 4,8,512,32 | causal | naive | 16/128 | 2.252656 | 0.013920 | 1.000 |
| 4,8,512,32 | causal | optimized | 16/128 | 0.640352 | 0.002544 | 3.518 |
| 4,8,512,64 | dense | naive | 16/128 | 8.728416 | 0.629952 | 1.000 |
| 4,8,512,64 | dense | optimized | 16/128 | 1.284576 | 0.015752 | 6.795 |
| 4,8,512,64 | causal | naive | 16/128 | 3.946992 | 0.216896 | 1.000 |
| 4,8,512,64 | causal | optimized | 16/128 | 1.013536 | 0.007936 | 3.894 |
| 4,8,512,128 | dense | naive | 16/128 | 15.021824 | 0.824096 | 1.000 |
| 4,8,512,128 | dense | optimized | 16/128 | 2.310176 | 0.024992 | 6.502 |
| 4,8,512,128 | causal | naive | 16/128 | 7.353440 | 0.011744 | 1.000 |
| 4,8,512,128 | causal | optimized | 16/128 | 1.797360 | 0.009216 | 4.091 |
| 4,8,1024,32 | dense | naive | 16/128 | 20.537584 | 0.444279 | 1.000 |
| 4,8,1024,32 | dense | optimized | 16/128 | 2.962576 | 0.015456 | 6.932 |
| 4,8,1024,32 | causal | naive | 16/128 | 10.332352 | 0.397584 | 1.000 |
| 4,8,1024,32 | causal | optimized | 16/128 | 2.391632 | 0.039520 | 4.320 |
| 4,8,1024,64 | dense | naive | 16/128 | 33.251345 | 0.537623 | 1.000 |
| 4,8,1024,64 | dense | optimized | 16/128 | 5.008608 | 0.078728 | 6.639 |
| 4,8,1024,64 | causal | naive | 16/128 | 17.173472 | 0.405048 | 1.000 |
| 4,8,1024,64 | causal | optimized | 16/128 | 3.911312 | 0.173184 | 4.391 |
| 4,8,1024,128 | dense | naive | 16/128 | 57.724768 | 0.742849 | 1.000 |
| 4,8,1024,128 | dense | optimized | 16/128 | 9.091648 | 0.220048 | 6.349 |
| 4,8,1024,128 | causal | naive | 16/128 | 31.176816 | 0.097424 | 1.000 |
| 4,8,1024,128 | causal | optimized | 16/128 | 6.969168 | 0.045176 | 4.474 |
| 4,8,2048,32 | dense | naive | 16/128 | 81.449135 | 1.876431 | 1.000 |
| 4,8,2048,32 | dense | optimized | 16/128 | 11.842480 | 0.425072 | 6.878 |
| 4,8,2048,32 | causal | naive | 16/128 | 45.170977 | 4.304505 | 1.000 |
| 4,8,2048,32 | causal | optimized | 16/128 | 9.457280 | 0.266592 | 4.776 |
| 4,8,2048,64 | dense | naive | 16/128 | 131.634705 | 1.853439 | 1.000 |
| 4,8,2048,64 | dense | optimized | 16/128 | 20.414145 | 0.521176 | 6.448 |
| 4,8,2048,64 | causal | naive | 16/128 | 68.058464 | 0.514263 | 1.000 |
| 4,8,2048,64 | causal | optimized | 16/128 | 15.880608 | 0.399983 | 4.286 |
| 4,8,2048,128 | dense | naive | 16/128 | 232.456352 | 0.920166 | 1.000 |
| 4,8,2048,128 | dense | optimized | 16/128 | 36.905584 | 0.193501 | 6.299 |
| 4,8,2048,128 | causal | naive | 16/128 | 123.937504 | 1.663675 | 1.000 |
| 4,8,2048,128 | causal | optimized | 16/128 | 28.322191 | 0.285463 | 4.376 |
| 4,8,4096,32 | dense | naive | 16/128 | 322.991287 | 5.318565 | 1.000 |
| 4,8,4096,32 | dense | optimized | 16/128 | 48.957008 | 1.222222 | 6.597 |
| 4,8,4096,32 | causal | naive | 16/128 | 162.615738 | 4.147785 | 1.000 |
| 4,8,4096,32 | causal | optimized | 16/128 | 38.301792 | 0.136785 | 4.246 |
| 4,8,4096,64 | dense | naive | 16/128 | 523.843384 | 3.999405 | 1.000 |
| 4,8,4096,64 | dense | optimized | 16/128 | 85.741951 | 0.832872 | 6.110 |
| 4,8,4096,64 | causal | naive | 16/128 | 272.618546 | 2.833076 | 1.000 |
| 4,8,4096,64 | causal | optimized | 16/128 | 64.501057 | 0.286123 | 4.227 |
| 4,8,4096,128 | dense | naive | 16/128 | 922.806335 | 7.577408 | 1.000 |
| 4,8,4096,128 | dense | optimized | 16/128 | 154.259377 | 0.613548 | 5.982 |
| 4,8,4096,128 | causal | naive | 16/128 | 497.233734 | 4.802155 | 1.000 |
| 4,8,4096,128 | causal | optimized | 16/128 | 116.231567 | 0.757023 | 4.278 |

Workspace bytes describe allocated buffers. They do not measure DRAM traffic.
The data represents warmed, resident-buffer execution. Profiler times are excluded.
Large timed shapes use representative CPU validation and finite-output checks.
All 60 Part A main-grid rows were measured with one frozen configuration.
Fused FlashAttention and Parts B/C remain outside this Part A measurement scope.
