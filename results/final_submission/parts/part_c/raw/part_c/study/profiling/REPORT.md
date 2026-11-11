# FA1 baselines and frozen FA2 profiling

B=4,H=8,d=64; 3 captures at N=[256, 4096]. Smoke=False. Profiler times are excluded from benchmark summaries.

Shared bytes include static+dynamic storage for new FA1-aware plans. Missing counters/zero sample denominators remain unavailable.

| Kernel | Config | N | Occupancy % | Eligible warps | Shared KiB | Excess wavefronts | Long scoreboard % | Short scoreboard % |
|---|---|---|---|---|---|---|---|---|
| fa1 | 0 | 256 | 2.08 | 0.10 | 36.00 | 37425152 | 5.58 | 67.31 |
| fa1 | 0 | 4096 | 2.08 | 0.09 | 36.00 | 8974303232 | 6.83 | 67.65 |
| fa1_ptx | 0 | 256 | 4.17 | 0.10 | 13.00 | 1529856 | 54.70 | 10.01 |
| fa1_ptx | 0 | 4096 | 4.17 | 0.07 | 13.00 | 350846976 | 66.02 | 8.25 |
| ptx | 2 | 256 | 14.93 | 0.21 | 9.00 | 1529856 | 55.10 | 10.00 |
| ptx | 2 | 4096 | 37.41 | 0.55 | 9.00 | 350846976 | 53.20 | 9.92 |
| vector | 2 | 256 | 28.19 | 0.74 | 20.00 | 0 | 19.83 | 10.97 |
| vector | 2 | 4096 | 32.89 | 0.84 | 20.00 | 0 | 24.33 | 10.18 |

Use these measured resource/traffic/scheduler values to test bottleneck hypotheses; sampled warp states are not runtime fractions. Explicit allocation is not traffic. Only d=64 is profiled. The two FA1 baselines are this project's implementations, not official FA1 parity measurements.

Regenerate offline: `python3 part_c/scripts/profile_frozen.py /home/buntr/Projects/Flash-Attention-CUDA/results/final_submission/parts/part_c/raw/part_c/study --analyze-only`.
