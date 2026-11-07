# Frozen winner profiling

B=4,H=8,d=64; three independent full Nsight captures at each N, medians below. These are fresh captures of the selected configurations. Instrumented timings are excluded from main results.

| Kernel | Config | N | Occupancy % | Eligible warps/scheduler | Shared actual/ideal | Excessive wavefronts | Long scoreboard % | Short scoreboard % |
|---|---|---|---|---|---|---|---|---|
| ptx | 2 | 256 | 14.90 | 0.21 | 3.976 | 1529856 | 54.76 | 10.18 |
| ptx | 2 | 4096 | 38.94 | 0.56 | 4.067 | 350846976 | 53.43 | 9.86 |
| skip | 4 | 256 | 46.81 | 1.16 | 1.000 | 0 | 18.35 | 7.06 |
| skip | 4 | 4096 | 65.02 | 1.66 | 1.000 | 0 | 17.70 | 4.89 |

## Traffic and resources at N=4096

| Kernel | Registers/thread | Shared KiB/block | DRAM read+write GiB | Local L1 sector traffic GiB | Tensor active % |
|---|---|---|---|---|---|
| ptx | 63 | 9.00 | 1.15 | 0.00 | 7.93 |
| skip | 40 | 24.00 | 11.17 | 137.17 | 0.00 |

The CUDA-core winner still generates substantial local-memory transactions despite a smaller per-thread local allocation than its default configuration. Smaller allocation size does not imply fewer executed transactions. Its measured occupancy is near the API resource ceiling, yet local-memory work remains an optimisation opportunity. PTX has no measured local sector traffic here; its long-scoreboard samples and shared excessive wavefronts identify different remaining constraints. These observations do not establish how much a future optimisation would accelerate either kernel.


Warp-state percentages describe sampled states, not runtime fractions. API resource estimates are in the main study CSVs; occupancy above is measured. Positive source excessive wavefronts support studying shared layouts, but do not predict the speedup from a layout change. DRAM bytes and local sectors are traffic, separate from the explicit allocation ledger. Clock management is disabled and caches are flushed by the profiler. Only d=64 is profiled here; do not extrapolate counters to other dimensions.

Regenerate offline: `python3 part_c/scripts/profile_frozen.py part_c/results/bounded_evaluation --analyze-only`.
