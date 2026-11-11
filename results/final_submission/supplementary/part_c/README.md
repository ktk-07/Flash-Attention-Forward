# Part C supplementary evidence

All native data, MIO/scoreboard counters, Nsight reports, logs, source snapshots and insights are retained byte for byte under [raw study](../../raw/part_c/study/). Only public column names are adapted.

The profiles table contains all raw Nsight metric columns plus the native derived metrics. Unavailable counters remain explicit. Sampled warp-state percentages are not percentages of runtime. Shared actual/ideal/excessive wavefronts diagnose source excess; aggregate conflict counters alone include arbitration. Local sector traffic is not allocated storage.

## Native reports

- [Measured study](../../raw/part_c/study/REPORT.md)
- [Frozen profiler findings](../../raw/part_c/study/profiling/REPORT.md)

## Additional generated profiler plots

- [occupancy.png](plots/occupancy.png)
- [occupancy.pdf](plots/occupancy.pdf)
- [eligible_warps.png](plots/eligible_warps.png)
- [eligible_warps.pdf](plots/eligible_warps.pdf)
- [tensor_activity.png](plots/tensor_activity.png)
- [tensor_activity.pdf](plots/tensor_activity.pdf)
- [warp_states.png](plots/warp_states.png)
- [warp_states.pdf](plots/warp_states.pdf)
- [shared_wavefronts.png](plots/shared_wavefronts.png)
- [shared_wavefronts.pdf](plots/shared_wavefronts.pdf)
- [traffic.png](plots/traffic.png)
- [traffic.pdf](plots/traffic.pdf)
