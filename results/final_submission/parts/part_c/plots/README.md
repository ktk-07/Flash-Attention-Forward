# Part C figure index

Plots use saved unprofiled timings; Nsight-instrumented durations never enter latency or speedup. Each PNG has a PDF counterpart. Native progression uses fixed defaults. FA1/FA2 comparisons preserve the recorded configuration policy; frozen PTX comparisons require matching study measurements. Historical bundles retain their recorded baselines.

## Main figures

- [cuda_core_progression](part_c/cuda_core_progression.png) ([PDF](part_c/cuda_core_progression.pdf))
- [device_memory](part_c/device_memory.png) ([PDF](part_c/device_memory.pdf))
- [fa1_cross_precision](part_c/fa1_cross_precision.png) ([PDF](part_c/fa1_cross_precision.pdf))
- [fa1_fa2_comparison](part_c/fa1_fa2_comparison.png) ([PDF](part_c/fa1_fa2_comparison.pdf))
- [fa1_fa2_latency](part_c/fa1_fa2_latency.png) ([PDF](part_c/fa1_fa2_latency.pdf))
- [frozen_family_comparison](part_c/frozen_family_comparison.png) ([PDF](part_c/frozen_family_comparison.pdf))
- [frozen_tuning_speedup](part_c/frozen_tuning_speedup.png) ([PDF](part_c/frozen_tuning_speedup.pdf))
- [matched_vector_load](part_c/matched_vector_load.png) ([PDF](part_c/matched_vector_load.pdf))
- [tensor_core_progression](part_c/tensor_core_progression.png) ([PDF](part_c/tensor_core_progression.pdf))
- [tuning_heatmap](part_c/tuning_heatmap.png) ([PDF](part_c/tuning_heatmap.pdf))

Matched latency and speedup inputs, exact case IDs/configurations and quartiles: [fa1_fa2_comparison.csv](../fa1_fa2_comparison.csv). Cross-precision ratios: [fa1_cross_precision.csv](../fa1_cross_precision.csv). Figure metadata: [fa1_comparison_index.json](../fa1_comparison_index.json).

Additional profiler charts and retained insights: [supplementary/part_c](../supplementary/part_c/README.md). Missing counters remain unavailable.
