# Generated figures

All paths refer to this bundle. Latency uses unprofiled medians; Q1–Q3 is within-case spread, not a confidence interval. Fixed-default progression, frozen comparisons and cross-precision tradeoffs are labelled separately. Historical plots retain their captured implementation.

## part_a

- [materialised_comparison](part_a/materialised_comparison.png) ([PDF](part_a/materialised_comparison.pdf)): Part A d=64 progression, medians and interquartile ranges. Tensor Core variants use FP16 operands and FP32 accumulation/output.
- [kernel_progression_fp32](part_a/kernel_progression_fp32.png) ([PDF](part_a/kernel_progression_fp32.pdf)): All final-grid variants; median and interquartile range. Precision contracts remain separate.
- [kernel_progression_fp16](part_a/kernel_progression_fp16.png) ([PDF](part_a/kernel_progression_fp16.pdf)): All final-grid variants; median and interquartile range. Precision contracts remain separate.
- [throughput_and_memory](part_a/throughput_and_memory.png) ([PDF](part_a/throughput_and_memory.pdf)): Measured forward throughput; explicit allocation ledger, not a device-wide peak.
- [numerical_error](part_a/numerical_error.png) ([PDF](part_a/numerical_error.pdf)): Errors from exact per-case CPU validation; full or sampled scope retained in CSV.
- [tuning](part_a/tuning.png) ([PDF](part_a/tuning.pdf)): Bounded candidate sweep, grouped by mode and configuration. Selection rules and frozen configurations are retained.
- [dram_traffic](part_a/dram_traffic.png) ([PDF](part_a/dram_traffic.pdf)): Median read/write traffic over three instrumented captures per stage. These cold-cache captures are separate from normal forward timing.
- [profiler_counters](part_a/profiler_counters.png) ([PDF](part_a/profiler_counters.pdf)): Median raw Nsight counter across captures. MIO sample counts are not percentages or runtime fractions. Missing counters are omitted and retained in profiles.csv with reasons.
- [stage_breakdown](part_a/stage_breakdown.png) ([PDF](part_a/stage_breakdown.pdf)): Independent stage passes; fused stages are unavailable, not zero.

## part_b

- [kernel_progression_fp32](part_b/kernel_progression_fp32.png) ([PDF](part_b/kernel_progression_fp32.pdf)): All final-grid variants; median and interquartile range. Precision contracts remain separate.
- [throughput_and_memory](part_b/throughput_and_memory.png) ([PDF](part_b/throughput_and_memory.pdf)): Measured forward throughput; explicit allocation ledger, not a device-wide peak.
- [numerical_error](part_b/numerical_error.png) ([PDF](part_b/numerical_error.pdf)): Errors from exact per-case CPU validation; full or sampled scope retained in CSV.
- [tuning](part_b/tuning.png) ([PDF](part_b/tuning.pdf)): Bounded candidate sweep, grouped by mode and configuration. Selection rules and frozen configurations are retained.
- [profiler_counters](part_b/profiler_counters.png) ([PDF](part_b/profiler_counters.pdf)): Median raw Nsight counter across captures. MIO sample counts are not percentages or runtime fractions. Missing counters are omitted and retained in profiles.csv with reasons.
- [latency](part_b/latency.png) ([PDF](part_b/latency.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [memory](part_b/memory.png) ([PDF](part_b/memory.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.

## part_c

- [kernel_progression_fp32](part_c/kernel_progression_fp32.png) ([PDF](part_c/kernel_progression_fp32.pdf)): All final-grid variants; median and interquartile range. Precision contracts remain separate.
- [kernel_progression_fp16](part_c/kernel_progression_fp16.png) ([PDF](part_c/kernel_progression_fp16.pdf)): All final-grid variants; median and interquartile range. Precision contracts remain separate.
- [throughput_and_memory](part_c/throughput_and_memory.png) ([PDF](part_c/throughput_and_memory.pdf)): Measured forward throughput; explicit allocation ledger, not a device-wide peak.
- [numerical_error](part_c/numerical_error.png) ([PDF](part_c/numerical_error.pdf)): Errors from exact per-case CPU validation; full or sampled scope retained in CSV.
- [tuning](part_c/tuning.png) ([PDF](part_c/tuning.pdf)): Bounded candidate sweep, grouped by mode and configuration. Selection rules and frozen configurations are retained.
- [profiler_counters](part_c/profiler_counters.png) ([PDF](part_c/profiler_counters.pdf)): Median raw Nsight counter across captures. MIO sample counts are not percentages or runtime fractions. Missing counters are omitted and retained in profiles.csv with reasons.
- [fa1_fa2_comparison](part_c/fa1_fa2_comparison.png) ([PDF](part_c/fa1_fa2_comparison.pdf)): CSV includes exact case IDs, precision labels, medians and quartiles; no official-library parity claim.
- [fa1_cross_precision](part_c/fa1_cross_precision.png) ([PDF](part_c/fa1_cross_precision.pdf)): CSV includes exact case IDs, precision labels, medians and quartiles; no official-library parity claim.
- [fa1_fa2_latency](part_c/fa1_fa2_latency.png) ([PDF](part_c/fa1_fa2_latency.pdf)): Matched FP32 and Tensor Core latency panels; median with Q1–Q3, using exact cases from fa1_fa2_comparison.csv. Quartiles are within-case spread, not confidence intervals.
- [cuda_core_progression](part_c/cuda_core_progression.png) ([PDF](part_c/cuda_core_progression.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [device_memory](part_c/device_memory.png) ([PDF](part_c/device_memory.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [frozen_family_comparison](part_c/frozen_family_comparison.png) ([PDF](part_c/frozen_family_comparison.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [frozen_tuning_speedup](part_c/frozen_tuning_speedup.png) ([PDF](part_c/frozen_tuning_speedup.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [matched_vector_load](part_c/matched_vector_load.png) ([PDF](part_c/matched_vector_load.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [tensor_core_progression](part_c/tensor_core_progression.png) ([PDF](part_c/tensor_core_progression.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [tuning_heatmap](part_c/tuning_heatmap.png) ([PDF](part_c/tuning_heatmap.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [eligible_warps](../supplementary/part_c/plots/eligible_warps.png) ([PDF](../supplementary/part_c/plots/eligible_warps.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [occupancy](../supplementary/part_c/plots/occupancy.png) ([PDF](../supplementary/part_c/plots/occupancy.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [shared_wavefronts](../supplementary/part_c/plots/shared_wavefronts.png) ([PDF](../supplementary/part_c/plots/shared_wavefronts.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [tensor_activity](../supplementary/part_c/plots/tensor_activity.png) ([PDF](../supplementary/part_c/plots/tensor_activity.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [traffic](../supplementary/part_c/plots/traffic.png) ([PDF](../supplementary/part_c/plots/traffic.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.
- [warp_states](../supplementary/part_c/plots/warp_states.png) ([PDF](../supplementary/part_c/plots/warp_states.pdf)): Retained part-specific figure; source data, recorded configurations and processing provenance remain in the child bundle.

## combined

- [cross_part](combined/cross_part.png) ([PDF](combined/cross_part.pdf)): Separate dense FP32, causal FP32 and causal FP16 panels. Frozen selections on identical inputs; target variants and head dimensions are explicit. FP16 probability rounding order differs.
- [materialised_fused_memory](combined/materialised_fused_memory.png) ([PDF](combined/materialised_fused_memory.pdf)): A dense, B dense and C causal explicit allocation ledgers, including separate C FA1 workspace. Capacity comparison across labelled workloads, not total process memory or traffic.
- [dense_causal_work](combined/dense_causal_work.png) ([PDF](combined/dense_causal_work.pdf)): Distinct dense/causal workloads on identical inputs; analytical pairs and measured traffic remain separate.

