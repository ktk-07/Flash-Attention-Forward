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

