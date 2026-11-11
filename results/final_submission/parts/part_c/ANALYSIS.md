# Part C report export

This bundle uses Part A column names with explicit precision, fingerprints and validation scope. All unprofiled timings and numerical checks are unchanged. See [native measured findings](raw/part_c/study/REPORT.md) and [supplementary profiler insights](supplementary/part_c/README.md).

Fused stage times, setup timing and inappropriate naive speedups are blank. Allocation fields measure explicit live buffers. Work.csv uses analytical valid pairs and source-derived CTA tile schedules, not measured executed FLOPs. The manifest records the actual oracle backend. Historical bundles keep their original reference provenance; fresh runs use the shared Part A CPU library.
