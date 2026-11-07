# Part C results

| Dataset | Purpose | Start here |
|---|---|---|
| `bounded_evaluation/` | Main common-input tuning, frozen benchmark, matched-load and memory study | [REPORT.md](bounded_evaluation/REPORT.md) |
| `bounded_evaluation/profiling/` | Fresh hardware counters for frozen d=64 winners | [REPORT.md](bounded_evaluation/profiling/REPORT.md) |
| `layout_validation/` | Earlier default-configuration regression, benchmark and sanitizer evidence | [README.md](layout_validation/README.md) |
| `nsight_analysis/` | Historical eight-kernel default profiles and source/SASS analysis | [FINDINGS.md](nsight_analysis/FINDINGS.md) |

Use `bounded_evaluation/figures/` for the seven new report figures. Raw case CSVs, configuration plans, input fingerprints, command records and source snapshots support reproduction; do not treat historical datasets as one combined benchmark. Part A's historical results were collected on another GPU.

Development attempts and capability probes are archived locally outside this result index. Large binary Nsight reports are excluded from Git and retain checksums in the datasets; restore them from the supplied profiling artifact before running full integrity audits.

## What to retain for submission

Keep the datasets cited by the report. The main recommended bundle is `bounded_evaluation/`: raw case/combined CSVs, plans and frozen selections, source/analysis manifests and snapshots, audits, report and figures. Retain `verification/` for numerical/sanitizer evidence and `profiling/` for the fresh winner counters. This supports reproduction; summary CSVs and plots alone are insufficient.

`layout_validation/` is optional older default-workflow evidence. Keep `nsight_analysis/` if the report uses its eight-kernel/source-SASS findings; otherwise it is supplementary. Local debug/probe runs and planning notes are outside this result index in ignored development archives. Large `.ncu-rep` binaries need not be committed; supply them separately with their recorded hashes when required for a full profiling audit. Ignore rules do not untrack previously tracked files.

A new root `run_all.sh` writes fresh requested-workflow results under `results/<run_id>/`; it does not overwrite or automatically import any dataset here.
