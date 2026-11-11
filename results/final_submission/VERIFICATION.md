# Full-run verification

The retained `final_submission` A/B/C experiment passes the saved-result checks. The measurements were produced on one NVIDIA GeForce RTX 3070 Ti Laptop GPU (SM86), UUID `GPU-f5bd2e3d-ddc7-7006-0401-dd3b93a77330`. Verification did not rerun GPU benchmarks or change recorded measurements.

## Evidence checked

| Check | Result |
|---|---|
| Root and A/B/C completion/audits | Passed; requested full grid present |
| Main grid | B=4, H=8; N=256/512/1024/2048/4096; d=32/64/128 |
| Numerical records | 1,657 PASS; zero exported failures/nonfinite outputs |
| Timing | 750 cases including tuning/diagnostics, 15,000 samples; 330 main-grid cases |
| Timing statistics and correctness joins | Recomputed medians/quartiles/throughput; exact case/config/input validation links |
| Shared inputs and physical GPU | Matching original/effective fingerprints and UUID |
| Fresh profiler captures | A=108, B=6, C=24; 138 total |
| Recorded commands | All 1,282 nested command records have exit code zero |
| Independent work/allocation arithmetic | All 750 valid-pair counts and 512 allocation ledgers agree with shape/precision formulas |
| FA1 comparison arithmetic | All 45 matched/cross-precision rows agree with their canonical cases and ratios |
| Kernel/header identity | Checked current A/B/C kernel/header hashes match the recorded source |
| Figures | All 41 indexed PNG/PDF pairs exist and pass image/PDF readability checks |
| Offline reproduction | Native re-extraction, aggregation and plot regeneration pass; all 33 root/child CSVs remain byte-identical |

A's recorded sanitizer suite contains 12 checks; B contains 32. C's default smoke suite contains four checks and its frozen-kernel verification contains 16, all successful. C also retains 424 default regression passes and eight explicitly unsupported dimension/kernel combinations. Unsupported cases are not treated as passing executions. Large timing cases use their recorded full/sampled reference scope plus all-output finiteness; these checks are not exhaustive validation of every large output value.

## Measured performance comparisons

Speedup below means baseline median / candidate median, over the 15 main-grid shapes in each row. Geometric means summarize those shapes, not independent experimental sessions. A's FP32 baseline here is `optimized`, not `naive`.

| Comparison | Speedup range | Geometric mean |
|---|---:|---:|
| A-dense FP32 / B-selected dense FP32 | 0.205–0.703× | 0.428× |
| A-causal FP32 / C-frozen CUDA-core FA2 | 0.547–1.179× | 0.781× |
| A-causal PTX / C-causal PTX FA2 | 1.182–6.131× | 2.483× |
| C-FP32 FA1 / C-frozen CUDA-core FA2 | 4.083–6.554× | 5.290× |
| C-PTX FA1 / C-PTX FA2 | 3.032–13.079× | 6.433× |

Thus dense FP32 fusion is slower than optimized materialised A in this run, and C's frozen FP32 winner does not consistently outperform optimized causal A. The results separate correctness and storage benefits from latency improvement. PTX FA2 improves on PTX FA1 across this grid. The A/C Tensor Core comparison still includes their different probability-rounding order; cross-precision FA1/FA2 ratios belong in the separately labelled `fa1_cross_precision.csv`.

Configurations were frozen using the designated tuning lengths and are not guaranteed to minimize latency at every reporting length. At N=4096,d=64, C `skip` is faster than the frozen family choice `vector`; the comparison retains the preselected configuration rather than selecting a new winner from the reporting grid.

## Variability and profiler limitations

Median within-case IQR/median across main-grid cases is approximately 1.01% for A, 0.43% for B and 0.80% for C. Some cases are noisier: C FP32 FA1 at N=512,d=64 has IQR/median 25.5%, and C WMMA at N=4096,d=32 has 22.8%. Preserve quartiles and avoid strong claims about small differences in noisy cases. These are single-session observations with unlocked clocks, not independent-session confidence intervals; no specific thermal explanation is established by these records.

There are 282 explicitly unavailable profiler rows: 18 in B and 264 in C. They concern optional triage/device/cache-configuration entries, while measured counters remain available in every part. Missing values must stay unavailable rather than becoming zero. Occupancy, eligible warps, stall samples, allocations and physical traffic measure different things; counters alone do not prove a causal bottleneck explanation.

The standalone C exporter still writes a legacy hardcoded `final_abc_integration_complete=false` metadata field. It does not evaluate root integration. The root `aggregation_audit.json` and `run_status.json` are authoritative and both establish the completed A/B/C experiment.

## Reporting fixes made during verification

- Split the combined latency/speedup chart into separate dense FP32, causal FP32 and causal FP16 rows, identifying target variants and preventing clipped/overlapping legends.
- Include dense Part B in the combined allocation chart and retain separate C FA1/FA2 series so FA1's m/l workspace is not silently collapsed into FA2's allocation. Use a logarithmic capacity axis so fused allocations remain visible.
- Regenerate the derived figures/data and add regression coverage. Original root/child measurement CSVs remain unchanged.

## Repeat the saved-result checks

From the repository root:

```bash
python3 scripts/prepare_submission.py check --results results/final_submission
bash run_all.sh --analyze-only results/final_submission
python3 scripts/prepare_submission.py check --results results/final_submission
python3 -m unittest discover -s tests -v
```

These commands validate/reprocess saved evidence and run CPU integration tests; they do not rerun the CUDA benchmarks. Preserve the recorded raw evidence and binary profiler reports at their relative paths for full integrity checks. The 66 original binary report files (approximately 304 MiB) are retained directly with this accepted bundle. A's 36 report files contain three captures each; B's six and C's 24 contain one each. Once committed, a checkout includes the 138 captures needed by these checks without a separate download. Other experimental binary reports remain ignored.

The separate historical B smoke recording `part_b/results/notebook_smoke_20261110_044515/` is still missing. It affects pinned historical replay tests and final source packaging, not this fresh full-run acceptance. Resolve it before claiming the complete submission package is ready. Final report, video, contributions, citations and accessible delivery links also remain team work.

## Submission-file reproduction check

A separate filesystem copy containing the tracked working-tree files and the 66 final-run reports proposed for inclusion passed the full saved-result audit before and after offline regeneration. All 33 root/child CSVs and all 66 report files remained byte-identical; all 41 indexed figure pairs were regenerated. No GPU benchmarks were executed. This verifies the proposed submission file set on the current analysis environment; it is not a claim of a fresh GitHub clone or an independent-machine test. After committing and pushing the report files, another member should verify the documented commands from the submitted revision.
