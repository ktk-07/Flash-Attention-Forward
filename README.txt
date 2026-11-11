CEG5206 TEAM PROJECT — FLASHATTENTION FORWARD
Reproduction, full-run handoff and submission packaging

README.md describes the project, repository layout, interfaces and plots.
This file gives the operator's command sequence. Run from the repository root.
No project runner or packaging helper executes Git or publishes anything.

1. SOURCE VERSION AND CURRENT STATUS

The A/B/C full-grid run is complete in results/final_submission/ on the
RTX 3070 Ti Laptop GPU. Saved-result validation and CPU-only regeneration pass:
1,657 passing correctness records, 15,000 timing samples and 138 profiler
captures (A108/B6/C24), with 41 indexed PNG/PDF figure pairs. See
results/final_submission/VERIFICATION.md for findings and limitations.
Use a fresh output directory when repeating the experiment.

The integrator supplies the reviewed source commit ID. The operator uses that
exact source and records the complete ID for packaging. The helper hashes source
files but does not consult Git or claim to verify that an ID matches the checkout.
Avoid changing kernels, dependencies, seeds or configuration policy during a run.

Known separate issue owned by Yichun: the historical B smoke replay bundle
part_b/results/notebook_smoke_20261110_044515/ is absent. This breaks three replay
unit tests and the exact historical smoke replay, but not the fresh root runner.
The supplied historical FULL bundle passes its pinned checks. Do not substitute
a new measurement for the missing recording or disable its integrity checks.

2. ENVIRONMENT AND SOURCE AUDIT

Use Linux/Bash, NVIDIA GPU SM80+, C++17 compiler, CUDA Toolkit/nvcc, CMake >=3.18,
Python 3.12, Compute Sanitizer and Nsight Compute. Archive the actual machine/tool
versions with the run. Part B's separate Windows notebook instructions are in
part_b/README.md; the root orchestration uses Bash.

cd /path/to/Flash-Attention-CUDA
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r part_c/requirements-analysis.txt
nvidia-smi
nvcc --version
cmake --version
compute-sanitizer --version
ncu --version

No external dataset or PyTorch is needed. A/B/C share SplitMix64 seed 5206,
contiguous row-major BHNd tensors and the independent CPU FP64 reference.

Audit a source-only filesystem copy, without Git, CUDA builds or GPU execution:
python3 scripts/prepare_submission.py source-check --output extra/source_audit

Use a fresh output directory. The audit copies the maintained repository files,
retained CSV fixtures and notebooks, excluding private extra/, environments,
compiled builds and historical binary profiler attachments. Checksum-pinned
historical source snapshots stay intact, even when a recorded path contains
an old extra/ subdirectory; current private drafts remain excluded. It runs the CPU
integration, C export, B adapter/replay tests, pinned B-full checks and root dry
run. Results/logs/source hashes stay under the requested ignored extra/ folder.
This is a filesystem source-snapshot check, not a Git clean-checkout guarantee.
The missing B smoke recording is reported as a failure; Yichun must resolve it
before the final source package can be declared complete. Review other failures
independently; do not suppress them as that known issue.

3. OPERATOR PRECHECK ON THE FINAL GPU

bash run.sh --arch auto --output results/full_plan --dry-run
bash run.sh --grid smoke --device 0 --build-root /tmp/attention-builds --output results/abc_smoke

Dry-run creates no output and executes no GPU commands. Smoke includes numerical
and sanitizer checks, tuning, timings and plots; it deliberately omits profiling.
Its report shapes are B1/H1/N33,65/d64, with tuning N17,33.

Before the long run, verify that Nsight can actually access counters on this
machine. After the smoke build, use a fresh probe directory:

mkdir -p extra/full_run_preflight
ncu --set basic --launch-count 1 --export extra/full_run_preflight/counter_probe /tmp/attention-builds/part_b/part_b_runner --B 1 --H 1 --N 33 --d 64 --variant tiled --warps 4 --key-tile 32 --pattern random --sampled 0 --timing 0 --profile 1

If access is denied, resolve device permissions with the machine administrator
before the full experiment. The probe stays outside report results. Never add
profiler-instrumented timings to ordinary performance samples.

4. EXACT FULL-RUN HANDOFF COMMAND — FULL-RUN OPERATOR ONLY

bash run.sh --parts a,b,c --grid full --device 0 --build-root /tmp/attention-builds --output results/final_submission

Choose a new output name if this one already exists. Architecture is detected;
override --arch only for a known supported target. Keep other GPU work idle.
The root runner executes A, B and C sequentially on the selected physical GPU.

Main grid: B4/H8, N256/512/1024/2048/4096, d32/64/128.
Tune on N256/1024, freeze per implementation/d before the reporting grid.
Use five warmups and twenty unprofiled CUDA-event samples per accepted case.

Do NOT use --no-profile or Part B's --no-sanitize for the final experiment.
If a stage fails, keep its logs and incomplete bundle, diagnose it, and use a
fresh destination for a new root attempt. Do not manually mark an audit passed,
concatenate historical timings, or relabel a partial run as complete.

Expected full profiler coverage: A=108 captures, B=6, C=24. Audits also require
same GPU/input identity, numerical validation and the complete requested grid.
These counts do not turn a smoke run into a full experiment.

5. VALIDATE AND REGENERATE THE COMPLETED EXPERIMENT

python3 scripts/prepare_submission.py check --results results/final_submission
bash run.sh --analyze-only results/final_submission
python3 scripts/prepare_submission.py check --results results/final_submission

The checks require completed A/B/C full-grid audits, seven valid canonical
schemas, unchanged child tables, matching GPU identity, shared inputs, required
profile coverage and indexed plots/evidence files. Offline regeneration verifies
native-to-public extraction and redraws figures without running CUDA kernels.
Preserve all binary reports at their original relative paths for these checks.

For a deliberate packaging test using a smoke bundle ONLY:
python3 scripts/prepare_submission.py check --results results/abc_smoke --allow-smoke

Check ANALYSIS.md and the underlying tables before writing conclusions. Review
matched FP32 and PTX FA1/FA2 results, materialised/fused latency and storage,
tuning, matched scalar/vector loading and profiler evidence. Dense B versus
causal C is a workload/implementation comparison, not a masking-only ablation.
Q1-Q3 measures within-case spread, not independent-session confidence intervals.

6. STANDALONE EXECUTION AND OFFLINE ANALYSIS

bash part_a/run.sh --output results/a_only
bash part_b/run.sh --output results/b_only
bash part_c/run.sh --output results/c_only

These are alternatives to the combined run. Add --grid smoke for a short check.
A root subset is explicit: bash run.sh --parts a,c --output results/ac_only
Standalone runners accept --build, --arch, --device, --grid, --no-profile,
--dry-run and --analyze-only. Part C retains its native study/pilot modes.

You may enter one part directory instead, for example:
cd part_b
bash run.sh --grid smoke --output ../results/b_only_smoke

Paths are relative to the invoking directory. Keep the complete repository:
all parts share the CPU reference; B also has small materialised-A cross-checks.

From the repository root, regenerate a completed standalone bundle:
bash part_a/run.sh --analyze-only results/a_only
bash part_b/run.sh --analyze-only results/b_only
bash part_c/run.sh --analyze-only results/c_only

Pinned historical B FULL replay is a separate diagnostic experiment:
python3 part_b/scripts/reproduce_notebook.py --grid full --check-only
bash part_b/reproduce.sh --grid full --output results/b_crosscheck_full

It reuses recorded choices rather than retuning. Its audit remains diagnostic
because replay omits sanitizer/profiler acceptance; crosscheck.json states
numerical replay success. See part_b/CROSS_CHECK.md. The absent historical smoke
recording is Yichun's separate restoration task.

7. WHERE RESULTS AND FIGURES ARE STORED

results/<run>/ contains correctness.csv, samples.csv, summary.csv, resources.csv,
profiles.csv, memory.csv and work.csv. comparison.csv records cross-part ratios.
fa1_fa2_comparison.csv supplies matched latency/speedup data;
fa1_cross_precision.csv separately labels PTX FA1 versus FP32 FA2.

parts/part_a/, parts/part_b/, parts/part_c/ retain complete individual bundles,
raw observations, configurations, commands, logs and source snapshots.
plots/README.md links all indexed figures. plots/part_a/, part_b/, part_c/ and
combined/ contain PNG/PDF figures; supplementary/part_c/ retains profiler charts.
The three main FA1 figures are fa1_fa2_latency, fa1_fa2_comparison and
fa1_cross_precision. Full PTX FA1 and PTX FA2 have equal four-candidate searches;
optional WMMA FA1 remains supplementary. See README.md for the complete plot map.

Keep all raw counter CSVs and native binary reports. Missing metrics are
unavailable, not zero. Historical part-local results keep their original GPU,
source and precision identity and are not automatically merged into new runs.

8. SUBMISSION REVIEW AND ARCHIVES

The briefing requires ONE report PDF per group, a 10-15-minute video link,
accessible GitHub repository link and final commit hash or release tag. The
report includes problem, implementation, correctness, experiments, results,
limitations, references, actual member contributions and AI-use disclosure.
There is no fixed page limit; keep the report concise and precise.

Human review before packaging:
- Replace all report figure/result placeholders with the reviewed final data.
- Explain optimisations mathematically, algorithmically and at hardware/memory
  level. Cite reused ideas/code and distinguish measured evidence from hypotheses.
- Label dense/causal work and FP32/FP16 comparisons accurately; use best measured
  bounded configurations, not unqualified claims of global optimality.
- Every member confirms contributions, AI assistance and ability to explain code.
- Put the accessible video link, repository URL and reviewed version IN THE PDF.
- Check the video's length and permissions without a team login. The helper
  validates URL syntax, not external accessibility or the video's content.
- Compare the report's actual PDF with the rubric and verify readable labels,
  units, legends, tables, bibliography and the precise submission deadline.

The private extra/main.tex is NOT packaged or edited. Supply the reviewed final
PDF explicitly; do not commit private drafts, scratch studies or build products.

Replace REVIEWED_FULL_COMMIT_ID, PDF_PATH and the two example URLs below with
actual reviewed values. Run only after Yichun resolves the missing replay bundle
and the final experiment/report are available:

python3 scripts/prepare_submission.py pack --results results/final_submission --source-version REVIEWED_FULL_COMMIT_ID --report PDF_PATH --video-url https://example.org/team-video --repository-url https://github.com/TEAM/REPOSITORY --output extra/submission_package/source_results_report.zip --profiles-output extra/submission_package/profiler_reports.zip

The source archive contains maintained source, reproduction instructions,
historical text/CSV fixtures, the selected COMPLETE result bundle and report.pdf.
Only raw .ncu-rep/.ncu-repz files from the selected run go to the optional companion
archive. No selected-run evidence is silently dropped. Both archives use the same
Flash-Attention-CUDA/ prefix: extract BOTH into the same directory before offline
regeneration. Omitting --profiles-output makes one self-contained ZIP instead.

submission_manifest.json records supplied version/URLs, per-file hashes, capture
scope and manual review items. SHA256SUMS and archive .sha256 sidecars accompany
the ZIPs. Verify transferred files with:

python3 scripts/prepare_submission.py verify-archive extra/submission_package/source_results_report.zip --profiles extra/submission_package/profiler_reports.zip

For a packaging test only, --validation-only permits a smoke bundle and an
uncommitted source label, and clearly marks the archive as NOT a final submission.
Final mode rejects smoke evidence, missing pinned references, missing report/URLs
and incomplete profiler coverage. It never manufactures a report, result or URL.

Finally, have another team member obtain the source from the supplied public
repository/version and reproduce setup/smoke/offline analysis. The filesystem
source audit assists this review; it does not replace that independent checkout.
Submit the PDF on Canvas and confirm every linked artifact is accessible. No
helper uploads anything, executes Git or changes repository history.
