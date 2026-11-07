CEG5206 TEAM PROJECT — FLASHATTENTION FORWARD
Reproduction instructions

Run from the repository root on Linux/Bash with a CUDA GPU. These are the
available Part C commands and root A/C orchestration. Final shared-oracle
integration and the Part B adapter are not implemented.
Part B's final source must be supplied before a complete A/B/C run is possible.
The current tested GPU is SM86 (RTX 3070 Ti Laptop). Adjust the CMake architecture
for another supported GPU. CUDA Toolkit, a C++17 compiler, CMake >=3.18,
Compute Sanitizer and Nsight Compute must be installed separately.

1. OPEN AND PREPARE

cd /path/to/Flash-Attention-CUDA
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r part_c/requirements-analysis.txt
nvidia-smi
nvcc --version

Inputs are generated internally. No dataset download is needed. New bounded
studies use Part A's SplitMix64 sequence, seed 5206. Do not change it when making
common-input comparisons. Historical Part C regression uses mt19937.

2. ROOT A/C RUNNER

Inspect commands without running CUDA or creating results:

bash run_all.sh --parts a,c --arch 86 --output results/my_run --dry-run

Run the existing A/C workflows sequentially on one GPU:

bash run_all.sh --parts a,c --arch 86 --output results/my_run

Use a fresh output directory. Add --build-root /tmp/my-builds to override builds.
The default --parts all fails before building because Part B is not present.
The runner records commands/environment, freezes A's existing FP32 tuning,
collects normalized CSVs at results/my_run/, and puts per-part figures under
plots/part_a/ and plots/part_c/. Part A runs both dense and causal modes.
Unchecked A rows stay UNCHECKED. Cross-part speedup plots are withheld while
shared-oracle, fingerprints and full validation are pending; see README.md
Section 6 and aggregation_audit.json. This command does not modify old results.

3. ONE-COMMAND PART C RUN

Use a NEW result directory. This runs the default regression and benchmark,
four sanitizer tools, legacy checks, then all candidate correctness checks,
bounded tuning, frozen main evaluation, matched scalar/vector loading,
isolated memory cases, CSV audit and PNG/PDF figures:

bash part_c/run.sh --complete part_c/results/my_complete_run /tmp/part_c-evaluation-build

Default workflow outputs: part_c/results/my_complete_run/regression/
Bounded-study outputs:    part_c/results/my_complete_run/main/

Do not run GPU benchmarks concurrently. Runtime depends on the GPU; use the
command logs and per-case PASS/FAIL records to monitor progress. A failed
correctness gate stops the study rather than entering a failed timing row in
plots. Existing result folders are retained.

4. RUN JUST THE NEW STUDY

bash part_c/run.sh --study part_c/results/my_bounded_study

The default build is /tmp/part_c-evaluation-build. Override it with:

PART_C_BUILD_DIR=/tmp/another-c-build bash part_c/run.sh --study part_c/results/another_study

The grid is B=4,H=8,N=256/512/1024/2048/4096,d=32/64/128. Tune only at
N=256/1024 before freezing. Twenty timed samples follow five warmups.

5. OPTIONAL FROZEN-WINNER SAFETY CHECKS AND PROFILING

After the study completes, run sequentially:

bash part_c/evaluate.sh --verify part_c/results/my_bounded_study
bash part_c/profile.sh --frozen part_c/results/my_bounded_study

These use /tmp/part_c-evaluation-build. The safety checks repeat the 380-pass
default regression and run four sanitizer tools on each frozen d=64 family
winner at B1,H1,N33. Profiling uses three independent captures per winner at
N256/4096,d64. These are representative checks, not exhaustive sanitizer or
profiler coverage. Profiling output is separate from unprofiled timing.

For another build directory, use PART_C_BUILD_DIR for evaluate.sh and append
--build /path/to/build to profile.sh --frozen.

6. READ AND REGENERATE RESULTS

Study files:
  plan.json                predefined grid, candidates and measurement policy
  frozen_selection.json    tuning choices saved before main evaluation
  manifest.json            measurement source/binary hashes and snapshots
  commands.jsonl           exact case commands, outputs and exit codes
  cases/*.csv              raw per-case measurements and numerical checks
  measurements.csv         combined raw rows
  summary.csv              medians, quartiles, throughput and resources
  comparison.csv           frozen core versus tensor results
  matched_ablation.csv     same-configuration scalar/vector comparison
  audit.json               completeness/correctness/input/allocation checks
  REPORT.md                measured findings and limitations
  figures/*.png,*.pdf      report figures
  profiling/               optional fresh frozen-winner captures
  verification/      optional regression/sanitizer logs

Regenerate figures and tables from the supplied study without a GPU:

python3 part_c/scripts/evaluate.py --build /tmp/part_c-evaluation-build --output part_c/results/bounded_evaluation --analyze-only

The integrity audits additionally require the checksummed .ncu-rep files. If
supplied separately, restore them to the recorded profiling/cases/ paths first.
Figure/table regeneration itself uses saved CSVs only.

Audit raw rows, snapshots and byte-identical offline figure regeneration:

python3 part_c/scripts/audit_evaluation.py part_c/results/bounded_evaluation

Regenerate historical default-profile findings without a GPU:

python3 part_c/scripts/analyze_profiles.py part_c/results/nsight_analysis
python3 part_c/scripts/interpret_profiles.py part_c/results/nsight_analysis
python3 part_c/scripts/audit_profiles.py part_c/results/nsight_analysis

The default --complete command generates benchmarks and plots, but optional
Nsight captures and additional frozen-winner sanitizers are Step 5.

7. OTHER RUNNERS AND LIMITATIONS

Part A: see part_a/README.md for its independent CMake/CLI commands. Its saved
RTX 5060 Ti results cannot be used as speedup denominators for Part C's laptop
GPU timings. The current work does not modify or execute Part A.

See README.md for the project structure and numerical/memory conventions,
part_c/README.md for kernel-specific details, part_c/PROFILING.md for metrics,
and README.md Section 6 for the planned common A/B/C runner.

No command above commits, merges or pushes Git changes.
