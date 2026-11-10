PART B EXACT REPLAY FOR A/C REVIEWERS
See part_b/CROSS_CHECK.md.
python3 part_b/scripts/reproduce_notebook.py --grid full --check-only
bash part_b/reproduce.sh --grid smoke --output results/b_crosscheck_smoke
bash part_b/reproduce.sh --grid full --output results/b_crosscheck_full
Replay is diagnostic; root safety/profile acceptance remains unchanged.

CEG5206 TEAM PROJECT — FLASHATTENTION FORWARD
Reproduction instructions

1. SETUP (Linux/Bash, NVIDIA GPU SM80+, CUDA Toolkit, CMake >=3.18,
C++17 compiler, Python, Compute Sanitizer and Nsight Compute)

cd /path/to/Flash-Attention-CUDA
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r part_c/requirements-analysis.txt
nvidia-smi
nvcc --version

No external dataset is needed. A/C use the same SplitMix64 generator, seed 5206,
row-major BHNd tensors and shared FP64 oracle. See README.md Sections 3 and 6.

2. CHECK THE AVAILABLE A/C INTEGRATION

bash run.sh --parts a,c --arch 86 --output results/preview --dry-run
bash run.sh --parts a,c --grid smoke --output results/ac_smoke

Dry-run creates no files and executes no commands. Smoke uses small shapes,
correctness, sanitizers, tuning, timings and plots; it omits profiling and is not
final report evidence. Use fresh output directories for each execution.

3. FULL AVAILABLE A/C EXPERIMENT

bash run.sh --parts a,c --output results/ac_full

Default architecture is detected from the selected GPU. Override with --arch 86
or another supported SM. --device 0 (or a GPU UUID) selects one physical GPU.
--build-root /tmp/attention-builds selects build storage. Jobs run sequentially.
Full runs include fresh Nsight profiling. --no-profile explicitly skips it and
records missing counter evidence. Avoid competing GPU workloads.

4. COMPLETE EXAMINER RUN (AFTER SAFETY/PROFILING ACCEPTANCE)

bash run.sh --output results/final_submission

The default requests A/B/C. Part B is implemented; see part_b/README.md for its commands and diagnostic evidence limitations. Final sanitizer/profiler acceptance and a common-machine run remain required. Do not treat an A/C subset as a complete submission.
run_all.sh is an alias for run.sh, using the identical workflow.

5. INDEPENDENT PARTS

bash part_a/run.sh --output results/a_only
bash part_c/run.sh --output results/c_only

Part B implements the shared interface:
bash part_b/run.sh --output results/b_only

Individual runners accept --build DIR, --arch, --device, --grid full|smoke,
--no-profile, --dry-run and --analyze-only BUNDLE. Part C's older positional,
--study and --complete commands are retained; see part_c/README.md.

You can also enter a part directory. Starting from the repository root, choose
one block (these are alternatives to the root-level commands above):

cd part_a
bash run.sh --output ../results/a_only

Or, separately starting from the repository root:
cd part_c
bash run.sh --output ../results/c_only

Output paths are relative to your invoking directory. Use a fresh output name;
do not run both alternatives into the same existing directory. Add --grid smoke
for a small check. Full runs are the default. Keep the repository intact because
the runners share the CPU reference source; B is not required for A or C.

View one standalone run from the repository root:
- results/a_only/ or results/c_only/: audit.json, run_status.json, ANALYSIS.md,
  summary.csv (latency/errors/throughput), correctness.csv and samples.csv.
- plots/part_a/ or plots/part_c/ inside that bundle: PNG/PDF figures and plot data.
- resources.csv, profiles.csv, memory.csv, work.csv: detailed evidence.
- raw/part_a/ or raw/part_c/, commands.jsonl and logs/: native records/commands.
- C additionally retains supplementary/part_c/ and plots/README.md.
Fresh C runs include FP32 FA1 and tuned PTX FA1 in all standard stages.
fa1_fa2_comparison.csv records matched-precision FA1/FA2 ratios;
fa1_cross_precision.csv labels FP16 FA1 versus FP32 FA2 separately. Their
PNG/PDF figures are in plots/part_c/. Full C profiling collects 24 captures
covering both FA1 baselines and the frozen d=64 FA2 CUDA-core winner and PTX FA2.
PTX FA1/FA2 receive equal four-configuration tuning budgets. WMMA FA1 is
available explicitly as supplementary evidence; see part_c/README.md.
Historical bundles retain their original kernel coverage.
Open CSVs with a CSV viewer and PNG/PDF figures with a normal image/PDF viewer.
For a combined root run, inspect its parts/part_a/ or parts/part_c/ child bundle,
or its top-level plots/part_a/ and plots/part_c/ folders. No rerun is needed.

6. OFFLINE ANALYSIS (NO GPU REQUIRED)

bash run.sh --analyze-only results/ac_full
bash part_a/run.sh --analyze-only results/a_only
bash part_c/run.sh --analyze-only results/c_only

Root analysis validates hashes/child tables and regenerates canonical aggregate
CSVs, applicable combined plots, per-part common plots and their CSV data.
Individual analysis re-extracts native records. Part C regenerates its native
and supplementary charts too. Binary Nsight reports are optional for CSV-only
charts; preserve supplied checksums and raw CSVs.

7. READ RESULTS

results/<run>/ contains standardized correctness.csv, samples.csv, summary.csv,
resources.csv, profiles.csv, memory.csv and work.csv. comparison.csv records
matched cross-part ratios and numerical contracts. See ANALYSIS.md,
figure_index.json, audit.json and aggregation_audit.json before citing results.
parts/part_a/ and parts/part_c/ retain complete standalone bundles including
native data, commands, logs, configuration selections and additional evidence.
Main figures are in plots/part_a/, plots/part_c/ and plots/combined/, PNG/PDF.
Additional C profiler evidence remains in supplementary/part_c/ and its child
bundle. Saved results in part_a/results/ and historical results in part_c/results/ remain separate;
never divide historical timings collected on different GPUs.

Twenty repetitions describe within-case spread. Large numerical checks sample
three query rows/head plus all-output finiteness; small suites check full O.
A and C Tensor Core paths use FP32 accumulation/output; probability rounding order differs.
The final report, video, references, contribution statement and common-machine
A/B/C full run are still team submission work. No runner performs Git actions.

PART A SAVED RESULTS

Part A includes FP32 naive/tiled/parallel/optimized and mixed-precision WMMA/PTX.
The saved full-grid RTX 5060 Ti run and compact report figures are in part_a/results.
Regenerate the figures with: python3 part_a/plot_results.py
A/C Tensor Core paths use FP32 accumulation/output and different probability rounding order.
Part A accepts compressed Nsight reports and supports profiling-only resume.
A new common-machine A/B/C run remains required after Part B integration.
