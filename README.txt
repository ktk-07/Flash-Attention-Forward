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

4. COMPLETE EXAMINER RUN (AFTER PART B IS SUPPLIED)

bash run.sh --output results/final_submission

The default requests A/B/C and currently stops before creating output because
Part B is absent. README.md Section 6 contains the exact B implementation and
adapter instructions. Do not treat an A/C subset as a complete submission.
run_all.sh is an alias for run.sh, using the identical workflow.

5. INDEPENDENT PARTS

bash part_a/run.sh --output results/a_only
bash part_c/run.sh --output results/c_only

After B implements the shared interface:
bash part_b/run.sh --output results/b_only

Individual runners accept --build DIR, --arch, --device, --grid full|smoke,
--no-profile, --dry-run and --analyze-only BUNDLE. Part C's older positional,
--study and --complete commands are retained; see part_c/README.md.

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
bundle. Historical results in part_a/results/ and part_c/results/ are preserved;
never divide historical timings collected on different GPUs.

Twenty repetitions describe within-case spread. Large numerical checks sample
three query rows/head plus all-output finiteness; small suites check full O.
FP16 A and C have different accumulation/output contracts, labelled explicitly.
The final report, video, references, contribution statement and common-machine
A/B/C full run are still team submission work. No runner performs Git actions.
