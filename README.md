# CEG5206 Team Project — FlashAttention Forward

**NUS CEG5206 — Algorithms & Performance Optimization for Computing**  
AY2026/2027 Semester 1 · Group 1

**Repository:** [ktk-07/Flash-Attention-Forward](https://github.com/ktk-07/Flash-Attention-Forward)

CUDA C++ implementations of materialised attention (Part A), dense fused FlashAttention (Part B) and causal fused FlashAttention (Part C). The project studies tiling, online softmax, fusion, causal tile skipping, memory behaviour and mixed-precision Tensor Core execution.

**Documentation:** this README describes the repository, shared interfaces, run commands, plots and evidence status. [README.txt](README.txt) provides the submission reproduction checklist. The full-run handoff and packaging commands are in [README.txt](README.txt), supported by [prepare_submission.py](scripts/prepare_submission.py). Implementation details are in the [Part A](part_a/README.md), [Part B](part_b/README.md) and [Part C](part_c/README.md) READMEs.

**Current status:** the full-grid A/B/C run in [results/final_submission](results/final_submission/) completed on the RTX 3070 Ti Laptop GPU and passed the saved-result audit and CPU-only regeneration: 1,657 passing correctness records, 15,000 timing samples and 138 profiler captures (A108/B6/C24). See [verification findings](results/final_submission/VERIFICATION.md). The bundle contains 41 PNG/PDF figure pairs. The historical Part B smoke recording has been restored; its pinned replay tests and the CPU source audit pass. Performance includes slowdowns against optimized Part A; fusion alone is not a guaranteed speedup.

**Presentation and demo:** the team reports a video duration of **14 minutes 58 seconds**, within the required 10–15 minutes. The live demonstration is [FlashAttention_Full_Run_Demo.ipynb](FlashAttention_Full_Run_Demo.ipynb); see [demo instructions](#live-cuda-demonstration) below. The final report must contain the accessible video URL and submitted commit hash or release tag.

The scope follows the approved FlashAttention proposal, Project Briefing and Team Project Rubrics. **Briefing deadline:** Friday, 13 November 2026, 11:59 PM. Submission requires a report PDF, an accessible 10–15-minute video link, the repository link and a final commit hash or release tag.

**Navigation:** [Demo](#live-cuda-demonstration) · [Repository layout](#repository-layout) · [Setup](#2-open-the-project-and-install-analysis-dependencies) · [Inputs](#3-inputs-and-computation-contract) · [Tests](#4-build-and-run-the-available-tests) · [Run A/B/C](#5-run-the-project) · [Integration and outputs](#6-shared-integration-and-results) · [Measurement](#7-measurement-conventions) · [Saved evidence](#8-read-the-results) · [Offline analysis](#9-recreate-figures-without-rerunning-cuda) · [Submission](#10-submission-and-outstanding-scope)

## Reproduce the full A/B/C experiment

Complete the [setup](#2-open-the-project-and-install-analysis-dependencies), then run this command from the repository root:

```bash
bash run_all.sh --parts a,b,c --grid full --device 0 \
  --output results/new_full_run
```

This builds and evaluates all three parts on GPU 0, including correctness checks, representative sanitizers, bounded tuning, benchmarks, memory/resource measurements, profiling and plot generation. CUDA, Compute Sanitizer and Nsight Compute with GPU-counter access are required. Use a fresh output directory; the verified `results/final_submission/` bundle is already supplied and should be preserved.

Aggregated CSVs are written to `results/new_full_run/`, complete individual bundles to `parts/part_a/`, `parts/part_b/` and `parts/part_c/` beneath that directory, and figures to `plots/part_a/`, `plots/part_b/`, `plots/part_c/` and `plots/combined/`. See [Section 5](#5-run-the-project) for the smoke check and individual runs, and [Section 6](#6-shared-integration-and-results) for the complete output layout.

## Live CUDA demonstration

Open [FlashAttention_Full_Run_Demo.ipynb](FlashAttention_Full_Run_Demo.ipynb) to reproduce the short demonstration. It builds repository CUDA kernels, generates fresh inputs, checks outputs against the shared CPU FP64 oracle and records five warmups plus 20 CUDA-event samples per implementation. It demonstrates:

- Causal A-PTX versus C-PTX FA2 on identical FP16-rounded inputs, including a small full-output correctness example.
- Sequence-length changes from N512 to N1024 at B4/H8/d64, with latency and explicit allocated tensor capacity shown separately.
- Dense FP32 A versus B, including memory savings even when B is slower.
- Validation of the accepted historical full-run bundle as a separate saved-data check.

Use an NVIDIA SM80+ GPU, CUDA Toolkit/nvcc, CMake **3.20+**, a C++17 host compiler and Python. On Windows, install the Visual Studio 2022 C++ build tools and Windows SDK; the helper can initialize the x64 compiler environment. The live recording used an RTX 4070 Ti; the report's full experiment used an RTX 3070 Ti Laptop GPU. The notebook detects the executing device. Other supported GPUs produce separate measurements.

From the repository root, activate your Python environment and run:

```bash
python -m pip install -r part_c/requirements-analysis.txt
python -m pip install jupyterlab
python -m jupyterlab FlashAttention_Full_Run_Demo.ipynb
```

Select that environment's Python kernel and run the cells in order. Launch JupyterLab **from the repository root**, including when the checkout is named `Flash-Attention-Forward`; otherwise set `FLASH_ATTENTION_REPO` to its absolute path. The first CUDA build may take several minutes; run it before recording a presentation. The notebook uses fixed demo configurations rather than the final experiment's independently frozen winners. Edit `SEED`, `N` or `N_LARGER` in the workload cell for a fresh experiment; keep `16 <= N < N_LARGER <= 2048`.

Builds are cached under `flash_attention_live_demo/build/<source-hash>/`. Fresh result JSON files, timing samples and `provenance.json` are stored under `flash_attention_live_demo/runs/<UTC_timestamp>_<suffix>/`. These live outputs do not replace or merge into `results/final_submission/`. The notebook also embeds the three demo-driver sources and materialises them under `flash_attention_live_demo/source/`. See [the demo README](flash_attention_live_demo/README.md) for platform notes and output interpretation.

## Team responsibilities

Lyu Sihan, Zhang Yichun and Kyaw Tun Khine contributed equally overall. Their primary implementation areas are listed below.

| Part | Member | Primary responsibility | Cross-review |
|---|---|---|---|
| A | Lyu Sihan | Independent C++ FP64 oracle; materialised QKᵀ, softmax and PV; baseline optimisation | Dense fused correctness and comparison fairness |
| B | Zhang Yichun | Dense fused forward; online softmax, tiled Q/K/V and output accumulation | Materialised indexing and memory access |
| C | Kyaw Tun Khine | Causal fused forward; future-tile skipping, diagonal masking and optimisation | Causal correctness and performance interpretation |

**Integration owner: Kyaw Tun Khine.** A, B and C retain separate GPU implementations and runners, connected through shared input, correctness and result conventions. Part B is supplied by Zhang Yichun; its source-pinned replay is documented in [CROSS_CHECK.md](part_b/CROSS_CHECK.md).

Each member contributes testing, tuning, profiling and analysis. Record actual contributions in the final report. AI assistance in Part A, Part B implementation/report preparation and Part C evaluation must be reviewed and disclosed as required by the briefing. Each member should confirm their actual contributions and assistance received.

## Repository layout

```text
README.md                                   # overview, shared conventions and result/figure index
README.txt                                  # submission reproduction checklist
FlashAttention_Full_Run_Demo.ipynb           # short live CUDA demo; fresh inputs/checks/timings
flash_attention_live_demo/
  README.md                                 # demo setup, run sequence and output locations
  source/                                   # CMakeLists.txt, demo_runner.cu and live_demo.py
  build/, runs/                             # generated demo builds and measurement records
run.sh                                      # sequential A/B/C evaluation and aggregation
run_all.sh                                  # compatibility alias for run.sh
scripts/
  run_all.py                                # builds, runner orchestration and aggregate audit
  results_schema.py                         # shared seven-table CSV schema and validation
  workflow.py                               # command logging, GPU identity and provenance
  ncu_csv.py                                # Nsight CSV parsing and metric aliases
  plot_combined.py                          # common per-part and combined plots/indexes
  prepare_submission.py                   # source audit, full-run checks and ZIP packaging
  fa1_comparison.py                         # matched FA1/FA2 latency and speedup figures
part_a/
  common/                                   # tensor types, deterministic inputs, reference API
  cpu_reference/                            # independent FP64 reference CMake library
  gpu_baseline/                             # materialised QK, softmax and PV CUDA kernels
  tests/                                    # shared reference agreement tests
  scripts/                                  # standard workflow and canonical export
  main.cpp                                  # materialised benchmark/test executable
  CMakeLists.txt
  run.sh                                    # standalone Part A evaluation
  plot_results.py                           # report-figure regeneration wrapper
  README.md
  results/                                  # retained RTX 5060 Ti measurements
part_b/
  include/flash_attention/
    dense_flash_attention.h                 # public dense-attention declarations
  src/
    dense_flash_attention.cu                # streaming, tiled and float4 FP32 definitions
  tests/
    evaluation_runner.cpp                   # oracle checks, A cross-checks, timings/resources
    test_adapter.py                         # canonical export and acceptance checks
    test_reference_replay.py                # pinned source/input/replay checks
  scripts/
    run_standard.py                         # fresh build/test/tune/profile/export workflow
    export_results.py                       # native B records to the shared CSV schema
    reproduce_notebook.py                   # exact recorded plan replay; no retuning
    analyze_delivery.py                     # retained-data verification and report assets
  notebooks/
    FlashAttention_Part_B_Lab.ipynb
    Analyze_Retained_Full_Run.ipynb
  CMakeLists.txt
  run.sh, run.cmd                           # fresh evaluation: Bash / Windows wrappers
  reproduce.sh, reproduce.cmd               # diagnostic exact-replay wrappers
  EXECUTION_REFERENCE.json                  # pinned reference/source/evidence identity
  CROSS_CHECK.md                            # A/C reviewer replay instructions
  CROSS_REVIEW.md                           # static cross-review notes and limitations
  README.md
  results/                                  # retained notebook evidence; not auto-aggregated
part_c/
  include/flash_attention/                  # public declarations and shared CUDA types
  src/
    flash_attention_1.cu                    # FP32 FA1 baseline
    flash_attention_1_tensor_core.cu        # PTX FA1 and supplementary WMMA FA1
    flash_attention_2_cuda_core.cu
    flash_attention_2_tensor_core.cu        # WMMA/PTX FA2
  tests/                                    # runners, FA1 invariants and input-contract checks
  scripts/                                  # tuning, export, plotting, audits and profiling
  CMakeLists.txt
  run.sh                                    # standalone standard workflow; legacy modes retained
  evaluate.sh, profile.sh                   # native study and separate profiler workflows
  requirements-analysis.txt                 # pinned NumPy/Matplotlib dependencies
  README.md, PROFILING.md
  results/                                  # retained historical studies and profiler evidence
tests/                                      # CPU-only shared integration/plotting checks
results/<run_id>/                           # new standalone or combined experiment bundles
extra/                                      # ignored private scratch; excluded from submission
```

Declarations live in headers; CUDA definitions live in `.cu` files. Part B's public namespace is `dense_attention`. Both FA1 Tensor Core implementations share Part C's Tensor Core source/header; WMMA FA1 is supplementary. Integration links the shared reference and standardizes host interfaces while each part retains its kernel source.

## 1. Software and platform requirements

The root A/B/C workflow uses Linux, Bash, a CUDA-capable NVIDIA GPU, a C++17 compiler, CMake and Python. The tested GPU is an RTX 3070 Ti Laptop with SM86. The named runners detect the selected GPU architecture by default; `--arch` overrides it. The integrated Tensor Core paths require SM80 or newer. Part A's saved results were collected separately on an RTX 5060 Ti. Part B's retained notebook run used Windows, an RTX 4070 Ti (SM89), CUDA 12.9/nvcc 12.9.41 and MSVC; see [Part B](part_b/README.md) for its separate Windows workflow and measured environment.

| Software | Recorded version / purpose |
|---|---|
| CUDA Toolkit / nvcc | 12.6; CUDA compilation and runtime |
| NVIDIA Nsight Compute | 2024.3; hardware counters and saved reports |
| Compute Sanitizer | CUDA Toolkit tool; memory and synchronization checks |
| CMake | 3.18 or newer; separate C++/CUDA compilation |
| Python | 3.12 used for analysis |
| NumPy | 1.26.4; saved-data analysis |
| Matplotlib | 3.10.8; PNG/PDF figures |

Exact compiler, driver and GPU records are stored with each run. Hardware counters and timings are machine-specific measurements.

## 2. Open the project and install analysis dependencies

Run from the repository root:

```bash
cd /path/to/Flash-Attention-Forward
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r part_c/requirements-analysis.txt
nvidia-smi
nvcc --version
```

CUDA, CMake, Nsight Compute and Compute Sanitizer must already be installed. Dependency installation requires available packages or network access. [README.txt](README.txt) gives the complete command sequence; the scripts do not perform Git operations.

## 3. Inputs and computation contract

Inputs are synthetic; no external dataset is needed. Tensors use contiguous `[B,H,N,d]`, score scale `1/sqrt(d)`, stable row softmax and output `softmax(QKᵀ/sqrt(d))V`. Causal attention permits key `j` for query `i` when `j <= i`.

**Shared external layout: contiguous row-major.** Q/K/V/O have shape `[B,H,N,d]`, with feature `r` contiguous. Part A's materialised S/P have shape `[B,H,N,N]`, with key `j` contiguous. All three parts must accept this layout at their public boundary:

```text
Q/K/V/O offset(b,h,i,r) = ((b*H + h)*N + i)*d + r
Q/K/V/O element strides = (H*N*d, N*d, d, 1)
S/P offset(b,h,i,j)      = ((b*H + h)*N + i)*N + j
S/P element strides     = (H*N*N, N*N, N, 1)
```

Strides above count elements, not bytes. K is stored as `[N,d]` per head; the transpose in QKᵀ is mathematical, not a requirement to supply a physically transposed K tensor. Shared-memory tile layouts and WMMA/PTX fragment layouts are internal implementation choices and do not change this input/output contract. Record `layout=BHNd_contiguous_row_major` in the common run metadata.

| Property | Common comparison setting |
|---|---|
| Main grid | B=4, H=8, N=256/512/1024/2048/4096, d=32/64/128 |
| Generator | Part A SplitMix64, seed 5206; all Q, then K, then V |
| Tuning subset | N=256/1024 for each d; bounded configuration list |
| Held-out sequence lengths | N=512/2048/4096 |
| Oracle | Independent CPU FP64 dot products, stable softmax and accumulation |
| FP16 inputs | Round the same generated values, then validate against the rounded-input oracle |

All standard runners use the shared generator and record input fingerprints; Part C additionally has a bitwise Part A/C input-contract test. Historical Part C regression/profile data uses its older mt19937 stream and retains that label. Equal seeds alone do not imply equal tensors.

## 4. Build and run the available tests

```bash
cmake -S part_c -B /tmp/part_c-evaluation-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/part_c-evaluation-build --parallel 2
/tmp/part_c-evaluation-build/part_c_input_contract
/tmp/part_c-evaluation-build/part_c_runner \
  --suite correctness --output part_c/results/my_correctness
```

The default regression checks all nine variants (including FP32 and PTX FA1 baselines), including tails, unusual head dimensions and deterministic patterns. Unsupported PTX dimensions are explicit skips. The bounded study separately validates every candidate configuration before tuning. Full small-case outputs are checked; large performance cases check queries 0, N/2 and N−1 in every head plus finiteness of every output. Sampled checks are not labelled full validation.

Fresh Part C runs include FP32 FA1 and tuned PTX FA1 throughout correctness, main-grid timing, memory, profiling and plots. `fa1_fa2_comparison.csv` records matched-precision ratios against frozen FA2 configurations; `fa1_cross_precision.csv` labels FP16 FA1 versus FP32 FA2 separately. Both have PNG/PDF figures in `plots/part_c/`. These compare our implementations, not official FA1 performance. Full C profiling now contains 24 captures: both FA1 baselines and the frozen d=64 FA2 CUDA-core winner and PTX FA2, two lengths and three repetitions. Historical bundles keep their recorded coverage. WMMA FA1 remains available as supplementary evidence. Both PTX implementations receive the same four-configuration tuning budget. See [Part C](part_c/README.md) for precision/state details.

CPU-only integration/export regression checks (no CUDA access required):

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s part_c/tests -p 'test_*.py' -v
python3 -m unittest discover -s part_b/tests -p 'test_*.py' -v
cmake -S part_a -B /tmp/attention-cpu-check -DPART_A_ENABLE_CUDA=OFF
cmake --build /tmp/attention-cpu-check --target attention_reference_test --parallel 2
/tmp/attention-cpu-check/cpu_reference/attention_reference_test
```

## 5. Run the project

Run commands from the repository root after setup. Choose a **fresh output directory** for each experiment. GPU work runs sequentially on one selected physical device.

### Complete A/B/C workflow

```bash
# Preview the full plan without GPU access, builds or output creation.
bash run.sh --arch 86 --output results/abc_preview --dry-run

# Test the combined pipeline with correctness, sanitizers, tuning and plots.
bash run.sh --grid smoke --output results/abc_smoke

# Full common-machine experiment, including fresh profiling.
bash run_all.sh --parts a,b,c --grid full --device 0 \
  --output results/new_full_run
```

Full runs execute tests, required sanitizer checks, bounded tuning, frozen main-grid timing, memory/resource studies, profiling, export and combined analysis. Smoke uses B=H=1, d64, reporting N33/65 and tuning N17/33; it omits profiling and is not final report evidence. The full grid is defined in Section 3. Do not bypass failed safety checks to label a diagnostic run complete.

To run an explicit subset through the same collector:

```bash
bash run.sh --parts a,c --grid smoke --output results/ac_smoke
bash run.sh --parts b --grid smoke --output results/b_root_smoke
```

The audit records the requested parts and marks subsets as incomplete A/B/C evidence. `run_all.sh` forwards to the same root runner.

### Run one part independently

```bash
bash part_a/run.sh --output results/a_only
bash part_b/run.sh --output results/b_only
bash part_c/run.sh --output results/c_only
```

Add `--grid smoke` for a short pipeline check. Full evaluation is the default. All three runners use the shared CPU reference source, so keep the repository intact; they do not require other parts' saved GPU results. B additionally links A's naive CUDA pipeline for supplementary small cross-checks.

You may instead enter the part directory; for example, starting from the repository root:

```bash
cd part_b
bash run.sh --grid smoke --output ../results/b_only_smoke
```

Output paths are relative to the invoking directory. This is an alternative to the root-level standalone command, not an extra required run.

| Option | Meaning |
|---|---|
| `--output PATH` | New bundle destination; existing experiment bundles are preserved |
| `--device INDEX_OR_UUID` | One physical GPU, selected consistently for all parts |
| `--arch auto\|SM` | Detect architecture by default; override with, for example, `86` or `89` |
| `--build-root PATH` / `--build PATH` | Root / standalone build storage |
| `--grid full\|smoke` | Final experimental grid / small pipeline check |
| `--no-profile` | Explicitly omit fresh counters; record the omission |
| `--dry-run` | Print the plan without creating results or running commands |
| `--analyze-only BUNDLE` | Re-extract saved data and regenerate analysis without CUDA |

Part B also offers `--no-sanitize` for **diagnostic** experiments; its audit remains incomplete and root aggregation rejects those bundles. Part C's older positional, `--study`, `--complete` and explicit `--fa1-pilot` modes remain available; see [Part C](part_c/README.md). Windows Part B commands and notebooks are documented in [Part B](part_b/README.md).

### Part B exact notebook replay for reviewers

The replay tools preserve recorded source, input patterns, case order and frozen launch choices; they do not retune for your GPU. See [CROSS_CHECK.md](part_b/CROSS_CHECK.md).

```bash
# Verify the pinned source, input/reference identity and retained evidence; no GPU.
python3 part_b/scripts/reproduce_notebook.py --grid full --check-only
# Replay the original 83-case smoke or 165-case full plan on your GPU.
bash part_b/reproduce.sh --grid smoke --output results/b_crosscheck_smoke
bash part_b/reproduce.sh --grid full --output results/b_crosscheck_full
```

The full replay has 90 timing cases / 1,800 samples. Inspect `crosscheck.json` for numerical replay success and `comparison_to_reference.csv` for separately labelled original/local observations. Replay intentionally has `audit.passed=false` because it does not collect safety/profiler evidence. Use `part_b/run.sh` for fresh tuning and acceptance, then the root runner for same-machine A/B/C comparisons.

### Inspect a standalone or combined run

For a standalone run, start at `results/a_only/`, `b_only/` or `c_only/`. For a combined run, the same complete bundles live under `results/<run_id>/parts/part_a/`, `part_b/` and `part_c/`.

| Files within a bundle | What to inspect |
|---|---|
| `audit.json`, `run_status.json` | Accepted validation, safety/profile scope and completion |
| `summary.csv` | Forward medians, quartiles, throughput and numerical error |
| `samples.csv`, `correctness.csv` | Individual timings and full/sampled oracle checks |
| `resources.csv`, `profiles.csv`, `memory.csv`, `work.csv` | Hardware resources, counters, storage and analytical work |
| `configs/part_<letter>.json`, `manifest.json`, `device.json` | Frozen selection, source/binary provenance and GPU identity |
| `plots/part_<letter>/` | Individual figures and available plotted CSV data |
| `ANALYSIS.md`, available figure indexes | Findings and figure navigation; B also writes `REPORT.md` |
| `raw/part_<letter>/`, `commands.jsonl`, `logs/` | Native records, source snapshots and executed commands |
| `supplementary/part_c/`, `plots/README.md` | Additional C profiling evidence and generated plot links |

Root additionally writes aggregate CSVs, `comparison.csv`, combined plots and `plots/README.md`; no rerun is needed to inspect one child part. Section 6 defines the interfaces and output layout, and Section 9 gives offline regeneration commands.

## 6. Shared integration and results

### Implementation and verification status

| Component | Current implementation | Evidence boundary |
|---|---|---|
| Part A | Materialised FP32 kernels plus WMMA/PTX; shared FP64 oracle and canonical adapter | Full common-machine run passed 728 correctness records and 108 profiler captures; historical standalone evidence remains separate |
| Part B | Dense FP32 streaming, tiled and float4 kernels; standard runner, pinned replay and canonical adapter | Full common-machine run passed 165 correctness records, 32 sanitizer checks and six profiler captures |
| Part C | FP32 FA1, tuned PTX FA1, FA2 CUDA-core/WMMA/PTX; native and standard workflows | Full common-machine run passed 764 canonical correctness records and 24 profiler captures; FA1 invariants and representative sanitizer checks are retained |
| Root runner | Builds A/B/C, runs each separately and validates/aggregates canonical tables | Full-grid same-machine A/B/C execution, saved-result audit and offline regeneration passed |
| Plotting | Individual, combined, matched FA1/FA2 and supplementary figures | All 41 full-run PNG/PDF figure pairs are indexed and readable; native re-extraction and combined regeneration passed |

Existing B replay/diagnostic bundles are not accepted final comparisons. Existing A/B/C historical measurements came from different GPUs and cannot be divided to claim a same-machine optimisation speedup. Private smoke/debug evidence stays in ignored `extra/`; final retained report evidence belongs under a fresh public `results/<run_id>/` bundle.

All three runners link the `attention_reference` target from [part_a/cpu_reference/CMakeLists.txt](part_a/cpu_reference/CMakeLists.txt). Part B's CMake also links A's actual naive materialised pipeline for supplementary small GPU comparisons; those buffers are freed before B timing and excluded from B's allocation ledger. The shared FP64 oracle remains the correctness authority.

### Shared input and correctness contract

Use [input_generation.hpp](part_a/common/input_generation.hpp) directly: `attention::synthetic_input(shape, 5206)`, `attention::apply_pattern`, and `attention::input_fingerprint`. Preserve the exact SplitMix64 stream, filling Q then K then V. Layout and strides are defined in Section 3. Fingerprints are FNV-1a-64 over canonical little-endian FP32 bytes of Q/K/V, serialized as 16 lowercase hexadecimal characters. FP16 paths round once outside timing and fingerprint those rounded values represented as floats as well.

The shared FP64 oracle is authoritative, rather than another kernel or a timing CSV:

```text
identical generated Q/K/V ── shared CPU reference ── expected O
                        └── selected GPU kernel ── actual O
                                   element-wise comparison outside timing
```

`attention::reference(input, causal)` returns full S/P/O and is retained for small materialised tests. `attention::reference_output(input, causal, sampled)` computes O without allocating quadratic S/P. Its full and sampled modes are checked against the original reference in both dense and causal modes; CPU tests include future-K/V isolation, tails and deterministic patterns. Large GPU cases check query rows 0, floor(N/2), N−1 in every head and check every output for finiteness. Record full versus sampled scope and checked elements. Sampling is not exhaustive large-output validation.

A checks S/P/O for its small suite. B and C check full O for small cases and validate tuning candidates, including boundary and stress patterns. B also runs supplementary small GPU-to-GPU checks against A's materialised pipeline; these do not replace the shared FP64 oracle. Every accepted timing row has a passing O record for the exact implementation, configuration and effective inputs. FP32 tolerances are atol=rtol=1e-4; the existing FP16 experiment uses atol=0.002, rtol=0.02. Do not widen these automatically to hide failures. GPU-to-GPU comparisons may be added as supplementary checks; CPU-reference comparisons establish correctness.

### Standard runner and adapter conventions

These are implemented interfaces for all parts and the maintenance contract for future changes:

| Boundary | Convention |
|---|---|
| Build | Separate `part_<letter>/CMakeLists.txt`, kernel library and executable; standalone runner builds itself and root can prebuild each part |
| Helpers | Reuse `scripts/results_schema.py`, `workflow.py`, `ncu_csv.py` and the common plotter; resolve repository paths from script locations |
| Labels | `part=part_a\|part_b\|part_c`, `mode=dense\|causal`, `layout=BHNd_contiguous_row_major`, `generator=splitmix64-fp32-v1`, seed `5206`; actual precision labels remain explicit |
| Selection | Freeze configuration per implementation/d before the reporting grid; B exposes its FP32 winner as `variant=selected` in `configs/part_b.json` |
| Identity | Build `case_id` from the shared string-valued common fields; every table preserves its case identity, fingerprints and exact configuration |
| Export | Always write all seven schema headers, even for an explicitly empty profiling table; never publish failed/unchecked timings as accepted summaries |
| Provenance | Record schema version, source/binary hashes, actual shared-oracle linkage, physical GPU UUID, grid and profiling request in manifest/audit records |
| Profiling | Keep instrumented runs separate; preserve raw metric names/units, unavailable reasons and bundle-relative evidence paths |
| Re-analysis | Rebuild public tables from retained native records without changing measured values or case identity; root checks child CSV hashes before/after extraction |

Part B's five bounded candidates cover streaming W4/BK32, tiled W4/BK32, vector W4/BK32, tiled W8/BK32 and vector W4/BK64. Its selected-kernel full profiling grid is N256/4096,d64 with three repetitions (six captures). Part C separately profiles FP32 FA1, frozen PTX FA1, frozen FA2 CUDA-core and frozen PTX FA2 (24 captures). Actual accepted capture counts are recorded in the bundle audits.

Root rejects incompatible schemas, failed audits, incomplete requested grids, invalid numerical joins, differing GPU UUIDs/input fingerprints and missing requested profiler captures. A standalone B run is not automatically merged into an earlier A/C bundle. Use the root all-parts run for final comparisons; keep native additional studies separate.

### Standard public tables

The canonical format extends Part A's existing column names. Every table starts with:

```text
experiment,precision,B,H,N,d,mode,variant,tile,softmax_threads,
run_id,case_id,part,config,pattern,layout,input_precision,accumulator_precision,
output_precision,generator,seed,input_digest,rounded_input_digest
```

The actual header is one CSV line. `mode` is dense/causal, `experiment` is the execution phase, `precision` is input storage precision. Fused kernels leave A-specific tile/softmax-thread fields blank; full launch parameters belong in the config registry. `experiment=proposal` is the frozen final comparison grid; C's fixed-default progression remains separately labelled `proposal_default` where a nonzero configuration was selected. `case_id` identifies one implementation/config/workload; cross-part matching uses workload and input fingerprints, not identical case IDs.

| File | Purpose |
|---|---|
| `correctness.csv` | Stage/O errors, tolerance, PASS/FAIL/UNSUPPORTED, reference scope, checked rows/elements, nonfinite counts |
| `samples.csv` | All 20 unprofiled forward samples; actual separate stage samples only where measured |
| `summary.csv` | Forward median, Q1/Q3/IQR, throughput, validation and allocation evidence |
| `resources.csv` | Registers, shared/local memory, launch geometry and theoretical occupancy where queried |
| `profiles.csv` | All exported raw metric names/units, values or unavailable reasons, capture repetition and evidence path |
| `memory.csv` | Explicit live device bytes, auxiliary global bytes, observed deltas if measured and measurement definition |
| `work.csv` | Valid attention pairs and any supported scheduled/skipped tile counts, distinguishing analytical work |

Exact field order and units are enforced by the shared schema. Timings are milliseconds, storage is bytes, percentages are 0–100. Missing values stay blank. `peak_workspace_allocated_bytes` retains A's name but means explicit live allocation ledger including inputs/output and intermediates, not a device-wide memory peak. Stage medians from separate passes do not sum to measured forward latency.

Root aggregation checks exact headers, schema version, passing output checks, 20 repeats, five warmups, recomputed statistics, full requested grid coverage, physical GPU UUID, original/effective input fingerprints and profile-to-timing joins. It rejects old unchecked A data and historical C exports that lack shared-oracle linkage. A's saved RTX 5060 Ti timings are not combined with C's historical laptop measurements.

### Analysis and output layout

```text
results/<run_id>/
  manifest.json, commands.jsonl, run_status.json, ANALYSIS.md
  correctness.csv, samples.csv, summary.csv
  resources.csv, profiles.csv, memory.csv, work.csv
  comparison.csv                  # matched cross-part ratios and contracts
  fa1_fa2_comparison.csv           # matched FP32 and PTX FA1/FA2 cases
  fa1_cross_precision.csv          # separately labelled FP16-FA1/FP32-FA2 ratios
  audit.json, aggregation_audit.json, figure_index.json
  configs/                        # each part's launch registry/frozen selection
  parts/
    part_a/                       # complete standalone bundle with raw/part_a/
    part_b/                       # complete standalone bundle with raw/part_b/
    part_c/                       # complete standalone bundle with raw/part_c/
  plots/
    README.md                     # clickable index of generated PNG/PDF files
    part_a/                       # individual figures and plotted data
    part_b/
    part_c/
    combined/                     # applicable shared-input comparisons
  supplementary/part_c/           # additional retained profiling/analysis
```

Only requested parts are included. Standalone bundles have the same seven canonical CSVs at their own root and write native evidence under `raw/part_<letter>/`; their destination is the supplied `--output` path, not a hard-coded part-local folder. Root preserves the complete child bundles, aggregates validated rows and copies applicable figures. See [the schema](scripts/results_schema.py) and each part's README for native details.

[plot_combined.py](scripts/plot_combined.py) consumes canonical tables only. It generates per-part FP32/FP16 kernel progression with IQR, throughput/storage, tuning and materialised stage breakdowns, plus applicable A-causal/C-causal and A-dense/B-dense comparisons. When B participates it also generates dense-B versus causal-C latency, analytical valid-pair counts and available measured DRAM traffic, explicitly labelling them as different workloads. C additionally regenerates its native CUDA progression, Tensor Core progression, frozen core-versus-Tensor-Core, controlled scalar/vector-load ablation, tuning, memory and profiler figures. Full numerical data and native profiler metrics remain available for further report analysis. Missing parts/counters are declared, not plotted as fabricated zeros. FP16 A/C comparisons use FP32 accumulation/output and label their different probability rounding order.

Part B standalone output includes `plots/part_b/latency.{png,pdf}` and `memory.{png,pdf}` with plotted CSV data. Root aggregation adds its common FP32 progression, throughput/storage, numerical-error, tuning and available counter figures. The separate `analyze_delivery.py` command creates `part_b_latency.png`, `part_b_loading_memory.png` and `part_b_tuning.png` from retained diagnostic evidence.

The current FA1/FA2 figure set is listed below. Paths are relative to the selected output bundle. All common figures have PNG/PDF versions. `plots/README.md` links every indexed image/PDF; `figure_index.json` includes common, retained native and supplementary figures; Part C's `plots/README.md` links its actual generated images.

| Figure / location | Interpretation and configuration policy |
|---|---|
| `plots/part_c/cuda_core_progression` | Fixed-default FP32 progression, starting at FP32 FA1 |
| `plots/part_c/tensor_core_progression` | Fixed-default PTX FA1, WMMA FA2 and PTX FA2; implementation progression, not a single-feature ablation |
| `plots/part_c/kernel_progression_fp32`, `kernel_progression_fp16` | Canonical reporting-grid latency with quartiles; frozen configurations for tuned variants |
| `plots/part_c/fa1_fa2_latency` | Direct FP32 FA1/core-FA2 and PTX FA1/PTX FA2 latency panels, median and Q1–Q3 |
| `plots/part_c/fa1_fa2_comparison` | Matched-precision FA1 median / FA2 median; PTX configurations independently frozen |
| `plots/part_c/fa1_cross_precision` | PTX FA1 / FP32 FA2, explicitly different input precision |
| `plots/part_c/frozen_family_comparison` | Selected FA2 CUDA-core versus Tensor Core; FP32/FP16 tradeoff |
| `plots/part_c/frozen_tuning_speedup`, `tuning_heatmap`, `tuning` | Default-versus-frozen gains and bounded candidate search, including PTX FA1 |
| `plots/part_c/matched_vector_load` | Controlled scalar/vector load comparison with matching launch knobs |
| `plots/part_c/device_memory`, `throughput_and_memory`, `numerical_error` | Allocated storage (including FA1 m/l), throughput and checked numerical error |
| `supplementary/part_c/plots/` | Achieved occupancy, eligible warps, Tensor Core activity, sampled warp states, shared wavefronts and memory traffic for recorded profiler targets |
| `plots/combined/` | Applicable A/B/C comparisons; Part A WMMA/PTX stay separately labelled; dense/causal work is not output equivalence |

The root runner adds the common `kernel_progression_*`, `throughput_and_memory`, `numerical_error` and `tuning` plots. A standalone C export supplies the seven native figures, three FA1 comparison figures and available profiler charts; its own `plots/README.md` lists that actual set.

`fa1_fa2_comparison.csv` supplies exact case IDs, configurations, medians and quartiles for both matched speedup and latency figures. `fa1_cross_precision.csv` is separate. Quartiles describe within-case timing spread, not confidence intervals; speedup curves are ratios of medians. Native fixed-default progression and frozen primary comparisons answer different questions. Optional WMMA FA1 appears only in explicit supplementary/pilot runs or historical bundles.

The scripts are updated for fresh experiments. Existing historical result folders keep their original kernels, timings and captured processors; they are not final plots for the new PTX baseline. Smoke and pilot plots verify the pipeline but do not replace the full-grid report experiment. No final-grid performance data are fabricated during offline regeneration.

Offline root analysis checks the recorded child table hashes, invokes each child’s native re-extraction/analysis, rechecks table hashes, aggregates, copies retained supplementary evidence and regenerates common figures. Individual analyze modes re-extract native data; C also regenerates native/supplementary plots. Large binary Nsight reports are optional for CSV-only figure reproduction and remain checksummed evidence for deeper inspection.

## 7. Measurement conventions

| Setting | Convention |
|---|---|
| Latency | 5 warmups, 20 CUDA-event forward samples; median, Q1, Q3 and IQR |
| Outside timing | Input creation/rounding, allocation, transfers, oracle, resource queries and file I/O |
| Complete forward | A includes QK+softmax+PV; fused kernels include both products and online softmax |
| Throughput | sequences/s = 1000×B/median_ms; tokens/s = 1000×B×N/median_ms |
| Selection | Tune first; save frozen configuration before the main grid |
| Main comparisons | Matching GPU/run/shape/mode with accepted validation and precision labels |
| Memory | Live explicit allocations separately from observed device deltas and profiler traffic |
| Profiling | Separate instrumented runs; profiler timings never enter main speedup tables |
| Uncertainty | Within-case IQR is not an independent-session confidence interval |

The current studies use fixed sequential ordering and unmanaged clocks; very close winners may change on rerun. Rotating implementation order across independent future sessions would strengthen uncertainty estimates.

Part A provides WMMA and direct PTX materialised baselines with FP16 Q/K/V/P and FP32 S, accumulation, softmax arithmetic and O. Part C also uses FP32 accumulation/output, but rounds tile probabilities before online normalization. Match A-WMMA with C-WMMA and A-PTX with C-PTX, and retain this rounding-order distinction. Never divide historical Part A timings by Part C timings from another GPU.

## 8. Read the results

The report's common-machine measurements are retained in [results/final_submission/](results/final_submission/). Validation scope, uncertainty and reproduction checks are documented in [VERIFICATION.md](results/final_submission/VERIFICATION.md). The table below summarizes baseline/candidate median latency ratios over the 15 main-grid shapes; a ratio greater than one favours the candidate. Configurations were frozen using the designated tuning lengths.

| Baseline / candidate | Ratio range | Geometric mean |
|---|---:|---:|
| A optimized dense FP32 / B selected dense FP32 | 0.205–0.703× | 0.428× |
| A optimized causal FP32 / C frozen CUDA-core FA2 | 0.547–1.179× | 0.781× |
| A causal PTX / C causal PTX FA2 | 1.182–6.131× | 2.483× |
| C FP32 FA1 / C frozen CUDA-core FA2 | 4.083–6.554× | 5.290× |
| C PTX FA1 / C PTX FA2 | 3.032–13.079× | 6.433× |

The FP32 fused implementations reduce intermediate storage, but do not consistently reduce latency relative to optimized materialised A on this GPU. C's query-resident FA2 implementations improve on the corresponding FA1 baselines across the measured grid. The A/C PTX comparison includes their different probability-rounding order. These observations establish the performance trade-offs; attribution to instruction issue, memory access or residency requires the accompanying profiler and source analysis. Quartiles describe within-case timing variation, and the geometric means do not establish uncertainty across independent sessions.

Earlier studies remain separate historical/local evidence:

| Location | Contents and scope |
|---|---|
| [part_a/results/](part_a/results/) | RTX 5060 Ti materialised study: 180 main configurations, 96 tuning configurations and 108 profiled kernel captures; standalone evidence |
| [part_b/results/notebook_full_20261110_044740/](part_b/results/notebook_full_20261110_044740/) | Exact measured RTX 4070 Ti notebook evidence: 165 numerical PASS records, 90 timing cases and 1,800 samples; sanitizer/profiler omission keeps `audit.passed=false` |
| [part_b/EXECUTION_REFERENCE.json](part_b/EXECUTION_REFERENCE.json) | Pinned B source/dependency/reference identity for exact replay |
| [part_c/results/layout_validation/](part_c/results/layout_validation/) | Historical fixed-default correctness, benchmarks, sanitizers and figures |
| [part_c/results/standardized/](part_c/results/standardized/) | Historical saved-data format adaptation, retaining counters and source/SASS evidence |
| [part_c/results/bounded_evaluation/](part_c/results/bounded_evaluation/) | Historical eight-kernel common-input study, frozen selection, matched-load ablation and memory; predates PTX FA1 |
| [part_c/results/nsight_analysis/](part_c/results/nsight_analysis/) | 48 historical fixed-default captures with source/SASS analysis and corrected wavefront interpretation |

The historical C bounded study passed 288 full candidate checks, 318 sampled benchmark checks (6,360 samples) and 30 sampled memory checks. At N4096,B4/H8, its CUDA-core versus PTX FA2 timings were:

| d | Frozen CUDA core | CUDA latency (ms) | PTX FA2 latency (ms) | Core / tensor |
|---|---|---:|---:|---:|
| 32 | `skip`, BQ16/BK32/4 warps | 65.716 | 5.042 | 13.03× |
| 64 | `skip`, BQ32/BK32/8 warps | 107.663 | 12.153 | 8.86× |
| 128 | `vector`, BQ32/BK32/8 warps | 150.936 | 36.106 | 4.18× |

These are FP32-core versus FP16-input/FP32-output measurements on the recorded laptop GPU, **not FA1→FA2 speedups or final A/B/C results**. See [the historical report](part_c/results/bounded_evaluation/REPORT.md) for configurations, tuning gains and limits. The current matched PTX FA1/FA2 measurements are retained in `results/final_submission/fa1_fa2_comparison.csv`.

Explicit storage, observed device deltas and profiler traffic remain distinct. The materialised S+P curve is an analytical model. Source-level excessive shared wavefronts support a layout investigation; aggregate conflict counters alone can include arbitration. The historical tiled/skip/vector captures show zero source excess, so they do not establish bank-conflicted layouts.

## 9. Recreate figures without rerunning CUDA

The accepted bundle includes its original Nsight binaries in Git: 66 report files (36 A, six B and 24 C), totalling approximately 304 MiB. Each A report contains three kernel captures, so the files collectively retain 138 captures. Only `results/final_submission/` is exempted from the profiler-file ignore rules; reports from other experimental runs remain ignored. No separate profiler download is needed once these files are committed and pushed.

After cloning the submitted revision and installing the Python analysis dependencies, validate and regenerate the supplied evidence from the repository root:

```bash
python3 scripts/prepare_submission.py check --results results/final_submission
bash run_all.sh --analyze-only results/final_submission
python3 scripts/prepare_submission.py check --results results/final_submission
```

These saved-data commands require neither an NVIDIA GPU nor Nsight installation. They read the supplied binary files for integrity checks and use the retained CSV exports for numerical processing and plots. Running a new full GPU experiment requires CUDA, Compute Sanitizer and Nsight Compute as described in Section 5, and produces new observations rather than identical timings.


```bash
# Re-extract each child, validate identities/hashes and regenerate joint figures.
bash run.sh --analyze-only results/final_submission
# Or process one completed standalone bundle.
bash part_a/run.sh --analyze-only results/a_only
bash part_b/run.sh --analyze-only results/b_only
bash part_c/run.sh --analyze-only results/c_only
# Verify retained B evidence and create its separate delivery report assets.
python3 part_b/scripts/analyze_delivery.py --verify-only
python3 part_b/scripts/analyze_delivery.py --output results/b_report_assets
```

These commands read saved measurements; they do not rerun CUDA kernels. Diagnostic B bundles retain a failing acceptance audit, so their standard re-extraction can exit nonzero even when numerical records are valid. Use the delivery-analysis/replay checks for their diagnostic evidence scope rather than treating them as accepted joint input.

Preserve previous exports by using a copied bundle or a fresh result destination. Historical native datasets should use their captured processors/configurations; current source does not retroactively turn an old FA1 timing into a new PTX FA1 measurement. Saved CSVs suffice for available counter plots; full binary Nsight integrity/source audits additionally require the checksummed `.ncu-rep` files. See [Part C profiling](part_c/PROFILING.md) and [Part B cross-checking](part_b/CROSS_CHECK.md).

## 10. Submission and outstanding scope

The full-run operator should follow [README.txt, Steps 1–5](README.txt) using the integrator's reviewed source version. It specifies the one-machine command, smoke/counter preflight, expected profiler coverage (A108/B6/C24), full-grid acceptance and offline regeneration. No full experiment is launched by the source or archive checks.

The CPU-only [submission helper](scripts/prepare_submission.py) provides:

| Command | Purpose |
|---|---|
| `source-check --output extra/source_audit` | Copy maintained source/fixtures to a clean filesystem snapshot; run CPU tests and a dry plan without Git/GPU |
| `check --results results/final_submission` | Require complete A/B/C full-grid tables, identities, profiler evidence and figures |
| `pack ...` | Include maintained source, complete selected results and explicitly supplied final report, version and URLs; optionally split raw profiler binaries |
| `verify-archive SOURCE.zip --profiles PROFILER.zip` | Verify transferred archive members and SHA-256 checksums |

Both pinned historical Part B smoke and full recordings are retained. The 13 November 2026 CPU audit passed 36 Python tests, the pinned full-recording check and the root dry plan; the shared CPU reference also built and passed its agreement tests. `--allow-smoke` checking and `--validation-only` packaging are explicitly labelled preparation tests, not final report evidence.

The Git repository includes the root live-demo notebook and `flash_attention_live_demo/source/`. The current archive helper selects the core A/B/C source/evidence and does not automatically include that additional notebook/demo directory. If distributing a source ZIP for the video demonstration, include those files and `flash_attention_live_demo/README.md` alongside the core archive, preserving their relative paths. Generated demo builds and runs are not required to execute a fresh demonstration.

Archive outputs and audit copies belong under ignored `extra/` or a separate delivery location. The private `extra/main.tex` is excluded; the final PDF must be supplied explicitly. Report/video contents, actual contributions, AI disclosure, public link access and the reviewed commit require human confirmation. Nothing is uploaded or committed by the helper.



Retain raw benchmark CSVs, selected configurations, metadata, plotting scripts, readable figures and reproduction instructions. Include source and source/binary hashes. The accepted final-run Nsight reports are included directly in Git; optional split archives remain available for other delivery formats. Do not submit compiled build directories as results.

The full-grid results and restored historical recordings are retained and verified. The recorded presentation is 14:58 according to the team. Before delivery, ensure the final PDF contains the renamed repository URL, accessible video URL and reviewed final revision, and confirm the links work without a team login. The report should distinguish materialised/fused latency and capacity, dense/causal workloads, and matched versus independently tuned optimisation comparisons. No fixed speedup target is required by the rubric; well-supported slowdowns and limitations are valid results.

