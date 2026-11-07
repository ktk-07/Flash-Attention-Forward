# CEG5206 Team Project — FlashAttention Forward

**Documentation:** This `README.md` describes the project, repository structure, implementation conventions and evaluation status. For step-by-step instructions to reproduce the available runs, use [README.txt](README.txt). Section 6 defines the required one-command reproduction workflow, identical public CSV contract and integration work for all three parts.

NUS CEG5206 — Algorithms & Performance Optimization for Computing, AY2026/2027 Semester 1, Group 1.

CUDA C++ implementations of materialised attention, dense fused FlashAttention and causal fused FlashAttention. The project studies tiling, online softmax, kernel fusion, causal tile skipping and GPU memory behaviour. Mixed precision and Tensor Cores are the optional extension.

The scope follows the approved FlashAttention proposal, Project Briefing and Team Project Rubrics.

**Deadline stated in the briefing:** Friday, 13 November 2026, 11:59 PM. Submission requires a report PDF, an accessible 10–15-minute video link, the repository link and a final commit hash or release tag.

## Team responsibilities

| Part | Member | Primary responsibility | Cross-review |
|---|---|---|---|
| A | Lyu Sihan | Independent C++ FP64 oracle; materialised QKᵀ, softmax and PV; baseline optimisation | Dense fused correctness and comparison fairness |
| B | Zhang Yichun | Dense fused forward; online softmax, tiled Q/K/V and output accumulation | Materialised indexing and memory access |
| C | Kyaw Tun Khine | Causal fused forward; future-tile skipping, diagonal masking and optimisation | Causal correctness and performance interpretation |

Each member contributes testing, tuning, profiling and analysis. Record actual contributions in the final report. AI assistance must be declared as required by the briefing; Part A and this Part C evaluation received Codex assistance and require team review.

## Repository layout

```text
README.md                         # project overview, conventions and result index
README.txt                        # executable reproduction instructions
run_all.sh                        # sequential A/C orchestration; B adapter pending
scripts/run_all.py                # command logging, CSV collection and audit
part_a/
  common/                         # layouts, deterministic inputs and FP64 oracle
  gpu_baseline/                   # materialised CUDA implementations
  cpu_reference/                  # independent FP64 oracle
  main.cpp                        # independent benchmark/test CLI
  results/                        # saved RTX 5060 Ti measurements
  README.md
part_c/
  include/flash_attention/        # public declarations and shared types
  src/                            # FA1, FA2 CUDA-core and Tensor Core definitions
  tests/                          # runners, shared support and input-contract check
  scripts/                        # tuning, plotting, auditing and profiling helpers
  run.sh                          # regression, --study or --complete entry point
  evaluate.sh                     # bounded-study build/run wrapper
  profile.sh                      # default or frozen-winner Nsight workflow
  requirements-analysis.txt       # pinned plotting dependencies
  results/                        # retained raw data, metadata, figures and reports
  README.md
  PROFILING.md
```

There is no separate `part_b/` directory in this checkout. The root `run_all.sh` can run A/C explicitly and collect their existing outputs; an all-parts request fails clearly until Part B is supplied. Shared-oracle integration and final validation gates remain outstanding. Helpers stay with their respective part.

## 1. Software and platform requirements

The documented Part C workflow uses Linux, Bash, a CUDA-capable NVIDIA GPU, a C++17 compiler, CMake and Python. The tested GPU is an RTX 3070 Ti Laptop with SM86. The scripts default to architecture 86; another GPU requires the appropriate architecture and supported instructions. Part A's saved results were collected separately on an RTX 5060 Ti.

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
cd /path/to/Flash-Attention-CUDA
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

The new Part C study checks Part A/C input bytes for equality and records input fingerprints. Historical Part C regression/profile data uses its older mt19937 stream and retains that label. Equal seeds alone do not imply equal tensors.

## 4. Build and run the available tests

```bash
cmake -S part_c -B /tmp/part_c-evaluation-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/part_c-evaluation-build --parallel 2
/tmp/part_c-evaluation-build/part_c_input_contract
/tmp/part_c-evaluation-build/part_c_runner \
  --suite correctness --output part_c/results/my_correctness
```

The default regression checks all eight variants, including tails, unusual head dimensions and deterministic patterns. Unsupported PTX dimensions are explicit skips. The bounded study separately validates every candidate configuration before tuning. Full small-case outputs are checked; large performance cases check queries 0, N/2 and N−1 in every head plus finiteness of every output. Sampled checks are not labelled full validation.

## 5. Run Part C

Use fresh output directories to preserve previous results.

```bash
# Single command: default regression/benchmark, sanitizer checks, legacy test,
# then bounded tuning, frozen evaluation, matched-load ablation, memory and plots.
bash part_c/run.sh --complete part_c/results/my_complete_run /tmp/part_c-evaluation-build

# Bounded study only:
bash part_c/run.sh --study part_c/results/my_bounded_study

# Existing default-only workflow:
bash part_c/run.sh part_c/results/my_regression /tmp/part_c-build
```

`--complete` writes the default workflow under `regression/` and the new study under `main/`. All GPU cases run sequentially. It does not run Parts A or B. Detailed standalone kernel commands are in [part_c/README.md](part_c/README.md).

Profile frozen d=64 winners separately after the study completes:

```bash
bash part_c/profile.sh --frozen part_c/results/my_bounded_study
bash part_c/evaluate.sh --verify part_c/results/my_bounded_study
```

These commands use `/tmp/part_c-evaluation-build` by default. The first captures Nsight counters; the second reruns the default regression and sanitizes the frozen d=64 winners. Avoid concurrent GPU workloads.

## 6. Integration handoff for each part and its coding agent

**Goal: standardize and integrate the software before collecting the final experiments.** Once Part B is ready, the user or examiner must be able to run one root `run.sh` on a selected GPU and obtain all correctness records, benchmarks, profiling/memory evidence, per-part plots and combined plots. Each part must also remain independently runnable using the same settings and public CSV format.

**Current status:** the existing `run_all.sh`/`scripts/run_all.py` is an A/C orchestration prototype with field mappings. The native A/C CSVs are not yet identical, the final shared correctness gate is incomplete, and the combined plotting script is not implemented. Part C now has a fixed-header, audited export and named standalone wrapper; A/B and the root prototype still need to adopt the same shared schema. The requirements below are the implementation contract for each member's coding agent; they are not claims that the final workflow already exists.

Keep kernels in their separate part directories. Share the generator, CPU reference, schema/writer and analysis utilities through includes/libraries/adapters. Do not copy kernels or require one combined C++ executable. Implement and test the adapters, schema checks and plotting logic now; collect fresh final GPU results only after integration and Part B are ready.

### Shared inputs and correctness

Every runner must use the same Q/K/V values for a given shape and pattern. Include Part A's [input generator](part_a/common/input_generation.hpp) through a CMake include directory and call `attention::synthetic_input(shape, 5206)`. This is a header-defined function; no Part A executable is involved. Preserve its SplitMix64 stream and Q-then-K-then-V order. Apply common test-pattern transformations identically and record original/effective input fingerprints using one agreed byte representation and fingerprint routine. FP16 paths round the same base inputs to FP16 once outside timing and give the oracle those rounded values represented as floats.

Use Part A's [CPU FP64 reference implementation](part_a/cpu_reference/attention_reference.cpp), declared in [attention_types.hpp](part_a/common/attention_types.hpp), as the shared numerical oracle. Expose that source as a reusable CMake library and link each test runner to it, retaining Part A's standalone executable. The function is called directly in the runner; GPU kernels remain in their own part directories.

```text
Shared deterministic Q/K/V
  ├── shared CPU FP64 reference ── expected output O
  └── selected part's GPU kernel ─ actual output O
                                      ↓
                        element-wise comparison outside timing
                                      ↓
                         correctness.csv records PASS/FAIL
```

| Implementation being checked | CPU oracle mode | Additional GPU cross-check |
|---|---|---|
| Part A dense | `attention::reference(input, false)` | Other validated A-dense variants |
| Part B dense | `attention::reference(input, false)` | Validated Part A FP32 dense output |
| Part A causal | `attention::reference(input, true)` | Other validated A-causal variants |
| Part C causal | `attention::reference(input, true)` | Validated Part A FP32 causal output |

The CPU oracle is authoritative. A GPU-to-GPU comparison is supplementary and must report its own precision/tolerance contract. Dense B and causal C generally have different outputs and must not be compared element-wise as the same operation. Benchmark CSVs contain timings and error summaries, not complete expected tensors; comparing their numbers does not validate a kernel.

Small tests compare every O element with `abs(actual-expected) <= atol + rtol*abs(expected)`. Retain A's S/P checks for materialised kernels; B/C only need to expose O. Start with the existing FP32 tolerances atol=rtol=1e-4 and FP16-input tolerances atol=0.002, rtol=0.02, recording the full precision contract. Report failures rather than automatically widening tolerances. Include N=1, partial tiles, multiple batches/heads, uniform attention, constant V and finite large-logit cases. Add a causal isolation test: changing future K/V rows must not affect earlier query outputs.

Part A's existing full oracle allocates FP64 S/P matrices. Keep it for small full checks. For large-grid validation, add a shared output-only sampled-row reference, checking rows 0,N/2,N−1 in every head, and first verify that helper against the original full oracle on small cases in both modes. Check every GPU output for finiteness. Save the sampled row indices and checked element count; sampled validation must not be labelled full-output validation. Validate every accepted final-grid row outside timing, not just representative shapes.

### Responsibilities before the final run

| Owner | Required integration work | Existing work to preserve |
|---|---|---|
| Lyu Sihan — A | Expose the CPU reference and validated sampled-row helper; add per-case correctness/fingerprints and the shared public CSV writer/adapter; provide the common standalone `part_a/run.sh` interface. | Existing CPU/GPU algorithms, standalone CLI, tests and historical results. No new kernel optimisation is required solely for integration. |
| Zhang Yichun — B | Supply dense kernels and `part_b/run.sh`; use the shared generator/oracle, public CSV schema and standalone CLI; provide validated tuning, benchmarks and profiling through that adapter. | B's own source files, headers, kernels and standalone tests. |
| Kyaw Tun Khine — C | Connect tests to the shared generator/oracle; adapt existing study outputs to the same public CSV headers and standalone CLI; retain frozen selection and safety checks. | C's kernel bodies/comments, existing tuning, safety checks and recorded datasets. The current input-contract test already confirms bitwise A/C base-input agreement. |
| Shared integration work | Implement root `run.sh`, the single versioned schema/writer/validator, extraction/aggregation, combined plotting and the complete run audit. Make `run_all.sh` a compatibility wrapper for the same workflow. | Per-part runners remain independently usable; current collection code lives in `scripts/run_all.py`. |

**Part A therefore has a small integration task, not a kernel rewrite.** Its runner currently validates selected representative performance cases; that is insufficient to mark the whole final grid as validated. The integration adapter or A's runner must close that gap. Existing A timing/error CSVs cannot substitute for executing the shared tests or rerunning A on the final machine.

Part A already supports `--mode dense|causal|both` for both the correctness suite and performance runs. This applies to its FP32 variants and its separate FP16 Tensor Core experiment. `--variants all` selects the four FP32 variants; use `--variants tensorcore` separately for FP16. The remaining A checklist is:

1. Keep the existing row-major GPU implementations and shared generator; export the CPU reference through a library target that other runners can link.
2. Add the agreed large-case sampled/output-only reference and check it against the existing full reference in both modes.
3. Validate every final-grid timing case, check all outputs for finiteness, and export validation scope/counts and original/effective input fingerprints.
4. Align its bounded tuning with the agreed experiment plan, freeze configurations, and rerun dense/causal FP32 plus the selected optional FP16 baseline on the common GPU.
5. Update its CSV metadata/adapter so the root collector can accept validated rows for final comparisons.

B/C must then call the shared reference with the appropriate mode and retain their own kernels. A complete all-parts run additionally needs B's implementation/adapter and the shared acceptance checks below; finishing A alone does not supply the missing Part B.

### Required root and individual entry points

The following are the **target interfaces for the complete integration**. The named Part C commands are now implemented; root `run.sh` and the A/B wrappers remain pending:

```bash
# Examiner: one command for all parts and all required result generation.
bash run.sh --arch 86 --output results/examiner_run

# Same pipeline for just one part; same tests, CSV headers and plot conventions.
bash run.sh --parts a --arch 86 --output results/part_a_run
bash run.sh --parts b --arch 86 --output results/part_b_run
bash run.sh --parts c --arch 86 --output results/part_c_run

# A part can also be invoked directly through its own wrapper.
bash part_a/run.sh --arch 86 --build /tmp/attention-build/a --output results/a_only
bash part_b/run.sh --arch 86 --build /tmp/attention-build/b --output results/b_only
bash part_c/run.sh --arch 86 --build /tmp/attention-build/c --output results/c_only

# Recreate all tables and plots from saved standardized data, without GPU work.
bash run.sh --analyze-only results/examiner_run
```

With no arguments, the final root `run.sh` must request A/B/C, select the visible CUDA device, detect its supported architecture and create a timestamped `results/<run_id>/` directory. `--arch`, `--device` and `--output` provide explicit overrides; `--build-root` selects build storage, and `--dry-run` prints the plan without creating results or running CUDA. Use the architecture appropriate to the selected GPU and a fresh output directory. Run GPU work sequentially on the selected device and fail before measurements if a required part/tool is missing. The final wrapper must support device selection consistently across child processes and record the selected device identity. Retain existing per-part CLI options as compatibility paths where practical. The final `run_all.sh` should forward to root `run.sh`, rather than maintain a different experiment pipeline.

Single-part execution must not depend on another part's GPU executable or results. B/C may depend on A's shared generator/reference source library. Each standalone wrapper must produce the same canonical files at its output root as the root runner, with only that part's rows. Common scripts then generate its plots; combined figures require the appropriate selected parts. An explicitly requested subset is a valid subset run, not a complete A/B/C submission run.

### Main report standard: Part A format with shared extensions

**Use Part A's existing result organization and column names as the base for the main report. Adapt B/C to that public format; do not rename A's timing columns merely to match C.** Part A already exports samples, timing statistics, S/P/O correctness, resources and profiling. Its remaining correctness/fingerprint/precision coverage still needs the additions described above; the existing files alone do not establish that every final case is validated.

All adapters and the root collector must publish schema version 1 with the same filenames and fixed header order through the shared `scripts/results_schema.py` module. Part C implements this through `part_c/scripts/export_results.py`; A/B and the root prototype remain to be adapted. Preserve existing native outputs under `raw/`. Adapters can reorder headers and add fields without rewriting kernel algorithms or deleting existing plots. Plotting for the main report reads the standardized tables, not independent per-part field-renaming rules.

Retain Part A's common columns first:

```text
experiment,precision,B,H,N,d,mode,variant,tile,softmax_threads
```

`precision` denotes input storage precision; retain it for compatibility and also record explicit input/accumulator/output precision below. `variant` identifies the implementation, so map C's native `kernel` field to `variant`. `tile` and `softmax_threads` retain their actual Part A meanings; B/C leave non-applicable values blank rather than relabel a different tile dimension. Their full BQ/BK/warp/register-tile parameters belong in the frozen config registry.

Append these common metadata columns, then the table-specific fields (wrapped here; the CSV has one header line):

```text
run_id,case_id,part,config,pattern,layout,input_precision,accumulator_precision,
output_precision,generator,seed,input_digest,rounded_input_digest
```

`part` is `part_a|part_b|part_c`; `mode` is `dense|causal`. Validation/benchmark/memory phases belong in `experiment`, not `mode`. `case_id` joins one implementation/config/workload's correctness, samples and resources within a run. Cross-part comparisons match workload/mode/precision and input fingerprints, not equal implementation case IDs. Record softmax precision, probability rounding, supported shapes and kernel symbols in the manifest's implementation registry.

Use one fingerprint routine: `input_digest` is FNV-1a-64 over Q then K then V as canonical little-endian FP32 bytes before rounding; `rounded_input_digest` applies the same routine to FP16-rounded values represented as FP32. Serialize as 16 lowercase hexadecimal characters, matching C's existing representation. The effective fingerprint is the original digest for FP32 and the rounded digest for FP16. Equal seeds alone are insufficient.

| Main report file | Fields after the common columns |
|---|---|
| `correctness.csv` | Keep A's `stage,atol,rtol,max_absolute,max_relative,failures,status`; add `oracle_input_precision,reference_scope,checked_query_rows,checked_elements,nonfinite` |
| `samples.csv` | Keep A's `repeat,forward_ms,qk_ms,softmax_ms,pv_ms`; fused kernels leave separately unmeasured stage times blank |
| `summary.csv` | Keep A's `atol,rtol,max_absolute,max_relative,warmup,repeats,forward_median_ms,forward_q1_ms,forward_q3_ms,forward_iqr_ms,qk_median_ms,qk_iqr_ms,softmax_median_ms,softmax_iqr_ms,pv_median_ms,pv_iqr_ms,sequences_per_second,speedup_vs_naive,peak_workspace_allocated_bytes,setup_host_ms`; add `tokens_per_second,validation_status,reference_scope` |
| `resources.csv` | Keep A's `stage,blocks,threads,registers_per_thread,shared_bytes,local_bytes_per_thread,active_blocks_per_sm,theoretical_occupancy_percent` |
| `profiles.csv` | Keep A's `kernel,metric,unit,value` for low-level profiled symbols/counters; add `stage,profile_repeat,availability,reason,source_report` |
| `memory.csv` | Shared extension: `explicit_live_device_bytes,auxiliary_global_bytes,observed_device_delta_bytes,measurement` |
| `work.csv` | Shared extension: `valid_attention_pairs,scheduled_tiles,skipped_tiles,evidence_kind`; distinguish analytical/scheduled work from measured executed instructions |

C's adapter maps `median_ms/q1_ms/q3_ms/iqr_ms` to A's `forward_median_ms/forward_q1_ms/forward_q3_ms/forward_iqr_ms`, and its recorded warmup setting to `warmup`. Keep C's original columns/files unchanged in its native output. The prototype collector currently uses intermediate names and must be updated to this final Part A-based schema.

Do not fabricate QK/softmax/PV timing breakdowns for fused kernels. A's stage samples are separate passes; never add their medians to derive full-forward latency. `speedup_vs_naive` is optional and only populated with a documented matching within-part baseline; combined speedups are computed from explicitly identified candidate/baseline rows. Retain `peak_workspace_allocated_bytes` for A compatibility and define it as the explicit live-buffer allocation peak including Q/K/V/O and any S/P, not device-wide peak or auxiliary storage alone. Detailed allocation components belong in `memory.csv`. Unmeasured setup time remains blank.

Missing/non-applicable values remain blank rather than zero. Metric availability is `available|unavailable|not_applicable`, with reasons where needed. Files use UTF-8, ordinary CSV quoting, decimal points and unrounded numeric values. Timings are milliseconds, allocation/traffic are bytes, and percentages use 0–100. Accepted timings must be positive and finite.

Correctness status is `PASS|FAIL|UNSUPPORTED`; transitional `UNCHECKED` cases cannot enter accepted final comparisons. Every timing case must link to passing O validation for the exact config and inputs. Keep A's S/P checks as additional records, allowing expected masked S values rather than treating them as invalid O values. Optional unsupported configurations remain visible; missing required FP32 coverage fails the complete-run audit.

### Preserve and generate supplementary per-part outputs

Standardization applies to the **main report tables and comparisons**, not every internal experiment. Keep each part's useful additional CSVs, source/SASS exports, specialized tests and individual plotting scripts. Do not discard information merely because A's base format has no column for it.

The complete root run must invoke the existing useful per-part generators as well as the common main-report analysis. Retain their native input files under `raw/<part>/` and expose extra data/figures under `supplementary/<part>/`, with an index recording source files and commands. Existing standalone paths may remain as compatibility outputs. Main per-part/report figures go under `plots/<part>/`; joint figures go under `plots/comparison/`. Supplementary results stay labelled and do not silently enter the fixed main comparison grid.

`--analyze-only` must regenerate both main and supplementary figures from their saved inputs. Required inputs for an individual plot are reproducibility evidence, not disposable scratch files. Only private planning/debug clutter belongs in ignored `extra/`.

### What one full run must do automatically

1. **Preflight and build:** check all requested sources, CUDA architecture/device, compiler, Python dependencies, Compute Sanitizer and Nsight requirements; build reusable CPU reference and all requested runners. Save tool versions, source/binary hashes, exact commands and exit codes in the manifest/logs.
2. **Shared correctness gates:** verify identical input generation/rounding and agreement of full/sampled CPU references; run small/full, boundary, stress, causal-isolation and representative sanitizer tests. Validate every large performance case against sampled CPU rows and check all outputs for finiteness, outside timing.
3. **Bounded tuning:** execute the agreed candidate sets on N256/1024 for d32/64/128, record invalid candidates explicitly and freeze each selection before the final grid. Label any fixed optional Tensor Core baseline as fixed rather than tuned.
4. **Final experimental grid:** run B4,H8,N256/512/1024/2048/4096,d32/64/128. A supplies dense/causal baselines, B dense fused kernels and C causal fused kernels. Use five warmups and 20 forward samples; keep allocation, transfers, references and file I/O outside timing.
5. **Memory/work/profiling studies:** collect explicit allocation evidence and representative small/large hardware counters using the frozen configurations. Save raw counter exports and report hashes. Instrumented timings stay out of `samples.csv`. Hardware-unavailable metrics need explicit availability/reasons; mandatory experiment failures must not be silently skipped.
6. **Extract and standardize:** keep native outputs under `raw/<part>/`; use the shared adapters to emit the public tables above. Enforce the same schema for individual and root runs, including dense/causal semantics and precision labels.
7. **Audit and aggregate:** check GPU/run identity, schema version/headers, input fingerprints, expected coverage, correctness/config links, sample counts and recomputed medians/quartiles/IQR. Record failures and stop acceptance instead of silently dropping rows.
8. **Generate all outputs:** call the main per-part, combined and retained supplementary plotting scripts, export supporting comparison CSVs, and write `ANALYSIS.md`, a figure index and `audit.json`. Mark the run complete only after all required artifacts and checks succeed. `--analyze-only` must repeat extraction/audits/analysis/plotting from saved records without executing GPU kernels.

None of these final software interfaces requires collecting new benchmark data today. Implement and test them using clearly identified fixtures first. Final reported values come from a fresh user/examiner run once Part B and integration are ready; fixtures and historical measurements must not be inserted as new results.

### Required combined plots and tables

Implement one `scripts/plot_combined.py` (or equivalent shared analysis module) that consumes only accepted standardized tables. It must work for a complete run and generate the applicable comparisons for an explicit subset, reporting which comparisons are unavailable. It must not parse each part's native schema separately.

| Figure/table | Required comparison |
|---|---|
| CUDA-core progression | Per-part FP32 variants, with A-dense/B-dense and A-causal/C-causal baselines matched by mode |
| Tensor Core progression | Available Tensor Core variants, with full precision contracts labelled |
| Frozen best CUDA-core versus Tensor Core | C's frozen family winners, plus compatible A/B context; label precision differences |
| Dense materialised versus fused | A-dense versus B-dense latency, IQR and speedup across N/d |
| Causal materialised versus fused | A-causal versus C-causal latency, IQR and speedup across N/d |
| Dense versus causal | B-dense versus C-causal work/latency/traffic; different workloads, not output equivalence |
| Memory and throughput scaling | Comparable explicit storage and throughput across shapes; measured/modelled values distinguished |
| Tuning and controlled ablations | Candidate/config choices and matched-parameter experiments; progression alone is not one-feature causality |
| Numerical error | Full/sampled scope and precision-specific error summaries, linked to accepted timings |
| Profiling/bottlenecks | Representative occupancy, scheduling and memory evidence for frozen implementations; unavailable counters stay explicit |

Export report figures as PNG/PDF and their plotted numeric data as CSV. Speedup denominators must match run, shape, mode and the intended numerical contract; cross-precision ratios require explicit labels. The final report must discuss slowdowns and missing evidence honestly rather than fabricate a complete series.

### Final output layout

```text
results/<run_id>/
  manifest.json                 # schema_version=1, plan, device, versions, hashes
  commands.jsonl                # all commands and exit codes
  correctness.csv
  samples.csv
  summary.csv
  resources.csv
  memory.csv
  profiles.csv
  work.csv
  audit.json
  ANALYSIS.md
  configs/                      # candidate observations and frozen selections
  raw/part_a/                   # original outputs, logs and profiling exports
  raw/part_b/
  raw/part_c/
  plots/README.md               # figure index and evidence links
  plots/part_a/
  plots/part_b/
  plots/part_c/
  plots/comparison/
  tables/                       # plotted data and combined comparison tables
  supplementary/part_a/         # extra native data/plots, indexed and retained
  supplementary/part_b/
  supplementary/part_c/
```

Keep private planning/scratch files in ignored `extra/`. Required numerical evidence, reproducibility metadata and figure data belong in the public run directory. Preserve earlier runs rather than overwrite them. Reproduction means the same inputs, validation and measurement procedure with regenerated evidence; another GPU is not expected to reproduce identical timing values.

### Implementation status and remaining work

| Component | Current status | Required completion |
|---|---|---|
| Shared row-major inputs | A generator exists; C's current stream passes bitwise agreement tests | All wrappers use the shared generator and fingerprint routine directly |
| Shared CPU reference | A's full reference exists; C currently has its own oracle | Reusable library, sampled helper and agreement tests, linked by every part |
| Root orchestration | `run_all.sh` runs existing A/C workflows sequentially | Root `run.sh` with the entire pipeline above; compatibility alias and individual part wrappers |
| Main CSV standardization | Shared schema module and lossless Part C export implemented; A/root still use native/intermediate schemas | Adapt A/B/root to the same fixed Part A-based headers and validation rules |
| Part A validation | Dense/causal small tests exist; some performance rows unchecked | Every final timing case validated and fingerprinted |
| Part B | Final source/adapter not present in this checkout | Dense implementation, common CLI/schema and shared validation |
| Per-part figures | Native A/C plots retained; C export also generates supplementary profiler charts | Connect A/B/root to the same public output and evidence-index conventions |
| Combined figures/analysis | Not implemented | Shared combined plotting, table extraction and analysis generation |
| Final acceptance | Prototype records `final_integration_ready=false` | Complete-run coverage, numerical/schema/metadata audits and all required artifacts |

The existing transitional commands are still `bash run_all.sh --parts a,c --arch 86 --output results/my_run --dry-run` and the same command without `--dry-run`. They are not the final examiner reproduction contract: they leave A's unchecked rows/fingerprints explicit and do not generate combined plots. Do not treat the target `run.sh` interfaces above as implemented until the corresponding code and tests are complete. After implementation, update this status table and `README.txt` to the verified commands.

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

Part C's current study uses fixed sequential ordering and unmanaged clocks; very close winners may change on rerun. A future shared runner should rotate implementation order across repeated runs.

Part A Tensor Core uses FP16 accumulators, intermediates, softmax and output. Part C Tensor Core uses FP16 Q/K/V, FP32 accumulators/state/output, and FP16 probability operands for PV. These comparisons show precision/performance trade-offs; they do not isolate fusion or hardware alone. Never divide historical Part A timings by Part C timings from another GPU.

## 8. Read the results

| Location | Contents |
|---|---|
| [part_a/results/](part_a/results/) | Historical materialised results on RTX 5060 Ti |
| [part_c/results/layout_validation/](part_c/results/layout_validation/) | Previous fixed-default correctness, benchmarks, sanitizers and figures |
| [part_c/results/standardized/](part_c/results/standardized/) | Part A-format saved-data export, retaining native counters and supplementary source/SASS evidence |
| [part_c/results/bounded_evaluation/](part_c/results/bounded_evaluation/) | New common-input bounded study, frozen selection, matched ablation and memory |
| [part_c/results/nsight_analysis/](part_c/results/nsight_analysis/) | 48 historical fixed-default captures, source/SASS analysis and corrected wavefront interpretation |

The bounded study writes per-case CSVs, `measurements.csv`, `summary.csv`, `comparison.csv`, `matched_ablation.csv`, configuration/plan/source manifests, exact command records, `audit.json`, `REPORT.md`, and PNG/PDF figures. Its figures cover CUDA-core progression, Tensor Core progression, frozen family comparison, tuning, matched loading and device storage. Profiling and additional verification live in separate subdirectories.

The recorded bounded study passed 288 full candidate checks, 318 sampled benchmark checks (6,360 timing samples), and 30 sampled memory checks. At the held-out N=4096, B=4,H=8:

| d | Frozen CUDA core | Latency ms | Frozen PTX MMA ms | Core / tensor |
|---|---|---|---|---|
| 32 | `skip`, scalar loads, BQ16/BK32/4 warps | 65.716 | 5.042 | 13.03× |
| 64 | `skip`, scalar loads, BQ32/BK32/8 warps | 107.663 | 12.153 | 8.86× |
| 128 | `vector`, float4 loads, BQ32/BK32/8 warps | 150.936 | 36.106 | 4.18× |

These are FP32-core versus FP16-input/FP32-accumulator/output Tensor Core results on the recorded laptop GPU. PTX uses BQ32/BK16/2 warps. See the [measured report](part_c/results/bounded_evaluation/REPORT.md) for tuning gains, allocation measurements and limitations.

Memory curves distinguish measured explicit storage from the analytical S+P storage model. Source-level excessive shared wavefronts establish a layout-conflict signal; aggregate L1 bank-conflict counters alone can include arbitration. The historical tiled/skip/vector captures show zero source excess, so those counters do not support calling their layouts bank-conflicted.

## 9. Recreate figures without rerunning CUDA

Figure regeneration needs the saved CSVs. Full profile-integrity audits additionally need the checksummed `.ncu-rep` files; restore separately supplied reports to their recorded `cases/` paths before running those audits.

```bash
python3 part_c/scripts/evaluate.py --build /tmp/part_c-evaluation-build \
  --output part_c/results/bounded_evaluation --analyze-only
python3 part_c/scripts/audit_evaluation.py part_c/results/bounded_evaluation
python3 part_c/scripts/analyze_profiles.py part_c/results/nsight_analysis
python3 part_c/scripts/interpret_profiles.py part_c/results/nsight_analysis
python3 part_c/scripts/audit_profiles.py part_c/results/nsight_analysis
```

See [part_c/PROFILING.md](part_c/PROFILING.md) for metric definitions, capture limitations and source analysis.

## 10. Submission and outstanding scope

Retain raw benchmark CSVs, selected configurations, metadata, plotting scripts, readable figures and reproduction instructions. Include source and source/binary hashes; large raw Nsight reports can be supplied as a separate accessible artifact with checksums. Do not submit compiled build directories as results.

Part C's local evaluation does not complete the project-wide requirements. The remaining shared work includes the final dense fused Part B source and tests, same-GPU A/B/C results, materialised-versus-fused validation, dense-versus-causal work/traffic analysis, the report, video, references and member contributions. The final comparison must include materialised versus fused latency/storage, dense versus causal work/traffic, bounded tuning and controlled optimisation comparisons. No fixed speedup target is required by the rubric; well-supported slowdowns and limitations are valid results.
