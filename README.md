# CEG5206 Team Project — FlashAttention Forward

**Documentation:** This `README.md` describes the project, repository structure, implementation conventions and evaluation status. For step-by-step instructions to reproduce the available runs, use [README.txt](README.txt). The root runner and remaining shared A/B/C integration work are described in Section 6 below.

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

**Status: root orchestration and CSV collection are implemented for existing A/C workflows; shared-oracle integration and the final A/B/C validation gates remain pending.** Keep the three implementations and their runners separate. `run_all.sh` builds and runs the requested available workflows sequentially, preserves their raw outputs and generates per-part plots. It does not copy or rewrite kernels. A single shared C++ executable is not required.

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
| Lyu Sihan — A | Expose the CPU reference as a reusable library; keep the generator header available; add or share the validated sampled-row reference; make every final-grid benchmark row carry a correctness record and common metadata/CSV mapping. | Existing CPU/GPU algorithms, standalone CLI, tests and historical results. No new kernel optimisation is required solely for integration. |
| Zhang Yichun — B | Supply the dense fused implementation and runner; use the shared generator/oracle; expose supported shapes and launch constraints; add common validation/CSV output and bounded tuning. | B's own source files, headers, kernels and standalone tests. |
| Kyaw Tun Khine — C | Connect the test runner to the shared generator/oracle, replacing the duplicated generator/reference logic after agreement checks; map existing study outputs to the shared schema and retain the frozen-selection process. | C's kernel bodies/comments, existing tuning, safety checks and recorded datasets. The current input-contract test already confirms bitwise A/C base-input agreement. |
| Shared integration work | Extend the existing root runner/collector with Part B, shared-oracle agreement, full validation/fingerprint coverage, aligned tuning and accepted cross-part comparison plots. | Per-part runners remain independently usable; current collection code lives in `scripts/run_all.py`. |

**Part A therefore has a small integration task, not a kernel rewrite.** Its runner currently validates selected representative performance cases; that is insufficient to mark the whole final grid as validated. The integration adapter or A's runner must close that gap. Existing A timing/error CSVs cannot substitute for executing the shared tests or rerunning A on the final machine.

Part A already supports `--mode dense|causal|both` for both the correctness suite and performance runs. This applies to its FP32 variants and its separate FP16 Tensor Core experiment. `--variants all` selects the four FP32 variants; use `--variants tensorcore` separately for FP16. The remaining A checklist is:

1. Keep the existing row-major GPU implementations and shared generator; export the CPU reference through a library target that other runners can link.
2. Add the agreed large-case sampled/output-only reference and check it against the existing full reference in both modes.
3. Validate every final-grid timing case, check all outputs for finiteness, and export validation scope/counts and original/effective input fingerprints.
4. Align its bounded tuning with the agreed experiment plan, freeze configurations, and rerun dense/causal FP32 plus the selected optional FP16 baseline on the common GPU.
5. Update its CSV metadata/adapter so the root collector can accept validated rows for final comparisons.

B/C must then call the shared reference with the appropriate mode and retain their own kernels. A complete all-parts run additionally needs B's implementation/adapter and the shared acceptance checks below; finishing A alone does not supply the missing Part B.

### Root runner and aggregate results

The available root entry point is:

```bash
# Inspect the complete A/C command sequence; creates no output and uses no GPU.
bash run_all.sh --parts a,c --arch 86 --output results/my_run --dry-run

# Execute the existing A/C workflows on this machine and collect their results.
bash run_all.sh --parts a,c --arch 86 --output results/my_run
```

Choose a fresh output directory. `--build-root PATH` overrides `/tmp/flash-attention-all-build`; `--arch` applies to both builds (SM80 or newer for Part C PTX MMA). With no `--parts`, the runner requests all three and currently stops before any build/run because Part B is missing. It never silently skips a requested part. Part B's future adapter must accept `bash part_b/run.sh --output DIR --build DIR --arch ARCH`, emit the normalized CSV tables below with `part=part_b`, dense mode and unique case IDs, record `gpu=<CUDA device name>` in `DIR/environment.txt`, and put its figures under `DIR/plots/`.

The current collector recomputes timing statistics from raw repetitions and writes `aggregation_audit.json`. A's unchecked benchmark rows remain `UNCHECKED`; absent fingerprints remain blank. The run manifest explicitly records `final_integration_ready=false`, and `plots/comparison/` explains why cross-part speedup plots are withheld. A's existing tuning uses N256/1024,d64 in both modes; its optional FP16 baseline remains fixed rather than independently tuned. These are current limitations for the integration agent to close, not completed final-experiment claims.

The final acceptance workflow remains:

1. Build all requested parts for the same GPU architecture and run shared-input/reference agreement tests plus each part's correctness and safety gates.
2. Tune the documented bounded candidates on the Section 3 tuning subset; save the selected configurations before running the final grid.
3. Run A in dense and causal modes, B in dense mode, and C in causal mode on the Section 3 main grid. Execute GPU work sequentially on one machine/GPU, rotating implementation order across cases or repeated runs where supported.
4. Preserve per-part raw files and normalize correctness, timing, resource and memory records. Do not merge data from the historical GPUs into this run. Failed/unsupported records remain explicit and do not enter speedup plots.
5. Validate completeness, precision/mode compatibility, input fingerprints, sample counts and configuration provenance before calculating speedups and generating plots. Capture profiler measurements separately from benchmark timing.

```text
results/<run_id>/
  manifest.json                 # common plan, GPU/toolchain, source/binary hashes
  correctness.csv               # numerical checks and validation scope
  samples.csv                   # individual unprofiled forward timings
  summary.csv                   # median, Q1/Q3/IQR and throughput
  resources.csv                 # launch settings and kernel resources
  memory.csv                    # allocation measurements; traffic kept separate
  configs/                      # tuning observations and frozen selections
  raw/part_a/                   # original per-part CSVs and supporting metadata
  raw/part_b/
  raw/part_c/
  plots/part_a/
  plots/part_b/
  plots/part_c/
  plots/comparison/             # same-run cross-part comparisons
```

Every normalized row needs a run/case ID, part, kernel/config ID, mode, B/H/N/d, pattern, input/output/accumulator precision, generator/seed and original/effective-input fingerprints. Correctness rows add oracle input precision, tolerances, maximum absolute/relative error, checked elements, nonfinite/failure counts and PASS/FAIL/UNSUPPORTED status. Sample rows add repeat and forward_ms and link to an accepted correctness record. Preserve unavailable metrics as missing. Keep stage timings separate from independently measured full-forward timings.

Acceptance requires bitwise input agreement, agreement of the sampled/full CPU references, small/full and large/sampled output checks, frozen tuning, and aggregate tables/plots regenerated from raw CSVs. Required FP32 main-grid cases must pass; unsupported cases expose missing coverage and do not count as completed requirements. Report A-dense versus B-dense and A-causal versus C-causal as matching-mode comparisons; label B-dense versus C-causal as different workloads and FP32 versus FP16 as a precision/performance trade-off. Private planning and scratch work belongs in ignored `extra/`; required result evidence belongs in the run directory above.

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
