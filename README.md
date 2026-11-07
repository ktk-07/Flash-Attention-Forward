# CEG5206 Team Project — FlashAttention Forward

**Documentation:** This `README.md` describes the project, repository structure, implementation conventions and evaluation status. For step-by-step instructions to reproduce the available runs, use [README.txt](README.txt). Section 6 documents the implemented A/C workflow and the exact Part B integration contract.

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

**Integration owner: Kyaw Tun Khine.** The A/C integration below updates shared helpers and host runners; it does not require Lyu Sihan to rewrite or reintegrate Part A. Zhang Yichun supplies Part B through the documented adapter contract.

Each member contributes testing, tuning, profiling and analysis. Record actual contributions in the final report. AI assistance must be declared as required by the briefing; Part A and this Part C evaluation received Codex assistance and require team review.

## Repository layout

```text
README.md                         # project overview, conventions and result index
README.txt                        # executable reproduction instructions
run.sh                            # shared A/C pipeline; defaults to A/B/C, B pending
run_all.sh                        # compatibility alias for run.sh
scripts/run_all.py                # command logging, CSV collection and audit
part_a/
  common/                         # layouts, deterministic inputs and FP64 oracle
  gpu_baseline/                   # materialised CUDA implementations
  cpu_reference/                  # independent FP64 oracle
  run.sh                          # standalone canonical evaluation and analysis
  scripts/                        # host workflow and canonical export
  main.cpp                        # independent benchmark/test CLI
  results/                        # saved RTX 5060 Ti measurements
  README.md
part_c/
  include/flash_attention/        # public declarations and shared types
  src/                            # FA1, FA2 CUDA-core and Tensor Core definitions
  tests/                          # runners, shared support and input-contract check
  scripts/                        # tuning, plotting, auditing and profiling helpers
  run.sh                          # standalone canonical evaluation; legacy paths retained
  evaluate.sh                     # bounded-study build/run wrapper
  profile.sh                      # default or frozen-winner Nsight workflow
  requirements-analysis.txt       # pinned plotting dependencies
  results/                        # retained raw data, metadata, figures and reports
  README.md
  PROFILING.md
```

Part B is absent. Root and standalone A/C runners now share the generator, FP64 reference library, fixed CSV schema and validation gates. An all-parts request fails before creating a run until `part_b/run.sh` exists. Use an explicit A/C subset while B is being developed. Kernels and native evidence stay in separate directories.

## 1. Software and platform requirements

The documented Part C workflow uses Linux, Bash, a CUDA-capable NVIDIA GPU, a C++17 compiler, CMake and Python. The tested GPU is an RTX 3070 Ti Laptop with SM86. The named runners detect the selected GPU architecture by default; `--arch` overrides it. The integrated Tensor Core paths require SM80 or newer. Part A's saved results were collected separately on an RTX 5060 Ti.

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

CPU-only integration/export regression checks (no CUDA access required):

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s part_c/tests -p 'test_*.py' -v
cmake -S part_a -B /tmp/attention-cpu-check -DPART_A_ENABLE_CUDA=OFF
cmake --build /tmp/attention-cpu-check --target attention_reference_test --parallel 2
/tmp/attention-cpu-check/cpu_reference/attention_reference_test
```

## 5. Run the available integration

Use fresh output directories. All GPU work runs sequentially on one selected device. Full runs include correctness, sanitizers, bounded tuning, the final grid, memory/resource evidence, fresh Nsight captures, standardized tables and plots.

```bash
# Check commands without creating files or accessing the GPU.
bash run.sh --parts a,c --arch 86 --output results/ac_preview --dry-run

# Small integration check, including correctness and sanitizers; no profiling.
bash run.sh --parts a,c --grid smoke --output results/ac_smoke

# Full A/C evaluation with fresh profiling.
bash run.sh --parts a,c --output results/ac_full

# Same canonical workflow, independently.
bash part_a/run.sh --output results/a_only
bash part_c/run.sh --output results/c_only

# Regenerate the combined analysis from saved child tables, without CUDA.
bash run.sh --analyze-only results/ac_full
# Re-extract one part's native data and regenerate its figures.
bash part_a/run.sh --analyze-only results/a_only
bash part_c/run.sh --analyze-only results/c_only

# After Part B implements Section 6, examiner's complete run:
bash run.sh --output results/final_submission
```

`run_all.sh` forwards to the same root runner. Options: `--device INDEX_OR_UUID`, `--arch SM` (default `auto`), `--build-root PATH` (root) / `--build PATH` (individual), `--grid full|smoke`, `--no-profile`, `--dry-run`. `--no-profile` explicitly omits fresh hardware counters and is recorded as incomplete profiling evidence. Smoke uses B=H=1, d=64, reporting N=33/65 and tuning N=17/33; it is not report data. Full uses the common grid in Section 3.

Part C's older positional, `--study` and `--complete` workflows remain supported; see [part_c/README.md](part_c/README.md). Keep their native outputs for additional studies. Do not mix their historical measurements into a new common-machine run.

## 6. Implemented integration and Part B handoff

The integration owner has connected A and C. GPU kernels remain in their own directories and separate executables. Neither part runs another part's GPU executable for correctness. Both link the reusable `attention_reference` CMake target from [part_a/cpu_reference/CMakeLists.txt](part_a/cpu_reference/CMakeLists.txt).

### Checked completion checklist

Checked against this checkout on 8 November 2026. The earlier list describing the shared oracle, root runner and combined plotting as pending is superseded by the implementation below. Integration is owned by Kyaw Tun Khine; no additional Part A integration task is assigned to Lyu Sihan.

| Requested work | Current status | Implementation / verification |
|---|---|---|
| 1. Part A shared interfaces | Implemented | [`attention_reference` library](part_a/cpu_reference/CMakeLists.txt), [output-only full/sampled oracle](part_a/cpu_reference/sampled_reference.cpp), [agreement tests](part_a/tests/reference_test.cpp), per-case O validation/fingerprints in [main.cpp](part_a/main.cpp), and [canonical export](part_a/scripts/export_results.py). |
| 2. Part B | Pending; deliberately not implemented here | Dense kernels, correctness, tuning and standalone adapter must follow the Part B contract below. No placeholder B runner is supplied. |
| 3. Part C shared correctness linkage | Implemented | [Active test support](part_c/tests/test_support.cpp) calls the shared generator, pattern transformations, fingerprint and FP64 output reference; all active runners link the library through [CMake](part_c/CMakeLists.txt). Original kernel bodies are unchanged. Historical scratch/legacy tests retain their provenance and are not the shared comparison path. |
| 4. Root experiment pipeline | Implemented for available parts | [run.sh](run.sh) invokes [scripts/run_all.py](scripts/run_all.py); standalone A/C workflows perform tests, sanitizers, tuning, timing, memory/resources, optional fresh profiling and export. Aggregation validates the requested grid and tables. Individual parts and CPU-only analysis are supported. [run_all.sh](run_all.sh) is now an alias, not a separate prototype. |
| 5. Combined plotting and analysis | Implemented | [plot_combined.py](scripts/plot_combined.py) generates applicable standardized comparisons and preserves individual figures. A/B and dense-B/causal-C paths are tested with temporary synthetic fixtures only; actual B measurements remain unavailable. |
| 6. Final common-machine experiment | Pending | After B passes its standalone and joint smoke checks, execute the complete full-grid A/B/C workflow with profiling. Review its actual outputs before writing final performance conclusions. |

**Verification scope:** the retained A/C GPU smoke run passed correctness, representative sanitizers, bounded tuning, timing, export and aggregation: 140 timing cases / 2,800 samples. It explicitly records `grid=smoke`, `all_three_parts_present=false` and no fresh profiling grid. CPU-only regeneration from the moved bundle reproduced 40 PNG/PDF figures byte for byte. The CPU regression suites cover schema rejection, fingerprints/GPU identity, missing B, offline exports and the future B plotting contract. A separate small real Part A Nsight capture verified QK/softmax/PV capture ordering and launch geometry.

**Still unverified as a complete experiment:** a fresh integrated full-grid run including all requested profiler captures, and any real B execution. Existing historical profiler records and the small capture do not substitute for that run. Private smoke/debug evidence stays in ignored `extra/integration_validation/`; final report evidence belongs under a new public `results/<run_id>/` bundle.


### Shared input and correctness contract

Use [input_generation.hpp](part_a/common/input_generation.hpp) directly: `attention::synthetic_input(shape, 5206)`, `attention::apply_pattern`, and `attention::input_fingerprint`. Preserve the exact SplitMix64 stream, filling Q then K then V. Layout and strides are defined in Section 3. Fingerprints are FNV-1a-64 over canonical little-endian FP32 bytes of Q/K/V, serialized as 16 lowercase hexadecimal characters. FP16 paths round once outside timing and fingerprint those rounded values represented as floats as well.

The shared FP64 oracle is authoritative, rather than another kernel or a timing CSV:

```text
identical generated Q/K/V ── shared CPU reference ── expected O
                        └── selected GPU kernel ── actual O
                                   element-wise comparison outside timing
```

`attention::reference(input, causal)` returns full S/P/O and is retained for small materialised tests. `attention::reference_output(input, causal, sampled)` computes O without allocating quadratic S/P. Its full and sampled modes are checked against the original reference in both dense and causal modes; CPU tests include future-K/V isolation, tails and deterministic patterns. Large GPU cases check query rows 0, floor(N/2), N−1 in every head and check every output for finiteness. Record full versus sampled scope and checked elements. Sampling is not exhaustive large-output validation.

A checks S/P/O for its small suite. C checks full O for its small suite and every tuning candidate, including boundary and stress patterns. Every accepted timing row has a passing O record for the exact implementation, configuration and effective inputs. FP32 tolerances are atol=rtol=1e-4; the existing FP16 experiment uses atol=0.002, rtol=0.02. Do not widen these automatically to hide failures. GPU-to-GPU comparisons may be added as supplementary checks; CPU-reference comparisons establish correctness.

### Part B: instructions for Zhang Yichun and the coding agent

Implement only B's dense fused kernels and its adapter. A/C integration and root aggregation are already maintained by Kyaw; no copying of A/C kernels or historical results is required.

1. Create `part_b/include/`, `part_b/src/`, `part_b/tests/`, `part_b/scripts/`, `part_b/CMakeLists.txt` and `part_b/run.sh`. Use normal declarations in headers and definitions in CUDA source files. Preserve useful standalone development tests under an appropriate `extra/` location when they are not submission evidence.
2. Link the shared oracle with `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../part_a/cpu_reference ${CMAKE_CURRENT_BINARY_DIR}/shared_reference)` and `target_link_libraries(YOUR_RUNNER PRIVATE attention_reference CUDA::cudart)`. Include the shared generator/header. Use `causal=false` for dense references; no causal mask is applied to B.
3. Accept the same CLI: `--output NEW_DIR --build DIR --arch auto|SM --device INDEX_OR_UUID --grid full|smoke --no-profile --dry-run --analyze-only BUNDLE`. No output directory overwriting. Select one device consistently via `CUDA_VISIBLE_DEVICES`; save its physical UUID using `scripts/workflow.py:device_info`. `--dry-run` must not create files or require GPU access. `--analyze-only` must not execute CUDA or require a build directory.
4. Use the shared inputs, row-major layout, patterns and FP64 checks above. Include N=1, tails around tile boundaries, multiple batches/heads, uniform attention, constant V and finite stress logits. For each accepted candidate/main case validate outside timing, checking all O values for finiteness and recording full/sample counts. Run Compute Sanitizer memcheck/racecheck/synccheck/initcheck on representative kernels and frozen winners. Keep failing candidates visible and stop if required FP32 coverage cannot pass.
5. Match the full grid B4,H8,N256/512/1024/2048/4096,d32/64/128, mode dense. Smoke is B1,H1,N33/65,d64. Tune on N256/1024 (smoke N17/33), per d, using a small documented candidate list. Freeze the choice before main measurements using geometric mean forward latency and deterministic tie-breaking. Publish one frozen winner as `variant=selected` for FP32 main rows, retaining its actual kernel name, symbol and launch configuration in `configs/part_b.json` and the manifest. Other variants can also be emitted with distinct names.
6. Take five warmups and 20 CUDA-event forward samples per accepted case; repeat indices 0–19. Exclude allocation, copies, CPU checks, resource queries, file writing and profilers. Fused QK/softmax/PV breakdown cells stay blank. Emit explicit live allocation and auxiliary-memory ledgers, analytical dense valid-pair count B×H×N², resource queries and fresh representative N256/4096,d64 Nsight captures (three repetitions), separately from timing. Import raw Nsight CSV with `--print-units base`; preserve every exported metric name/unit/value. Retain unavailable counters with blank values and reasons.
7. Export exactly the seven headers in [scripts/results_schema.py](scripts/results_schema.py), using its `case_id`, `write` and `validate`. Do not invent a second public schema. Native B-specific files and additional experiments stay under `raw/part_b/` and `supplementary/part_b/`. Write `configs/part_b.json`, `audit.json`, `manifest.json`, command logs and per-part plots. Required manifest fields: `schema_version=1`, `part=part_b`, `grid`, `shared_cpu_reference_linked=true`, `device` containing physical `uuid`, `profile_requested`, source/binary hashes and a registry describing numerical precision and selected kernels. The audit must set `passed=true` only after validation and record `profile_captures` (at least six when profiling is requested: two lengths × three captures). Never claim shared linkage unless the compiled runner actually links it.
8. Test `bash part_b/run.sh --grid smoke --output /tmp/b_smoke`, CPU-only analyze mode, and `bash run.sh --parts a,b,c --grid smoke --output /tmp/abc_smoke`. Root already discovers `part_b/run.sh`, requires dense FP32 grid coverage, validates the schema/input/GPU identity and plots `variant=selected` against A-dense. Fix B's adapter rather than changing shared semantics to make an invalid run pass. After acceptance, run the final full experiment on the same machine for all three parts.

Part B may add FP16/Tensor Core experiments, but FP32 dense coverage is required. Preserve precision labels and unsupported shapes; do not compare B-dense output with C-causal output as equal operations. Their workload/traffic comparison is a separate study.

#### B adapter details required by the existing collector

These conventions let B plug into the current root runner without rewriting A/C or the collector:

| Boundary | Exact convention |
|---|---|
| Build | Root configures `cmake -S part_b -B <build> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=<SM>` before measurements; standalone B must also configure/build itself. |
| Shared imports | Python helpers live in root `scripts/`: `results_schema.py`, `workflow.py`, `ncu_csv.py` and `plot_combined.py`. Resolve paths from the script location, not the caller's working directory. |
| Common labels | `part=part_b`, `mode=dense`, `layout=BHNd_contiguous_row_major`, `generator=splitmix64-fp32-v1`, `seed=5206`; FP32 input/accumulator/output labels are `FP32`. |
| Frozen main series | `experiment=proposal`, `variant=selected`; exactly one selected FP32 row per required shape. A configuration selected for each d is reused across that d's reporting lengths. |
| Other experiments | Use `experiment=tuning` for candidates; retain all additional variants/experiments under distinct case identities and in native files. Never publish failed/unchecked timings as accepted summaries. |
| Identity | Represent every common field as its CSV string value (including dimensions and seed; blanks are empty strings), then set `case_id=results_schema.case_id(row)`. JSON integers in the identity would hash differently after CSV readback. All tables for that case share those common fields; stage/profile-repeat fields do not alter its identity. |
| Required CSVs | Always write all seven tables using `results_schema.HEADERS` / `write`, including a header when a table has no applicable rows. Empty profiling is allowed only when explicitly skipped; it is not complete profiling evidence. |
| Profiler evidence paths | `source_report` is relative to the standalone bundle, for example `raw/part_b/profiles/<case>/raw.csv`; the target must exist. Root prefixes the child-bundle location automatically. |
| Re-analysis | Rebuild canonical tables and figures from retained native records without changing their values, order or case identities. Root compares child CSV hashes before and after re-extraction. Preserve immutable source data and configuration freezes. |

Minimum metadata structure (replace illustrative values with actual measurements/provenance):

```json
{
  "schema_version": 1,
  "part": "part_b",
  "grid": "full",
  "shared_cpu_reference_linked": true,
  "device": {"uuid": "GPU-<actual-uuid>"},
  "profile_requested": true,
  "sources": {"part_b/src/<kernel>.cu": "<actual-sha256>"},
  "binary_sha256": "<actual-sha256>",
  "implementation_registry": {"selected": "<actual kernel, precision and selection policy>"}
}
```

`audit.json` must include `passed` as a JSON Boolean and `profile_captures` as an integer. A full profiling run must provide at least six real selected-kernel captures (N256/4096 × three repetitions); a smoke or `--no-profile` run sets `profile_requested=false` and records the omission. `configs/part_b.json` must retain the bounded candidate settings, tuning observations and frozen per-d selection. The seven CSVs are the machine interface; extra B-specific detail can be retained alongside them without changing their headers.

Acceptance order for B: standalone smoke → standalone CPU-only re-analysis → joint A/B/C smoke → complete full A/B/C run. The existing temporary-fixture test exercises the schema/plots only; it does not validate B's kernels.


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
  manifest.json, commands.jsonl, run_status.json
  correctness.csv, samples.csv, summary.csv
  resources.csv, profiles.csv, memory.csv, work.csv
  comparison.csv                 # matched cross-part speedups and contracts
  audit.json, aggregation_audit.json, ANALYSIS.md, figure_index.json
  configs/                       # per-part configuration registries and freezes
  parts/part_a/                  # full standalone bundle, including raw/part_a/
  parts/part_b/                  # created only when B participates
  parts/part_c/                  # full native study, safety checks, counters, insights
  plots/part_a/                  # main figures and their plotted CSV data
  plots/part_b/
  plots/part_c/
  plots/combined/                # applicable same-input cross-part comparisons
  supplementary/part_c/          # retained additional profiling and analysis
```

Each standalone bundle has the same seven canonical CSVs at its root. Keeping complete child bundles preserves extra results needed for the report. Private scratch/history stays in ignored `extra/`; numerical evidence does not.

[plot_combined.py](scripts/plot_combined.py) consumes canonical tables only. It generates per-part FP32/FP16 kernel progression with IQR, throughput/storage, tuning and materialised stage breakdowns, plus applicable A-causal/C-causal and A-dense/B-dense comparisons. Once B is present it also generates dense-B versus causal-C latency, analytical valid-pair counts and available measured DRAM traffic, explicitly labelling them as different workloads. C additionally regenerates its native CUDA progression, Tensor Core progression, frozen core-versus-Tensor-Core, controlled scalar/vector-load ablation, tuning, memory and profiler figures. Full numerical data and native profiler metrics remain available for further report analysis. Missing parts/counters are declared, not plotted as fabricated zeros. FP16 A/C comparisons explicitly label their different accumulation/output contracts.

Offline root analysis checks the recorded child table hashes, invokes each child’s native re-extraction/analysis, rechecks table hashes, aggregates, copies retained supplementary evidence and regenerates common figures. Individual analyze modes re-extract native data; C also regenerates native/supplementary plots. Large binary Nsight reports are optional for CSV-only figure reproduction and remain checksummed evidence for deeper inspection.

### Status and final collection

A/C software integration is implemented; Part B's implementation/adapter and the final full common-machine experiment remain outstanding. Smoke runs exercise the pipeline, not report performance. A requested A/C subset can pass its audit while `all_three_parts_present=false`. `--no-profile` leaves the absence of fresh counters explicit. Completing A/C does not complete the report, video or final submission.

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
