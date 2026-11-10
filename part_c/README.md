# Part C — Causal FlashAttention

Part C implements causal attention with online softmax, progressing from scalar CUDA kernels to cooperative warps, register tiling, vector loads, WMMA and PTX MMA. All kernels compute a complete forward pass without materialising the full score/probability matrices in device memory.

For shared setup and submission reproduction, see [../README.md](../README.md) and [../README.txt](../README.txt). A/C integration is implemented by Kyaw Tun Khine; Part B remains pending.

## Source layout

```text
part_c/
  include/flash_attention/       # shared types and public launch declarations
  src/
    flash_attention_1.cu        # FA1 Algorithm-1 K/V-outer FP32 shared-memory baseline
    flash_attention_1_tensor_core.cu # maintained WMMA FA1 baseline
    flash_attention_2_cuda_core.cu
    flash_attention_2_tensor_core.cu
  tests/                        # C++ runners, reference and regression checks
    kernel_bodies.json          # baseline hashes used by verification
  scripts/                      # native analysis plus Part A-format export
    export_results.py            # lossless standardized report adapter
    run_standard.py              # named standalone workflow
  CMakeLists.txt
  run.sh                        # regression, --study and --complete workflows
  evaluate.sh                   # bounded study / frozen-winner verification
  profile.sh                    # separate Nsight captures
  requirements-analysis.txt
  results/                      # measured datasets and generated figures
  PROFILING.md                  # profiler methodology and interpretation
```

Headers declare the interfaces; `.cu` files define kernels and dispatch. CMake builds `part_c_attention` and links the C++ runners. No kernel-definition headers are included by the runners. The optional original coding file remains locally under `src/extra/`; it is outside the maintained build and submission workflow.

## Kernels and numerical contract

Inputs are contiguous `[B,H,N,d]`, scaled by `1/sqrt(d)`. Query `i` attends to keys `j <= i`, including the diagonal. No dropout, backward pass, KV cache or offset causal alignment is implemented.

| ID | Implementation | Inputs / accumulation / output |
|---|---|---|
| `fa1` | One CTA/head; K/V outer, Q inner; shared Q/K/V/S/O and global m/l | FP32 / FP32 / FP32 |
| `fa1_wmma` | FA1 K/V-outer WMMA; normalised global O/m/l | FP16 / FP32 / FP32 |
| `scalar` | FA2-style scalar baseline | FP32 / FP32 / FP32 |
| `warp` | Cooperative warp per query | FP32 / FP32 / FP32 |
| `tiled` | Shared Q/K/V, register tiling; already skips future tiles | FP32 / FP32 / FP32 |
| `skip` | Historical scalar-load register-tiled variant | FP32 / FP32 / FP32 |
| `vector` | Aligned float4 Q/K/V loads with scalar fallback | FP32 / FP32 / FP32 |
| `wmma` | Fused WMMA QKᵀ and PV | FP16 / FP32 / FP32 |
| `ptx` | Fused explicit PTX MMA QKᵀ and PV | FP16 / FP32 / FP32 |

FA1/FA2 names identify educational implementations, not equivalence to official library releases. The maintained FA1 now follows the original Algorithm-1 schedule without Tensor Cores; older saved FA1 results refer to the previous query-parallel scalar kernel and must not be relabelled. `tiled` versus `skip` is not a clean causal-skipping ablation. WMMA/PTX also round probability operands to FP16 for PV. PTX packs operands manually; the maintained kernels do not implement `ldmatrix`, swizzling or double buffering.

Instantiated head dimensions are 7,16,32,33,40,64,65,96,128. PTX requires d>=16 divisible by 16; unsupported cases are explicit skips. WMMA pads internally. Tensor configurations require BQ=16×warps and BK divisible by 16.

## Standardized report output, with all native details retained

Part C now exports the agreed Part A-based CSV headers through the shared `scripts/results_schema.py` module. This changes report interfaces, not kernel algorithms. The native study, all Nsight/NCU counter exports, MIO/scoreboard samples, source/SASS insights, configuration history and existing plots are retained byte for byte under `raw/part_c/`.

```bash
# CPU-only conversion of the saved study, including older source/SASS evidence.
# Choose a fresh destination; this does not rerun GPU measurements.
bash part_c/run.sh --export part_c/results/bounded_evaluation \
  --output part_c/results/my_report \
  --supplementary-profile part_c/results/nsight_analysis

# Fresh standalone run: correctness, sanitizers, bounded study, frozen-winner
# profiling, native plots, standardized CSVs and extra profiler charts.
bash part_c/run.sh --output results/my_c_run --build /tmp/part_c-build --arch 86

# Preview that workflow without running CUDA or creating results.
bash part_c/run.sh --output results/my_c_run --arch 86 --dry-run

# Regenerate public tables and all exported plots from the copied raw data.
bash part_c/run.sh --analyze-only part_c/results/my_report
```

`--grid smoke` selects small reporting/tuning shapes and omits profiling; correctness and sanitizers still run. `--arch auto` detects the GPU architecture by default.

No-argument `bash part_c/run.sh` now selects the standardized standalone workflow. `--device` selects one physical index/UUID; by default it honors the first existing `CUDA_VISIBLE_DEVICES` entry. Full named runs require Nsight Compute and Compute Sanitizer. `--no-profile` is an explicitly labelled partial run without fresh profiler evidence, not fabricated zero counters. Existing positional, `--study` and `--complete` interfaces remain native-format compatibility workflows.

The published [saved-data export](results/standardized/ANALYSIS.md) is a format adaptation of the existing measurements, not a new GPU experiment. It contains:

- Fixed-header `correctness.csv`, `samples.csv`, `summary.csv`, `resources.csv`, `memory.csv`, `profiles.csv` and `work.csv`.
- Part A names such as `variant`, `forward_median_ms`, `forward_q1_ms`, `warmup` and `precision`, with explicit precision/fingerprint/validation metadata added.
- All 6,360 unprofiled samples and 318 timing cases; the 60 frozen per-variant main cases use `experiment=proposal`. Default progression, tuning and matched-load experiments remain distinctly labelled.
- All raw metrics from the twelve matching frozen-winner captures, plus derived MIO/short/long-scoreboard percentages. Instrumented durations remain profiler metrics and never enter forward samples.
- Seven original study figures in `plots/part_c/`, plus six extra profiler figure pairs under `supplementary/part_c/plots/` and an evidence index.
- The complete native study and optional historical profiling datasets, including source/SASS, reports, logs, CSVs, snapshots and all supplied `.ncu-rep` files. Historical counters are supplementary and are not relabelled as counters for tuned configs or the common input stream.

Fused QK/softmax/PV timings and unmeasured setup time stay blank. Raw missing Nsight counters stay explicitly unavailable. Per-thread resource estimates are distinct from measured occupancy, source shared excess is distinct from aggregate arbitration counts, and warp-state percentages are not runtime fractions. Shared-memory/local traffic is not allocated storage.

The exporter verifies native case hashes and metadata, fingerprints, sample/summary statistics, O-validation joins, profiler CSV hashes, supplied binary report hashes and byte-identical evidence copies. `.ncu-rep` files may be supplied separately; when absent, saved CSV evidence still exports and the missing binaries/expected checksums are explicit. New runs link the shared Part A CPU reference directly. Historical exports retain their original oracle provenance. Part B and the final common-machine A/B/C experiment remain pending.

CPU-only adapter regression tests:

```bash
python3 -m unittest discover -s part_c/tests -p 'test_export_results.py' -v
```

## Both FA1 baselines in fresh standard runs

The normal runner checks all nine kernels (428 passes and four unsupported PTX cases), then runs dedicated FP32 and WMMA FA1 suites (236 cases each). Both suites check full output and final m/l state. Frozen verification runs four Compute Sanitizer tools on both FA1 baselines and both selected FA2 family winners (16 checks). FA2 tuning lists are unchanged; FA1 baselines use fixed config 0 and are not selected using reporting timings.

Both FA1s appear as `experiment=proposal` beside the frozen FA2 variants. Their exact memory measurements include the two FP32 B×H×N state arrays. New full profiling captures these two baselines plus the two frozen d=64 FA2 family winners, at N256/4096 with three repetitions (24 captures). Standard smoke runs omit profiling.

Fresh standalone and combined bundles add:

- `fa1_fa2_comparison.csv`: FP32 FA1 versus the frozen CUDA-core family winner, and FP16 FA1 versus each frozen WMMA/PTX variant.
- `fa1_cross_precision.csv`: FP16 FA1 versus FP32 FA2, explicitly labelled as different precision.
- `plots/part_c/fa1_fa2_comparison.{png,pdf}` and `fa1_cross_precision.{png,pdf}`, plus an index linking exact case IDs and configurations.

Ratios use unprofiled medians after input-fingerprint checks. Tensor Core progression now includes FA1 WMMA; native CUDA progression includes FP32 FA1. Historical eight-kernel bundles retain their original coverage and are not filled with invented FA1 WMMA measurements. Root A/C comparisons continue to use the frozen FA2 family winners.

## Native-format compatibility workflows

Use Linux/Bash, CMake >=3.18, C++17, CUDA, Python and the pinned analysis packages. The recorded machine uses an RTX 3070 Ti Laptop (SM86), CUDA 12.6 and Nsight Compute 2024.3. Run commands from the repository root with a fresh output directory.

```bash
python3 -m pip install -r part_c/requirements-analysis.txt

# Full Part C workflow: default regression/benchmark/sanitizers/legacy checks,
# then candidate validation, tuning, frozen evaluation, memory and figures.
bash part_c/run.sh --complete part_c/results/my_complete_run /tmp/part_c-evaluation-build

# Bounded study only:
bash part_c/run.sh --study part_c/results/my_bounded_study

# Default regression/benchmark workflow only:
bash part_c/run.sh part_c/results/my_regression /tmp/part_c-build
```

`--complete` places default-workflow outputs in `regression/` and the bounded study in `main/`. GPU cases run sequentially. It does not execute Parts A/B. `PART_C_BUILD_DIR` overrides the study build directory; `PART_C_CUDA_ARCHITECTURES` overrides its default SM86 architecture. The older default workflow explicitly targets SM86.

Individual commands:

```bash
cmake -S part_c -B /tmp/part_c-evaluation-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/part_c-evaluation-build --parallel 2
/tmp/part_c-evaluation-build/part_c_input_contract
/tmp/part_c-evaluation-build/part_c_runner \
  --suite correctness --output part_c/results/my_correctness
/tmp/part_c-evaluation-build/part_c_runner \
  --suite benchmark --grid full --output part_c/results/my_default_benchmark
```

The default runner accepts `--kernel fa1|scalar|warp|tiled|skip|vector|wmma|ptx` and `--grid representative` for a shorter benchmark.

## FA1 Algorithm-1 baseline (rewritten 9 November 2026)

The former `fa1` used online softmax but had no shared-memory staging and one CTA per query tile. It was not the original FA1 IO schedule. The maintained implementation now follows the original K/V-outer, Q-inner schedule:

1. One CTA owns a complete batch/head, cooperatively loading a K/V tile once.
2. It iterates over the applicable query tiles, staging Q and the previous normalised O in shared memory.
3. One CUDA thread per query row forms a shared score tile, computes its stable softmax and merges the previous maximum/normaliser/output.
4. It writes O and the row statistics back to global memory before processing the next K/V tile. Entirely future tile pairs are skipped and diagonal entries are masked in causal mode.

This is a readable FP32 CUDA-core implementation of FA1's algorithmic schedule, not a performance replica of the official implementation. It does not call WMMA, MMA or CUTLASS. Both dense and causal modes are exposed by the FA1 API; normal Part C evaluation remains causal. Barriers are collective even for tails; every forward initialises its own O/m/l state in the first K/V iteration, so no reset launch or host memset is hidden outside timing.

The caller preallocates two FP32 arrays of B×H×N elements through `FlashAttention1Workspace` and keeps them alive until the selected stream finishes. Concurrent forwards need disjoint output/state buffers. The wrapper never allocates in the timed path. Live allocation is **16×B×H×N×d + 8×B×H×N bytes**, including Q/K/V/O and m/l. With BQ=BK=32, shared Q/K/V/S/O storage is `4*(2*BQ*d + 2*BK*d + BQ*BK)` bytes: 69,632 bytes at d=128. First-use setup opts into larger dynamic shared memory when needed; insufficient hardware capacity fails explicitly.

A single CTA/head intentionally exposes FA1's reduced parallelism and per-tile O/state traffic relative to the query-resident FA2-style kernels. The new kernel uses coalesced cooperative staging, but simple thread-per-row arithmetic is not a tuned GEMM microkernel. FA1-to-FA2 progression changes scheduling, state placement and compute mapping; it is not a single-feature ablation.

After the normal CMake build:

```bash
/tmp/part_c-build/part_c_fa1_test --output /tmp/fa1_full_checks
# A short set suitable for each Compute Sanitizer tool:
compute-sanitizer --tool racecheck --error-exitcode 3 \
  /tmp/part_c-build/part_c_fa1_test --smoke --output /tmp/fa1_racecheck
```

Substitute the actual build path. The dedicated tests compare dense/causal full O and final m/l against FP64; exercise all supported head dimensions, partial tiles, multiple heads, poisoned initial memory, repeat calls, a nondefault stream and future-K/V isolation. The standardized runner executes this suite automatically. Use a fresh output path per command.

New study/profiler plans record `fa1_implementation=kv_outer_shared_fp32_v2`. The exporter gives this FA1 a distinct configuration identity, includes workspace bytes and the one-CTA/head resource grid, and retains dynamic shared-memory queries. Historical bundles have no such flag and keep their old resource/allocation interpretation. Their saved samples and source snapshots are unchanged; regenerate fresh measurements for new FA1 ratios. FA2 CUDA-core and Tensor Core kernel bodies are unchanged by this rewrite.

## FA1 Tensor Core baseline and targeted pilot

Keep the FP32 FA1 baseline and compare FP32 FA1/FA2 separately from mixed precision. `fa1_wmma` is the maintained Tensor Core FA1 baseline: K/V outer and Q inner, one CTA/head, BQ=BK=32, two warps, WMMA for both products. Q/K/V and probability operands are FP16; accumulation, scaling, softmax, row state and O are FP32. Normalised O/m/l are exchanged through global memory between K/V blocks, unlike the query-resident FA2 variants. Workspace is caller-owned and each forward starts fresh internally. It is **not** the official FA1 library or a demonstrated performance-parity implementation.

Both FA1 baselines run in the normal nine-kernel workflow. The targeted pilot remains available for a smaller fixed-configuration comparison:

```bash
bash part_c/run.sh --fa1-pilot --output results/my_fa1_pilot
# Print the plan without touching the GPU or creating files.
bash part_c/run.sh --fa1-pilot --arch 86 --output results/preview_fa1 --dry-run
# Revalidate saved native records and regenerate figures without CUDA.
bash part_c/run.sh --fa1-pilot --analyze-only results/my_fa1_pilot
```

The default pilot uses B4/H8/d64, N256/1024 and config 0 for every implementation. It first runs FP32 and Tensor Core FA1 full output/state suites, the default all-kernel regression, four focused Tensor Core FA1 sanitizer checks and SASS verification. Then it takes five warmups/20 unprofiled samples per case, rotating kernel order between shapes. No winner is chosen on the reporting grid and no unsuccessful speedup case is discarded. `--grid full` is an explicit later expansion; default is the requested small run. `--arch`, `--device` and `--build` work as in the other runners.

- `comparison.csv`: FP32 `fa1` versus `skip`; FP16 `fa1_wmma` versus `wmma` and `ptx`. The matched FP16 contract includes FP32 accumulators/softmax/O, but normalisation/rounding history and WMMA/PTX fragment mappings can differ.
- `precision_tradeoffs.csv`: FP32 FA1 versus WMMA FA1, explicitly a hardware-plus-precision comparison, **not** an FA2 speedup.
- Seven standard CSVs at the output root; `experiment=fa1_pilot` keeps these cases out of the core final-grid series.
- `plots/part_c/{fa1_fa2_speedup,pilot_latency}.{png,pdf}`, with numeric inputs in the CSVs, plus `ANALYSIS.md`, `manifest.json`, `audit.json` and raw records/source snapshots in `raw/part_c/fa1_pilot/`.

Both `fa1` and `fa1_wmma` are included in default regression, main timing, explicit memory measurements, profiling and exports. Either can also be selected explicitly with `part_c_runner --kernel fa1_wmma` or `part_c_evaluation fa1_wmma 0 ...`. The dedicated executable accepts `--tensorcore`; CTest registers its invariant suite. Existing FA1/FA2 kernels, earlier saved results and the fixed main comparison contract are preserved.

The first [recorded small pilot](../results/fa1_pilot_20261110/ANALYSIS.md) ran on the RTX 3070 Ti Laptop GPU (SM86), with B4/H8/d64 and config 0. Speedup means baseline median divided by candidate median:

| Comparison | N=256 | N=1024 |
|---|---:|---:|
| FP32 FA1 / FP32 FA2-style `skip` | 3.60× | 4.28× |
| FA1 WMMA / FA2-style WMMA | 2.21× | 2.88× |
| FA1 WMMA / FA2-style PTX | 4.26× | 7.87× |

These measurements motivated including both baselines in fresh standard runs. The results are two-shape, fixed-default observations, not full-grid tuning conclusions. Median/IQR values, 200 raw timing samples, fingerprints, numerical checks, source hashes and PNG/PDF figures are retained in that bundle. The FP32-to-Tensor-Core FA1 improvement is stored separately and is not attributed to FA2.

The official FA1 v1.0.9 release supports FP16/BF16 and has different implementation/measurement details. It was not benchmarked by this pilot. An actual official comparison would require a pinned compatible installation and separately validated inputs/timing boundaries; our own kernel must not be relabelled as official performance.

## Validation and tuning

An independent CPU FP64 reference computes stable causal softmax. Small cases check every output, including tails and random/uniform/constant-V/rising-logit/stress inputs. Large benchmark cases check query rows 0,N/2,N−1 in every head and finiteness of all outputs. Outputs are prefilled with NaNs to catch missing stores.

FP32 uses atol=rtol=1e-4. Tensor Core paths use atol=0.002, rtol=0.02 against the FP64 oracle on the same FP16-rounded inputs. Original FP32 and rounded input fingerprints are saved. Relative error divides by max(abs(reference),1e-12).

The new study uses Part A's SplitMix64 sequence, seed 5206, filling Q then K then V. `part_c_input_contract` verifies bitwise equality with Part A. Historical saved regression/profile inputs retain the older mt19937 stream. Newly executed regression and study runners both use the shared generator and FP64 oracle library.

The bounded search tests five configurations for each of `skip/vector` and four for each of `wmma/ptx`. All candidates pass small-case checks before tuning at B=4,H=8,N=256/1024,d=32/64/128. Per-variant/per-d configurations and family selections are saved before the main grid. Family selection covers skip/vector and WMMA/PTX, not an exhaustive search of all kernels. N=512/2048/4096 is held out from selection.

Timing uses five warmups and 20 CUDA-event repetitions. Allocation, copies, rounding, reference checks, resource queries and file writes stay outside timing. CSVs report median, Q1, Q3, IQR and throughput. Fixed sequential ordering and unmanaged clocks limit conclusions about close winners.

## Results and figures

Start with [results/README.md](results/README.md) and the [bounded-study report](results/bounded_evaluation/REPORT.md).

The historical eight-kernel study passed 288 full candidate checks, 318 sampled benchmark cases (6,360 samples), and 30 sampled memory cases. Additional verification passed 380 default checks with four explicit unsupported cases, plus eight representative sanitizer runs. Twelve fresh Nsight captures cover the frozen d=64 winners at N=256/4096.

| d | Frozen CUDA core | CUDA ms at N4096 | Frozen PTX ms | Core / tensor |
|---|---|---|---|---|
| 32 | `skip`, config 2 | 65.716 | 5.042 | 13.03× |
| 64 | `skip`, config 4 | 107.663 | 12.153 | 8.86× |
| 128 | `vector`, config 4 | 150.936 | 36.106 | 4.18× |

These compare FP32 CUDA-core and FP16-input Tensor Core paths. They are not precision-isolated hardware speedups. Tuning improved the selected core kernels over their own defaults by 1.55×/1.23×/1.36× and PTX by 1.10×/1.16×/1.52× at N4096 for d32/64/128.

Seven PNG/PDF figures cover CUDA progression, Tensor Core progression, frozen family comparison, default-versus-tuned gains, tuning heatmap, matched scalar/vector loading, and device storage. The matched-load experiment uses identical launch parameters for `skip/vector`; other progression comparisons may combine changes.

The frozen FA2-family memory cases allocate Q/K/V/O: 16 bytes/element for FP32, 10 for FP16 inputs with FP32 output. The rewritten FA1 additionally allocates 8×B×H×N bytes for m/l; its benchmark allocation ledger and canonical memory records include them. The ledger measures their exact explicit live peak; compiler/runtime allocations are outside it. Three `cudaMemGetInfo` observations are separately labelled, not a continuous total-process peak. The materialised S+P curve is an analytical model, not measured Part A storage. Profiler traffic is a separate metric.

## Additional safety checks and profiling

After the bounded study, run these sequentially:

```bash
bash part_c/evaluate.sh --verify part_c/results/my_bounded_study
bash part_c/profile.sh --frozen part_c/results/my_bounded_study
```

Both default to `/tmp/part_c-evaluation-build`; the profiler also accepts `--build PATH`. Verification runs four sanitizer tools on each frozen d=64 winner at B1,H1,N33; coverage is representative, not exhaustive. See [PROFILING.md](PROFILING.md) for the independent default-kernel capture workflow and metric definitions.

## Regenerate saved figures and audit results

```bash
python3 part_c/scripts/evaluate.py --build /tmp/unused-build \
  --output part_c/results/bounded_evaluation --analyze-only
python3 part_c/scripts/audit_evaluation.py part_c/results/bounded_evaluation
python3 part_c/scripts/profile_frozen.py part_c/results/bounded_evaluation --analyze-only
```

No GPU is needed for saved-data processing. Full profile-integrity audits require the checksummed raw `.ncu-rep` files, supplied separately if excluded from Git. Plans, frozen configs, raw samples, command records, numerical checks and source/analysis manifests are retained with each dataset. The small `tests/kernel_bodies.json` manifest lets active audits verify the declared current kernel fingerprints (FA1 rewritten; FA2 bodies preserved) without depending on private development history.
