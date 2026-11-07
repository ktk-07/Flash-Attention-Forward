# Part C — Causal FlashAttention

Part C implements causal attention with online softmax, progressing from scalar CUDA kernels to cooperative warps, register tiling, vector loads, WMMA and PTX MMA. All kernels compute a complete forward pass without materialising the full score/probability matrices in device memory.

For shared setup and submission reproduction, see [../README.md](../README.md) and [../README.txt](../README.txt). A/C integration is implemented by Kyaw Tun Khine; Part B remains pending.

## Source layout

```text
part_c/
  include/flash_attention/       # shared types and public launch declarations
  src/
    flash_attention_1.cu        # educational FA1-style scalar baseline
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
| `fa1` | One thread per query, scalar online softmax | FP32 / FP32 / FP32 |
| `scalar` | FA2-style scalar baseline | FP32 / FP32 / FP32 |
| `warp` | Cooperative warp per query | FP32 / FP32 / FP32 |
| `tiled` | Shared Q/K/V, register tiling; already skips future tiles | FP32 / FP32 / FP32 |
| `skip` | Historical scalar-load register-tiled variant | FP32 / FP32 / FP32 |
| `vector` | Aligned float4 Q/K/V loads with scalar fallback | FP32 / FP32 / FP32 |
| `wmma` | Fused WMMA QKᵀ and PV | FP16 / FP32 / FP32 |
| `ptx` | Fused explicit PTX MMA QKᵀ and PV | FP16 / FP32 / FP32 |

FA1/FA2 names identify educational implementations, not equivalence to official library releases. `tiled` versus `skip` is not a clean causal-skipping ablation. WMMA/PTX also round probability operands to FP16 for PV. PTX packs operands manually; the maintained kernels do not implement `ldmatrix`, swizzling or double buffering.

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

## Validation and tuning

An independent CPU FP64 reference computes stable causal softmax. Small cases check every output, including tails and random/uniform/constant-V/rising-logit/stress inputs. Large benchmark cases check query rows 0,N/2,N−1 in every head and finiteness of all outputs. Outputs are prefilled with NaNs to catch missing stores.

FP32 uses atol=rtol=1e-4. Tensor Core paths use atol=0.002, rtol=0.02 against the FP64 oracle on the same FP16-rounded inputs. Original FP32 and rounded input fingerprints are saved. Relative error divides by max(abs(reference),1e-12).

The new study uses Part A's SplitMix64 sequence, seed 5206, filling Q then K then V. `part_c_input_contract` verifies bitwise equality with Part A. Historical saved regression/profile inputs retain the older mt19937 stream. Newly executed regression and study runners both use the shared generator and FP64 oracle library.

The bounded search tests five configurations for each of `skip/vector` and four for each of `wmma/ptx`. All candidates pass small-case checks before tuning at B=4,H=8,N=256/1024,d=32/64/128. Per-variant/per-d configurations and family selections are saved before the main grid. Family selection covers skip/vector and WMMA/PTX, not an exhaustive search of all kernels. N=512/2048/4096 is held out from selection.

Timing uses five warmups and 20 CUDA-event repetitions. Allocation, copies, rounding, reference checks, resource queries and file writes stay outside timing. CSVs report median, Q1, Q3, IQR and throughput. Fixed sequential ordering and unmanaged clocks limit conclusions about close winners.

## Results and figures

Start with [results/README.md](results/README.md) and the [bounded-study report](results/bounded_evaluation/REPORT.md).

The recorded study passed 288 full candidate checks, 318 sampled benchmark cases (6,360 samples), and 30 sampled memory cases. Additional verification passed 380 default checks with four explicit unsupported cases, plus eight representative sanitizer runs. Twelve fresh Nsight captures cover the frozen d=64 winners at N=256/4096.

| d | Frozen CUDA core | CUDA ms at N4096 | Frozen PTX ms | Core / tensor |
|---|---|---|---|---|
| 32 | `skip`, config 2 | 65.716 | 5.042 | 13.03× |
| 64 | `skip`, config 4 | 107.663 | 12.153 | 8.86× |
| 128 | `vector`, config 4 | 150.936 | 36.106 | 4.18× |

These compare FP32 CUDA-core and FP16-input Tensor Core paths. They are not precision-isolated hardware speedups. Tuning improved the selected core kernels over their own defaults by 1.55×/1.23×/1.36× and PTX by 1.10×/1.16×/1.52× at N4096 for d32/64/128.

Seven PNG/PDF figures cover CUDA progression, Tensor Core progression, frozen family comparison, default-versus-tuned gains, tuning heatmap, matched scalar/vector loading, and device storage. The matched-load experiment uses identical launch parameters for `skip/vector`; other progression comparisons may combine changes.

Memory cases allocate only Q/K/V/O: 16 bytes/element for FP32, 10 for FP16 inputs with FP32 output. The ledger measures their exact explicit live peak; compiler/runtime allocations are outside it. Three `cudaMemGetInfo` observations are separately labelled, not a continuous total-process peak. The materialised S+P curve is an analytical model, not measured Part A storage. Profiler traffic is a separate metric.

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

No GPU is needed for saved-data processing. Full profile-integrity audits require the checksummed raw `.ncu-rep` files, supplied separately if excluded from Git. Plans, frozen configs, raw samples, command records, numerical checks and source/analysis manifests are retained with each dataset. The small `tests/kernel_bodies.json` manifest lets active audits verify preserved kernel bodies without depending on private development history.
