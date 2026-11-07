# Flash-Attention-CUDA

CEG5206 Group 1 project: implement and evaluate exact attention forward kernels, progressing from materialised attention to fused dense and causal FlashAttention.

This README describes the project and the planned integrated evaluation. It does not claim that the shared harness or the final comparison results are complete. See [Part C](part_c/README.md) for the kernel inventory, progression figures, validation gates, and integration work remaining.

## Scope and repository

The core scope is FP32 materialised attention, optimised materialised kernels, fused dense attention, fused causal attention with tile skipping, and reproducible correctness/performance evaluation. Mixed precision and Tensor Cores are the selected optional extension. Existing libraries are labelled references, not substitutes for the team's kernels.

| Location | Purpose and current status |
|---|---|
| [Part A](part_a/README.md) | Independent C++ reference with FP64 accumulation; dense/causal materialised CUDA baselines; benchmark CLI and saved results. |
| Part B | Dense fused attention responsibility; no separate `part_b/` directory is currently present. Its final source/runner must be identified during integration. |
| [Part C](part_c/README.md) | Split causal CUDA-core/WMMA/PTX kernels; standalone build, tests, benchmarks, sanitizer checks and measured plots. Part A integration remains deferred. |
| [Integration](flash_attn_integration/) | PyTorch extension sources; not yet the agreed cross-part benchmark interface. |
| [Figure notes](FIGURES.MD) | Earlier figure suggestions. The experiment plan here and in Part C specifies the intended comparisons. |

Inputs and outputs are logically contiguous `[B,H,N,d]`. Causal self-attention allows key `j` for query `i` exactly when `j <= i`; scale is `1/sqrt(d)`. A kernel's actual stride arguments must be adapted explicitly: some Part C code uses `N` for head count and `H` for head dimension. Do not infer physical K transposition from an older parameter name.

## How Part A enters the comparison

Part A is the conventional attention baseline for both dense and causal results. Compare complete forward passes: its QKᵀ, softmax, and PV kernels together against the complete fused kernel.

| Experiment | Part A comparator | Fused comparator | Interpretation |
|---|---|---|---|
| FP32 causal | `optimized`, with `naive` as historical baseline | Validated Part C FP32 kernels | Main evidence for fusion and CUDA-core optimisation. |
| FP32 dense | `optimized`, with `naive` as historical baseline | Validated dense fused kernel | Separate core-scope experiment; causal results cannot replace it. |
| FP16 causal | `tensorcore` | Part C fused WMMA and PTX MMA | Materialised versus fused Tensor Core implementations, with accumulator/output differences reported. |
| Best implementation comparison | Part A FP32 `optimized` and FP16 `tensorcore` as context | Frozen best CUDA-core and Tensor Core versions | Practical performance/accuracy trade-off; not an isolated hardware speedup. |

Part A's current `tensorcore` implementation stores Q/K/V/S/P/O in FP16, uses FP16 WMMA accumulators, and performs half-precision softmax. Part C's fused WMMA/PTX implementations take FP16 Q/K/V and use FP32 accumulators, softmax state, and output. These are different numerical contracts. Report errors and the complete precision configuration; a strictly matched accumulation experiment would require an additional implementation and is not currently available.

Saved Part A results were collected on an RTX 5060 Ti. Rerun Part A and all fused variants on the same selected GPU before computing cross-part speedups. Historical timings may be cited with their original environment, but must not be mixed into a same-GPU comparison.

## Shared experiment protocol

1. Register each implementation with a unique ID, source file/symbol, commit, attention mode, input/output/accumulator precision, layout, supported shapes, and launch constraints. Exclude empty placeholders and duplicate copies.
2. Reuse Part A's deterministic input generator (`seed=5206`) and independent CPU oracle. A matching seed alone is insufficient if generators differ. For FP16 comparisons, round the same inputs once and give the oracle those rounded values.
3. Validate before accepting performance rows. Compare S/P/O for small materialised cases and O for fused cases. Store errors, tolerances, and pass/fail. Unsupported shapes are explicit skips, not passes.
4. Tune on a bounded subset, freeze configurations and best-version selection, then run the main grid. Do not select a new winner for every final plot point unless explicitly presenting a per-shape selection policy.
5. Preallocate buffers; use 5 warmups and 20 CUDA-event repetitions, reporting median, Q1, Q3, and IQR. Exclude generation, allocations, transfers, validation, and file I/O. Include required GPU preprocessing or report its reusable setup cost and assumptions separately. Time full forwards independently of stage breakdowns.
6. Record GPU/software/flags, launch configurations, seeds, source revision, and available clock/power information. Run without competing GPU workloads; rotate implementation order to reduce order effects.
7. Save raw samples before plotting. Measure allocated bytes separately from profiler DRAM traffic. Report sequences/s as `B / time_seconds`; tokens/s is optional.

| Grid | Values |
|---|---|
| Core performance | `B=4`, `H=8`, `N=256,512,1024,2048,4096`, `d=32,64,128`, FP32 |
| Attention modes | Dense and causal, compared with matching references |
| Tuning subset | `B=4,H=8,N=256/1024,d=64`; a small documented configuration list |
| Progression panels | `B=4,H=8,d=64,N=256/1024/4096` |
| Profiling | At least small `N=256` and large `N=4096`, with `B=4,H=8,d=64` |
| Tensor Core extension | Same supported performance shapes, FP16 inputs; precision and numerical errors recorded |

Part A's current CLI does not automatically validate every main-grid row. Its representative validations and blank error fields must not be interpreted as full-grid validation. The integrated harness must identify which rows were actually validated.

## Figures and tables

The three kernel plots are separate report figures, as requested. The detailed specification is in [Part C's figure plan](part_c/README.md#three-kernel-comparison-figures).

| Figure | Evidence |
|---|---|
| CUDA-core progression | Part A naive/optimised context plus Part C's successive FP32 causal kernels, median latency and IQR on representative workloads. |
| Tensor Core progression | Part A materialised WMMA plus distinct validated Part C fused Tensor Core stages; numerical-contract differences labelled. |
| Best CUDA-core versus best Tensor Core | Frozen best Part C versions, latency versus N in panels for d=32/64/128, with Part A optimised/WMMA context curves. |
| Full-project latency and speedup | Main-grid scaling, dense and causal panels, matching baseline for every speedup. |
| Peak allocated memory | Materialised versus fused, showing total live buffers and auxiliary allocation separately where possible. |
| Controlled ablation | One-feature comparisons, including causal tile skipping and vectorised loads. Progression alone does not establish individual causality. |
| Tile-size study | Bounded configuration heatmap/table with invalid configurations marked. |
| Causal work and runtime | Processed/skipped tiles versus latency; distinguish causal semantics from the effect of tile skipping. |
| Profiling and stage breakdown | Small/large bottlenecks and materialised QKᵀ/softmax/PV breakdown beside fused total time. |
| Numerical accuracy | Correctness table for all variants; error versus N for the mixed-precision extension. |
| Design diagram | Materialised global intermediates versus on-chip tiled online softmax and output accumulation. |

Combine related panels and move detailed tables into an appendix as needed. Retain measured regressions and explain them; a fixed speedup target is not required by the rubric.

## Reproduction and deliverables

Part A has an existing build/run interface in its [README](part_a/README.md). From the repository root, this is an example Linux build for an RTX 3090 (SM86); use the architecture of the actual evaluation GPU:

```bash
cmake -S part_a -B build/part_a -DPART_A_ENABLE_CUDA=ON -DPART_A_PROFILE_INFO=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build/part_a --parallel
./build/part_a/attention --grid correctness --output build/part_a/check-fp32
./build/part_a/attention --grid proposal --mode causal --variants main --tile 16 --softmax-threads 128 --output build/part_a/proposal-fp32
./build/part_a/attention --grid correctness --variants tensorcore --output build/part_a/check-fp16
./build/part_a/attention --grid proposal --mode causal --variants tensorcore --tile 16 --softmax-threads 128 --output build/part_a/proposal-fp16
```

These commands describe existing Part A functionality; they were not run for this documentation update. Each output directory must be new. A single cross-part reproduction command is still to be implemented. Part C now has its own CMake target and standalone reproduction command:

```bash
bash part_c/run.sh part_c/results/my_run /tmp/part_c-build
```

See [current Part C results](part_c/results/layout_validation/README.md) for the verified conventional source layout, or [the initial results and corrections](part_c/results/local_validation/RESULTS.md). Public launch declarations are in `part_c/include/flash_attention/*.h`; the three corrected implementations are `part_c/src/flash_attention_1.cu`, `flash_attention_2_cuda_core.cu`, and `flash_attention_2_tensor_core.cu`. CMake builds them as a library used by the C++ runner. The original coding file is retained with a minimal standalone test at `part_c/src/extra/causal_flash_attn.cu`; redundant development copies have been removed and the legacy test source moved into `part_c/tests/`. No Part A integration is performed by this command.

Planned integrated artifacts, separate from the historical `part_a/results/`, are `results/<run_id>/environment.json`, `manifest.json`, `configs.json`, `correctness.csv`, `samples.csv`, `summary.csv`, `tuning.csv`, `resources.csv`, `causal_work.csv`, profiler reports/metrics, plotting scripts, and exported PDF/PNG figures. See Part C for required fields and validation.

The final submission includes source/dependencies/build instructions, reproducible raw results, a report PDF, a 10–15 minute video, and a contribution record. The report must contain accessible clickable repository/video links and a final commit hash or release tag. Cite adapted code, papers, and figures, distinguish team contributions, and declare AI assistance. The briefing gives the deadline as **13 November 2026, 11:59 PM** and imposes no fixed report page limit; the proposal suggests 8–10 pages excluding references.

## Project documents

- [Project briefing](Project%20Briefing.pdf): submission requirements.
- [Proposal](FlashAttention_Proposal.pdf): implementation scope, experiment grid, and required evidence.
- [Rubric](Team%20Project%20Rubrics.pdf): correctness 20, optimisation depth 25, evaluation/profiling 25, analysis 15, report/video 15; up to 5 bonus marks for a substantial validated extension.

Core-scope completion takes priority over optional work. The rubric rewards supported explanations of overheads, trade-offs, and slowdowns as well as improvements.

## Hardware-counter analysis

See the [Part C profiling workflow](part_c/PROFILING.md) for the Python capture/processing scripts, Nsight commands, saved raw reports, and interpretation rules. This analysis is separate from ordinary latency benchmarking and does not integrate Part A.

The completed [measured bottleneck findings](part_c/results/nsight_analysis/FINDINGS.md) use 48 validated Nsight captures (8 kernels × 2 shapes × 3 repetitions), with raw reports, Python processing scripts and correlated CUDA/SASS. Hardware-counter measurements are kept separate from the ordinary latency dataset.
