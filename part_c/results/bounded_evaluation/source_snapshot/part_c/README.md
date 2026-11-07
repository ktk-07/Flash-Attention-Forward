# Part C — Causal FlashAttention

Part C develops causal attention forward kernels using scalar CUDA cores, cooperative warps, register tiling, WMMA, and explicit PTX MMA. The evaluation will show both the progression within each family and the comparison against [Part A's materialised baselines](../part_a/README.md).

**Status:** a standalone Part C build, test/benchmark runner, CSV export, sanitizer workflow, and plotting scripts now exist. The split kernels are derived from the updated `src/causal_flash_attn.cu`, with documented corrections found during testing. The original coding file is retained as a minimal standalone smoke test under `src/extra/`, and its kernel comments are preserved; superseded copies and built executables have been removed from `src/`. This work does not integrate Part A or PyTorch. Cross-part comparisons below remain a future plan.

## Standalone build, tests, and plots

The implementation uses conventional separate compilation: public declarations under `include/flash_attention/`, CUDA definitions under `src/`, and a C++ test runner that includes only public headers.

```text
part_c/
├── include/flash_attention/
│   ├── types.h
│   ├── flash_attention_1.h
│   ├── flash_attention_2_cuda_core.h
│   └── flash_attention_2_tensor_core.h
├── src/
│   ├── extra/causal_flash_attn.cu  # Original coding file with a minimal test
│   ├── flash_attention_1.cu
│   ├── flash_attention_2_cuda_core.cu
│   └── flash_attention_2_tensor_core.cu
├── tests/
│   ├── kernel_runner.cpp
│   └── causal_flash_attn_full_test.cu
├── scripts/
│   ├── plot_results.py
│   ├── inspect_binary.py
│   └── audit_results.py
├── CMakeLists.txt
└── run.sh
```

| Source file | Definitions |
|---|---|
| [`src/flash_attention_1.cu`](src/flash_attention_1.cu) | FA1-style scalar kernel and its public launch function. |
| [`src/flash_attention_2_cuda_core.cu`](src/flash_attention_2_cuda_core.cu) | FA2 scalar, warp-per-query, register-tiled, tile-skipping, and vectorised kernels; CUDA-core launch dispatch. |
| [`src/flash_attention_2_tensor_core.cu`](src/flash_attention_2_tensor_core.cu) | Fused WMMA and PTX MMA kernels, helper definitions, and Tensor Core launch dispatch. |

[`include/flash_attention/`](include/flash_attention/) contains only the shared type/enum definitions and public function declarations. CUDA templates and their dimension dispatch are private to the `.cu` implementations. CMake compiles the three `.cu` files into `part_c_attention`, then links [`tests/kernel_runner.cpp`](tests/kernel_runner.cpp) against that library. The runner does not include `.cu` files or kernel-definition headers.

The original coding file is now [`src/extra/causal_flash_attn.cu`](src/extra/causal_flash_attn.cu), with its comments retained, known missing definitions/fixes applied, and a small standalone `main()`. To run only its basic four-kernel CUDA-core check:

```bash
cd part_c/src/extra
nvcc causal_flash_attn.cu -o ./test && ./test
```

See [`src/extra/README.md`](src/extra/README.md) for the coding history and smoke-test limits. This file is not a CMake build input. The top-level `src/` contains only the three maintained implementation files and `extra/`. The legacy regression harness is under `tests/`; provenance and the original-to-smoke patch are under [`docs/kernel_provenance/`](docs/kernel_provenance/).

Run from the repository root on an Ampere GPU (the recorded run uses SM86). No Git commands are included:

```bash
# One command: all tests, full benchmark, sanitizers, plots, and command/output logs.
bash part_c/run.sh part_c/results/my_run /tmp/part_c-build
```

Use a fresh result directory for independent runs. With no arguments, `bash part_c/run.sh` uses a timestamped result directory and `part_c/build/`. You can also run `./run.sh` from inside `part_c/`. Its first two arguments override the results and build directories. The old `scripts/run_all.sh` is only a compatibility forwarder. For another GPU architecture, change the explicit architecture in the script or use the individual commands below with the correct value.

```bash
cmake -S part_c -B /tmp/part_c-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON
cmake --build /tmp/part_c-build --parallel 2
/tmp/part_c-build/part_c_runner --suite correctness --output part_c/results/my_run/correctness
/tmp/part_c-build/part_c_runner --suite benchmark --grid full --output part_c/results/my_run/benchmark

# Optional shorter benchmark, not the full proposal shape grid:
/tmp/part_c-build/part_c_runner --suite benchmark --grid representative --output part_c/results/quick/benchmark

# Small boundary-case safety checks across all eight variants:
compute-sanitizer --tool memcheck --error-exitcode 3 /tmp/part_c-build/part_c_runner --suite smoke --output part_c/results/my_run/memcheck
compute-sanitizer --tool racecheck --error-exitcode 3 /tmp/part_c-build/part_c_runner --suite smoke --output part_c/results/my_run/racecheck
compute-sanitizer --tool synccheck --error-exitcode 3 /tmp/part_c-build/part_c_runner --suite smoke --output part_c/results/my_run/synccheck
compute-sanitizer --tool initcheck --error-exitcode 3 /tmp/part_c-build/part_c_runner --suite smoke --output part_c/results/my_run/initcheck

python3 part_c/scripts/plot_results.py part_c/results/my_run/benchmark
python3 part_c/scripts/inspect_binary.py /tmp/part_c-build/part_c_runner part_c/results/my_run/binary_inspection
python3 part_c/scripts/audit_results.py part_c/results/my_run
```

The runner also accepts `--kernel fa1|scalar|warp|tiled|skip|vector|wmma|ptx` for an individual variant. CUDA GPU access is needed for execution; compilation and plotting do not require GPU access. Matplotlib is required for plotting. Sanitizer checks use `B=1,H=1,N=33,d=64`; they are representative checks, not exhaustive sanitizer coverage of the full grid.

### Actual standalone test coverage

- Full-output comparisons against an independent Part C FP64 CPU oracle, with nonfinite output detection and output prefilled with NaNs to catch missing stores.
- N=1/15/16/17/31/32/33/63/64/65/97 at d=32/64/128; additional d=7/16/33/40/65/96 cases with B=2,H=3, and a B=2,H=3,N=97,d=64 case.
- Random, zero-Q/K uniform-prefix attention, constant V, increasing logits across KV tiles, and finite large-logit stress inputs. The additional patterns run at d=64,N=97.
- FP32 `atol=rtol=1e-4`; FP16-input Tensor Core `atol=0.002,rtol=0.02`. Tensor Core oracle inputs are rounded to the same FP16 values. Relative error uses `max(abs(reference),1e-12)`.
- PTX dimensions below 16 or not divisible by 16 are explicit `UNSUPPORTED` rows. The current suite produces 380 supported checks and 4 unsupported cases.
- Benchmark grid: B=4,H=8,N=256/512/1024/2048/4096,d=32/64/128, all eight causal kernels: 120 configurations, 5 warmups and 20 repeats each (2,400 samples).
- Benchmark validation checks query rows 0, N/2, N-1 in **every** batch/head against the CPU oracle and checks **all** outputs for finiteness. This is recorded as `sampled_queries`; it is not a full-output FP64 validation of the large grid.

All kernels currently use fixed educational configurations, not a completed tuning search. `fa1/scalar`: BQ=32,BKV=32,32 threads; `warp`: BQ=4,BKV=32,4 warps; `tiled/skip/vector`: BQ=32,BKV=32,4 warps,THREAD_Q=3,THREAD_K=5; `wmma/ptx`: BQ=32,BKV=32,2 warps. The main file's tiled kernel already includes causal skipping: the separate `skip` snapshot is not a clean additional optimisation stage.

### Recorded results and limitations

The initial measured run and correction history remain under [`results/local_validation/`](results/local_validation/). A separate [layout-validation result summary](results/layout_validation/README.md) verifies the conventional `.h`/`.cu` layout without overwriting those results. Its commands, logs, CSVs, and generated figures use the same directory structure. The local GPU is an RTX 3070 Ti Laptop GPU with CUDA Toolkit 12.6; timings are local development evidence, not the RTX 3090/5060 Ti measurements used elsewhere. Final counts and observed timings are summarised in [`results/local_validation/RESULTS.md`](results/local_validation/RESULTS.md).

The standalone plots are C1 CUDA-core progression, C2 WMMA/PTX progression, C3 best-observed CUDA/Tensor comparison, all-kernel scaling, and sampled numerical error. Each is exported as PDF and PNG. C3 selects an exploratory best observed kernel using geometric-mean latency at d=64,N=256/1024; it is **not** a separately tuned, frozen winner. Both families accumulate/output in FP32, but their inputs differ in precision. Part A is absent from these plots because integration is intentionally deferred.

Each run saves full commands and exit codes in `commands.log`, stdout/stderr under `logs/`, and source/binary hashes in `logs/source_hashes.log`. Correctness, timing samples and summaries live in separate suite subdirectories. Compiler register/shared/stack reports and per-function HMMA counts are under `binary_inspection/`; these are static evidence, not achieved occupancy or measured DRAM traffic. The runner records logical kernel Q/K/V/O storage separately from all harness allocations, which retain both FP32 and FP16 inputs. Neither is a device-wide peak-memory measurement.

## Future cross-part evaluation plan

The remaining sections specify future integration/report experiments. Their planned names and schemas are broader than the standalone runner above; they are not claims that Part A is integrated or that tuning/profiling studies are complete.

## Computation and precision

The evaluation uses contiguous `[B,H,N,d]` Q/K/V/O, scale `1/sqrt(d)`, and causal visibility `key_position <= query_position`. No dropout, backward pass, KV cache, or offset causal alignment is included. Some source signatures use `N` for head count and `H` for head dimension; adapters must map these explicitly and preserve each function's stride order.

| Family | Q/K/V | Accumulation and online-softmax state | Output |
|---|---|---|---|
| CUDA-core fused | FP32 | FP32 | FP32 |
| Fused WMMA / PTX MMA | FP16 | FP32 | FP32 |
| Part A `optimized` | FP32 | FP32 | FP32 |
| Part A `tensorcore` | FP16 | FP16 WMMA accumulation and half-precision softmax | FP16 |

For Tensor Core correctness, use the same FP16-rounded inputs in the CPU reference. Separately report error against attention on original FP32 inputs when evaluating the end-to-end precision trade-off. The Part A FP16 comparison includes accumulation, softmax, and output-precision differences; do not attribute its entire timing difference to fusion.

## Kernel inventory and registration

The current runner uses these IDs. The source files are compiled by CMake; `src/extra/causal_flash_attn.cu` retains the original coding history with a minimal standalone check and is not built by CMake. FA1/FA2 names describe educational variants rather than equivalence to official library releases.

| ID | Source / symbol | Role |
|---|---|---|
| `fa1` | `src/flash_attention_1.cu`: `flash_attention_v1_baseline` | FA1-style scalar online-softmax baseline. |
| `scalar` | `src/flash_attention_2_cuda_core.cu`: `flash_attention_v2_scalar_baseline` | One thread owns a query row and online state. |
| `warp` | Same file: `flash_attention_v2_warp_per_query` | Cooperative warp-per-query computation and shared K/V. |
| `tiled` | Same file: `flash_attention_v2_cuda_core_tiled` | Register tiling and shared Q/K/V reuse; already includes causal skipping. |
| `skip` | Same file: `tile_skip::casual_flash_attention_forward_gpu_optimised_for_cuda_core` | Historical tile-skipping variant retained within the compiled implementation. |
| `vector` | Same file: `vector_load::casual_flash_attention_forward_gpu_optimised_for_cuda_core` | Aligned float4 Q/K/V loads with scalar fallback. |
| `wmma` | `src/flash_attention_2_tensor_core.cu`: `flash_attention_wmma_fused` | Fused WMMA QKᵀ and PV. |
| `ptx` | Same file: `flash_attention_v2_tensor_core_ptx_mma_fully_fused` | Explicit MMA with register-resident score softmax/output. |

WMMA/PTX require `BLOCK_Q=16*NUM_WARPS`. PTX additionally requires `BLOCK_KV>=16` divisible by 16, `HEAD_DIM>=16` divisible by 16, and 1–8 warps. WMMA pads head/KV dimensions internally. PTX packs operands manually; it does not currently use `ldmatrix`.

Superseded combined, annotated, placeholder, and test-copy sources were removed from `src/`, along with two built executables. The 13-case legacy regression harness was moved into `tests/` and `run.sh` updated accordingly. Historical filenames/hashes are recorded in [`docs/kernel_provenance/src_cleanup.json`](docs/kernel_provenance/src_cleanup.json); old measurement logs retain their original command paths.

## Three kernel comparison figures

All three figures use causal self-attention and measured complete-forward CUDA-event latency. Each bar/point must have a passing validation record and a source/configuration identity. Retain slower variants; missing or unsupported entries are labelled rather than plotted as zero.

### Figure C1 — CUDA-core progression

- Plot ordered kernel versions on the x-axis and median latency in milliseconds on the y-axis, with Q1–Q3 error bars.
- Use one figure with small/medium/large panels: `B=4,H=8,d=64,N=256/1024/4096`.
- Show Part A `naive` and `optimized` as a distinct materialised baseline group, then `c_scalar`, `c_warp`, `c_tiled`, `c_skip`, `c_vector` as the fused development sequence.
- Annotate speedup against Part A `optimized` for the same shape. Use a logarithmic latency axis if needed and label it.
- These are FP32 results. Verify what each stage actually changes: earlier versions may already skip some causal work. A development sequence is not a one-feature ablation.
- Proposed export: `c1_cuda_core_progression.pdf` and `.png`.

### Figure C2 — Tensor Core progression

- Use the same representative shapes, latency units, error bars, and stage order convention as C1.
- Show Part A `tensorcore` as the materialised WMMA baseline, followed by distinct validated fused stages. The currently identified stages are `tc_wmma` and `tc_ptx`.
- Label FP16 inputs, accumulator/softmax/output precision, and materialised versus fused execution. Compare the two fused stages directly with their matching FP16/FP32 contract.
- If a fused WMMA-QK/scalar-PV intermediate is later implemented and validated, insert it as its own stage. Do not invent a measurement for that planned stage.
- Standalone GEMM is a development check, not an attention-forward timing bar. Annotated/duplicate WMMA files are not separate progression stages unless they implement a measurable design change.
- Proposed export: `c2_tensor_core_progression.pdf` and `.png`.

### Figure C3 — Best CUDA-core versus best Tensor Core

- Plot sequence length `256,512,1024,2048,4096` against median latency, with panels for `d=32,64,128`, at `B=4,H=8`.
- Highlight the best validated Part C CUDA-core and Tensor Core implementations. Choose the winners using geometric-mean latency over the fixed tuning subset, then freeze their IDs and configurations before the main run. Declare a tie-break rule in the run manifest.
- Include thinner context curves for Part A FP32 `optimized` and FP16 `tensorcore`; keep the two selected Part C curves visually prominent.
- Use shared original inputs, with explicit FP16 rounding for Tensor Core paths. Label this as a precision/performance trade-off and accompany it with maximum-error results. It does not isolate the benefit of Tensor Core hardware alone.
- Report the latency ratio `best_cuda_ms / best_tensor_ms`; values below one mean the selected CUDA-core implementation was faster. Retain crossovers and regressions.
- Proposed export: `c3_best_cuda_vs_tensor.pdf` and `.png`.

Part A's historical measurements are from an RTX 5060 Ti. Rebuild and rerun Part A alongside Part C on the same evaluation GPU; never divide its historical latency by a new RTX 3090 latency to claim a kernel speedup.

## Additional report evidence

The three figures above are the kernel-development story. The proposal also calls for full-project FP32 dense/causal scaling, peak memory, a controlled ablation, tile-size tuning, and small/large-case profiling. See the [root figure plan](../README.md#figures-and-tables).

For Part C, prepare these supporting analyses:

- **Ablation:** hold shape, precision, and launch configuration constant while changing one feature. Compare full causal KV traversal with masking against causal tile skipping; compare scalar loads against float4 loads with all other choices fixed. Dense-versus-causal timing is a separate semantics/work comparison. Record any other implementation changes rather than claiming perfect isolation.
- **Memory:** count all live device allocations, including scratch, padding, and FP16/FP32 output differences. Report auxiliary storage separately. Shared-memory/register usage and profiler DRAM bytes are different metrics from allocated global memory.
- **Causal work:** count CTA-level processed, fully skipped, and diagonal-intersecting tiles, stating whether counts represent loaded tiles or actual MMA/FMA work. For partial tiles, tile counts do not equal useful interaction counts. Collect instrumentation separately from latency runs.
- **Tuning:** predefined candidates such as `(BLOCK_Q,BLOCK_KV,NUM_WARPS)=(32,32,2),(32,64,2),(64,32,4),(64,64,4)` for the WMMA/PTX family, subject to resource checks. CUDA-core families need their own valid candidates, including `THREAD_Q/THREAD_K`; do not force Tensor Core launch mappings onto them.
- **Profiling:** small `N=256` and large `N=4096` at `B=4,H=8,d=64`. Save DRAM traffic/bandwidth, registers, shared memory, theoretical/achieved occupancy, spills/local traffic, and relevant stalls where available. Inspect generated instructions to confirm Tensor Core use. Interpret these measurements alongside latency rather than assuming occupancy alone predicts performance.
- **Accuracy:** maximum absolute and relative errors, pass/fail tolerances, and mixed-precision error versus N. Define the relative-error denominator/floor; near-zero references can produce large relative errors while passing the element-wise absolute-plus-relative criterion.

## Correctness gates and test plan

Reuse `part_a/common/input_generation.hpp` and the independent `part_a/cpu_reference/attention_reference.cpp` through shared adapters. Development demos use their own inputs and smoke-test criteria; their output does not establish a passing integrated grid.

1. **Materialised oracle agreement:** compare Part A S/P/O for small deterministic cases; compare every fused final O against the CPU oracle and matching Part A mode. CPU timing is not a performance baseline.
2. **Batch/head indexing:** include `B=1/2/4` and `H=1/3/8` on small cases to detect mixing between heads/batches.
3. **Sequence boundaries:** test N=1,15,16,17,31,32,33,63,64,65 and lengths around every selected query/KV tile. Cover partial query tiles, partial KV tiles, and multiple KV iterations.
4. **Head dimensions:** test d=32/64/128 plus supported boundary dimensions such as 7,33,65 for scalar/WMMA kernels. PTX's current non-multiple-of-16 dimensions are explicit unsupported cases, not silent fallbacks or passes.
5. **Causal semantics:** check the first row equals V[0] within tolerance, diagonal inclusion, tails, and independence from changes to future K/V for earlier outputs. Keep all threads participating safely in required barriers.
6. **Online softmax:** use zero Q/K (uniform distribution over the valid prefix), constant V (same constant output), deterministic random inputs, and finite stress inputs that cause the running maximum to change across KV tiles. Verify finite output and correct rescaling.
7. **Memory/load safety:** test aligned float4 loads and the supported scalar fallback, tails, and bounds. Once integrated, run Compute Sanitizer memory/race/synchronisation checks on representative boundary cases, outside timed runs.
8. **Representative and main-grid validation:** validate each benchmark configuration against the oracle before admitting its timings. FP64 oracle runs can be separate and cached by input identity; exclude their cost from GPU timing. Record timeout/OOM/unsupported/failure distinctly and omit invalid measurements.

Use `abs(output-reference) <= atol + rtol*abs(reference)` per element. Part A currently uses FP32 `atol=rtol=1e-4` and FP16 `atol=0.002,rtol=0.02`; use these as documented initial policies, frozen before integrated runs. Any justified revision must be recorded and rerun consistently. Do not replace this criterion with the loose maximum-error smoke-test threshold in a demo. Report accumulation error on rounded inputs separately from input-quantisation error on original inputs.

## Benchmark integration and execution order

Implementation work remaining, in dependency order:

1. Extract callable launches from demos, resolve duplicate names, and register the selected kernels and their constraints. Preserve the educational sources and provenance.
2. Add a shared harness that reuses Part A generation/oracle and owns preallocated buffers. Normalise shape, strides, scaling, precision conversion, error reporting, and output contracts.
3. Complete validation and boundary checks before performance acceptance. Confirm the dense fused implementation separately for the project's Part B requirement.
4. Run bounded tuning on `B=4,H=8,N=256/1024,d=64`. Save trials, the selection rule, winning implementations/configurations, and then freeze them. If different family-specific configurations are used, document them; one-feature ablations keep the configuration fixed.
5. Rerun Part A and Part C in the same GPU environment. Use 5 warmups, 20 timed full forwards, median/Q1/Q3/IQR, and the main grid in the root README. Rotate implementation order; keep profiling and instrumentation out of timing runs.
6. Profile representative small/large cases, export raw results, then generate C1/C2/C3 and supporting plots from saved data. Review slowdowns and the accuracy trade-off before writing conclusions.

Use `bash part_c/run.sh` from the repository root for the supported complete workflow. The retained legacy regression source is `tests/causal_flash_attn_full_test.cu`; it is also run by the script. Removed historical demos are not part of the current reproduction instructions.

## Planned result files

Use a new `results/<run_id>/` for each integrated run, retaining Part A's existing result directory as historical data. Filenames below are planned artifacts, not files generated by this documentation update.

| File | Required content |
|---|---|
| `environment.json` | GPU/device UUID, architecture, toolkit/compiler/flags, available power/clocks, timing settings, seeds, and revision. |
| `manifest.json` | Kernel ID, source/symbol, parent stage, optimisation description, mode, precision contract, supported shapes, selection policy, and run status. |
| `configs.json` | Frozen launch settings per family, tuning subset, best kernel IDs, and tie-break rule. |
| `correctness.csv` | Run/kernel/config/input IDs, B/H/N/d, mode, precisions, reference identity, atol/rtol, relative-error definition, errors, nonfinite count, and pass/fail/skip reason. |
| `samples.csv` | Same identity fields, repeat index, and full-forward milliseconds; Part A stage samples tagged as separately timed. |
| `summary.csv` | Identity, validated status, warmups/repeats, median/Q1/Q3/IQR, sequences/s, allocated/auxiliary bytes, and named speedup comparator. |
| `tuning.csv` | Every bounded candidate, validity/resource failures, correctness result, timings, and selection score. |
| `resources.csv` | Blocks/threads/warps, registers/thread, shared/local bytes, and theoretical occupancy. |
| `causal_work.csv` | Tile sizes, counting unit/method, processed/skipped/diagonal tiles, and dense-work comparator. |
| `profiles/` | Saved profiler reports plus exported metrics with units, GPU/kernel/config identity, and metric availability. |
| `figures/` | C1/C2/C3 and supporting PDF/PNG figures generated from the CSVs by versioned plotting scripts. |

Part A already writes `samples.csv`, `summary.csv`, `correctness.csv`, and `resources.csv`; its saved results also include profiler data and an environment record. The integration should map these into a common schema while preserving original values. Join on run/kernel/config/input identity, shape, mode, and all precision fields, not just sequence length. Recompute cross-part speedups from matching same-GPU full-forward medians rather than reusing Part A's `speedup_vs_naive` field.

## Original development notes

The notes below preserve the learning history, exploratory pseudocode, and earlier plans. They are not the current build instructions or evidence of validated performance; use the sections above for the evaluation plan.

# How we will be progressing towards the kernels

Approach | Thread mapping and computation | Performance potential
A. | Thread-per-query	One thread owns a query row, its online-softmax statistics \(m,l\), and output accumulator acc[d]. Scalar QKᵀ/PV. | Low
B. | Warp-per-query	One warp owns a query row. Cooperative K/V loading, shared-memory tiles, warp reductions, and shuffles.	Moderate
C. | 2D register-tiled CUDA-core	Threads own small 2D score subtiles; thread groups process multiple query rows and key positions. Register blocking and Q/K/V reuse, scalar FMA computation. | Potentially high for CUDA cores
D. | Tensor Core FlashAttention	Warps own distributed MMA fragments. Matrix multiplication uses Tensor Core instructions instead of scalar FMAs. | Highest potential for suitable workloads

Stage 1 — Correctness-first FlashAttention
Objective: Understand FlashAttention Algorithm  
One CUDA thread computes one query row. Scalar QKᵀ and PV multiplication, online softmax, causal masking, and output normalization.  
Algorithm implemented; compilation and PyTorch correctness testing still need to be verified.

Stage 2 — Shared-memory tiled FlashAttention - General Purpose GPU Optimisations
Objective: Understand general GPU optimization and cooperative parallel programming. It teaches you how to program GPUs efficiently.
Task: We will redesign the kernel so multiple threads cooperate on matrix tiles, using coalesced memory access, shared-memory reuse, synchronization, and parallel reductions.

Concepts to know:
    - Memory coalescing: Adjacent threads access adjacent global-memory addresses.
    - Shared-memory tiling: Load data once and reuse it across multiple computations.
    - Register tiling: Keep frequently accessed values in registers.
    - Parallel reductions: Cooperatively compute row-wise maximums and sums.
    - Synchronization: Coordinate threads sharing data.

1. Thread mapping: Which thread computes which output elements?
2. Shared memory: How do threads cooperatively load and reuse Q/K/V?
3. Synchronization: When is it safe for threads to consume the loaded data?
4. Parallel reductions: How do threads calculate the maximum and denominator for each query row?

Stage 3 — Tensor Core FlashAttention inspired - Hardware Optimisation
Objective: Understand architecture-aware optimization, especially Tensor Core programming on Ampere. We are exploiting a particular GPU architecture efficiently by doing Hardware Specific Optmisation.
Replace scalar matrix multiplication with warp-level MMA, then optimize layouts, register usage, memory pipelining, and occupancy.

# Resource Consolidation
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
[Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
[CUDA and C++ extension with PyTorch](https://github.com/pytorch/extension-cpp)

# Immplementing General GPU Optimisations to the CUDA kernel
Before even reading the optimisation.
## First Thoughts
- We should recap the articles on How to Optimise a CUDA Matmul Kernel for cuBLAS-like Performance
- We Should probably be reading the kernel bottlenecks via nsight or something. (Ill probably do this)

## Understanding limitations of Kernel

This are just some limitations I there there might be?
1.How we decide GridDim and BlockDimGrid and Threadblock we are launching rightblock
    - Currently we have 1 thread computing 1 row of the output.
    - gridDim (B*N, ((S + BLOCK_Q - 1) / BLOCK_Q), blockDim(BLOCK_Q)
    - Tile_r = (S + BLOCK_Q - 1) / BLOCK_Q)
2.How we are indexing might not be optimised for memory coalescing
3.We are computing the 2 matmuls, we are directly reading from HBM instead of SMEM
4.It seems like 1 thread for computing multiple rows of output is quite of alot , we can perhaps profile this to see if the arithmetic intensity is already high?
5.Maybe we can use GPU Profiling with NSIGHT, we found that 

## GPU Profiling of my Correctness CUDA Kernel

## Understanding and Reading the compileed PTX/SASS Code to find loopholes


# My Thought Process
## 1.Incorporating Global Memory Coalescing (GMEM)
It is about indexing that threads in the same warp group, so that when the GPU (or was it the warp scheduler) does a memory transaction that requires less amount of reads to and from memory.

I believe the advantage here would be the loading of Q, K_tranposed,V and O in a memory coalesced manner.
Q * K^T && PV
so i believe K^T and V is where w
Lets take into account Q, K^T and V, O are stored in a row major memory format (Im not too sure if this is the actual convention especailly for K^T, as K might be in row major memory format and K^T is not)
    - I will just try to give the convention based on this

Indexing in the non-optimsed kernel
```
b_idx = blockIdx.x // N
n_idx = blockIdx.x % N
tile_r_idx = blockIdx.y
s_idx = tile_r_idx * BLOCK_Q + threadIdx.x # This acts as the row and col

q_base_ptr = b_idx * stride + n_idx * stride + s_idx
k_base_ptr = b_idx * stride + n_idx * stride
v_base_ptr = b_idx * stride + n_idx * stride
o_base_ptr = b_idx * stride + n_idx * stride + s_idx
```

In the kv looop
```
for (int i = 0 ; i < col && i <= s_idx ; i += BLOCK_KV){ // Causal attention: query at position s_idx can only attend to keys at positions <= s_idx (including itself).
    // S = QK^T - First Matmul
    int col_start = i;
    float matmul_acc[BLOCK_KV] = {};
    // K/V Loop
    for (int j = 0 ; j < BLOCK_KV; j++){
        int col_idx = col_start + j;
        if (col_idx >= col || col_idx > s_idx){
            matmul_acc[j] = -INFINITY;
            continue;
        }
        
        // Casual Mask add code here later
        int cur_k_base_offset = k_base_offset + col_idx * stride_ks;
        // Head Dimension Loop, which is the same as me passing in the inner_dimension no? is this variable redundant?
        for (int k = 0 ; k < HEAD_DIM ; k++){
            int q_ptr = q_base_offset + k * stride_qh;
            int k_ptr = cur_k_base_offset + k * stride_kh;
            matmul_acc[j] += *(Q + q_ptr) * *(K_transposed + k_ptr);
        }
    }
    
    ... // Softmax recurrence we can skip

    // O = PV - Second Matmul, O is not normalised yet
    for (int j = 0 ; j < HEAD_DIM ; j++){
        for (int k = 0 ; k < BLOCK_KV ; k++){
            int col_idx = i + k;
            if (col_idx >= col || col_idx > s_idx) continue; // No need to compute future tokens
            int cur_v_base_offset = v_base_offset + col_idx * stride_vs;
            int v_ptr = cur_v_base_offset + j * stride_vh;
            final_acc[j] +=  matmul_acc[k] * *(V + v_ptr);
        }
    }
}
```
My analysis is that
Since we are doing Q @ K^T && PV, so i believe K^T and V is where we can try to access it in a memroy coalesced manner for reading
For O since we are like writing, I dont think there is much to do? or memory coalescing is for writing also? I think it is 

But anyways our current kernel, i think the problem right now is we are loading the KV manually? 
so every thread we are manually accessing the K and V memory and does not even uses different threads to load the KV no?

so we are using already them with threadidx.x no?
Hmm is there anyway? idk
Isnt Q already memory coalesced tho? nope nvm cause, q is the row, and threadIdx.x is being used to compute the row, which makes the warps access nonlinear memory


If Q,K_transposed,V,0 are in row major format
K should take the slow changing index
Q should take the slow changing inded

If only Q, K (not K_tranposed we which are passing in), V,O are in row major format

nvm i think it was never abou the K and V, cause that is what the internal thread computes
I also dont think there is any point is memory coalecsing Q it doesnt make sense, the output Q x dim
Yea so coaleasing doesnt matter

I think i figured out a way to do it.
the gridDim doesnt really matter it is jsut the number of like threadblocks we are launching.
Initally my old approach, 1 thread block has BLOCK_Q 
Now my current approach, 1 threadblock has BLOCK_Q * BLOCK_KV
1 cuda threadblock is still responsible for 1 block q row of output, no changes;
SO 1 cuda thread 1 row of output , so that single thread does everything.
it is just that now we have more threads working rather than just 1 thread working on the KV/LOOP to comptue softmax and the 2 matmuls

nah nah nah im retarded HAHAHA this is definitely wrong. caue each thread is launch in parallel to do work separately no? or is it each threa block does work  separately // im pretty sure
waittt then it should be correct no?
cause eachh individual thread is no accessing the BLOCK K, 
yes i think it might be correct

Unless, our threadBlock is BLOCK_Q * BLOCK_KV (just need to make sure BLOCK_Q * BLOCK_KV <= 1024, cause 1 threadblock only 1024 threads)
```
int tile_r_idx = blockIdx.y
int sq_idx = threadDim.x / BLOCK_KV
int sk_idx = threadDim.x % BLOCK_KV

int s_idx = sq_idx

tile_c = ( col + BLOCK_KV - 1 ) / BLOCK_KV
# In the kv loop
for (int i = 0 ; i < tile_c && i <= s_idx ; i++){ // Causal attention: query at position s_idx can only attend to keys at positions <= s_idx (including itself).
    // S = QK^T - First Matmul
    int col_start = i * BLOCK_KV + sk_idx;
    float matmul_acc[BLOCK_KV] = {};
    // K/V Loop
    for (int j = 0 ; j < BLOCK_KV; j++){
        int col_idx = col_start + j;
        if (col_idx >= col || col_idx > s_idx){
            matmul_acc[j] = -INFINITY;
            continue;
        }
        
        // Casual Mask add code here later
        int cur_k_base_offset = k_base_offset + col_idx * stride_ks;
        // Head Dimension Loop, which is the same as me passing in the inner_dimension no? is this variable redundant?
        for (int k = 0 ; k < HEAD_DIM ; k++){
            int q_ptr = q_base_offset + k * stride_qh;
            int k_ptr = cur_k_base_offset + k * stride_kh;
            matmul_acc[j] += *(Q + q_ptr) * *(K_transposed + k_ptr);
        }
    }
    
    ... // Softmax recurrence we can skip, doesnt affect it

    // O = PV - Second Matmul, O is not normalised yet
    for (int j = 0 ; j < HEAD_DIM ; j++){
        for (int k = 0 ; k < BLOCK_KV ; k++){
            int col_idx = col_start + k;
            if (col_idx >= col || col_idx > s_idx) continue; // No need to compute future tokens
            int cur_v_base_offset = v_base_offset + col_idx * stride_vs;
            int v_ptr = cur_v_base_offset + j * stride_vh;
            final_acc[j] +=  matmul_acc[k] * *(V + v_ptr);
        }
    }
}
```

### My Approach After Reading Hints
So after reading the hints, I realise, there is a step before memory coalescing which is to redesign the thread ownership before touching memory coalescing.

Instead of 1 thread 1 query row, we are essentially doing 1 warp for 1 row of q first before 1 warp for BLOCK_Q rows of q
Im pretty sure this is like quite similar to what im doing, but I guess my does not have proper ownership ?

Before that we should understand how the underlying hardware works
1 warp 32 lanes, each lane 
1 Compute Thread Array (CTA) = 1 ThreadBlock

Since every lane is memory coalescing, which is exactly what we want BLOCK_KV which is stored in row-major format to do
BLOCK_KV should be 32 lanes, every warp should be loading the same BLOCK_KV tile (we can optimise this with shared memory cache later ?)

So im assuming if we use more threads to compute a row, that much, our tile_size_r will increase
more threads assign to compute BLOCK_Q, mean more TILE_R are needed ?
eh not really, I was only thinking this way cause our inital launching of cuda kernel uses in our blockdim
However, if i just launch with blockDim(BLOCK_Q * no_of_threads_per_warp) it should be no issue.
So no_of_threads_per_warp should correspond to the 

Threads ownership Initially in Stage 1
```
int TILE_R = ( S + BLOCK_Q - 1 ) / BLOCK_Q;
dim3 gridDim(B*N,TILE_R);
dim3 blockDim(BLOCK_Q)
```

Thread ownership Now my approach in Stage 2.
```
int num_warps = BLOCK_Q;
int threads_in_warp = 32; # Standard
int BLOCK_SIZE = N; # some value doesnt matter
int TILE_R = ( S + BLOCK_Q - 1 ) / BLOCK_Q;
dim3 gridDim(B*N,TILE_R);
dim3 blockDim(num_warps * threads_in_warp) 
```

Only problem here is that num_warps * threas_in_warpcs cannot be more than 1024 cause 1 threadBlock or 1 CTA is 1024 max
My actual solution here

For simplicity sake, we just keep BLOCK_Q == NUM_WARPs
For the BLOCK_KV can be more thatn warp_size.
It is because we eliminate the need for an explicit loop over individual query rows, not the loops over the head dimension or K/V tiles.
```
int warp_size = 32;
int USER_BLOCK_Q;
int max_warps = 1024 / warp_size;
int num_warps = min(USER_BLOCK_Q,max_warps)
BLOCK_Q = num_warps
int TILE_R = ( S + BLOCK_Q - 1 ) / BLOCK_Q;
dim3 gridDim(B*N,TILE_R);
dim3 blockDim(num_warps * warp_size);
```

With regards to the indexing in the KV loop
```
int warp_size = 32;
int tile_r_idx = blockIdx.y
int warp_idx = threadIdx.x / warp_size // Should corresponed to sq_idx
int warp_lane_idx = threadIdx.x % warp_size // Should correspond to the sk_idx, 

int s_idx = tile_r_idx * BLOCK_Q + warp_idx; // which is just sq_idx


// Defining Shared Memory for Memory coalescing
__shared__ float K_shared[BLOCK_KV][HEAD_DIM];
__shared__ float V_shared[BLOCK_KV][HEAD_DIM];
# In the kv loop
int tile_c = ( col + BLOCK_KV - 1 ) / BLOCK_KV
for (int i = 0 ; i < tile_c ; i++){ // Causal attention: query at position s_idx can only attend to keys at positions <= s_idx (including itself).
    // Maybe we would be defining another loop for SMEM during the compute later, but right now we focus on memory coalescing

    // The problem is the K is stored physically in B x N x S x H
    // We want to load into shared memory B x N x H x S

    int tile_c_start = i * BLOCK_KV;  
    // Loading Loop
    // Each warp cooperatively loads K rows, and each lane within that warp loads a different feature of those rows. Once the shared-memory tile is populated, every warp uses the entire K tile to compute attention scores for its own query row.
    # Since other warps are responsibel for other row, take into account for that too
    int no_iterations = (HEAD_DIM + warp_size -1) / warp_size;
    int no_row_iterations = (BLOCK_KV + num_warps -1) / num_warps;
    for (int j = 0 ; j < no_iterations; j++){
        int inner_dim_idx = 32 * j + warp_lane_idx;
        int col_idx = inner_dim_idx;

        if (inner_dim_idx >= HEAD_DIM ){
            continue;
        }

        // Why is there a need for this? Why cant we just do for(int k = 0; k < BLOCK_KV; k++)
        for (int k = 0 ; k < no_row_iterations ; k++){
            int row_local = warp_idx + k * num_warps;
            int row_idx = tile_c_start + row_local;

            if (row_local >= BLOCK_KV)
                continue;

            if (row_idx >= col) {
                K_shared[row_local][inner_dim_idx] = 0.0f;
                V_shared[row_local][inner_dim_idx] = 0.0f;
                continue;
            } 
            
            int k_ptr = k_base_offset + row_idx * stride_ks + inner_dim_idx * stride_kh;
            int v_ptr = v_base_offset + row_idx * stride_vs + inner_dim_idx * stride_vh;
            K_shared[row_local][inner_dim_idx] = *(K_transposed + k_ptr);
            V_shared[row_local][inner_dim_idx] = *(V + v_ptr);
        }
    }
    __syncthreads();

    // Compute Loop
    // S = QK^T - First Matmul
    int no_key_groups = (BLOCK_KV + warp_size - 1) / warp_size;
    float matmul_acc[no_key_groups] = {}; // Because each lane is responsible for computing it, we store this in the registers of each threads
    for (int j = 0; j < no_key_groups ; j++){
        int matmul_idx = j * warp_size + warp_lane_idx; 
        int col_idx = tile_c_start + matmul_idx; // Your sk_idx

        if (col_idx >= col || col_idx > s_idx || matmul_idx >= BLOCK_KV){
            matmul_acc[j] = -INFINITY;
            continue;
        }
        
        // Head Dimension Loop
        // We need to take into account that K is mathematically transposed and not physically transposed.
        // Why do we need use warp_idx here?
        // Each warp owns one query row, and every warp needs to multiply that query row against all the K rows in the shared-memory tile
        for (int k = 0 ; k < HEAD_DIM ; k++){
            int q_ptr = q_base_offset + k * stride_qh;
            matmul_acc[j] += *(Q + q_ptr) * K_shared[matmul_idx][k];
        }

        // Softmax scaling 
        matmul_acc[j] *= softmax_scale;
    }

    __syncthreads();

    ... Online Softmax Recurrence

    ... O = PV
}
```


After cooperative loading K/V with memory coalescing then computing QK^T we can now move onto Warp-level softmax and online recurrence
Because we store the matmul results of S = QK^T into the registers of each warp lane, we need to use warp-level softmax
Each warp computes 1 row of q.

[Cooperative-groups](https://developer.nvidia.com/blog/cooperative-groups/)
[Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
[warp-shuffle-functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

Outside of the tile_c loop, we have to define some global variables, but do we store the in the registers or the shared memory
```
float global_max, global_denominator;
global_denominator = 0.0f;
global_max = -INFINITY; // In C++ FLT_MIN, in CUDA they already have this pre-built macro
float final_acc[HEAD_DIM] = {};
```

How do we call this funciton tho? so every warp calls it? how does it even work?, like we randomly in the code all it? and somehow the GPU threads will pause all other execution and call this?
```
// __forceinline__ to encourage compiler to inline these tiny functions, avoiding unncesary function call overhead
__device__ __forceinline__ float warp_max(float val, int warp_size){
    // 0xFFFFFFFF means all 32 threads participate
    constexpr unsigned FULL_MASK = 0xffffffffu; // Mask is just to identify which warplane in the warp is particpating in the warp function
    for (int offset = 16; offset > 0; offset /= 2){
        float remote_val = __shfl_down_sync(FULL_MASK, val, offset);
        val = fmaxf(val, remote_val);
    }
    int srcLane = 0; // LaneIdx which contains teh value
    int width = warp_size; // logical shuffle subgroup width
    return __shfl_sync(FULL_MASK, val, srcLane, width); // Lane 0 holds the max so we broadcast
}

__device__ __forceinline__ float warp_sum(float val, int warp_size){
    // 0xFFFFFFFF means all 32 threads participate
    constexpr unsigned FULL_MASK = 0xffffffffu; // Mask is just to identify which warplane in the warp is particpating in the warp function
    for (int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    int srcLane = 0; // LaneIdx which contains teh value
    int width = warp_size; // logical shuffle subgroup width
    return __shfl_sync(FULL_MASK, val, srcLane, width); // Lane 0 holds the max so we broadcast
}
```

```
    // P = softmax(S) Online Softmax Softmax
    // Defining local running statistics
    float new_max = -INFINITY;        
    float local_max = -INFINITY;

    // Get the max of the values in the registers
    float warp_lane_max = -INFINITY;
    for (int j = 0 ; j < no_key_groups; j++){
        warp_lane_max = max(matmul_acc[j], warp_lane_max);
    }

    // Get the warp max, which is the equivalent of our local_max in the stage1 kernel
    float warp_max_val = warp_max(warp_lane_max);
    local_max = warp_max_val;

    // Get the new_max, dont update first cause we need to rescale
    new_max = max(global_max, local_max);

    // Adjust the rescaling factor,
    // Problem is if new_max == -INFINITY, we might get NAN so need to change
    float rescaling_factor = (new_max == -INFINITY) ? 0.0f : expf(global_max - new_max);

    // Rescaling running statistic, we need to determine
    // first what is the dimensions for final_acc
    // final_acc[NUM_OUTPUT_GROUPS], cause 1 warps for now is responsible for 1 query

    constexpr int NUM_OUTPUT_GROUPS = (HEAD_DIM + warp_size -1) / warp_size;
    for (int j = 0; j < NUM_OUTPUT_GROUPS; j++) {
        final_acc[j] *= rescaling_factor;
    }

    global_denominator *= rescaling_factor;

    // Update the global_max
    global_max = new_max;

    // Compute the score = exp(matmul_acc - global_max), which is P
    // We will overwrite the matmul_acc
    // Accumulate the global denominator in the same loop

    float local_warp_lane_denominator = 0.0f;
    for (int j = 0 ; j < no_key_groups; j++){
        matmul_acc[j] = (global_max == -INFINITY) ? 0.0f : expf(matmul_acc[j] - global_max);
        local_warp_lane_denominator += matmul_acc[j];
    }

    // Im not sure if we can go global_denominator += warp_sum(local_warp_lane_denominator); i think no issue but i just did the below, so yes the answer is you can do that
    // We need to do a warp sum of like the current sequences of denominaotrs;
    local_warp_lane_denominator = warp_sum(local_warp_lane_denominator);
    global_denominator += local_warp_lane_denominator;

```

Now onto PV accumulation
The matmul_acc is stored in each warps's threads as well as the final_acc
So i guess we can do the initial accumulation first
then we do the warp level accumulation
No we cant do that.

if HEAD_DIM and BLOCK_KV= warp_size im tryin gto take the simplest example, this way even if HEAD_DIM && BLOCK_KV > warp_size, we just need to loop throught the group
each warp lane contains the Pik, so what we want to do is
take each warplanes's Pik and multipl

each lane also stores one Pik which is in matmul__acc
each lane is one Vkj right? then we move it down ? and accumualte 

we want that Pik to multiple with othe rother warps 's Vkj via broadcast
For PV, we want to broadcast each attention weight Pqk to all 32 lanes, while each lane independently accumulates its own output features.
acumulated into their own respective Pik then shift the Vkj to V(k+1) +j

this way the final_acc is S x H
no need to use warp_idx, cause each_warp_idx is already resposinbel for its own row of Q


```
        // O = PV - Second Matmul, O is not normalised yet 

        // We want to essentially broadcast the probabiities
        constexpr unsigned mask = 0xffffffffu;
        for (int j = 0 ; j < num_key_groups ; j++){
            // We want to essentially broadcast the probabiities
            // This happens sequentially across key positions
            for (int k = 0 ; k < warp_size; k++){
                int k_local = k +  warp_size * j;
                int k_global_idx = tile_c_start + k_local;
                int srcLane = k;
                int width = 32;
                float broadcasted_probability = __shfl_sync(mask, matmul_acc[j], srcLane, width);

                for (int l = 0 ; l < NUM_OUTPUT_GROUPS; l++){
                    // All 32 lanes peform PV computations in parallel for each broadcast
                    int d = warp_lane_idx + warp_size * l;  
                    final_acc[l] += broadcasted_probability * V_shared[k_local][d];
                }
            }
        }

        __syncthreads();

        // 
```

### Actual Approach

Step	Implementation
0   Redesign Thread Ownership Before touching memory coalescing
1	Cooperative Coalesced K/V loading + shared-memory caching	
2	Warp-level softmax reductions	
3	Online softmax recurrence across K/V tiles	
4	PV accumulation	
5	Validate complete forward kernel against PyTorch	
6	Resolve shared-memory bank conflicts	
7	Register tiling, vectorization, profiling	


```
One CTA processes BLOCK_Q query rows
    |
    |-- Each warp owns one query row
    |
    |-- Initialize global_max, global_denominator, final_acc
    |
    |-- For each KV tile:
    |      |
    |      |-- Cooperative K/V global-to-shared loading
    |      |-- __syncthreads()
    |      |
    |      |-- Compute QK^T scores
    |      |-- Apply scaling and attention masks
    |      |
    |      |-- Warp max reduction
    |      |-- Online softmax rescaling
    |      |-- Warp sum reduction
    |      |
    |      |-- PV accumulation using warp shuffles
    |      |
    |      |-- __syncthreads()
    |
    |-- Normalize final_acc by global_denominator
    |
    |-- Write output to global memory
```

So after reading the hints, I realise, there is a step before memory coalescing which is to redesign the thread ownershiper before touching memory coalescing.
Upon even more reading, if we design thread ownship to only do memory coalescing only, then the thread ownship would significantly differ from the actual appraoch, thus we had ot combine step 0. 1. and 2. In addition, There is also a new Warp-level softmax and online recurrence stage, i initially thought there is not need to change the online softmax, but I was heavily wrong
It is basically to enforce independent row computations -> cooperative tiled matrix multiplications

So mutliple threads cooperate to 

What is a warp lane? that a schduler has mutliple warp lanes? eahc warp 32 threads.
Physcically warp shceudler is 1 physical core??

The trick is to separate two different operations:
1. Loading ownership: Which CUDA thread loads each element of K from global memory?
    - We can also load V at the same time
2. Compute ownership: Which CUDA thread uses that element to calculate an attention score?

Phase 1: Cooperative Loading the memory coalesced data into the Shared Memory
    - Memory coalescing describes how threads access global memory.
    - Shared-memory caching describes where data is stored temporarily so it can be reused without repeatedly accessing global memory.

Phase 2: QK^T Computation


Shared-memory bank conflicts


# Implementing 2D Blocked Matrix Multiplication

Multiple query rows per warp.
Multiple warps per CTA.
2D register tiling with explicit operand reuse.
Vectorized, coalesced global-memory loading.
Shared-memory tiling for Q/K/V.
Online softmax and causal tile skipping.


1. Each CTA is responsible for BLOCK_Q query rows.
2. All warps cooperatively load those query rows into shared memory, forming a [BLOCK_Q, HEAD_DIM] tile.
3. The CTA iterates over K/V tiles, each containing BLOCK_KV rows.
4. For every K/V tile, the warps compute QKᵀ, update online softmax, and accumulate PV.
5. The same Q tile is reused across every K/V iteration.

```
// Initialize running statistics and output accumulator

// Load Q_SHARED
__syncthreads();

for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {

    // Load K_SHARED and V_SHARED
    __syncthreads();

    for (int q_base = 0; q_base < query_tile_size; q_base += THREAD_Q) {

        // Load Q_REG

        for (int k_base = 0; k_base < BLOCK_KV; k_base += THREAD_K) {

            // Load K_REG
            // Compute QK^T
            // Warp reduce
            // Apply scale
            // Apply causal mask
            // Update online softmax
            // Accumulate PV
        }
    }

    __syncthreads();
}

// FINAL WRITE: outside all K/V and register-tile loops
for (int j = 0; j < query_tile_size; j++) {
    // Normalize and write O
}
```

In Stage 2B, multiple warps process multiple query rows, so there are two reasonable approaches:
- Shared-memory Q: Cooperatively load the entire query tile into shared memory, then have warps read their assigned query rows.
- Register-resident Q: Each warp loads its own query rows directly into registers, avoiding shared-memory Q altogether.
For your first Stage 2B implementation, I think shared-memory Q is a reasonable educational choice. It makes the data movement explicit and gives you a natural progression toward Tensor Core programming.

K is transposed in shared memory.
- For QKᵀ, each thread should reuse Q and K operands across its register tile. With row-major K, accessing the same feature across multiple keys can produce shared-memory bank conflicts.
- A transposed K layout makes accesses across key positions contiguous, although the exact benefit depends on the lane mapping. The extra column provides padding for some bank-conflict patterns; it is not a universal guarantee of conflict-free accesses.
- V remains row-major because the PV computation accesses features across the head dimension.


# Immplementing TensorCore FlashAttention
- After the General GPU Optimisatiosn, I will stop, I will only implement the Tensor Core Flash Attention Once, My code can work with part_a
- So we will pause here, integrate wth part_a under flash_attn_folder before moving on
- I think my goal here is to try different ways to beat out all the other implementation for Flash Attention on this the rtx 3090

## My analysis 
Fully Fused WMMA FlashAttention
QK^T: Tensor Core WMMA
Softmax: CUDA cores
PV: Tensor Core WMMA
WMMA accumulator lives in registers distributed
across the warp, but its element mapping is opaque.
Store score tile to shared memory for row-wise softmax.
Store probabilities to shared memory for WMMA PV.
No full attention matrix written to HBM.
One warp executes each WMMA operation, but a CUDA thread block can contain multiple warps, each computing different tiles. You aren't limited to one warp per entire FlashAttention kernel.

So the parameters of my flash attention can be the same no?
I get why we need to do 2d register tiling right ? cause if we dont then it because like a its not going to work with the wmma or like mma.sync at all cause when we do thread tiling, 

So in order for wmma to work, 1 warp would be working on that 1 matrix tiles.

How do we derive the ldm (leading dimension of a matrix) ? It is basically the actuall amount of elemented need to traverse before moving down the next row/col depending on the memory format.
ldm represents the actual number of elements (or bytes) you need to skip in memory to move from the start of one row (or column) to the next.

How is LDM determined here ?
Cause our matrix is B x N x S x H


Global Q/K/V
    ↓
Cooperative loading
    ↓
Shared-memory Q/K/V tiles
    ↓
WMMA QKᵀ
    ↓
Online softmax
    ↓
WMMA PV
    ↓
Output


Fully Fused PTX MMA Optimized FlashAttention
QK^T: PTX mma.sync
Softmax: CUDA cores, operating on register values
PV: PTX mma.sync
Explicitly manage accumulator register mapping.
Use warp shuffles / register rearrangements.
Minimize intermediate shared-memory staging.
No full attention matrix written to HBM.

The crucial difference is that PTX provides documented register layouts for supported MMA instructions, allowing you to design softmax around the distributed accumulator. However, PTX does not automatically eliminate shared-memory staging. Preparing the probability fragments for PV can still require data rearrangement, and shared memory may be the most efficient solution.

## Hardware-counter analysis

See the [reproducible profiling workflow](PROFILING.md) for the Python capture/processing scripts, Nsight commands, saved raw reports, and interpretation rules. This analysis is separate from ordinary latency benchmarking and does not integrate Part A.

The completed [measured bottleneck findings](results/nsight_analysis/FINDINGS.md) use 48 validated Nsight captures (8 kernels × 2 shapes × 3 repetitions), with raw reports, Python processing scripts and correlated CUDA/SASS. Hardware-counter measurements are kept separate from the ordinary latency dataset.
