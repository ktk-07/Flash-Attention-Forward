# Original causal FlashAttention coding file

[`causal_flash_attn.cu`](causal_flash_attn.cu) was the original file used to develop the kernels and record the coding/thought-process comments, before splitting the implementations and integrating a full test and benchmark harness.

It is retained here as a standalone learning file. The original comments are preserved; the unfinished `main()` has been replaced with a minimal runnable correctness check. Missing definitions and the previously discovered Tensor Core fixes have also been carried over. The three maintained implementations used by CMake remain one directory above this folder.

## Quick compile and run

From this directory:

```bash
nvcc causal_flash_attn.cu -o ./test && ./test
```

From the repository root:

```bash
cd part_c/src/extra
nvcc causal_flash_attn.cu -o ./test && ./test
```

Requires the CUDA Toolkit and access to an NVIDIA GPU. No CMake, Python, project headers, or other source files are required.

## What the minimal test does

Uses deterministic FP32 Q/K/V with `B=1`, `H=1`, `N=33`, `d=32`, including a partial sequence tile. It compares every output element against an independent CPU implementation with FP64 accumulation and numerically stable causal softmax:

- FA1-style scalar kernel.
- FA2 scalar kernel.
- FA2 warp-per-query kernel.
- FA2 register-tiled CUDA-core kernel.

Each kernel prints `PASS` or `FAIL` and its maximum absolute error. The element-wise tolerance is `abs(error) <= 1e-4 + 1e-4 * abs(reference)`. Missing stores/nonfinite outputs and CUDA errors fail the test. Exit codes: `0` for all passing, `2` for a numerical failure, `1` for a CUDA/runtime error.

The Tensor Core learning implementations remain in the source, with architecture guards so the simple default compile works. This minimal test **does not launch Tensor Core kernels**, measure performance, write CSVs, or generate plots. Use `part_c/run.sh` for the complete maintained workflow.

The original pre-fix hash and a record of changes are retained under [`../../docs/kernel_provenance/`](../../docs/kernel_provenance/). Historical result logs still contain the file's previous `src/causal_flash_attn.cu` path.

Verified locally: the plain `nvcc` build and all four checks passed. The largest observed absolute error was `5.68773e-08`. Build warnings, commands and test output are saved in [`../../results/original_smoke/`](../../results/original_smoke/).
