# Part A: CPU reference and GPU attention baseline

Part A implements forward scaled dot-product attention in FP32, with an independent FP64 CPU reference. Both dense and causal modes are supported.

## Implementation

- `common/`: tensor shapes, deterministic FP32 inputs and input fingerprints
- `cpu_reference/`: materialized FP64 attention
- `gpu_baseline/`: separate QK, softmax and PV kernels, including tiled matrix multiplication and parallel reduction
- `validation/`: absolute-plus-relative error comparison
- `benchmarks/`: correctness checks, CUDA timing, statistics and plots
- `scripts/`: Windows experiment and profiling commands
- `results/`: measured data and environment records

Q/K/V/O use contiguous `[B,H,N,d]` storage. S/P use `[B,H,N,N]` storage. The computation is `S=QK^T/sqrt(d)`, `P=softmax(S)` and `O=PV`. Causal attention includes the diagonal and masks `j>i`.

## Build and test

Requires CMake 3.18+, C++17 and a compatible compiler. GPU builds also require an NVIDIA GPU and a compatible CUDA Toolkit. For RTX 5060 Ti, use a Toolkit supporting `sm_120` and its supported MSVC toolset.

CPU example with MinGW:

```powershell
cmake -S . -B build-cpu -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

GPU example from an x64 Visual Studio 2022 developer shell:

```powershell
cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64 -DPART_A_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-gpu --config Release
ctest --test-dir build-gpu -C Release --output-on-failure
```

Use separate build directories for different toolchains. Set `CMAKE_CUDA_ARCHITECTURES` for the actual GPU. CPU builds register 13 test groups. CUDA builds add four variant groups and one configuration-validation group, for 18 total. All numerical and boundary cases remain inside these groups.

## Experiments

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1
```

This builds, tests and measures representative cases. The main proposal grid uses `B=4`, `H=8`, `N={256,512,1024,2048,4096}`, `d={32,64,128}`, FP32 and both modes. See [benchmark commands](benchmarks/README.md) and [measured results](results/README.md).

The default tile is 16 and the parallel softmax block has 256 threads. Input generation, allocation, copies and CPU validation are outside GPU timing. Backward attention, KV-cached decoding, mixed precision and fused FlashAttention are outside this Part A implementation. The measured implementation from `e2487da` passed 18/18 test groups, three sanitizer tools, and both profiling runs. The full 60-row main grid was measured using the frozen experimental configuration (16,128), with representative, tuning and ablation data.

The [Release validation](results/validation/2026-10-15T052608Z-fa1f796/README.md) of `fa1f796` passed 18 CTest groups, 7 analysis tests and the representative benchmark at (16,128). Full-grid and profile measurements remain from `e2487da`.
