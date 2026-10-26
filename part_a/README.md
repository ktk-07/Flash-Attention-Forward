# Part A: CPU reference and GPU baselines

Part A implements materialized FP32 attention with an independent FP64 CPU reference. Dense and causal modes are supported.

## Implementation

- `common/`: tensor shapes and deterministic FP32 Q/K/V inputs
- `cpu_reference/`: straightforward FP64 QK, row-wise softmax and PV
- `gpu_baseline/`: naive and optimized CUDA kernels with reusable device buffers
- `validation/`: absolute-plus-relative error comparison
- `benchmarks/`: correctness comparisons, CUDA timing, statistics and plotting
- `scripts/`: experiment and profiling commands
- `results/`: selected measurements and their interpretation

Q/K/V/O use `[B,H,N,d]`; S/P use `[B,H,N,N]`. The computation is `S=QK^T/sqrt(d)`, `P=softmax(S)`, `O=PV`. Causal attention includes the diagonal and masks `j>i`.

| Variant | QK and PV | Softmax |
| --- | --- | --- |
| naive | One output element per thread | One row per thread |
| tiled | Shared-memory tiles | One row per thread |
| parallel | One output element per thread | Block reduction |
| optimized | Shared-memory tiles | Block reduction |

The default tile is 16 and softmax uses 256 threads. Recorded main-grid measurements explicitly use (16,128), selected from a bounded tuning study.

## Build and run

Requires CMake 3.18+ and C++17. CUDA builds also need a compatible CUDA Toolkit and host compiler. Run from `part_a/`.

CPU library with MinGW:

```powershell
cmake -S . -B build-cpu -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
```

GPU build from an x64 Visual Studio 2022 developer shell:

```powershell
cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64 -DPART_A_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-gpu --config Release
```

Set the architecture for the actual GPU; 120 is used for RTX 5060 Ti.

Check all variants and candidate configurations against the CPU reference:

```powershell
.\build-gpu\benchmarks\Release\attention_benchmark.exe --validate-only --variants all --configs study --output results/local/correctness-run
```

The output directory must be new. CPU S/P/O comparisons use identical FP32 inputs, FP64 arithmetic and `atol=rtol=1e-4`. Representative shapes include nonmultiples and tile boundaries. Timed shapes with N<=257 receive exact-shape comparisons before and after measurement; larger shapes use representative comparisons and finite-output checks.

## Experiments

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid tuning -Configs study
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid proposal -Tile 16 -SoftmaxThreads 128
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid ablation -Tile 16 -SoftmaxThreads 128
```

The main grid uses B=4, H=8, N={256,512,1024,2048,4096}, d={32,64,128}, FP32 and both modes. It compares naive and optimized implementations in 60 rows. Ablation measures all four variants at N=256/1024,d=64. Tuning compares (8,256), (16,256), (32,256) and (16,128); select one configuration before the main grid.

Buffers are preallocated and uploaded once. CUDA events measure full forward latency and separate QK/softmax/PV passes. Five warmups and 20 repeats are used by default. Allocation, copies, CPU comparisons and file I/O are outside timing. CSV files report median, IQR, throughput, workspace allocation and kernel resources.

Profile small and large dense cases, separately from normal timing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile_gpu.ps1 -N 256 -Tile 16 -SoftmaxThreads 128
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile_gpu.ps1 -N 2048 -Tile 16 -SoftmaxThreads 128
```

Profiling requires Nsight Compute and access to GPU counters. Plot a measured summary with Python and matplotlib:

```powershell
python benchmarks/plot_results.py results/gpu-benchmarks/RUN/benchmark/summary.csv --output results/local/figures
```

See [measured results](results/README.md). This component provides materialized baselines; fused attention belongs to Parts B/C. Finite inputs must keep FP32 products and accumulations representable. A workspace result can be downloaded only after a successful forward execution.
