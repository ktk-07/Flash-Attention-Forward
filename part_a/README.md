# Part A

CPU FP64 reference and materialized GPU attention baselines for dense and causal modes. ChatGPT Codex assisted development.

## Common-machine integration

Integration is maintained by Kyaw Tun Khine. The materialised GPU kernels remain unchanged. From the repository root:

```bash
bash part_a/run.sh --grid smoke --output results/a_smoke
bash part_a/run.sh --output results/a_full
bash part_a/run.sh --analyze-only results/a_full
bash run.sh --parts a,c --output results/ac_full
```

The wrapper builds CUDA and the reusable `attention_reference` library, runs the shared CPU contract, full small-shape dense/causal S/P/O checks, deterministic pattern tests and representative Compute Sanitizer checks. It tunes optimized FP32 and FP16 Tensor Core configurations separately per mode/head dimension, freezes before the main grid, validates every timing case and exports the shared seven CSVs. Full runs also collect fresh stage-specific Nsight metrics; `--no-profile` records their omission. Architecture defaults to GPU detection (SM80+); `--arch`, `--device`, `--build` and `--dry-run` are supported. See the main [README](../README.md) for the common schema and Part B handoff.

Native data and source snapshots stay under `raw/part_a/`, public tables at the bundle root, and plotted data/PNG/PDF figures under `plots/part_a/`. Historical `results/` below remain separate RTX 5060 Ti evidence.

## Build and run

Requires CMake 3.18+, C++17, CUDA and Python with Matplotlib for figures. Saved measurements use an RTX 5060 Ti with CUDA architecture 120.

```powershell
cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64 -DPART_A_ENABLE_CUDA=ON -DPART_A_PROFILE_INFO=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-gpu --config Release
.\build-gpu\Release\attention.exe --grid correctness --output build-gpu/correctness
.\build-gpu\Release\attention.exe --shape 4 8 256 64 --mode both --tile 16 --softmax-threads 128 --output build-gpu/representative
```

Each run writes to a new output directory. Build directories keep new run records separate from the saved results. For CPU-only execution, configure without CUDA and run `attention --cpu --shape 1 1 2 2`.

## Implementation

Q/K/V/O use row-major `[B,H,N,d]`; S/P use `[B,H,N,N]`. Each forward pass runs separate QK, softmax and PV kernels.

- `naive`: one score/output element per thread and one softmax row per thread
- `tiled`: shared-memory matrix multiplication with naive softmax
- `parallel`: naive matrix multiplication with block-parallel softmax
- `optimized`: tiled matrix multiplication and block-parallel softmax
- `tensorcore`: FP16 Q/K/V/S/P/O, WMMA 16x16x16 multiplication with FP16 accumulators, and CUDA half-precision softmax

`main` selects naive and optimized; `all` selects the four FP32 variants. The Tensor Core extension changes precision and matrix hardware together. Its CPU reference reads the same FP16-rounded inputs.

Small shapes compare S/P/O against FP64. Every new performance case validates O against the shared output-only FP64 oracle: full rows for N<=97, otherwise queries 0,N/2,N−1 per head plus all-output finiteness checks. The saved historical measurements retain their original validation scope. The criterion is `abs(error)<=atol+rtol*abs(reference)`: FP32 uses atol=rtol=0.0001; FP16 uses atol=0.002 and rtol=0.02.

Seed=5206, 5 warmups, 20 repeats. CUDA events measure forward and stage times in separate passes, excluding input preparation, allocation, transfers and comparison. Results report medians, IQR, sequences/s and allocated Q/K/V/S/P/O bytes.

## Results

CSV tables use `experiment` and `precision` to distinguish runs.

| File | Contents |
|---|---|
| `results/correctness.csv` | FP32/FP16 small-shape S/P/O comparisons |
| `results/samples.csv` | Individual timing samples for the main grid, tuning and ablation |
| `results/summary.csv` | Timing statistics, errors, throughput and allocated workspace |
| `results/profiles.csv` | Measured DRAM traffic, bandwidth, registers, shared memory and occupancy |
| `results/resources.csv` | Launch settings and kernel resource usage |
| `results/environment.txt` | Environment, configurations, tuning selection and rerun details |
| `results/figures/` | Six report figures |

The main grid uses B=4, H=8, N=256/512/1024/2048/4096 and d=32/64/128 in both modes. The frozen configuration is tile=16, softmax threads=128.

The saved data contains 330 passing S/P/O records, 122 performance configurations, 2440 timing samples and 12 profiling cases. FP32 optimized is 3.18-6.96x faster than naive. FP16 Tensor Core is 1.27-1.78x faster than FP32 optimized, with a geometric mean of 1.42x and half the allocated workspace. The largest recorded main-grid FP16 output absolute error is 0.000919.

Six FP16 performance cases were repeated; their complete 20-sample runs replaced the earlier timings. Forward IQR/median is at most 5.30% for FP32 and 7.19% for FP16. Saved output errors come from the original numerical validation. These historical results are preserved. The final shared A/B/C experiment must be collected anew on one machine.

```powershell
python -m pip install matplotlib
python plot_results.py
```

Use `python plot_results.py --select-config` to print the best FP32 tuning configuration. Plotting reads `results/summary.csv` and writes only to `results/figures/`.
