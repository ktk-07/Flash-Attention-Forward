# Part A

CPU FP64 reference and materialised CUDA attention baselines for dense and causal attention.

## Implementations

- `naive`: direct FP32 products and one thread per softmax row
- `tiled`: shared-memory FP32 products
- `parallel`: block-parallel FP32 softmax
- `optimized`: tiled products and parallel softmax
- `wmma`: FP16 matrix operands with WMMA 16x16x16 products
- `ptx`: FP16 matrix operands with direct PTX m16n8k16 products

Both Tensor Core variants use FP16 Q/K/V/P and FP32 S, accumulation, softmax arithmetic and O. A rounds normalized P; Part C rounds tile weights before online normalization. Compare matching WMMA/PTX variants and state this rounding-order difference.

## Run

Requires CMake, a C++17 compiler, CUDA, Compute Sanitizer, Nsight Compute, NumPy and Matplotlib. Tensor Core paths require SM80 or newer. From the repository root:

```bash
python3 part_a/scripts/run_standard.py --output build-a-run --publish part_a/results
```

Choose a fresh output directory. On Windows, use an x64 Visual Studio 2022 developer terminal, replace `python3` with `python`, and add `--arch 120 --build build-a`. CUDA and profiling tools must be on PATH.

The full run checks small S/P/O, tunes bounded configurations and records the proposal grid with 5 warmups and 20 repetitions. Preparation, transfers, comparison and profiling are outside the CUDA-event timings. Forward and stage timings use separate passes.

For a common-machine A/C evaluation: `bash run.sh --parts a,c --output results/ac_full`. Part B can join after integration.

## Results

The saved RTX 5060 Ti run completed 180 main configurations, 96 tuning configurations and 32 input-pattern configurations. All 728 correctness records passed. The saved data contains 6160 timing samples and 108 kernel profiling captures. All final configurations use tile=16 and 128 softmax threads. See `results/environment.txt` for the environment and selection method.

Small shapes and N=256,d=64 compare full output against FP64. Other large cases compare queries 0,N/2,N-1 per head and check every output for finiteness. FP16-input references use rounded Q/K/V. FP32 tolerances are 1e-4 absolute/relative; mixed precision uses 0.002 absolute and 0.02 relative.

`results/` contains seven CSVs: correctness, samples, summary, profiles, resources, memory and work, plus environment settings and report figures. Profiling retains selected report counters with their original names; readers accept old and new Nsight DRAM names. Full captures and diagnostic output stay in the local run directory. Saved CSVs use ordinary Git files; Git LFS is not required.

For E=BHNd and M=BHN^2, allocated device buffers occupy 16E+8M bytes for FP32 and 10E+6M bytes for the Tensor Core paths. Allocated bytes and profiler memory traffic are separate measurements.

Regenerate saved figures without CUDA:

```bash
python3 part_a/plot_results.py
```

Use `--analyze-only build-a-run --publish part_a/results` with `run_standard.py` to re-export a complete local run. `--resume-profiles build-a-run --build BUILD --publish part_a/results` resumes missing profiler work without repeating timings. Both compressed and uncompressed Nsight reports are supported.
