# GPU experiments

The recorded implementation `e2487da` was executed on RTX 5060 Ti: 18/18 tests, all four experiments, three sanitizer tools, and small/large profiling. The full 60-row grid uses the experimentally frozen (16,128) configuration; the library default remains (16,256). See [actual results and commands](../results/README.md). Analysis coverage recognizes the single frozen configuration in each timing run.

## Method

Use identical FP32 Q/K/V inputs, layout, shape and mode for every implementation. `synthetic_input` uses SplitMix64, seed 5206 and fixed FP32 conversion. It generates Q, then K, then V. Values lie in `[-1,1)`. Results record the generator, seed and input fingerprint. Parts B/C should reuse this input contract and the FP64 oracle.

Each selected implementation/configuration passes representative CPU S/P/O checks before measurement. Timed shapes with `N<=257` also receive exact-shape checks before and after timing. Larger shapes use the representative gate and finite-output checks; this scope is recorded explicitly.

Preallocate Q/K/V/S/P/O, upload once, warm up 5 calls and collect 20 repeats by default. CUDA events measure complete forward latency and a separate QK/softmax/PV breakdown. Timing excludes allocation, copies, input generation, CPU validation and file I/O. Buffers remain resident without cache flushing. Stage medians need not sum to the separately measured forward median.

Report median and IQR using linear interpolation at `(n-1)*p`. For median `L` milliseconds, sequence throughput is `1000*B/L`, token throughput is `1000*B*N/L`, and speedup is matched naive latency divided by variant latency. Dense and causal comparisons stay separate.

## Run

From the Part A root in an x64 Visual Studio developer PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1
```

This performs one full test run and a representative benchmark. `-SkipTests` is available for further experiments on the same validated build. Each benchmark still runs its CPU correctness gate. Avoid concurrent GPU workloads and record power/clock state. The script does not change GPU settings or upload results.

Bounded tuning changes one parameter at a time:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid tuning -Configs study
```

Candidates are `(tile,softmax_threads)=(8,256),(16,256),(32,256),(16,128)`. Tuning shapes are `(4,8,256,64)` and `(4,8,1024,64)`, in both modes. The measured study selected (16,128), which was frozen before the main grid. The following commands reproduce that configuration.

The proposal grid measures naive and optimized on all required shapes, producing 60 rows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid proposal -Tile 16 -SoftmaxThreads 128
```

The grid uses `B=4`, `H=8`, `N={256,512,1024,2048,4096}`, `d={32,64,128}`, FP32 and both modes. Representative ablation measures all four variants:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_gpu_experiments.ps1 -Grid ablation -Tile 16 -SoftmaxThreads 128 -SkipTests
```

A direct custom run is also supported:

```powershell
.\build-gpu\benchmarks\Release\attention_benchmark.exe --shape 4 8 256 64 --variants main --tile 16 --softmax-threads 128 --output results/custom-run
```

`--variants main` selects naive and optimized. `all` selects all four; individual variants are also available. `--validate-only` runs correctness without timing. `--profile-only` warms and runs a forward without collecting latency samples. Output directories must be new. Memory failures are recorded; dimensions are never silently reduced.

Cases exceeding the workspace memory budget are marked `NOT_RUN_MEMORY`; other shapes are still attempted. A partial grid returns a nonzero exit code. The analysis command accepts complete normal timing runs only.

## Output

The runner saves environment and build/test logs under `results/gpu-benchmarks/<run-id>/`. Its `benchmark/` subdirectory contains:

- `metadata.txt`: source/build identity, hardware and measurement settings
- `status.txt`: completion state
- `plan.csv`: intended configuration rows
- `correctness.csv`: stage errors and tolerance outcomes
- `samples.csv`: raw forward and stage samples
- `summary.csv`: latency, throughput, speedup, setup time and workspace bytes
- `resources.csv`: launch size, registers, shared/local memory and theoretical occupancy
- `case_status.csv`: completed, profiled or memory-limited cases
- `failure.txt`: error and interrupted case when a run fails

Peak workspace allocation is `16*B*H*N*d+8*B*H*N*N` bytes. It covers the six simultaneous device buffers, excluding runtime/context overhead. It is distinct from measured DRAM traffic. Setup host time is reported separately.

## Profiling and plots

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile_gpu.ps1 -N 256 -Tile 16 -SoftmaxThreads 128
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile_gpu.ps1 -N 2048 -Tile 16 -SoftmaxThreads 128
python benchmarks/analyze_results.py results/gpu-benchmarks/RUN/benchmark --output results/gpu-benchmarks/RUN/analysis
```

Profiling requires a compatible Nsight Compute CLI and permission to access counters. It uses a separate profile build and saves logs, counter CSVs and the report. Inspect registers, shared memory, occupancy, traffic and bottlenecks. Profiler durations are not normal latency samples. Plotting requires Python and matplotlib. The analysis command checks measured samples and complete representative/exact S/P/O report coverage, writes an English summary and creates charts. It rejects duplicate reports and inconsistent correctness fingerprints. It rejects incomplete normal benchmark runs.

Run analysis regression checks with `python benchmarks/tests/analyzer_tests.py`.
After changing the workspace API, rebuild the CUDA target and run CTest.
Variant tests cover workspace state and recovery.
