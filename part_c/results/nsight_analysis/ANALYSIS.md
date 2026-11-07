# Reproducible Nsight Compute analysis

48 captured kernels: all eight Part C variants, B=4,H=8,d=64,N=256/4096, 3 independent profiles each. No Part A integration. All captures passed sampled CPU-FP64 validation before profiling. Full small-case correctness validation is also saved.

## Measurement contract

Each process performs one validated kernel launch, five warmups, then one target launch. Nsight uses `--launch-skip 6 --launch-count 1`. The exporter verifies the captured grid and the validation record. The `full` section set collects scheduler, occupancy, memory, source/branch, instruction, and roofline information. Explicit FP32 FADD/FMUL/FFMA counters are added. Raw `.ncu-rep`, raw-metric CSV, detail/rule CSV, all commands, environment, snapshots, and SHA256 hashes are retained.

`--cache-control all` flushes caches between replay passes. `--clock-control none` leaves GPU clocks unmanaged; measured frequencies and repeat ranges are saved. Profiler durations are not substituted for the normal warm-cache CUDA-event benchmark. Profile repeats measure counter/sampling variability; they do not make individual counter values bitwise deterministic.

Stall percentages below use each state's samples divided by the sum of the all-samples warp-state counts, including selected/issuing and not-selected warps. They are not percentages of kernel wall time. Missing metrics remain unavailable, never fabricated as zero.

## Occupancy, scheduling, and memory

| N | Kernel | Theoretical / achieved occupancy % | Eligible warps / scheduler | Registers/thread | Static shared KiB | DRAM MiB | Local L1 sector MiB | Uniform branches % | Predicated-on threads/warp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | fa1 | 25 / 7.25 | 0.05 | 154 | 0 | 9.56 | 9.22 | 93.9 | 28.5 |
| 256 | scalar | 33.3 / 7.35 | 0.12 | 64 | 0 | 31.8 | 572 | 84.9 | 28.5 |
| 256 | warp | 41.7 / 40 | 0.46 | 72 | 16 | 34.1 | 0 | 99.8 | 29.1 |
| 256 | tiled | 33.3 / 25.3 | 0.48 | 69 | 24 | 41.4 | 578 | 100 | 30.6 |
| 256 | skip | 33.3 / 25.1 | 0.48 | 69 | 24 | 41.7 | 578 | 100 | 30.6 |
| 256 | vector | 33.3 / 25.3 | 0.47 | 55 | 24 | 42 | 578 | 98.3 | 30.2 |
| 256 | wmma | 12.5 / 10.3 | 0.14 | 40 | 25.4 | 4.66 | 0 | 98.5 | 30.6 |
| 256 | ptx | 29.2 / 14.9 | 0.2 | 64 | 13 | 4.67 | 0 | 98.9 | 30.7 |
| 4096 | fa1 | 25 / 23.4 | 0.08 | 154 | 0 | 8.83e+03 | 2.07e+03 | 99.5 | 31.6 |
| 4096 | scalar | 33.3 / 30.3 | 0.2 | 64 | 0 | 1.81e+04 | 1.41e+05 | 98.9 | 31.4 |
| 4096 | warp | 41.7 / 41.6 | 0.36 | 72 | 16 | 5.84e+04 | 0 | 100 | 29.5 |
| 4096 | tiled | 33.3 / 32.5 | 0.63 | 69 | 24 | 1.23e+04 | 1.31e+05 | 100 | 30.6 |
| 4096 | skip | 33.3 / 32.6 | 0.63 | 69 | 24 | 1.23e+04 | 1.31e+05 | 100 | 30.6 |
| 4096 | vector | 33.3 / 32.5 | 0.61 | 55 | 24 | 1.24e+04 | 1.31e+05 | 98.5 | 30.2 |
| 4096 | wmma | 12.5 / 11.9 | 0.14 | 40 | 25.4 | 642 | 0 | 99.8 | 30.7 |
| 4096 | ptx | 29.2 / 27 | 0.38 | 64 | 13 | 752 | 0 | 99.9 | 30.8 |

## Warp-state evidence and bank conflicts

| N | Kernel | MIO throttle % | Short scoreboard % | Long scoreboard % | Barrier % | Math-pipe throttle % | Shared load/store conflicts |
|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | fa1 | 0 | 0.032 | 83 | 0 | 0.00458 | 0 / 0 |
| 256 | scalar | 0 | 0.0164 | 52.9 | 0 | 0.0861 | 0 / 0 |
| 256 | warp | 19.2 | 11.6 | 40.5 | 4.15 | 0.683 | 1.47e+07 / 1.96e+05 |
| 256 | tiled | 1.62 | 30.6 | 19.9 | 2.51 | 0.327 | 4.34e+05 / 9.03e+03 |
| 256 | skip | 1.72 | 30.3 | 20.2 | 2.61 | 0.353 | 4.33e+05 / 9.02e+03 |
| 256 | vector | 0.609 | 32.7 | 19.5 | 2.6 | 0.506 | 4.56e+05 / 7.69e+03 |
| 256 | wmma | 0.675 | 34.1 | 34.1 | 0.579 | 1.35 | 5.1e+06 / 2.51e+06 |
| 256 | ptx | 1.71 | 12.4 | 54.7 | 0.689 | 0.73 | 1.49e+06 / 3.94e+04 |
| 4096 | fa1 | 0.000245 | 0.00387 | 66.7 | 0 | 0.051 | 0 / 0 |
| 4096 | scalar | 0.000381 | 0.00616 | 56.7 | 0 | 0.508 | 0 / 0 |
| 4096 | warp | 7.44 | 6.54 | 62.8 | 2.6 | 0.562 | 3.76e+09 / 3.9e+07 |
| 4096 | tiled | 2.58 | 30.4 | 23 | 2.12 | 0.417 | 1.27e+08 / 1.71e+06 |
| 4096 | skip | 2.6 | 30.5 | 22.9 | 2.1 | 0.419 | 1.27e+08 / 1.71e+06 |
| 4096 | vector | 0.989 | 32.3 | 22.8 | 2.13 | 0.649 | 1.34e+08 / 1.47e+06 |
| 4096 | wmma | 0.599 | 31.7 | 42.8 | 0.648 | 1.37 | 1.17e+09 / 5.76e+08 |
| 4096 | ptx | 1.68 | 11.6 | 55.7 | 0.946 | 1.53 | 3.42e+08 / 9.2e+06 |

## Arithmetic intensity and operation counts

Useful attention work is `4*B*H*d*N*(N+1)/2` FLOPs (causal QK plus PV, excluding softmax). Dividing it by measured DRAM read+write bytes gives an algorithm-useful DRAM intensity. This includes the effects of caching and spills in measured traffic and does not count masked/padded work. It is not the executed Tensor Core FLOP count or a substitute for the appropriate hardware roofline in the raw report. Measured scalar FP32 operations are `2*FFMA + FADD + FMUL`; for Tensor Core kernels these exclude matrix operations. Compare intensity at the same memory level.

## Reading the results and choosing experiments

1. **Global-memory reuse:** compare `fa1` with `tiled/skip` at matching shapes. Shared Q/K/V staging and register reuse change the source-level access pattern; use DRAM and L1/local traffic plus issue metrics to determine which layer improved. A lower DRAM byte count by itself does not prove that bandwidth was limiting.

2. **MIO and shared memory:** MIO throttle means pressure on the MIO instruction path, which also serves special math and dynamic branching. Short scoreboard signals dependencies that can involve shared memory but is not proof of bank conflicts. Check actual shared conflict counts and wavefronts before proposing swizzling. Reducing redundant shared-memory operations/register reuse is a separate hypothesis from eliminating bank conflicts.

3. **Latency hiding:** achieved occupancy is the measured resident-warp fraction; theoretical occupancy is the launch/resource ceiling. Neither equals utilisation or eligible warps. Inspect the register/shared/block/warp limits and eligible warps together. Higher occupancy alone is not a performance objective.

4. **Local traffic:** local load/store sectors indicate thread-local memory accesses, including spilled or dynamically indexed arrays. They do not all reach DRAM. Correlate with compiler stack/local resources and SASS/source lines before attributing every local transaction to register spilling. Scalar/register-array indexing is a concrete candidate for compile-time unrolling/specialisation experiments.

5. **Divergence and predication:** uniform-branch percentage only describes branch behaviour. Predicated-off lanes can waste work even with 100% uniform branches. Compare active and predicated-on threads per warp, the causal/tail guards, and grid size. A causal mask does not automatically imply that divergence is the bottleneck.

6. **Vectorisation:** compare `skip/tiled` against `vector` using global load instruction/sector counts, local traffic, resources, and issue rates. Fewer load instructions or 128-bit loads need not shorten the critical path. The current fixed launch mapping may leave loader lanes idle; an alternative cooperative loader is an experiment, not an established improvement.

7. **WMMA versus PTX:** compare shared traffic/wavefronts, scoreboard/MIO states, local traffic, and tensor utilisation. PTX keeps score/output state in registers and avoids some WMMA shared-memory round trips, but occupancy/register trade-offs must be measured. FP16-input Tensor Core results must not be presented as a precision-isolated speedup over FP32 CUDA-core kernels.

8. **Double buffering:** consider only after identifying a useful load/compute overlap opportunity. Additional K/V buffers consume shared memory and may lower occupancy. Swizzling and double buffering are not implemented by this profiling workflow.

These are observation-driven experiment candidates, not proven causes or promises of speedup. Establish causality with one-change variants, revalidation, ordinary CUDA-event timing, and repeat profiling. Existing kernel progression versions can change several features simultaneously.

## Reproduce

```bash
bash part_c/profile.sh --output part_c/results/my_profile --build /tmp/part_c-profile-build --repeats 3
# Regenerate all tables, plots and this report without accessing the GPU:
python3 part_c/scripts/analyze_profiles.py /home/buntr/Projects/Flash-Attention-CUDA/part_c/results/nsight_analysis
```

Use a fresh output directory. `--resume` requires matching source hashes and the same plan. Build/runtime/tool versions and GPU state are in `logs/`; exact argv, working directory, UTC times and exit codes are in `commands.jsonl`. Source snapshots are under `source_snapshot/`. The initial capability probe is separate and is not used as a representative workload.

## Artifacts

- `cases/*/profile.ncu-rep`: open in Nsight Compute GUI for source/SASS correlation and rooflines.
- `cases/*/raw.csv` and `details.csv`: original exports.
- `all_metrics.csv`: lossless long-form metric values and units.
- `metric_samples.csv`: derived per-profile values and sampling denominators.
- `metric_summary.csv`: medians and min/max across independent profiles.
- `rules.csv`: Nsight advisory text, preserved as suggestions rather than proof.
- `unavailable_metrics.csv`: explicit metric availability.
- `figures/`: occupancy, scheduler eligibility, lane activity, warp states, memory traffic, and useful DRAM intensity (PNG/PDF).

Definitions follow NVIDIA's [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) and [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
