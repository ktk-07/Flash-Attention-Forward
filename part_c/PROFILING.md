# Reproducible kernel profiling

The ordinary benchmarks establish latency. This separate workflow collects hardware evidence for why variants differ: occupancy/resource limits, eligible warps, issue rates, warp states, branch uniformity and predication, DRAM/local/shared traffic, bank conflicts, scalar operations, and Tensor Core activity.

The completed common-machine experiment retains 24 Part C captures under `../results/final_submission/parts/part_c/`: FP32 FA1, PTX FA1, the frozen CUDA-core FA2 winner and PTX FA2 at d64, N256/4096, with three repetitions. Generated counters and figures are retained in the child bundle and `../results/final_submission/supplementary/part_c/`. See [verification](../results/final_submission/VERIFICATION.md) for coverage and unavailable metrics. The optional all-variant workflow below has a broader capture plan and is not required to reproduce the completed experiment's figures.

## Capture and process

From the repository root, with CUDA, Nsight Compute, CMake, a C++ compiler, Python 3.12, and an accessible NVIDIA GPU:

```bash
# Install plotting dependencies in your chosen environment if needed.
python3 -m pip install -r part_c/requirements-analysis.txt

# Fresh output directory; builds and validates before collecting counters.
bash part_c/profile.sh --output part_c/results/my_profile --build /tmp/part_c-profile-build --repeats 3

# Additional source/SASS correlation, using saved reports and source snapshots.
python3 part_c/scripts/analyze_source.py part_c/results/my_profile
python3 part_c/scripts/interpret_profiles.py part_c/results/my_profile
```

`profile_kernels.py` drives all nine maintained kernel variants (including both FA1 baselines; 54 captures) at `B=4,H=8,d=64,N=256/4096`, with three independent captures per kernel/shape. The default uses SM86, matching the local Ampere GPU. This is a representative small/large study, not exhaustive profiling of all benchmark shapes.

The existing full small-case correctness suite runs first. Each capture also checks query rows `0,N/2,N-1` in every batch/head against CPU FP64 attention and checks all outputs for finiteness. Tensor Core references use identical FP16-rounded inputs.

The `profile` runner mode has exactly seven kernel launches: one validation, five warmups, and one target. Nsight uses `--launch-skip 6 --launch-count 1`; the processor verifies that exactly one capture with the expected grid was exported. No benchmark medians are derived from this profiling mode.

## Python scripts

| Script | Role |
|---|---|
| [`scripts/profile_kernels.py`](scripts/profile_kernels.py) | Builds, validates, captures Nsight counters, exports raw/details CSVs, and invokes aggregate analysis. |
| [`scripts/analyze_profiles.py`](scripts/analyze_profiles.py) | Normalises raw metrics, checks capture identity/validation, computes medians/ranges, generates figures and `ANALYSIS.md`. |
| [`scripts/analyze_source.py`](scripts/analyze_source.py) | Exports CUDA/SASS correlation for repetition 0 of each case; deduplicates source mappings by PC and identifies sampled stall PCs/operation groups. |
| [`scripts/interpret_profiles.py`](scripts/interpret_profiles.py) | Generates workload-specific findings from aggregate metrics and source hotspots, including LSU/FMA pipeline comparisons and candidate experiments. |

To regenerate analysis without running kernels or accessing the GPU:

```bash
python3 part_c/scripts/analyze_profiles.py part_c/results/my_profile
python3 part_c/scripts/analyze_source.py part_c/results/my_profile
python3 part_c/scripts/interpret_profiles.py part_c/results/my_profile
```

Aggregate analysis requires Python/Matplotlib and the saved CSVs. Source correlation additionally needs the Nsight Compute CLI to read `.ncu-rep` files; it does not recollect GPU counters.

## Saved evidence

- `plan.json`: fixed workloads, seeds, repetitions, profiling options, and SHA256 hashes of the build/analysis sources.
- `source_snapshot/`: source files used for the run, including CMake, headers, kernels, runner, and processing scripts.
- `binary.json`: executable path and SHA256; tool/compiler/GPU details are in `logs/`.
- `commands.jsonl`: exact argument arrays, working directory, UTC timestamps and exit codes. `invocation.txt` records the workflow command.
- `cases/*/profile.ncu-rep`: raw reports for Nsight Compute GUI, source/SASS investigation, and collected roofline sections.
- `cases/*/{raw,details}.csv`: original exports, with units and profiler suggestions preserved.
- `metric_samples.csv`, `metric_summary.csv`, `all_metrics.csv`: per-capture data, medians/min/max, and complete long-form metric exports.
- `rules.csv`, `unavailable_metrics.csv`: advisory text and explicit missing metrics. Missing counters are not replaced by zero.
- `figures/`: occupancy, scheduler eligibility, predicated-on lanes, warp states, memory traffic, and useful DRAM intensity (PDF and PNG).
- `source_analysis/`: correlated CUDA/SASS exports, hotspot tables, operation groups, processing script/hash, and `SOURCE_ANALYSIS.md`.

The recorded run is under [`results/nsight_analysis/`](results/nsight_analysis/): start with [FINDINGS.md](results/nsight_analysis/FINDINGS.md), then the [complete tables](results/nsight_analysis/ANALYSIS.md) and [source/SASS report](results/nsight_analysis/source_analysis/SOURCE_ANALYSIS.md). Python/library versions are recorded in `python_environment.json`.

Use a new directory for a new experiment. `--resume` skips completed captures only when the source hashes and plan match. Resume on the same GPU/toolchain; use a fresh directory after hardware or software changes. Preserve original reports and logs when publishing an analysis.

## Interpretation rules

1. **Profiler versus benchmark timing:** counter replay and instrumentation can alter execution. Nsight uses `--cache-control all` for controlled cache flushing and `--clock-control none` because clocks are unmanaged here. Keep measured frequencies/repetition ranges, and use the ordinary CUDA-event dataset for reported speedups. Warm-cache benchmark behaviour and cold-cache profiling traffic are different experimental conditions.
2. **Occupancy:** distinguish the theoretical launch/resource ceiling, achieved resident warps, eligible warps, and issue rate. Occupancy is not GPU utilisation, and a higher value alone does not establish an improvement.
3. **MIO:** MIO throttle can involve shared-memory instructions, special math, and dynamic branching. Short-scoreboard dependencies do not prove bank conflicts. Use source-level actual minus ideal shared wavefronts to diagnose layout conflicts. Aggregate L1 conflict counters can include arbitration. The historical tiled/skip/vector captures have zero source-level excess; WMMA/PTX/warp have positive excess.
4. **Divergence:** branch uniformity is different from useful lane activity. Predicated-off threads and masked work can exist even when branches are uniform.
5. **Local memory:** local sectors can come from spills or dynamically indexed thread-local arrays. They are not allocated workspace and do not all reach DRAM. Inspect actual instructions and compiler resources before assigning a cause.
6. **Arithmetic intensity:** useful attention work is `4*B*H*d*N*(N+1)/2` FLOPs, excluding softmax. Dividing by measured DRAM bytes gives a specifically labelled algorithm-useful DRAM intensity. It is not executed Tensor Core FLOPs. A shared-memory/MIO bottleneck requires analysis at that level, rather than simply assuming that higher DRAM intensity fixes it.
7. **Sampling:** warp-state percentages describe sampled warp states, including selected and not-selected warps; they are not percentages of wall-clock runtime. Sampling varies. Source-PC attribution requires examining producer/consumer dependencies.
8. **Causality:** a profiler suggests an optimisation experiment. Proving an improvement requires a controlled code change, renewed correctness checks, ordinary timing, and counter comparison. No swizzling or double buffering is introduced by the profiling workflow.

Reproducibility means retained code, conditions, commands, raw observations and deterministic processing of those observations. It does not mean identical GPU timings, PC samples, timestamps or PDF creation metadata across independent runs.

Definitions: NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) and [Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

The saved analysis includes source shared actual/ideal/excessive wavefronts. Historical fixed-default profiles do not describe newly tuned configurations.

## Profiling frozen configurations

The bounded study profiles FP32 FA1, tuned PTX FA1, its frozen d=64 FA2 CUDA-core winner and tuned PTX FA2. This is separate from the all-kernel fixed-default profiler:

```bash
bash part_c/profile.sh --frozen part_c/results/my_bounded_study
# Offline regeneration from captured CSVs:
python3 part_c/scripts/profile_frozen.py part_c/results/my_bounded_study --analyze-only
```

This uses the same evaluation binary hash recorded by the study, one module-warmup launch, one validation launch, five warmups, then `--launch-skip 7 --launch-count 1` to capture the first timed launch. All timings produced under Nsight are excluded from the unprofiled study. Three independent captures per target at N=256/4096 produce 24 raw reports and source-level actual/ideal/excessive shared wavefronts. The input sequence is the new Part A-compatible stream; historical profiles retain their older stream. Avoid treating cross-study counter differences as a precision-controlled or input-controlled ablation.

## FA1 schedule/version note

Fresh all-kernel captures record `fa1_grid=one_CTA_per_batch_head` and
`fa1_implementation=kv_outer_shared_fp32_v2`. FA1's grid is now `(B*H,1,1)`;
Both FA1 baselines use that grid; FA2 kernels retain their query-tile grids. FP32 FA1 uses dynamic shared
storage, so static shared bytes alone are not its shared-memory footprint.
The maintained processor exports dynamic shared bytes separately. Historical
plans without these flags retain their original query-parallel FA1 grid;
no saved source snapshots or counter records are relabelled. Profiling still
selects one forward after validation and five warmups (skip 6): initialisation
is inside FA1's forward, not an additional launch.

Earlier WMMA FA1 plans record `fa1_wmma_implementation=kv_outer_wmma_v1`.
Its shared arrays are static. Frozen profiling exports static and dynamic
shared bytes and their sum for both FA1 baselines and FA2 targets.
Legacy plans with two targets still require twelve captures. Offline processing
uses recorded target lists and capture counts, preserving historical coverage.

To exercise the fresh four-target capture/processing path on a **smoke** study:

```bash
bash part_c/profile.sh --frozen /path/to/smoke/study --build /path/to/matching/build --smoke
```

Use a study without an existing profiling directory. This records the study's
small sequence lengths and B/H in the profile plan. It validates the pipeline;
it is not evidence for the final N256–4096 grid.

Fresh default plans now use `fa1_ptx_implementation=kv_outer_ptx_v1`. Its kernels share `flash_attention_1_tensor_core.cu` with the supplementary WMMA implementation. PTX FA1 uses static shared arrays and one CTA/head; the frozen config comes from the same four-candidate budget as PTX FA2. Historical WMMA-aware plans keep their recorded targets.
