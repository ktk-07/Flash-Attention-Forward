# Part B: dense FP32 FlashAttention forward

**A/C reviewers:** see [CROSS_CHECK.md](CROSS_CHECK.md) for source-pinned replay of the exact recorded notebook case order, inputs and launch configurations. `reproduce.sh` reuses the frozen winners; `run.sh` performs a new local tuning/acceptance study.

This directory contains the **measured streaming/tiled/float4 implementation**
used in the recorded notebook run `20261110_044740_8e7cd654`. Public namespace:
`dense_attention`. It is different from the other b0-b3 implementation previously
in the checkout. Its results must not be assigned to that other implementation.

Zhang Yichun, A0240469B, CEG5206 Group 1. Implementation and report preparation
were AI-assisted and require member review/disclosure before submission.

## Algorithms and layout

Inputs/output are FP32 contiguous row-major `[B,H,N,d]`, element offset
`((b*H+h)*N+i)*d+r`. Output is `softmax(QK^T/sqrt(d))V`, with no causal mask.
The independent linked C++ FP64 oracle is the correctness authority.

One CTA owns W4/W8 query rows within one batch/head. One warp owns each row;
lanes own adjacent features and further coordinates at stride 32. Q and the
output numerator stay in registers. Tiled variants cooperatively stage BK x d
K/V tiles and use a private shared score/probability row per warp. Online softmax
rescales both the normaliser and accumulated output whenever the running maximum
changes. Tail keys receive negative infinity. Inactive tail-query warps still
participate in all block barriers.

| Variant | K/V path | Update | Role |
|---|---|---|---|
| streaming | Global per-query loads | Per key | Starting point |
| tiled | Cooperative scalar shared loads | Per tile | Reuse and fewer state updates |
| vector | Cooperative float4 shared loads | Per tile | Controlled loading comparison |

Candidates: streaming W4/BK32; tiled W4/BK32; vector W4/BK32; tiled W8/BK32;
vector W4/BK64. Selection minimises geometric mean median latency on N256/1024
and is frozen per d before reporting. N512/2048/4096 are held out. W4/BK32
scalar/vector differ in loading; streaming/tiled change multiple mechanisms.

Shared storage for tiled variants is `4*(2*BK*d + W*BK)` bytes per CTA. Requests
over 48 KiB use explicit opt-in. The public scalar interface supports d1..256
subject to resources; float4 requires d divisible by four. BK64/d256 is outside
the bounded study. No Tensor Core, asynchronous-copy or double-buffered path
is claimed for this implementation. B allocates no global S/P or auxiliary array.

## Common-machine results

The integrated RTX 3070 Ti Laptop run is retained in [../results/final_submission/parts/part_b/](../results/final_submission/parts/part_b/). It passed 165 correctness records, 32 sanitizer checks and six profiler captures, and contains 90 timing cases with 1,800 samples. The selected configurations are vector W4/BK64 at d32, vector W4/BK32 at d64 and streaming W4/BK32 at d128.

Against optimized materialised A on identical dense FP32 inputs, the A/B median-latency ratio is 0.205–0.703×, with geometric mean 0.428× across the 15 main-grid shapes. B uses linear Q/K/V/O storage and avoids global S/P, but is slower across this grid. Determining whether instruction dependencies, memory access or resource limits dominate requires source-correlated profiling; the latency result alone does not establish the cause. Do not substitute historical timings from another GPU for this comparison.

The per-part figures are in `../results/final_submission/plots/part_b/`; joint comparisons are in `plots/combined/` under the same root. See [verification findings](../results/final_submission/VERIFICATION.md) for reference scope and timing variability.

## Historical notebook results and evidence status

`results/notebook_full_20261110_044740/` contains the exact full-run evidence:
165 PASS records (75 full, 90 sampled), 75 small direct A/B CUDA comparisons,
90 timing cases and 1,800 samples. Each timed case uses five warmups and 20
CUDA-event samples. Large cases sample three query rows per head plus
all-output finiteness checks. Allocation, transfers, validation, resource queries
and file output are outside timing. Fused stage timing cells are blank.

The actual measured toolkit is **CUDA 12.9 / nvcc 12.9.41**, with MSVC 19.41,
RTX 4070 Ti SM89 and driver 621.01. See native logs for the exact commands.
The notebook's historical reference run used a different toolkit; those older
reference values are not substituted for this full run.

**The historical notebook bundle is diagnostic evidence.** That run explicitly used `--no-sanitize
--no-profile`, so `audit.passed=false`. Zero numerical failures does not establish
memory/race safety. No fresh counter captures, measured DRAM savings, measured
active-warp occupancy or same-machine A/B/C speedup are claimed. Retained source
and evidence hashes are immutable; derived submission figures live separately.

## Reproduce on Linux

From the complete repository root, install the common CUDA/CMake/Python/NumPy/
Matplotlib prerequisites and provide Compute Sanitizer; full profiling also
requires Nsight Compute and permission to access GPU counters.

```bash
# Small default pipeline, including required sanitizer checks.
bash part_b/run.sh --grid smoke --output results/b_smoke
# Full numerical/tuning/performance/safety/profiling study.
bash part_b/run.sh --output results/b_full
# Explicit diagnostic study, matching the retained evidence scope.
bash part_b/run.sh --grid full --no-sanitize --no-profile --output results/b_diagnostic
# Complete common-machine integration after standalone acceptance.
bash run.sh --parts a,b,c --grid smoke --output results/abc_smoke
bash run.sh --output results/abc_full
```

Use fresh output directories. Full grid: B4/H8, N256/512/1024/2048/4096,
d32/64/128. Smoke: B1/H1, N33/65, d64. `--dry-run` creates no files or GPU
access. Options also include `--build`, `--arch`, `--device`, `--analyze-only`.
The root collector rejects `--no-sanitize` diagnostic bundles by design.

## Reproduce on Windows

The Python adapter avoids Bash. Run in an x64 Visual Studio developer terminal
with CUDA/nvcc, CMake and Ninja available. For your RTX 4070 Ti:

```powershell
cmake -G Ninja -S part_b -B build/part_b -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_POLICY_DEFAULT_CMP0141=NEW -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
python part_b/scripts/run_standard.py --build build/part_b --arch 89 --grid smoke --no-sanitize --no-profile --output results/b_windows_smoke
```

The run.cmd wrapper forwards to the same Python CLI. The self-contained
`notebooks/FlashAttention_Part_B_Lab.ipynb` also discovers the x64 compiler and
creates an isolated build. Its saved outputs are an older smoke example; choose
GRID=full and run all cells for fresh measurements. The recorded notebook full run is
retained separately in the part-local results directory. If an older toolkit rejects your compiler,
choose a compatible toolchain; the notebook's explicit unsupported-compiler
workaround is not a general compatibility guarantee.

## Inspect or regenerate without CUDA

```bash
# Verify schema, raw evidence, source snapshot and numerical joins.
python part_b/scripts/analyze_delivery.py --verify-only
# Regenerate exact report figures and tables without building/running CUDA.
python part_b/scripts/analyze_delivery.py --output results/b_report_assets
# Original seven-table re-extraction; diagnostic audit remains false, exit 1.
python part_b/scripts/export_results.py --analyze-only part_b/results/notebook_full_20261110_044740
python -m unittest discover -s part_b/tests -p 'test_*.py' -v
```

Open summary.csv for medians/IQR/throughput, correctness.csv for scope and error,
resources.csv for API estimates, memory.csv for explicit allocation and observed
device deltas, and configs/part_b.json for the immutable selection. The empty
profiles.csv is evidence of omitted captures, not zero traffic. Native per-case
JSON, source snapshot, manifest and commands remain authoritative evidence.

At N4096 the frozen d32/d64/d128 medians are 112.053/125.168/163.780 ms.
At d64 this uses exactly 128 MiB Q/K/V/O and zero auxiliary global bytes. These
values are observations rather than required timing targets. For the matched
W4/BK32 comparison at N4096, float4 reduced latency by 3.9%/8.3%/20.4% for
d32/d64/d128. Larger tiles did not universally help; streaming won d128 tuning.
No historical A/C measurements from other GPUs are used as speedup denominators.

## Files and disclosure

Headers contain declarations, CUDA source contains kernels, and the runner
links the shared oracle plus A's actual naive materialised kernels for small
supplementary checks. Those A allocations are freed before timing and excluded
from B's ledger. CMake never copies or edits A/C kernels. New packaging/analysis
files are distinguished from measured code in the delivery manifest.

See CROSS_REVIEW.md for a static review of A indexing/accesses; it is not another
member's signed review. The integrated common-machine run has passed sanitizer/profiler acceptance;
results are linked above. Interpretation, report/video preparation and contribution
confirmation remain submission work.
