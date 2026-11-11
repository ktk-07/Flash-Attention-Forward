# A/C reviewers: reproduce B's exact notebook execution

This is the measured `dense_attention` streaming/tiled/float4 implementation.
The kernels, public header, evaluation runner, CMake and original adapter match
the code executed in Yichun's notebook. It is not the separate b0-b3 code that
previously appeared in the checkout. The recorded user runs are preserved under
results/; do not merge them into a new common-machine experiment.

## First verify the delivered source and data (no GPU needed)

From the complete repository root:

```bash
python3 part_b/scripts/reproduce_notebook.py --grid full --check-only
python3 -m unittest discover -s part_b/tests -p 'test_*.py' -v
```

The check pins B source content, shared input/oracle/schema dependencies, the
reference manifest and every indexed raw record. For source text only, CRLF/LF
line endings are treated as equivalent. Reference evidence is byte-preserved by
part_b/.gitattributes and checked byte-for-byte. Functional source edits require
a new experiment, not relabelling the old measurements.

## Exact smoke replay on your GPU

Linux prerequisites: NVIDIA CUDA GPU SM80+, CUDA Toolkit/nvcc, C++17 compiler,
CMake, Python 3.10+ with NumPy and Matplotlib. No PyTorch or data download is
needed. This diagnostic replay does not require Compute Sanitizer or Nsight.

```bash
bash part_b/reproduce.sh --grid smoke --device 0 --arch auto \
  --output results/b_crosscheck_smoke
```

It builds locally, runs the shared FP64 oracle self-test and executes **the same
83 cases in recorded order**, including the same patterns, input fingerprints,
head dimensions, warps, key tiles and reference scopes. It reuses the recorded
configuration selection; it does not pick a different winner for your GPU.

Expected: 83 numerical PASS records, 18 timing cases, 360 samples, and
`crosscheck.json` with `numerical_replay_passed=true` and
`input_fingerprints_matched=true`. Correctness tolerance remains
`abs(error) <= 1e-4 + 1e-4*abs(FP64 reference)`.

## Exact full replay

```bash
bash part_b/reproduce.sh --grid full --device 0 --arch auto \
  --output results/b_crosscheck_full
```

Expected: 165 numerical PASS records (75 exhaustive, 90 sampled), 90 timing
cases and 1,800 samples. The full main grid is B4/H8, N256/512/1024/2048/4096,
d32/64/128. The frozen winners remain vector W4/BK64 for d32, vector W4/BK32
for d64 and streaming W4/BK32 for d128. All outputs are checked for finiteness;
large cases validate three query rows per head rather than every output value.
The 75 small cases also compare directly with A's actual materialised CUDA
pipeline. Those supplementary buffers are outside B's timed allocation ledger.

The original frozen scores are reference-only information. Local tuning cases
are replayed and timed but are not used to replace the reference selection.

## Inspect your output

* `crosscheck.json`: case completion, pinned source, fingerprints and validation.
* `comparison_to_reference.csv`: original/local error and latency in **separate
  columns**, retaining case identity, shape, configuration and experiment.
* The seven standard CSVs: exclusively your locally measured timing/error,
  resource, memory and analytical work records.
* `device.json` / `manifest.json`: your physical GPU UUID and newly built binary
  hash; the original GPU is separately labelled as the reference device.
* `commands.jsonl`, `logs/`, `raw/part_b/`: actual local commands and evidence.
* `REPLAY.md`, `REPORT.md`, `plots/part_b/`: interpretation and local figures.

Input fingerprints, case plans and tolerances should match. Timings, floating
point reduction errors, compiler resource allocation and observed memory deltas
can differ. Bitwise output/timing equality is not required. Different hardware
timings are portability observations, not an optimisation speedup.

**Diagnostic scope:** `audit.passed=false` is expected in exact replay, because
this command does not run sanitizers or capture hardware counters. The root
collector must continue to reject this bundle for final acceptance. The
numerical replay result is separately explicit in crosscheck.json.

## Windows

Use an x64 Visual Studio developer terminal with CUDA, CMake and Ninja. Python
can invoke the same replay without Bash. On the RTX 4070 Ti, SM89 is appropriate;
use your detected architecture on another GPU:

```powershell
cmake -G Ninja -S part_b -B build/part_b_replay -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_POLICY_DEFAULT_CMP0141=NEW -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
python part_b/scripts/reproduce_notebook.py --grid smoke --arch 89 --build build/part_b_replay --output results/b_crosscheck_smoke
```

Choose a CUDA-supported host compiler. The original notebook passed the explicit
unsupported-compiler workaround; it is not a general compatibility guarantee.
Use fresh output directories and isolate GPU activity. Kernel work is CUDA C++;
Python only builds/launches the executable and analyses its saved data.

## Fresh accepted comparison after cross-checking

For local tuning, required sanitizer checks and fresh profiling, use the normal
adapter. These are separate new experiments, not exact configuration replays:

```bash
bash part_b/run.sh --grid smoke --output results/b_accepted_smoke
bash run.sh --parts a,b,c --grid smoke --output results/abc_smoke
bash run.sh --output results/abc_full
```

Compute Sanitizer must work; full runs also require permitted Nsight counters.
The root collector checks the same GPU/input identities across A/B/C. Never
combine Yichun's historical timings with your own GPU timings to claim a
materialised/fused speedup. Report unsupported tools or numerical failures
explicitly. AI assistance and individual review/contributions remain disclosed.
