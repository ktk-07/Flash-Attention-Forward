# Release CUDA validation

Validated source: `fa1f796a2b50ccf95c9fdc0de479cec4ec42d5b9`.

## Results

- Release CUDA build passed with architecture 120 and profiling disabled
- All 18 CTest groups passed, including workspace state and recovery checks
- All 7 analysis regression tests passed
- All 8 representative configurations passed with tile 16 and softmax 128
- Both shapes, (1,1,33,32) and (4,8,256,64), passed CPU S/P/O checks before and after timing
- Each configuration used 5 warmups and 20 repeats

For (4,8,256,64), forward medians in milliseconds were:

| Mode | Naive | Optimized |
| --- | ---: | ---: |
| Dense | 1.720800 | 0.320800 |
| Causal | 0.980544 | 0.267056 |

Environment: Windows build 26200, MSVC 19.44, CUDA Toolkit 13.4.59,
CMake 4.4.4, Python 3.13.11, RTX 5060 Ti 8 GiB and driver 610.88.
Full-grid timing and profiling were not repeated. Earlier measurements remain unchanged.

## Records

[Exit codes](exit-codes.txt), [CTest output](ctest.log), [analysis tests](analyzer-tests.log)
and [benchmark metadata](benchmark/metadata.txt) record the successful checks.
Preliminary environment and command-wrapper failures are retained separately;
no product check failed. Python path and toolkit code-page warnings were nonblocking.

[validate.ps1.txt](validate.ps1.txt) is the archived invocation script.
For new runs, use [the experiment runner](../../../scripts/run_gpu_experiments.ps1)
with a new output directory. English log transcriptions retain diagnostic codes,
locations, measurements and exit codes; original hashes are in
[log_sources.csv](../../log_sources.csv).
