# Materialized GPU attention

The GPU implementation uses separate kernels for scaled QK^T, stable softmax and PV. It stores complete S/P matrices in global memory.

| Variant | Matrix multiplication | Softmax |
| --- | --- | --- |
| naive | One score/output element per thread | One row per thread |
| tiled | Shared-memory tiles | One row per thread |
| parallel | One score/output element per thread | Block reduction |
| optimized | Shared-memory tiles | Block reduction |

The default matrix tile is 16x16. Parallel softmax uses 256 threads per block. Tiles 8/16/32 and softmax blocks 128/256 are supported. Tiled kernels zero-fill partial tiles and keep all participating threads at barriers. Causal QK skips computation of fully masked tiles while writing masked scores.

`baseline(input,causal,variant)` allocates, uploads, runs and downloads S/P/O for diagnosis. `BaselineWorkspace` allocates and uploads once, then reuses buffers for correctness and timing. Both use the same dispatch functions and default stream. The workspace is single-threaded and tied to the CUDA device on which it was created.

```powershell
.\build-gpu\gpu_baseline\Release\gpu_baseline_tests.exe all optimized
```

Each variant test runs all numerical and boundary cases. Configuration validation also checks the preallocated path against the CPU oracle. Missing devices, runtime errors and numerical mismatches fail validation.

Kernel resources are recorded in benchmark `resources.csv`. Theoretical occupancy is a resource-based limit; achieved occupancy and actual memory traffic require Nsight Compute. Build and measurement commands are in the root and benchmark READMEs.

The measured main grid, ablation and profiling use explicit tile 16 and 128 softmax threads. The library defaults remain tile 16 and 256 softmax threads. See [measured results](../results/README.md).

`download()` requires a completed successful `run()` or `measure()`. A failed forward invalidates the result; a later successful forward restores it. `warmup(...,0)` does not produce a result.

Inputs must be finite, and their magnitude must keep FP32 products and accumulations representable. Finite-input validation alone cannot guarantee this. Subtracting the row maximum stabilizes softmax exponentiation after finite scores are formed; it cannot repair QK overflow. The measured generator uses values in `[-1,1)`. Arbitrarily large finite inputs are outside the supported numerical contract.
