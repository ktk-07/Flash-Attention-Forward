# Conventional .h/.cu layout verification

The public API is declared in `part_c/include/flash_attention/*.h`. Definitions and template dispatch are compiled from `part_c/src/flash_attention_1.cu`, `flash_attention_2_cuda_core.cu`, and `flash_attention_2_tensor_core.cu` into the `part_c_attention` library. The C++ runner links the library and includes only public headers. Kernel computations, comments, test cases, launch configurations, and the prior correctness fixes are retained.

The complete workflow passed after the layout change:

- **380 full-output correctness checks passed**, 4 PTX cases explicitly unsupported.
- **120 benchmark configurations passed sampled query validation**, with **2,400 measured samples**; all output elements checked finite.
- **Memcheck, Racecheck, Synccheck, and Initcheck passed** on the smoke workload. Racecheck: zero errors, warnings, or hazards.
- **13 legacy test configurations / 104 comparisons passed**.
- CSV audit verified completeness, recomputed medians/quartiles, and confirmed **308 preserved original kernel comments** and the unchanged original `causal_flash_attn.cu` hash.
- All five figures regenerated as PNG and PDF; compiler resources and Tensor Core instruction evidence saved.

The GPU/environment is recorded in [benchmark/environment.txt](benchmark/environment.txt). These are standalone Part C measurements on an RTX 3070 Ti Laptop GPU, with fixed launch configurations. There is no Part A integration or comparison. Benchmark oracle checks use queries 0, N/2, N-1 in each head, rather than full-output FP64 checks of the large grid.

## Figures

- [CUDA-core progression](benchmark/figures/c1_cuda_core_progression.png)
- [Tensor Core progression](benchmark/figures/c2_tensor_core_progression.png)
- [Best observed CUDA-core versus Tensor Core](benchmark/figures/c3_best_cuda_vs_tensor.png)
- [All-kernel scaling](benchmark/figures/all_kernel_scaling.png)
- [Sampled numerical error](benchmark/figures/sampled_numerical_error.png)

C3 uses exploratory selection on the d=64,N=256/1024 subset. It compares FP32-input CUDA-core kernels with FP16-input Tensor Core kernels; it is not a separately tuned or precision-isolated hardware speedup claim.

## Reproduction

From the repository root:

```bash
bash part_c/run.sh part_c/results/my_run /tmp/part_c-layout-build
```

Or from inside `part_c/`, run `./run.sh` for a timestamped results directory and a local `build/` directory. The recorded run invoked the same workflow through its compatibility forwarder:

```bash
bash part_c/scripts/run_all.sh part_c/results/layout_validation /tmp/part_c-layout-build
```

[commands.log](commands.log) records build/test/benchmark/sanitizer/Python commands and exit codes. [logs/](logs/) contains stdout/stderr and source/binary hashes. [audit.json](audit.json) contains verification counts. The previous [local-validation dataset](../local_validation/RESULTS.md) is retained unchanged as historical measurement evidence; only its documentation links were updated for the new source layout.
