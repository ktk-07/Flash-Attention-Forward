# Split kernel provenance and corrections

The authoritative extraction source was `../../src/causal_flash_attn.cu`, as requested. That file has since been moved to `src/extra/causal_flash_attn.cu` and fixed at the user's request to run a minimal standalone test. Its historical pristine hash remains recorded, and `original_to_smoke.patch` records the changes. The three `src/flash_attention_*.cu` implementation files contain the original kernel bodies and comments, with the corrections below. The CUDA-core `.cu` file also retains the tile-skipping and vectorised optimisation snapshots in namespaces. Public declarations are in `include/flash_attention/*.h`; template instantiation and runtime dimension dispatch are implemented inside the `.cu` files. The C++ test runner links against the compiled library.

All 308 comments in the original main file's kernel sections are preserved in the split family files. The original naming header is also retained. Demonstration `main()` code is excluded from the new implementation files but remains in the untouched original file. The audit checks comment retention and the recorded standalone source hash. The original three extracted sections, before corrections, were recorded by their hashes in `provenance.json`.

## Corrections made during actual testing

1. **Build completeness:** restore the WMMA namespace alias and missing `row_in_warp=lane/2`, `pair_lane=lane%2` definitions. The missing `fa_pack_half2` and `fa_mma_m16n8k16` helper definitions were copied from the historical `flash_attention_ptx_mma_integrated.cu` demo before that redundant demo was removed. Its hash remains in `provenance.json`. The kernels themselves come from the main file, not this older demo.
2. **PTX A-fragment ordering:** the original assignments swapped the second and third packed A registers in both QKᵀ and PV. The correct packed-register order is row0/low-K, row1/low-K, row0/high-K, row1/high-K. The first test run had 41 PTX failures. The corrected version passes the supported suite. See NVIDIA's [m16n8k16 floating-point fragment mapping](https://docs.nvidia.com/cuda/archive/11.5.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).
3. **WMMA shared-memory ordering:** Racecheck reported a read/write hazard between reading the old `running_m` and the even lane updating it. Add `__syncwarp()` before the row-statistics writes so all participating lanes complete their reads first. Shuffle collectives alone do not provide the required shared-memory ordering.

Initial failures are archived under `../../results/local_validation/attempts/`. Final supported-check results and safety-check logs are under `../../results/local_validation/`. `corrections.patch` is the historical correction record relative to the first mechanically extracted Tensor Core header (including helper restoration). The subsequent layout refactor moved those corrected definitions into `src/flash_attention_2_tensor_core.cu` without changing their computations. `provenance.json` retains historical hashes and current source/header hashes. Part A sources and PyTorch integration files are unchanged. The original coding file was subsequently relocated and fixed for a minimal standalone smoke test. Superseded source copies were subsequently removed; see the cleanup record below.

## What the labels mean

The FA1/FA2 labels identify the educational organisation used in these sources; they do not claim feature/performance equivalence to the official library releases. The updated main file's register-tiled CUDA-core version already bounds KV traversal to the causal region. The separate tile-skipping snapshot should not be treated as an isolated new optimisation added on top of that updated version.

The PTX kernel currently packs half operands with scalar shared-memory loads; it uses explicit `mma.sync` but does not use `ldmatrix`. WMMA and PTX both take FP16 inputs and maintain FP32 accumulation/softmax/output. Neither materialises the full score/probability matrices in global memory.

## Source-directory cleanup

`src/` now contains the three current `flash_attention_*.cu` implementations and `extra/`, where the original coding file has a minimal standalone test. The legacy regression harness was moved to `tests/causal_flash_attn_full_test.cu`, and `run.sh` uses that path. Ten redundant source/demo/test copies or built executables were removed. [src_cleanup.json](src_cleanup.json) records filenames and hashes. Historical command logs are retained as recorded; their old source paths describe the pre-cleanup layout.

## Minimal original-file smoke test

At the user's request, the coding file was moved to `src/extra/causal_flash_attn.cu`. Its unfinished main was replaced with a four-variant CUDA-core check against CPU FP64 attention. The previously tested WMMA definitions, PTX helpers/register order, and WMMA barrier fix were restored here too. Architecture guards let plain `nvcc causal_flash_attn.cu -o ./test` compile without enabling Tensor Core targets; the minimal main does not launch those kernels. `original_to_smoke.patch` preserves the changes from the original file, and `source_sha256` remains its historical pristine hash. The audit verifies the new smoke-file hash and retains the original 308-comment preservation check against the maintained three implementations.
