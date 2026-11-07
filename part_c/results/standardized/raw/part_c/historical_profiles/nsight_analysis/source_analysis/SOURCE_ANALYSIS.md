# CUDA/SASS correlation of sampled stalls

Representative repetition 0 for every kernel and workload. All source exports resolve the captured source snapshots; no live source modification is needed. The aggregate report uses three independent captures per case.

A sampled PC identifies the instruction associated with a waiting warp; inspect its data dependencies and neighbouring producers before assigning a cause. In particular, a short-scoreboard sample at an arithmetic instruction can arise from a preceding shuffle or shared-memory load. It does not mean that the arithmetic instruction itself accessed shared memory. Percentages below are within the named stall category, not kernel runtime. Sampling and attribution can vary between captures. Optimised/inlined source mappings are approximate; duplicate mappings to the same PC are counted once, with consistency checks and project source preferred over library headers. Counts are in export_audit.csv.

| Kernel | N | Stall | % within category | CUDA line | SASS at sampled PC |
|---|---:|---|---:|---|---|
| fa1 | 256 | stall_short_sb | 7.1 | flash_attention_1.cu:135 | `FFMA R7, R0, -R3, 1` |
| fa1 | 256 | stall_short_sb | 4.8 | flash_attention_1.cu:135 | `FFMA R9, R0, -R3, 1` |
| fa1 | 256 | stall_long_sb | 4.3 | flash_attention_1.cu:126 | `FFMA R14, R25, R29, R14` |
| fa1 | 256 | stall_long_sb | 4.2 | flash_attention_1.cu:100 | `FFMA R0, R25, R27, R0` |
| fa1 | 4096 | stall_short_sb | 44.5 | flash_attention_1.cu:112 | `FMUL R23, R23, R0` |
| fa1 | 4096 | stall_short_sb | 23.3 | flash_attention_1.cu:64 | `LEA R11, R11, R2, 0x5` |
| fa1 | 4096 | stall_long_sb | 4.8 | flash_attention_1.cu:121 | `FSETP.NEU.AND P0, PT, R25, -INF , PT` |
| fa1 | 4096 | stall_long_sb | 3.8 | flash_attention_1.cu:100 | `FFMA R0, R25, R27, R0` |
| fa1 | 4096 | stall_mio | 11.1 | flash_attention_1.cu:100 | `LDG.E R31, [R136.64]` |
| fa1 | 4096 | stall_mio | 11.1 | flash_attention_1.cu:100 | `LDG.E R27, [R140.64]` |
| scalar | 256 | stall_short_sb | 18.5 | flash_attention_2_cuda_core.cu:121 | `FMUL R29, R29, R8` |
| scalar | 256 | stall_short_sb | 14.8 | cmath:223 | `FMUL R32, R35, R38` |
| scalar | 256 | stall_long_sb | 3.6 | flash_attention_2_cuda_core.cu:97 | `FFMA R32, R51, R32, R38` |
| scalar | 256 | stall_long_sb | 2.5 | flash_attention_2_cuda_core.cu:145 | `@!P5  FFMA R50, R4, R39, R49` |
| scalar | 4096 | stall_short_sb | 18.4 | flash_attention_2_cuda_core.cu:121 | `FMUL R29, R29, R8` |
| scalar | 4096 | stall_short_sb | 17.6 | flash_attention_2_cuda_core.cu:134 | `FFMA.SAT R10, R37, R58, 0.5` |
| scalar | 4096 | stall_long_sb | 6.2 | flash_attention_2_cuda_core.cu:145 | `@!P5  FFMA R50, R4, R39, R49` |
| scalar | 4096 | stall_long_sb | 2.9 | flash_attention_2_cuda_core.cu:145 | `FFMA R48, R31, R38, R48` |
| scalar | 4096 | stall_mio | 21.4 | flash_attention_2_cuda_core.cu:58 | `S2R R0, SR_TID.X` |
| scalar | 4096 | stall_mio | 7.1 | flash_attention_2_cuda_core.cu:97 | `LDG.E R49, [R8.64]` |
| warp | 256 | stall_short_sb | 12.3 | flash_attention_2_cuda_core.cu:361 | `@!P2  FFMA R28, R37, R23, R28` |
| warp | 256 | stall_short_sb | 6.5 | flash_attention_2_cuda_core.cu:293 | `FFMA R61, R16, R61, R7` |
| warp | 256 | stall_long_sb | 10.8 | flash_attention_2_cuda_core.cu:268 | `STS [R58+0x1c00], R37` |
| warp | 256 | stall_long_sb | 10.7 | flash_attention_2_cuda_core.cu:268 | `STS [R58+0x1800], R27` |
| warp | 256 | stall_mio | 2.6 | flash_attention_2_cuda_core.cu:361 | `@!P2  LDS R21, [R24+0x200]` |
| warp | 256 | stall_mio | 2.5 | flash_attention_2_cuda_core.cu:355 | `SHFL.IDX PT, R33, R5, R16, 0x1f` |
| warp | 4096 | stall_short_sb | 19.3 | flash_attention_2_cuda_core.cu:361 | `@!P2  FFMA R28, R37, R23, R28` |
| warp | 4096 | stall_short_sb | 5.1 | flash_attention_2_cuda_core.cu:174 | `FMNMX R5, R4, R5, !PT` |
| warp | 4096 | stall_long_sb | 11.8 | flash_attention_2_cuda_core.cu:268 | `STS [R58+0x1c00], R37` |
| warp | 4096 | stall_long_sb | 11.7 | flash_attention_2_cuda_core.cu:268 | `STS [R58+0x1000], R37` |
| warp | 4096 | stall_mio | 3.9 | flash_attention_2_cuda_core.cu:361 | `@!P3  LDS R27, [R24+0x380]` |
| warp | 4096 | stall_mio | 3.8 | flash_attention_2_cuda_core.cu:361 | `@!P3  LDS R25, [R24+0x300]` |
| tiled | 256 | stall_short_sb | 1.9 | flash_attention_2_cuda_core.cu:186 | `FADD R11, R10, R11` |
| tiled | 256 | stall_short_sb | 1.8 | flash_attention_2_cuda_core.cu:186 | `FADD R44, R9, R44` |
| tiled | 256 | stall_long_sb | 7.1 | flash_attention_2_cuda_core.cu:597 | `FMNMX R42, R8, R11, !PT` |
| tiled | 256 | stall_long_sb | 7.1 | flash_attention_2_cuda_core.cu:597 | `FMNMX R37, R9, R10, !PT` |
| tiled | 256 | stall_mio | 3.3 | flash_attention_2_cuda_core.cu:543 | `@!P6  LDS R43, [R14+0x480]` |
| tiled | 256 | stall_mio | 2.8 | flash_attention_2_cuda_core.cu:618 | `@!P5  LDS R40, [R14+0x2480]` |
| tiled | 4096 | stall_short_sb | 1.7 | flash_attention_2_cuda_core.cu:560 | `FMUL R11, R39, c[0x0][0x1d0]` |
| tiled | 4096 | stall_short_sb | 1.7 | flash_attention_2_cuda_core.cu:560 | `FMUL R30, R53, c[0x0][0x1d0]` |
| tiled | 4096 | stall_long_sb | 9.0 | flash_attention_2_cuda_core.cu:597 | `FMNMX R37, R9, R10, !PT` |
| tiled | 4096 | stall_long_sb | 8.7 | flash_attention_2_cuda_core.cu:597 | `FMNMX R42, R8, R11, !PT` |
| tiled | 4096 | stall_mio | 2.6 | flash_attention_2_cuda_core.cu:543 | `@!P6  LDS R43, [R14+0x480]` |
| tiled | 4096 | stall_mio | 1.9 | flash_attention_2_cuda_core.cu:543 | `@!P6  LDS R8, [R14+0x400]` |
| skip | 256 | stall_short_sb | 1.9 | flash_attention_2_cuda_core.cu:845 | `FMUL R10, R49, c[0x0][0x1d0]` |
| skip | 256 | stall_short_sb | 1.9 | flash_attention_2_cuda_core.cu:845 | `FMUL R30, R53, c[0x0][0x1d0]` |
| skip | 256 | stall_long_sb | 7.4 | flash_attention_2_cuda_core.cu:882 | `FMNMX R37, R9, R10, !PT` |
| skip | 256 | stall_long_sb | 7.1 | flash_attention_2_cuda_core.cu:882 | `FMNMX R42, R8, R11, !PT` |
| skip | 256 | stall_mio | 2.9 | flash_attention_2_cuda_core.cu:828 | `@!P6  LDS R43, [R14+0x480]` |
| skip | 256 | stall_mio | 2.2 | flash_attention_2_cuda_core.cu:903 | `@!P4  LDS R44, [R14+0x2380]` |
| skip | 4096 | stall_short_sb | 1.7 | flash_attention_2_cuda_core.cu:845 | `FMUL R30, R53, c[0x0][0x1d0]` |
| skip | 4096 | stall_short_sb | 1.7 | flash_attention_2_cuda_core.cu:186 | `FADD R49, R30, R49` |
| skip | 4096 | stall_long_sb | 9.0 | flash_attention_2_cuda_core.cu:882 | `FMNMX R37, R9, R10, !PT` |
| skip | 4096 | stall_long_sb | 8.7 | flash_attention_2_cuda_core.cu:882 | `FMNMX R42, R8, R11, !PT` |
| skip | 4096 | stall_mio | 2.6 | flash_attention_2_cuda_core.cu:828 | `@!P6  LDS R43, [R14+0x480]` |
| skip | 4096 | stall_mio | 2.0 | flash_attention_2_cuda_core.cu:828 | `@!P6  LDS R8, [R14+0x400]` |
| vector | 256 | stall_short_sb | 1.7 | flash_attention_2_cuda_core.cu:186 | `FADD R47, R48, R47` |
| vector | 256 | stall_short_sb | 1.6 | flash_attention_2_cuda_core.cu:1161 | `FMUL R24, R47, c[0x0][0x1d0]` |
| vector | 256 | stall_long_sb | 14.9 | flash_attention_2_cuda_core.cu:1088 | `STS.128 [R11], R12` |
| vector | 256 | stall_long_sb | 14.6 | flash_attention_2_cuda_core.cu:1080 | `STS.128 [R26], R12` |
| vector | 256 | stall_mio | 3.2 | flash_attention_2_cuda_core.cu:1219 | `@!P3  LDS R48, [R23+0x380]` |
| vector | 256 | stall_mio | 3.2 | flash_attention_2_cuda_core.cu:1222 | `STL [R17+0x64], R39` |
| vector | 4096 | stall_short_sb | 1.6 | flash_attention_2_cuda_core.cu:186 | `FADD R38, R27, R38` |
| vector | 4096 | stall_short_sb | 1.5 | flash_attention_2_cuda_core.cu:186 | `FADD R41, R38, R41` |
| vector | 4096 | stall_long_sb | 14.6 | flash_attention_2_cuda_core.cu:1088 | `STS.128 [R11], R12` |
| vector | 4096 | stall_long_sb | 14.5 | flash_attention_2_cuda_core.cu:1080 | `STS.128 [R26], R12` |
| vector | 4096 | stall_mio | 2.6 | flash_attention_2_cuda_core.cu:1222 | `STL [R17+0x64], R39` |
| vector | 4096 | stall_mio | 2.5 | flash_attention_2_cuda_core.cu:1222 | `STL [R17+0x60], R44` |
| wmma | 256 | stall_short_sb | 9.2 | flash_attention_2_tensor_core.cu:281 | `FFMA R7, R8, R7, R13` |
| wmma | 256 | stall_short_sb | 8.4 | flash_attention_2_tensor_core.cu:281 | `FFMA R11, R8, R11, R13` |
| wmma | 256 | stall_long_sb | 74.8 | flash_attention_2_tensor_core.cu:166 | `STS.U16 [R13+0x1000], R4` |
| wmma | 256 | stall_long_sb | 21.5 | flash_attention_2_tensor_core.cu:137 | `STS.U16 [R7], R10` |
| wmma | 256 | stall_mio | 19.8 | flash_attention_2_tensor_core.cu:281 | `LDS R11, [R16+-0x18]` |
| wmma | 256 | stall_mio | 18.9 | flash_attention_2_tensor_core.cu:281 | `LDS R10, [R3+0xc00]` |
| wmma | 4096 | stall_short_sb | 9.6 | flash_attention_2_tensor_core.cu:281 | `FFMA R7, R8, R7, R13` |
| wmma | 4096 | stall_short_sb | 9.0 | flash_attention_2_tensor_core.cu:281 | `FFMA R11, R8, R11, R13` |
| wmma | 4096 | stall_long_sb | 93.0 | flash_attention_2_tensor_core.cu:166 | `STS.U16 [R13+0x1000], R4` |
| wmma | 4096 | stall_long_sb | 5.5 | flash_attention_2_tensor_core.cu:167 | `STS.U16 [R13+0x2000], R5` |
| wmma | 4096 | stall_mio | 22.0 | flash_attention_2_tensor_core.cu:281 | `LDS R5, [R16+-0x10]` |
| wmma | 4096 | stall_mio | 18.1 | flash_attention_2_tensor_core.cu:281 | `LDS R11, [R16+-0x18]` |
| ptx | 256 | stall_short_sb | 9.1 | flash_attention_2_tensor_core.cu:33 | `HMMA.16816.F32 R44, R36, R52, R44` |
| ptx | 256 | stall_short_sb | 7.1 | flash_attention_2_tensor_core.cu:33 | `HMMA.16816.F32 R44, R36, R52, R44` |
| ptx | 256 | stall_long_sb | 82.8 | flash_attention_2_tensor_core.cu:393 | `STS.U16 [R43+0x2000], R39` |
| ptx | 256 | stall_long_sb | 12.3 | flash_attention_2_tensor_core.cu:358 | `STS.U16 [R11], R6` |
| ptx | 256 | stall_mio | 5.4 | flash_attention_2_tensor_core.cu:492 | `LDS R36, [R49]` |
| ptx | 256 | stall_mio | 4.8 | flash_attention_2_tensor_core.cu:416 | `LDS R39, [R48+0x430]` |
| ptx | 4096 | stall_short_sb | 7.6 | flash_attention_2_tensor_core.cu:33 | `HMMA.16816.F32 R44, R36, R52, R44` |
| ptx | 4096 | stall_short_sb | 7.5 | flash_attention_2_tensor_core.cu:33 | `HMMA.16816.F32 R44, R36, R52, R44` |
| ptx | 4096 | stall_long_sb | 92.9 | flash_attention_2_tensor_core.cu:393 | `STS.U16 [R43+0x2000], R39` |
| ptx | 4096 | stall_long_sb | 5.4 | flash_attention_2_tensor_core.cu:391 | `STS.U16 [R43+0x1000], R38` |
| ptx | 4096 | stall_mio | 8.8 | flash_attention_2_tensor_core.cu:393 | `STS.U16 [R43+0x2000], R39` |
| ptx | 4096 | stall_mio | 6.5 | flash_attention_2_tensor_core.cu:394 | `@!P2  LDG.E.U16.CONSTANT R39, [R36.64]` |

Full correlated source/SASS CSV exports, all eight top PCs per category, and operation-group distributions are in this directory. The main `plan.json` and `source_snapshot/` identify the exact code; `processing.json` records this processing script hash.

```bash
python3 part_c/scripts/analyze_source.py part_c/results/nsight_analysis
```
