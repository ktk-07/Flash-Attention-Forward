# Part B cross-review of materialised indexing and memory access

Static review performed during the B implementation; this is not another
member's signed review or a substitute for runtime validation.

* `part_a/gpu_baseline/matmul_tiled.cu`: Q uses `(bh*N+i)*d+r`, K uses
  `(bh*N+j)*d+r`, and S uses `(bh*N+i)*N+j`. K remains physically row-major;
  `k_tile[tx][r]` implements the mathematical transpose. Feature-tail zero fill
  and output guards match the external layout. Fully masked causal tiles write
  negative infinity to valid S entries before a block-uniform `continue`.
* PV reads P at `(bh*N+i)*N+j` and V at `(bh*N+j)*d+r`. Both input tiles are
  bounds-checked and padded with zero; barriers protect reuse between key tiles
  and between grid-stride work items. N and d tails are separately guarded.
* `softmax_parallel.cu`: one block reduces each score row, then normalises it.
  Supported 128/256 thread counts are powers of two. The final barrier protects
  the shared sum before another grid-stride row overwrites it. Dense keys span
  N; causal keys span `i+1`, which is always nonempty for positive N.
* Global loads traverse adjacent features/keys where possible. Shared K padding
  alone does not prove reduced bank conflicts, especially for tile16 mappings;
  counters and source-level access analysis are needed for such a claim.

B's small-case runner additionally links A's actual naive materialised CUDA
QK/softmax/PV kernels and checks its dense output against B at the same tolerance.
The independent FP64 oracle remains authoritative. This supplementary equivalence
check does not establish performance fairness of A's tuned materialised kernels;
see the completed [common-machine comparison](../results/final_submission/VERIFICATION.md) for the performance evidence.
