# Combined evaluation

Requested parts: part_a, part_b, part_c. Grid: full.

All accepted timing rows have output validation against the shared Part A FP64 oracle. Benchmark checks use three query rows per head plus all-output finiteness checks; Part A additionally checks full output for small shapes and N=256,d=64. Small correctness suites compare full outputs.

Input fingerprints and physical GPU UUIDs agree. FP32 A/C comparisons match numerical precision. FP16 comparisons use identical rounded inputs and FP32 accumulation/output in A and C. A rounds normalized P; C rounds tile probabilities before online normalization. Configuration selection precedes the final grid; full-grid N=512/2048/4096 are held out from tuning.

Twenty repetitions describe within-case variation, not independent-session uncertainty. Explicit allocation ledgers are not total process peaks. Fused stage timings are unavailable. Nsight instrumented timings are excluded. Missing parts and missing counters are declared in aggregation_audit.json; an A/C or smoke run is not the final A/B/C experiment. Native data, commands and additional insights remain in parts/.
