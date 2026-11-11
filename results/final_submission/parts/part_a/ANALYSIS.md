# Part A

All timings link to shared FP64 oracle validation. N=256,d=64 uses full O comparison. Other large outputs use three sampled query rows per head plus all-output finiteness checks. Both Tensor Core variants use FP16 Q/K/V/P and FP32 S, accumulation, softmax arithmetic and O. Materialized normalized P rounding differs from Part C tile-wise unnormalized P rounding. Stage timings are independent passes, not components to sum into a forward median. All native outputs and profiler exports remain under raw/part_a/.
