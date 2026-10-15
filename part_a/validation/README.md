# Numerical validation

`compare.hpp` compares FP32 GPU results with the FP64 CPU reference. Materialized implementations compare S/P/O. Fused implementations can use the same comparator for O.

Acceptance is `abs(actual-reference)<=atol+rtol*abs(reference)`, with `atol=rtol=1e-4`. Maximum absolute and relative errors are reported. Relative reporting uses `abs(error)/max(abs(reference),1e-12)`; this denominator does not affect acceptance.

Matching negative infinity is accepted only for masked scores. NaN, unexpected infinity, size mismatch and invalid tolerances fail. Tests check the acceptance rule, nonfinite handling, error reporting and invalid arguments. See [measured results](../results/README.md) for completed GPU validation.
