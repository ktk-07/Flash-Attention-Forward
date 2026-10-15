# CPU reference

The CPU reference computes scaled QK^T, stable row-wise softmax and PV with double-precision multiplication and accumulation. Inputs are the same FP32 values supplied to GPU implementations. The reference supports dense and causal attention and returns S, P and O.

```cpp
#include "attention_reference.hpp"
#include "input_generation.hpp"

const auto input = attention::synthetic_input({1,1,33,32}, 5206);
const auto output = attention::reference(input, false);
```

The CPU is a numerical oracle. Its runtime is excluded from GPU latency and speedup comparisons. Tests cover known values, singleton and uniform cases, stability, head isolation, causal prefixes, irregular shapes and invalid inputs.

Build from the Part A root. This directory also supports a standalone build using `cmake -S cpu_reference -B build-cpu-only`. The demo prints S, P and O for a small dense and causal example.
