#pragma once
#include "attention_types.hpp"
namespace attention {
struct Result { std::vector<double> scores, probabilities, output; };
// FP64 oracle with S/P layout [B,H,N,N]
Result reference(const Input& input, bool causal = false);
}
