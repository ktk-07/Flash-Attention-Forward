#pragma once
#include "attention_types.hpp"
#include <string>
namespace attention {
enum class BaselineVariant { Naive, TiledMatmul, ParallelSoftmax, Optimized };
struct GpuResult {
    std::vector<float> scores, probabilities, output;
};
// Diagnostic forward with allocation and copies
GpuResult baseline(const Input& input, bool causal = false,
                   BaselineVariant variant = BaselineVariant::Naive);
std::string gpu_description();
}
