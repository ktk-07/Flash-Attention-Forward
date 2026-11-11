#pragma once
#include "attention_types.hpp"
#include <cuda_runtime.h>
#include <string>
namespace dense_attention {
struct Config { std::string variant="tiled"; int warps=4; int key_tile=32; };
// Contiguous BHNd FP32 inputs/output. No NxN global workspace; d <= 256.
void forward(const float* q,const float* k,const float* v,float* out,
             attention::Shape shape,const Config& config,cudaStream_t stream=nullptr);
attention::KernelResources resources(attention::Shape shape,const Config& config);
std::string symbol(const Config& config);
}
