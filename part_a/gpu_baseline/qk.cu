#include "attention_types.hpp"
#include <cmath>
namespace attention::detail {
__global__ void qk_kernel(const float* q, const float* k, float* s,
                          Shape shape, bool causal, float scale, std::size_t count) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t t = start; t < count; t += stride) {
        // flat index to head,row,column
        const auto bh = t / (shape.N * shape.N);
        const auto local = t % (shape.N * shape.N);
        const auto i = local / shape.N;
        const auto j = local % shape.N;

        // skip future keys
        if (causal && j > i) {
            s[t] = -CUDART_INF_F;
            continue;
        }

        const auto base = bh * shape.N * shape.d;
        float sum = 0.0f;
        // One score per thread
        for (std::size_t r = 0; r < shape.d; ++r)
            sum += q[base + i * shape.d + r] * k[base + j * shape.d + r];
        s[t] = sum * scale;
    }
}

void launch_qk(const float* q, const float* k, float* s, Shape shape, bool causal) {
    const auto count = shape.B * shape.H * shape.N * shape.N;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.d));
    qk_kernel<<<blocks_for(count), threads_per_block>>>(q, k, s, shape, causal, scale, count);
}

KernelResources qk_resources(Shape shape) {
    return kernel_info("QK",qk_kernel,threads_per_block,blocks_for(shape.B*shape.H*shape.N*shape.N));
}
}
