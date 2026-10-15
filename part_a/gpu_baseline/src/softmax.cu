#include "kernels.cuh"
namespace attention::detail {
__global__ void softmax_kernel(const float* s, float* p, Shape shape,
                               bool causal, std::size_t rows) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t row = start; row < rows; row += stride) {
        const auto offset = row * shape.N;
        const auto valid = causal ? row % shape.N + 1 : shape.N;
        float maximum = -CUDART_INF_F;
        // One row per thread
        for (std::size_t j = 0; j < valid; ++j)
            maximum = fmaxf(maximum, s[offset + j]);
        float sum = 0.0f;
        for (std::size_t j = 0; j < valid; ++j) {
            const float value = expf(s[offset + j] - maximum);
            p[offset + j] = value;
            sum += value;
        }
        for (std::size_t j = 0; j < valid; ++j) p[offset + j] /= sum;
        // Write masked P=0
        for (std::size_t j = valid; j < shape.N; ++j) p[offset + j] = 0.0f;
    }
}
void launch_softmax(const float* s, float* p, Shape shape, bool causal) {
    const auto rows = shape.B * shape.H * shape.N;
    softmax_kernel<<<blocks_for(rows), threads_per_block>>>(s, p, shape, causal, rows);
}
KernelResources softmax_resources(Shape shape) {
    return kernel_info("softmax",softmax_kernel,threads_per_block,blocks_for(shape.B*shape.H*shape.N));
}
}
