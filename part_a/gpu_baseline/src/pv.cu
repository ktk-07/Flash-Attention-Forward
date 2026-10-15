#include "kernels.cuh"
namespace attention::detail {
__global__ void pv_kernel(const float* p, const float* v, float* o,
                          Shape shape, std::size_t count) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t t = start; t < count; t += stride) {
        const auto bh = t / (shape.N * shape.d);
        const auto local = t % (shape.N * shape.d);
        const auto i = local / shape.d;
        const auto r = local % shape.d;
        const auto p_base = (bh * shape.N + i) * shape.N;
        const auto v_base = bh * shape.N * shape.d;
        float sum = 0.0f;
        // One output element per thread
        for (std::size_t j = 0; j < shape.N; ++j)
            sum += p[p_base + j] * v[v_base + j * shape.d + r];
        o[t] = sum;
    }
}
void launch_pv(const float* p, const float* v, float* o, Shape shape) {
    const auto count = shape.B * shape.H * shape.N * shape.d;
    pv_kernel<<<blocks_for(count), threads_per_block>>>(p, v, o, shape, count);
}
KernelResources pv_resources(Shape shape) {
    return kernel_info("PV",pv_kernel,threads_per_block,blocks_for(shape.B*shape.H*shape.N*shape.d));
}
}
