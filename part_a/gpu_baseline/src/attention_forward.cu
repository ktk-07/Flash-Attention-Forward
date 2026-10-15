#include "attention_baseline.hpp"
#include "launch.cuh"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace attention {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Shape/byte count overflow");
    return a * b;
}
// RAII device allocation
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) (void)cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* data() const { return data_; }
    void release_checked() {
        if (data_) { check(cudaFree(data_), "cudaFree"); data_ = nullptr; }
    }
private:
    float* data_ = nullptr;
};
}
std::string gpu_description() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp info{};
    check(cudaGetDeviceProperties(&info, device), "cudaGetDeviceProperties");
    return std::string(info.name) + "; compute capability " + std::to_string(info.major)
        + "." + std::to_string(info.minor) + "; VRAM bytes " + std::to_string(info.totalGlobalMem);
}
GpuResult baseline(const Input& input, bool causal, BaselineVariant variant) {
    detail::validate_configuration(variant);
    const auto [B,H,N,d] = input.shape;
    if (!B || !H || !N || !d) throw std::invalid_argument("Dimensions must be positive");
    const auto heads = product(B,H);
    const auto elements = product(product(heads,N),d);
    const auto matrix_elements = product(product(heads,N),N);
    const auto tensor_bytes = product(elements,sizeof(float));
    const auto matrix_bytes = product(matrix_elements,sizeof(float));
    for (const auto* tensor : {&input.q, &input.k, &input.v}) {
        if (tensor->size() != elements) throw std::invalid_argument("Input size mismatch");
        for (float value : *tensor)
            if (!std::isfinite(value)) throw std::invalid_argument("Inputs must be finite");
    }
    DeviceBuffer q(tensor_bytes), k(tensor_bytes), v(tensor_bytes);
    DeviceBuffer s(matrix_bytes), p(matrix_bytes), o(tensor_bytes);
    check(cudaMemcpy(q.data(),input.q.data(),tensor_bytes,cudaMemcpyHostToDevice), "copy Q");
    check(cudaMemcpy(k.data(),input.k.data(),tensor_bytes,cudaMemcpyHostToDevice), "copy K");
    check(cudaMemcpy(v.data(),input.v.data(),tensor_bytes,cudaMemcpyHostToDevice), "copy V");
    // Ordered default-stream launches
    detail::qk_stage(q.data(),k.data(),s.data(),input.shape,causal,variant);
    detail::softmax_stage(s.data(),p.data(),input.shape,causal,variant);
    detail::pv_stage(p.data(),v.data(),o.data(),input.shape,variant);
    check(cudaDeviceSynchronize(), "execute attention kernels");
    GpuResult result{std::vector<float>(matrix_elements),std::vector<float>(matrix_elements),
                     std::vector<float>(elements)};
    check(cudaMemcpy(result.scores.data(),s.data(),matrix_bytes,cudaMemcpyDeviceToHost), "copy S");
    check(cudaMemcpy(result.probabilities.data(),p.data(),matrix_bytes,cudaMemcpyDeviceToHost), "copy P");
    check(cudaMemcpy(result.output.data(),o.data(),tensor_bytes,cudaMemcpyDeviceToHost), "copy O");
    q.release_checked(); k.release_checked(); v.release_checked();
    s.release_checked(); p.release_checked(); o.release_checked();
    return result;
}
}
