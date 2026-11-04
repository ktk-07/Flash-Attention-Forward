#pragma once
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#ifdef PART_A_ENABLE_CUDA
#include <cuda_fp16.h>
#endif

namespace attention {

// Shared shapes and arrays
struct Shape { std::size_t B, H, N, d; };
struct Input { Shape shape; std::vector<float> q, k, v; };
struct Result { std::vector<double> scores, probabilities, output; };
enum class BaselineVariant { Naive, TiledMatmul, ParallelSoftmax, Optimized, TensorCore };
struct KernelConfig { unsigned tile=16, softmax_threads=256; };
struct KernelResources {
    std::string stage;
    unsigned blocks, threads;
    int registers_per_thread, active_blocks_per_sm;
    std::size_t shared_bytes, local_bytes_per_thread;
    double theoretical_occupancy_percent;
};
// FP64 reference
Result reference(const Input& input, bool causal=false);
namespace detail {

// GPU entry points
void launch_qk(const float* q, const float* k, float* s, Shape shape, bool causal);
void launch_softmax(const float* s, float* p, Shape shape, bool causal);
void launch_pv(const float* p, const float* v, float* o, Shape shape);
void launch_qk_tiled(const float* q, const float* k, float* s, Shape shape, bool causal, unsigned tile = 16);
void launch_softmax_parallel(const float* s, float* p, Shape shape, bool causal, unsigned threads = 256);
void launch_pv_tiled(const float* p, const float* v, float* o, Shape shape, unsigned tile = 16);
#ifdef PART_A_ENABLE_CUDA
void launch_qk_tensorcore(const __half* q,const __half* k,__half* s,Shape shape,bool causal);
void launch_pv_tensorcore(const __half* p,const __half* v,__half* o,Shape shape);
void launch_softmax_fp16(const __half* s,__half* p,Shape shape,bool causal,unsigned threads);
KernelResources qk_tensorcore_resources(Shape shape);
KernelResources pv_tensorcore_resources(Shape shape);
KernelResources softmax_fp16_resources(Shape shape,unsigned threads);
#endif
KernelResources qk_resources(Shape shape);
KernelResources softmax_resources(Shape shape);
KernelResources pv_resources(Shape shape);
KernelResources qk_tiled_resources(Shape shape,unsigned tile);
KernelResources pv_tiled_resources(Shape shape,unsigned tile);
KernelResources softmax_parallel_resources(Shape shape,unsigned threads);
}
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>
namespace attention::detail {
constexpr unsigned threads_per_block = 256;
inline unsigned blocks_for(std::size_t count) {
    // Bounded grid-stride launch
    const auto blocks = count / threads_per_block + (count % threads_per_block != 0);
    return static_cast<unsigned>(std::min<std::size_t>(blocks, 65535));
}

inline void cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

template<class Kernel>
KernelResources kernel_info(const char* stage,Kernel kernel,unsigned threads,unsigned blocks) {
    // resource usage
    cudaFuncAttributes attributes{};
    cudaDeviceProp device{};
    int index,active;
    cuda_check(cudaGetDevice(&index),"current device");
    cuda_check(cudaGetDeviceProperties(&device,index),"device properties");
    cuda_check(cudaFuncGetAttributes(&attributes,kernel),"kernel attributes");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,0),"kernel occupancy");
    return {stage,blocks,threads,attributes.numRegs,active,attributes.sharedSizeBytes,attributes.localSizeBytes,
            100.0*active*threads/device.maxThreadsPerMultiProcessor};
}
}
#endif
