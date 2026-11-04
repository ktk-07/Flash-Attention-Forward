#include "attention_types.hpp"
#include <stdexcept>
namespace attention::detail {
template<unsigned threads>
__global__ void softmax_parallel_kernel(const float* s, float* p, Shape shape,
                                        bool causal, std::size_t rows) {
    __shared__ float partial[threads];
    const unsigned tid = threadIdx.x;
    // One row per block
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto offset = row * shape.N;
        const auto valid = causal ? row % shape.N + 1 : shape.N;
        float local_max = -CUDART_INF_F;
        for (std::size_t j = tid; j < valid; j += blockDim.x)
            local_max = fmaxf(local_max,s[offset+j]);

        partial[tid] = local_max;
        __syncthreads();
        // Power-of-two maximum reduction
        for (unsigned step = blockDim.x / 2; step; step /= 2) {
            if (tid < step) partial[tid] = fmaxf(partial[tid],partial[tid+step]);
            __syncthreads();
        }
        const float maximum = partial[0];
        __syncthreads(); // Protect maximum before reuse

        // local exponent sum
        float local_sum = 0.0f;
        for (std::size_t j = tid; j < shape.N; j += blockDim.x) {
            const float value = j < valid ? expf(s[offset+j]-maximum) : 0.0f;
            p[offset+j] = value;
            local_sum += value;
        }

        partial[tid] = local_sum;
        __syncthreads();
        for (unsigned step = blockDim.x / 2; step; step /= 2) {
            if (tid < step) partial[tid] += partial[tid+step];
            __syncthreads();
        }

        // normalise this row
        const float sum = partial[0];
        for (std::size_t j = tid; j < shape.N; j += blockDim.x) p[offset+j] /= sum;
        __syncthreads(); // Protect sum before next row
    }
}

void launch_softmax_parallel(const float* s,float* p,Shape shape,bool causal,unsigned threads) {
    const auto rows=shape.B*shape.H*shape.N;
    const auto blocks=static_cast<unsigned>(std::min<std::size_t>(rows,65535));
    if(threads==128) softmax_parallel_kernel<128><<<blocks,128>>>(s,p,shape,causal,rows);
    else if(threads==256) softmax_parallel_kernel<256><<<blocks,256>>>(s,p,shape,causal,rows);
    else throw std::invalid_argument("Unsupported softmax block");
}

KernelResources softmax_parallel_resources(Shape shape,unsigned threads) {
    const auto blocks=static_cast<unsigned>(std::min<std::size_t>(shape.B*shape.H*shape.N,65535));
    if(threads==128) return kernel_info("softmax",softmax_parallel_kernel<128>,128,blocks);
    if(threads==256) return kernel_info("softmax",softmax_parallel_kernel<256>,256,blocks);
    throw std::invalid_argument("Unsupported softmax block");
}
// FP16 arithmetic and reduction
template<unsigned threads>
__global__ void softmax_fp16_kernel(const __half* s,__half* p,Shape shape,bool causal,std::size_t rows) {
    __shared__ __half partial[threads];
    const unsigned tid=threadIdx.x;
    const __half zero=__float2half(0.0f);
    for(std::size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        const auto offset=row*shape.N;
        const auto valid=causal ? row%shape.N+1 : shape.N;
        __half maximum=__float2half(-CUDART_INF_F);
        for(std::size_t j=tid;j<valid;j+=threads) maximum=__hmax(maximum,s[offset+j]);

        // row maximum
        partial[tid]=maximum;
        __syncthreads();
        for(unsigned step=threads/2;step;step/=2) {
            if(tid<step) partial[tid]=__hmax(partial[tid],partial[tid+step]);
            __syncthreads();
        }
        maximum=partial[0];
        __syncthreads();

        // keep the sum in FP16
        __half sum=zero;
        for(std::size_t j=tid;j<shape.N;j+=threads) {
            const __half value=j<valid ? hexp(__hsub(s[offset+j],maximum)) : zero;
            p[offset+j]=value;
            sum=__hadd(sum,value);
        }

        partial[tid]=sum;
        __syncthreads();
        for(unsigned step=threads/2;step;step/=2) {
            if(tid<step) partial[tid]=__hadd(partial[tid],partial[tid+step]);
            __syncthreads();
        }

        // divide by the row sum
        sum=partial[0];
        for(std::size_t j=tid;j<shape.N;j+=threads) p[offset+j]=__hdiv(p[offset+j],sum);
        __syncthreads();
    }
}

// not sure if 256 threads will be faster for longer rows
void launch_softmax_fp16(const __half* s,__half* p,Shape shape,bool causal,unsigned threads) {
    const auto rows=shape.B*shape.H*shape.N;
    const auto blocks=static_cast<unsigned>(std::min<std::size_t>(rows,65535));
    if(threads==128) softmax_fp16_kernel<128><<<blocks,128>>>(s,p,shape,causal,rows);
    else softmax_fp16_kernel<256><<<blocks,256>>>(s,p,shape,causal,rows);
}

KernelResources softmax_fp16_resources(Shape shape,unsigned threads) {
    const auto rows=shape.B*shape.H*shape.N;
    const auto blocks=static_cast<unsigned>(std::min<std::size_t>(rows,65535));
    if(threads==128) return kernel_info("softmax",softmax_fp16_kernel<128>,128,blocks);
    return kernel_info("softmax",softmax_fp16_kernel<256>,256,blocks);
}

}
