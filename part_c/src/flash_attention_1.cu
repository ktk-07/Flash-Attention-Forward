#include "flash_attention/flash_attention_1.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <stdexcept>

// Algorithmic basis: Dao et al., FlashAttention (2022), Algorithm 1.
// https://arxiv.org/abs/2205.14135
// Educational reference: Peter Kim, flash-attention-minimal (Apache-2.0).
// https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
// Independently written for this project's contiguous CUDA C++ interface:
// cooperative staging, separate Q/O SRAM tiles, safe tails, causal/dense modes,
// collective barriers, internal first-tile initialisation and caller-owned m/l.
// No PyTorch, WMMA, inline MMA, or Tensor Core instructions are used.
//
// FA1's distinctive schedule retains K/V in SRAM while iterating over Q tiles.
// Normalised O and row statistics m/l are read/written in global memory between
// K/V iterations. This is different from the query-resident FA2-style variants.
// One CTA owns an entire batch/head so there are no cross-CTA state races.
// It is a readable Algorithm-1 baseline, not an optimised official FA1 kernel.
namespace {
constexpr int block_q = 32;
constexpr int block_k = 32;
constexpr int threads = 32;

template<int D, bool Causal>
__global__ void flash_attention_v1_tiled(const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        float* __restrict__ o,
                                        float* __restrict__ maxima,
                                        float* __restrict__ normalizers,
                                        int n, float scale) {
    extern __shared__ float storage[];
    float* q_tile = storage;
    float* k_tile = q_tile + block_q * D;
    float* v_tile = k_tile + block_k * D;
    float* scores = v_tile + block_k * D;
    float* o_tile = scores + block_q * block_k;
    const int tid = threadIdx.x;
    const std::size_t head = blockIdx.x;
    const std::size_t base = head * std::size_t(n) * D;
    const std::size_t state_base = head * std::size_t(n);

    for (int key_start = 0; key_start < n; key_start += block_k) {
        // Linear cooperative loads keep adjacent lanes on adjacent features.
        // Invalid K/V rows are zero-filled; invalid score entries are masked.
        for (int index = tid; index < block_k * D; index += threads) {
            const int row = key_start + index / D;
            k_tile[index] = row < n ? k[base + std::size_t(row) * D + index % D] : 0.0f;
            v_tile[index] = row < n ? v[base + std::size_t(row) * D + index % D] : 0.0f;
        }
        __syncthreads();

        // All lanes use this same causal bound. Fully future block pairs do
        // not touch Q/O or m/l. Dense mode visits every query/key block pair.
        const int first_query = Causal ? (key_start / block_q) * block_q : 0;
        for (int query_start = first_query; query_start < n; query_start += block_q) {
            for (int index = tid; index < block_q * D; index += threads) {
                const int row = query_start + index / D;
                const auto address = base + std::size_t(row) * D + index % D;
                q_tile[index] = row < n ? q[address] : 0.0f;
                // Never read allocator leftovers (including NaNs) on tile 0.
                o_tile[index] = row < n && key_start != 0 ? o[address] : 0.0f;
            }
            __syncthreads();

            // One worker owns each query row and its shared score/P row.
            // Tail workers stay in the CTA and reach every barrier.
            const int query = query_start + tid;
            if (query < n) {
                float tile_max = -CUDART_INF_F;
                for (int column = 0; column < block_k; ++column) {
                    const int key = key_start + column;
                    float score = -CUDART_INF_F;
                    if (key < n && (!Causal || key <= query)) {
                        float dot = 0.0f;
                        for (int feature = 0; feature < D; ++feature)
                            dot += q_tile[tid * D + feature] * k_tile[column * D + feature];
                        score = dot * scale;
                    }
                    scores[tid * block_k + column] = score;
                    tile_max = fmaxf(tile_max, score);
                }
                // Guard an empty score row before subtracting -infinity.
                // For the current equal tiles, every valid causal row has a
                // valid key; this guard also documents the recurrence invariant.
                if (tile_max != -CUDART_INF_F) {
                    float tile_sum = 0.0f;
                    for (int column = 0; column < block_k; ++column) {
                        const float weight = expf(scores[tid * block_k + column] - tile_max);
                        scores[tid * block_k + column] = weight;
                        tile_sum += weight;
                    }
                    const auto state = state_base + query;
                    const float old_max = key_start == 0 ? -CUDART_INF_F : maxima[state];
                    const float old_sum = key_start == 0 ? 0.0f : normalizers[state];
                    const float new_max = fmaxf(old_max, tile_max);
                    const float old_weight = old_sum == 0.0f ? 0.0f : old_sum * expf(old_max - new_max);
                    const float tile_weight = expf(tile_max - new_max);
                    const float new_sum = old_weight + tile_weight * tile_sum;
                    for (int feature = 0; feature < D; ++feature) {
                        float pv = 0.0f;
                        for (int column = 0; column < block_k; ++column)
                            pv += scores[tid * block_k + column] * v_tile[column * D + feature];
                        o_tile[tid * D + feature] =
                            (old_weight * o_tile[tid * D + feature] + tile_weight * pv) / new_sum;
                    }
                    maxima[state] = new_max;
                    normalizers[state] = new_sum;
                }
            }
            __syncthreads(); // All row owners finish before cooperative O stores.
            for (int index = tid; index < block_q * D; index += threads) {
                const int row = query_start + index / D;
                if (row < n) o[base + std::size_t(row) * D + index % D] = o_tile[index];
            }
            // Protect Q/O/S reuse and the next outer iteration's K/V overwrite.
            __syncthreads();
        }
    }
}
void verify(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

namespace part_c {
namespace {
std::size_t query_count(Shape s) {
    if (s.b <= 0 || s.h <= 0 || s.n <= 0 || s.d <= 0)
        throw std::invalid_argument("FA1 requires positive B,H,N,d");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (std::size_t(s.b) > limit / s.h || std::size_t(s.b) * s.h > limit / s.n / 2)
        throw std::invalid_argument("FA1 workspace extent overflow");
    if (std::size_t(s.b) * s.h > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("FA1 batch/head grid exceeds supported extent");
    return std::size_t(s.b) * s.h * s.n;
}
template<int D, bool Causal>
LaunchConfig launch_impl(Shape s, const DevicePointers& x,
                         FlashAttention1Workspace workspace, cudaStream_t stream,
                         KernelResources* resources) {
    constexpr std::size_t shared_bytes = sizeof(float) *
        (2 * block_q * D + 2 * block_k * D + block_q * block_k);
    auto kernel = flash_attention_v1_tiled<D, Causal>;
    // First-use setup occurs before measured repetitions in the runners.
    // Cache per template/device in the calling thread, never allocate in launch.
    static thread_local int prepared_device = -1;
    int device = 0;
    verify(cudaGetDevice(&device));
    if (prepared_device != device) {
        int default_limit = 0;
        verify(cudaDeviceGetAttribute(&default_limit, cudaDevAttrMaxSharedMemoryPerBlock, device));
        if (shared_bytes > std::size_t(default_limit)) {
            int optin_limit = 0;
            verify(cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            if (shared_bytes > std::size_t(optin_limit))
                throw std::invalid_argument("FA1 Q/K/V/S/O shared tile exceeds this GPU's per-block limit");
            verify(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared_bytes)));
        }
        prepared_device = device;
    }
    if (resources) {
        cudaFuncAttributes attributes{};
        cudaDeviceProp properties{};
        int active = 0;
        verify(cudaFuncGetAttributes(&attributes, kernel));
        verify(cudaGetDeviceProperties(&properties, device));
        verify(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, shared_bytes));
        resources->registers_per_thread = attributes.numRegs;
        resources->shared_bytes = attributes.sharedSizeBytes + shared_bytes;
        resources->local_bytes_per_thread = attributes.localSizeBytes;
        resources->active_blocks_per_sm = active;
        resources->occupancy_estimate_percent = 100.0 * active * threads / properties.maxThreadsPerMultiProcessor;
    }
    flash_attention_v1_tiled<D, Causal><<<dim3(unsigned(s.b * s.h)), threads, shared_bytes, stream>>>(
        x.q, x.k, x.v, x.o, workspace.maxima, workspace.normalizers,
        s.n, 1.0f / std::sqrt(float(D)));
    return {block_q, block_k, 1, 1, 1};
}
template<int D>
LaunchConfig select_mode(Shape s, const DevicePointers& x, FlashAttention1Workspace w,
                         bool causal, cudaStream_t stream, KernelResources* resources) {
    return causal ? launch_impl<D, true>(s,x,w,stream,resources)
                  : launch_impl<D, false>(s,x,w,stream,resources);
}
}
std::size_t flash_attention_1_workspace_elements(Shape s) { return 2 * query_count(s); }
LaunchConfig launch_flash_attention_1(Shape s, const DevicePointers& x,
                                     FlashAttention1Workspace w, bool causal,
                                     cudaStream_t stream, KernelResources* resources) {
    query_count(s);
    if (!x.q || !x.k || !x.v || !x.o || !w.maxima || !w.normalizers)
        throw std::invalid_argument("FA1 requires Q/K/V/O and caller-owned maxima/normalizer workspace");
    switch (s.d) {
#define PART_C_DIM(D) case D: return select_mode<D>(s,x,w,causal,stream,resources);
    PART_C_DIM(7) PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(33) PART_C_DIM(40)
    PART_C_DIM(64) PART_C_DIM(65) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default: throw std::invalid_argument("FA1 head dimension not instantiated");
    }
}
}
