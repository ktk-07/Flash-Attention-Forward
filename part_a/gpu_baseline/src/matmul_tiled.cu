#include "kernels.cuh"
#include <cmath>
#include <stdexcept>
namespace attention::detail {
namespace {
std::size_t tile_count(std::size_t n, unsigned tile) { return n / tile + (n % tile != 0); }
unsigned tile_blocks(std::size_t n) {
    return static_cast<unsigned>(std::min<std::size_t>(n, 65535));
}
}
template<unsigned tile>
__global__ void qk_tiled_kernel(const float* q, const float* k, float* s,
                               Shape shape, std::size_t tiles_n,
                               std::size_t total_tiles, float scale, bool causal) {
    __shared__ float q_tile[tile][tile];
    // Padded K tile
    __shared__ float k_tile[tile][tile + 1];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    for (std::size_t block = blockIdx.x; block < total_tiles; block += gridDim.x) {
        const auto bh = block / (tiles_n * tiles_n);
        const auto local = block % (tiles_n * tiles_n);
        const auto query_start = (local / tiles_n) * tile;
        const auto key_start = (local % tiles_n) * tile;
        const auto i = query_start + ty, j = key_start + tx;
        // Uniform skip for masked tiles
        if (causal && key_start > query_start + tile - 1) {
            if (i < shape.N && j < shape.N)
                s[(bh * shape.N + i) * shape.N + j] = -CUDART_INF_F;
            continue;
        }
        const auto base = bh * shape.N * shape.d;
        float accumulator = 0.0f; // Register accumulator
        for (std::size_t feature = 0; feature < shape.d; feature += tile) {
            // Coalesced feature loads
            q_tile[ty][tx] = i < shape.N && feature + tx < shape.d
                ? q[base + i * shape.d + feature + tx] : 0.0f;
            const auto key_row = key_start + ty;
            k_tile[ty][tx] = key_row < shape.N && feature + tx < shape.d
                ? k[base + key_row * shape.d + feature + tx] : 0.0f;
            __syncthreads(); // Wait for tile loads
            #pragma unroll
            for (unsigned r = 0; r < tile; ++r)
                accumulator += q_tile[ty][r] * k_tile[tx][r];
            __syncthreads(); // Protect tile reuse
        }
        if (i < shape.N && j < shape.N)
            s[(bh * shape.N + i) * shape.N + j] = causal && j > i
                ? -CUDART_INF_F : accumulator * scale;
        // Shared storage ready for reuse
    }
}
template<unsigned tile>
__global__ void pv_tiled_kernel(const float* p, const float* v, float* o,
                               Shape shape, std::size_t tiles_n,
                               std::size_t tiles_d, std::size_t total_tiles) {
    __shared__ float p_tile[tile][tile];
    __shared__ float v_tile[tile][tile];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    for (std::size_t block = blockIdx.x; block < total_tiles; block += gridDim.x) {
        const auto bh = block / (tiles_n * tiles_d);
        const auto local = block % (tiles_n * tiles_d);
        const auto i = (local / tiles_d) * tile + ty;
        const auto r = (local % tiles_d) * tile + tx;
        float accumulator = 0.0f;
        for (std::size_t key = 0; key < shape.N; key += tile) {
            p_tile[ty][tx] = i < shape.N && key + tx < shape.N
                ? p[(bh * shape.N + i) * shape.N + key + tx] : 0.0f;
            v_tile[ty][tx] = key + ty < shape.N && r < shape.d
                ? v[(bh * shape.N + key + ty) * shape.d + r] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (unsigned j = 0; j < tile; ++j)
                accumulator += p_tile[ty][j] * v_tile[j][tx];
            __syncthreads();
        }
        if (i < shape.N && r < shape.d)
            o[(bh * shape.N + i) * shape.d + r] = accumulator;
    }
}
template<unsigned tile>
void qk_dispatch(const float* q,const float* k,float* s,Shape shape,bool causal) {
    const auto tiles_n=tile_count(shape.N,tile);
    const auto count=shape.B*shape.H*tiles_n*tiles_n;
    const float scale=1.0f/std::sqrt(static_cast<float>(shape.d));
    qk_tiled_kernel<tile><<<tile_blocks(count),dim3(tile,tile)>>>(q,k,s,shape,tiles_n,count,scale,causal);
}
template<unsigned tile>
void pv_dispatch(const float* p,const float* v,float* o,Shape shape) {
    const auto tiles_n=tile_count(shape.N,tile),tiles_d=tile_count(shape.d,tile);
    const auto count=shape.B*shape.H*tiles_n*tiles_d;
    pv_tiled_kernel<tile><<<tile_blocks(count),dim3(tile,tile)>>>(p,v,o,shape,tiles_n,tiles_d,count);
}
void launch_qk_tiled(const float* q,const float* k,float* s,Shape shape,bool causal,unsigned tile) {
    switch(tile) {
    case 8: qk_dispatch<8>(q,k,s,shape,causal); break;
    case 16: qk_dispatch<16>(q,k,s,shape,causal); break;
    case 32: qk_dispatch<32>(q,k,s,shape,causal); break;
    default: throw std::invalid_argument("Unsupported QK tile");
    }
}
void launch_pv_tiled(const float* p,const float* v,float* o,Shape shape,unsigned tile) {
    switch(tile) {
    case 8: pv_dispatch<8>(p,v,o,shape); break;
    case 16: pv_dispatch<16>(p,v,o,shape); break;
    case 32: pv_dispatch<32>(p,v,o,shape); break;
    default: throw std::invalid_argument("Unsupported PV tile");
    }
}
template<unsigned tile>
KernelResources tiled_info(Shape shape,bool qk) {
    const auto n=tile_count(shape.N,tile),d=tile_count(shape.d,tile);
    if(qk) return kernel_info("QK",qk_tiled_kernel<tile>,tile*tile,tile_blocks(shape.B*shape.H*n*n));
    return kernel_info("PV",pv_tiled_kernel<tile>,tile*tile,tile_blocks(shape.B*shape.H*n*d));
}
KernelResources tile_resources(Shape shape,unsigned tile,bool qk) {
    switch(tile) {
    case 8: return tiled_info<8>(shape,qk);
    case 16: return tiled_info<16>(shape,qk);
    case 32: return tiled_info<32>(shape,qk);
    default: throw std::invalid_argument("Unsupported tile");
    }
}
KernelResources qk_tiled_resources(Shape shape,unsigned tile) { return tile_resources(shape,tile,true); }
KernelResources pv_tiled_resources(Shape shape,unsigned tile) { return tile_resources(shape,tile,false); }
}
