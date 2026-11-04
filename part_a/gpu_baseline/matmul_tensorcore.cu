#include "attention_types.hpp"
#include <mma.h>

namespace attention::detail {
namespace {
constexpr unsigned tile = 16;
std::size_t tiles(std::size_t n) { return (n+tile-1)/tile; }
unsigned grid(std::size_t count) { return static_cast<unsigned>(std::min<std::size_t>(count,65535)); }
}

// One warp per output tile
// not sure if more warps per block will be faster
template<bool qk>
__global__ void matmul_fp16_kernel(const __half* a,const __half* b,__half* out,
                                  Shape shape,bool causal,std::size_t columns,
                                  std::size_t inner,std::size_t column_tiles,std::size_t count) {
    // WMMA-aligned tiles
    __shared__ __align__(32) __half left[tile*tile];
    __shared__ __align__(32) __half right[tile*tile];
    __shared__ __align__(32) __half product[tile*tile];
    const unsigned lane=threadIdx.x;
    const auto row_tiles=(shape.N+tile-1)/tile;
    const __half zero=__float2half(0.0f);
    const __half scale=hrsqrt(__float2half_rn(static_cast<float>(shape.d)));
    const __half masked=__float2half(-CUDART_INF_F);

    for(std::size_t block=blockIdx.x;block<count;block+=gridDim.x) {
        const auto bh=block/(row_tiles*column_tiles);
        const auto local=block%(row_tiles*column_tiles);
        const auto row_start=(local/column_tiles)*tile;
        const auto col_start=(local%column_tiles)*tile;

        // skip future tiles
        if(qk && causal && col_start>row_start+tile-1) {
            for(unsigned t=lane;t<tile*tile;t+=32) {
                const auto i=row_start+t/tile,j=col_start+t%tile;
                if(i<shape.N && j<columns) out[(bh*shape.N+i)*columns+j]=masked;
            }
            continue;
        }

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::row_major> af;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::row_major> bf;
        // not sure if S,P,O should all use FP16
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,__half> acc;
        nvcuda::wmma::fill_fragment(acc,zero);

        for(std::size_t k=0;k<inner;k+=tile) {
            // load and pad edges
            for(unsigned t=lane;t<tile*tile;t+=32) {
                const auto r=t/tile,c=t%tile;
                const auto i=row_start+r,j=col_start+c;
                left[t]=i<shape.N && k+c<inner ? a[(bh*shape.N+i)*inner+k+c] : zero;
                if constexpr(qk)
                    right[t]=j<columns && k+r<inner ? b[(bh*shape.N+j)*inner+k+r] : zero;
                else
                    right[t]=k+r<inner && j<columns ? b[(bh*inner+k+r)*columns+j] : zero;
            }
            __syncwarp();

            // warp matrix multiply
            nvcuda::wmma::load_matrix_sync(af,left,tile);
            nvcuda::wmma::load_matrix_sync(bf,right,tile);
            nvcuda::wmma::mma_sync(acc,af,bf,acc);
            __syncwarp();
        }

        // write the tile back
        nvcuda::wmma::store_matrix_sync(product,acc,tile,nvcuda::wmma::mem_row_major);
        __syncwarp();
        for(unsigned t=lane;t<tile*tile;t+=32) {
            const auto i=row_start+t/tile,j=col_start+t%tile;
            if(i<shape.N && j<columns) {
                if constexpr(qk)
                    out[(bh*shape.N+i)*columns+j]=causal && j>i ? masked : __hmul(product[t],scale);
                else out[(bh*shape.N+i)*columns+j]=product[t];
            }
        }
        __syncwarp();
    }
}

void launch_qk_tensorcore(const __half* q,const __half* k,__half* s,Shape shape,bool causal) {
    const auto count=shape.B*shape.H*tiles(shape.N)*tiles(shape.N);
    matmul_fp16_kernel<true><<<grid(count),32>>>(q,k,s,shape,causal,shape.N,shape.d,tiles(shape.N),count);
}

void launch_pv_tensorcore(const __half* p,const __half* v,__half* o,Shape shape) {
    const auto count=shape.B*shape.H*tiles(shape.N)*tiles(shape.d);
    matmul_fp16_kernel<false><<<grid(count),32>>>(p,v,o,shape,false,shape.d,shape.N,tiles(shape.d),count);
}

KernelResources qk_tensorcore_resources(Shape shape) {
    return kernel_info("QK",matmul_fp16_kernel<true>,32,grid(shape.B*shape.H*tiles(shape.N)*tiles(shape.N)));
}

KernelResources pv_tensorcore_resources(Shape shape) {
    return kernel_info("PV",matmul_fp16_kernel<false>,32,grid(shape.B*shape.H*tiles(shape.N)*tiles(shape.d)));
}
}
