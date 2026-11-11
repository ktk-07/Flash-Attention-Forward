#include "attention_types.hpp"
#include <mma.h>

namespace attention::detail {
namespace {
constexpr unsigned tile = 16;
std::size_t tiles(std::size_t n) { return (n+tile-1)/tile; }
unsigned grid(std::size_t count) { return static_cast<unsigned>(std::min<std::size_t>(count,65535)); }
}

// Register layout follows NVIDIA m16n8k16 and Part C
__device__ __forceinline__ unsigned pack_half2(__half low,__half high) {
    return unsigned(__half_as_ushort(low)) | (unsigned(__half_as_ushort(high))<<16);
}
__device__ __forceinline__ void mma(float (&c)[4],const unsigned (&a)[4],const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    asm volatile("trap;");
#endif
}

// One warp per output tile
// not sure if more warps per block will be faster
template<bool qk,bool ptx>
__global__ void matmul_fp16_kernel(const __half* a,const __half* b,float* out,
                                  Shape shape,bool causal,std::size_t columns,
                                  std::size_t inner,std::size_t column_tiles,std::size_t count) {
    // WMMA-aligned tiles
    __shared__ __align__(32) __half left[tile*tile];
    __shared__ __align__(32) __half right[tile*tile];
    __shared__ __align__(32) float product[tile*tile];
    const unsigned lane=threadIdx.x;
    const auto row_tiles=(shape.N+tile-1)/tile;
    const __half zero=__float2half(0.0f);
    const float scale=rsqrtf(float(shape.d));
    const float masked=-CUDART_INF_F;

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
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc;
        if constexpr(!ptx) nvcuda::wmma::fill_fragment(acc,0.0f);
        float registers[2][4] = {};

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

            if constexpr(ptx) {
                const unsigned group=lane/4,lo=2*(lane%4),hi=lo+8;
                unsigned af[4]={pack_half2(left[group*tile+lo],left[group*tile+lo+1]),
                    pack_half2(left[(group+8)*tile+lo],left[(group+8)*tile+lo+1]),
                    pack_half2(left[group*tile+hi],left[group*tile+hi+1]),
                    pack_half2(left[(group+8)*tile+hi],left[(group+8)*tile+hi+1])};
                for(unsigned part=0;part<2;++part) {
                    const unsigned col=part*8+group;
                    unsigned bf[2]={pack_half2(right[lo*tile+col],right[(lo+1)*tile+col]),
                                    pack_half2(right[hi*tile+col],right[(hi+1)*tile+col])};
                    mma(registers[part],af,bf);
                }
            } else {
                nvcuda::wmma::load_matrix_sync(af,left,tile);
                nvcuda::wmma::load_matrix_sync(bf,right,tile);
                nvcuda::wmma::mma_sync(acc,af,bf,acc);
            }
            __syncwarp();
        }

        // write the tile back
        if constexpr(ptx) {
            for(unsigned part=0;part<2;++part) for(unsigned r=0;r<4;++r)
                product[(lane/4+(r/2)*8)*tile+part*8+2*(lane%4)+r%2]=registers[part][r];
        } else nvcuda::wmma::store_matrix_sync(product,acc,tile,nvcuda::wmma::mem_row_major);
        __syncwarp();
        for(unsigned t=lane;t<tile*tile;t+=32) {
            const auto i=row_start+t/tile,j=col_start+t%tile;
            if(i<shape.N && j<columns) {
                if constexpr(qk)
                    out[(bh*shape.N+i)*columns+j]=causal && j>i ? masked : product[t]*scale;
                else out[(bh*shape.N+i)*columns+j]=product[t];
            }
        }
        __syncwarp();
    }
}

void launch_qk_tensorcore(const __half* q,const __half* k,float* s,Shape shape,bool causal,bool ptx) {
    const auto count=shape.B*shape.H*tiles(shape.N)*tiles(shape.N);
    if(ptx) matmul_fp16_kernel<true,true><<<grid(count),32>>>(q,k,s,shape,causal,shape.N,shape.d,tiles(shape.N),count);
    else matmul_fp16_kernel<true,false><<<grid(count),32>>>(q,k,s,shape,causal,shape.N,shape.d,tiles(shape.N),count);
}

void launch_pv_tensorcore(const __half* p,const __half* v,float* o,Shape shape,bool ptx) {
    const auto count=shape.B*shape.H*tiles(shape.N)*tiles(shape.d);
    if(ptx) matmul_fp16_kernel<false,true><<<grid(count),32>>>(p,v,o,shape,false,shape.d,shape.N,tiles(shape.d),count);
    else matmul_fp16_kernel<false,false><<<grid(count),32>>>(p,v,o,shape,false,shape.d,shape.N,tiles(shape.d),count);
}

KernelResources qk_tensorcore_resources(Shape s,bool ptx) {
    const auto blocks=grid(s.B*s.H*tiles(s.N)*tiles(s.N));
    if(ptx) return kernel_info("QK",matmul_fp16_kernel<true,true>,32,blocks);
    return kernel_info("QK",matmul_fp16_kernel<true,false>,32,blocks);
}
KernelResources pv_tensorcore_resources(Shape s,bool ptx) {
    const auto blocks=grid(s.B*s.H*tiles(s.N)*tiles(s.d));
    if(ptx) return kernel_info("PV",matmul_fp16_kernel<false,true>,32,blocks);
    return kernel_info("PV",matmul_fp16_kernel<false,false>,32,blocks);
}
}
