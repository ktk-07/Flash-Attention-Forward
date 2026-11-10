#include "flash_attention/flash_attention_1_tensor_core.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <stdexcept>

// FA1 Algorithm-1-style schedule implemented with the WMMA API. This is an
// additional educational comparator, not code from the official FA1 library.
// Unlike the FA2-style query-resident kernels, a single CTA holds a K/V block
// while visiting Q blocks, and writes normalised O/m/l back after each visit.
// Shape/fragment choices intentionally match the existing WMMA baseline where
// possible. No change to the maintained FP32 FA1 or FA2 kernels is required.
namespace {
constexpr int BQ=32, BK=32, NW=2, M=16, N=16, K=16, THREADS=32*NW;
namespace wmma=nvcuda::wmma;

template<int D, bool Causal>
__global__ void flash_attention_v1_wmma(const half* __restrict__ q,
                                       const half* __restrict__ k,
                                       const half* __restrict__ v,
                                       float* __restrict__ o,
                                       float* __restrict__ maxima,
                                       float* __restrict__ normalizers,
                                       int length, float scale) {
    constexpr int PAD=(D+15)/16*16;
    __shared__ __align__(32) half qs[BQ][PAD],ks[BK][PAD],vs[BK][PAD];
    __shared__ __align__(32) float os[BQ][PAD];
    __shared__ __align__(32) float score[NW][M][N],product[NW][M][N];
    __shared__ __align__(32) half probability[NW][M][N];
    __shared__ float row_max[BQ],row_sum[BQ];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int local_row=warp*M+lane/2,row_lane=lane%2;
    const std::size_t head=blockIdx.x;
    const std::size_t base=head*std::size_t(length)*D,state_base=head*std::size_t(length);

    for(int key_start=0;key_start<length;key_start+=BK) {
        for(int i=tid;i<BK*PAD;i+=THREADS) {
            const int key=key_start+i/PAD,feature=i%PAD;
            const auto address=base+std::size_t(key)*D+feature;
            ks[i/PAD][feature]=key<length && feature<D?k[address]:__float2half(0.f);
            vs[i/PAD][feature]=key<length && feature<D?v[address]:__float2half(0.f);
        }
        __syncthreads();
        const int first_query=Causal?(key_start/BQ)*BQ:0;
        for(int query_start=first_query;query_start<length;query_start+=BQ) {
            for(int i=tid;i<BQ*PAD;i+=THREADS) {
                const int query=query_start+i/PAD,feature=i%PAD;
                const auto address=base+std::size_t(query)*D+feature;
                qs[i/PAD][feature]=query<length && feature<D?q[address]:__float2half(0.f);
                os[i/PAD][feature]=query<length && feature<D && key_start!=0?o[address]:0.f;
            }
            if(tid<BQ) {
                const int query=query_start+tid;
                row_max[tid]=query<length && key_start!=0?maxima[state_base+query]:-CUDART_INF_F;
                row_sum[tid]=query<length && key_start!=0?normalizers[state_base+query]:0.f;
            }
            __syncthreads();
            for(int key_local=0;key_local<BK;key_local+=N) {
                wmma::fragment<wmma::accumulator,M,N,K,float> scores;
                wmma::fill_fragment(scores,0.f);
                for(int feature=0;feature<PAD;feature+=K) {
                    wmma::fragment<wmma::matrix_a,M,N,K,half,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,M,N,K,half,wmma::col_major> b;
                    wmma::load_matrix_sync(a,&qs[warp*M][feature],PAD);
                    wmma::load_matrix_sync(b,&ks[key_local][feature],PAD);
                    wmma::mma_sync(scores,a,b,scores);
                }
                wmma::store_matrix_sync(&score[warp][0][0],scores,N,wmma::mem_row_major);
                __syncwarp();

                const int query=query_start+local_row;
                float tile_max=-CUDART_INF_F;
                for(int column=row_lane;column<N;column+=2) {
                    const int key=key_start+key_local+column;
                    const float s=query<length && key<length && (!Causal || key<=query)
                        ?score[warp][lane/2][column]*scale:-CUDART_INF_F;
                    score[warp][lane/2][column]=s;
                    tile_max=fmaxf(tile_max,s);
                }
                tile_max=fmaxf(tile_max,__shfl_xor_sync(0xffffffffu,tile_max,1));
                const float old_max=row_max[local_row],old_sum=row_sum[local_row];
                const float merged_max=fmaxf(old_max,tile_max);
                // An entirely masked subtile leaves an existing row unchanged;
                // empty tail rows stay zero without evaluating -inf - -inf.
                const float alpha=old_sum>0.f?expf(old_max-merged_max):0.f;
                float partial_sum=0.f;
                for(int column=row_lane;column<N;column+=2) {
                    const float s=score[warp][lane/2][column];
                    const float p=s==-CUDART_INF_F?0.f:expf(s-merged_max);
                    partial_sum+=p;
                    probability[warp][lane/2][column]=__float2half_rn(p);
                }
                const float tile_sum=partial_sum+__shfl_xor_sync(0xffffffffu,partial_sum,1);
                const float merged_sum=alpha*old_sum+tile_sum;
                __syncwarp();
                for(int feature=0;feature<PAD;feature+=N) {
                    wmma::fragment<wmma::matrix_a,M,N,K,half,wmma::row_major> p;
                    wmma::fragment<wmma::matrix_b,M,N,K,half,wmma::row_major> values;
                    wmma::fragment<wmma::accumulator,M,N,K,float> pv;
                    wmma::fill_fragment(pv,0.f);
                    wmma::load_matrix_sync(p,&probability[warp][0][0],N);
                    wmma::load_matrix_sync(values,&vs[key_local][feature],PAD);
                    wmma::mma_sync(pv,p,values,pv);
                    wmma::store_matrix_sync(&product[warp][0][0],pv,N,wmma::mem_row_major);
                    __syncwarp();
                    for(int column=row_lane;column<N;column+=2) {
                        const int output_feature=feature+column;
                        os[local_row][output_feature]=merged_sum>0.f?
                            (alpha*old_sum*os[local_row][output_feature]+product[warp][lane/2][column])/merged_sum:0.f;
                    }
                    __syncwarp(); // Product scratch is reused by the next output fragment.
                }
                if(row_lane==0) {row_max[local_row]=merged_max;row_sum[local_row]=merged_sum;}
                __syncwarp(); // Publish row state before the next 16-key subtile.
            }
            __syncthreads();
            for(int i=tid;i<BQ*D;i+=THREADS) {
                const int query=query_start+i/D;
                if(query<length)o[base+std::size_t(query)*D+i%D]=os[i/D][i%D];
            }
            if(tid<BQ && query_start+tid<length) {
                maxima[state_base+query_start+tid]=row_max[tid];
                normalizers[state_base+query_start+tid]=row_sum[tid];
            }
            __syncthreads(); // All Q/O/row-state reads/writes finish before reuse.
        }
    }
}
void verify(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}

namespace part_c {
namespace {
template<int D,bool Causal>
LaunchConfig launch(Shape s,const DevicePointers& x,FlashAttention1Workspace w,
                    cudaStream_t stream,KernelResources* resources) {
    auto kernel=flash_attention_v1_wmma<D,Causal>;
    if(resources) {
        cudaFuncAttributes a{};cudaDeviceProp prop{};int device,active;
        verify(cudaGetDevice(&device));verify(cudaGetDeviceProperties(&prop,device));
        verify(cudaFuncGetAttributes(&a,kernel));
        verify(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,0));
        resources->registers_per_thread=a.numRegs;resources->shared_bytes=a.sharedSizeBytes;
        resources->local_bytes_per_thread=a.localSizeBytes;resources->active_blocks_per_sm=active;
        resources->occupancy_estimate_percent=100.0*active*THREADS/prop.maxThreadsPerMultiProcessor;
    }
    flash_attention_v1_wmma<D,Causal><<<dim3(unsigned(s.b*s.h)),THREADS,0,stream>>>(
        x.hq,x.hk,x.hv,x.o,w.maxima,w.normalizers,s.n,1.f/std::sqrt(float(D)));
    return {BQ,BK,NW,1,1};
}
template<int D>
LaunchConfig mode(Shape s,const DevicePointers& x,FlashAttention1Workspace w,bool causal,
                  cudaStream_t stream,KernelResources* resources) {
    return causal?launch<D,true>(s,x,w,stream,resources):launch<D,false>(s,x,w,stream,resources);
}
}
LaunchConfig launch_flash_attention_1_tensor_core(Shape s,const DevicePointers& x,
    FlashAttention1Workspace w,bool causal,cudaStream_t stream,KernelResources* resources) {
    if(s.b<=0 || s.h<=0 || s.n<=0 || s.d<=0 || std::size_t(s.b)*s.h>std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("FA1 WMMA requires positive supported dimensions");
    if(!x.hq || !x.hk || !x.hv || !x.o || !w.maxima || !w.normalizers)
        throw std::invalid_argument("FA1 WMMA requires half Q/K/V, float O and caller-owned m/l");
    switch(s.d) {
#define PART_C_DIM(D) case D:return mode<D>(s,x,w,causal,stream,resources);
    PART_C_DIM(7) PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(33) PART_C_DIM(40)
    PART_C_DIM(64) PART_C_DIM(65) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default:throw std::invalid_argument("FA1 WMMA head dimension not instantiated");
    }
}
}
