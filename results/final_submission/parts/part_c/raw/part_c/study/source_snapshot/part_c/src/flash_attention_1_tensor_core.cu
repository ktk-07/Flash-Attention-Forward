#include "flash_attention/flash_attention_1_tensor_core.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
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


// FA1 Algorithm-1 K/V-outer schedule with the same m16n8k16 fragment mapping,
// shared scalar operand loads and FP32 register accumulators as our PTX FA2.
// One CTA/head reuses each K/V tile across Q tiles. Normalised O and FP32 m/l
// are written to global memory after each K/V tile. No N*N HBM buffer exists.
// The four launch configurations match FA2's bounded PTX tuning budget.
// This educational implementation is not the official FA1 library.
namespace {
__device__ __forceinline__ uint32_t fa1_pack_half2(half low, half high) {
    return static_cast<uint32_t>(__half_as_ushort(low)) |
           (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

// Four FP32 accumulator registers are read/updated in-place by one Tensor
// Core instruction. Unlike WMMA, the register-to-matrix mapping is explicit.
__device__ __forceinline__ void fa1_mma_m16n8k16(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#else
    // Refuse unsupported device architectures rather than silently omitting MMA.
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#endif
#endif
}


template<int BLOCK_Q,int BLOCK_KV,int HEAD_DIM,int NUM_WARPS,bool Causal>
__global__ void flash_attention_v1_ptx_mma(
    const half* __restrict__ Q,const half* __restrict__ K,const half* __restrict__ V,
    float* __restrict__ O,float* __restrict__ maxima,float* __restrict__ normalizers,
    int row,float softmax_scale) {
    static_assert(BLOCK_Q==16*NUM_WARPS && BLOCK_KV%16==0);
    static_assert(HEAD_DIM>=16 && HEAD_DIM%16==0);
    constexpr int MMA_M=16,MMA_N=8,MMA_K=16,OUTPUT_TILES=HEAD_DIM/MMA_N;
    const int warp_id=threadIdx.x/32,lane_id=threadIdx.x%32;
    const int lane_group=lane_id/4,lane_in_group=lane_id%4;
    const int query_local_start=warp_id*MMA_M;
    const int query_row_0=query_local_start+lane_group,query_row_1=query_row_0+8;
    const int col=row,stride_qs=HEAD_DIM,stride_qh=1,stride_ks=HEAD_DIM,stride_kh=1;
    const int stride_vs=HEAD_DIM,stride_vh=1,stride_os=HEAD_DIM,stride_oh=1;
    const size_t q_head_offset=size_t(blockIdx.x)*row*HEAD_DIM;
    const size_t k_head_offset=q_head_offset,v_head_offset=q_head_offset,o_head_offset=q_head_offset;
    const size_t state_head_offset=size_t(blockIdx.x)*row;
    __shared__ half q_shared[BLOCK_Q][HEAD_DIM];
    __shared__ half k_shared[BLOCK_KV][HEAD_DIM],v_shared[BLOCK_KV][HEAD_DIM];
    __shared__ half prob_shared[NUM_WARPS][MMA_M][16];

    for(int key_block_start=0;key_block_start<row;key_block_start+=BLOCK_KV) {
        for(int idx=threadIdx.x;idx<BLOCK_KV*HEAD_DIM;idx+=blockDim.x) {
            const int kr=idx/HEAD_DIM,d=idx%HEAD_DIM,kg=key_block_start+kr;
            k_shared[kr][d]=kg<row?K[k_head_offset+size_t(kg)*HEAD_DIM+d]:__float2half(0.f);
            v_shared[kr][d]=kg<row?V[v_head_offset+size_t(kg)*HEAD_DIM+d]:__float2half(0.f);
        }
        __syncthreads();
        // For BK=16,BQ=32 the containing Q tile may precede the key start;
        // per-element masking handles its earlier rows without dropping work.
        const int first_query=Causal?(key_block_start/BLOCK_Q)*BLOCK_Q:0;
        for(int query_block_start=first_query;query_block_start<row;query_block_start+=BLOCK_Q) {
            const int query_global_0=query_block_start+query_row_0;
            const int query_global_1=query_block_start+query_row_1;
            for(int idx=threadIdx.x;idx<BLOCK_Q*HEAD_DIM;idx+=blockDim.x) {
                const int qr=idx/HEAD_DIM,d=idx%HEAD_DIM,qg=query_block_start+qr;
                q_shared[qr][d]=qg<row?Q[q_head_offset+size_t(qg)*HEAD_DIM+d]:__float2half(0.f);
            }
            __syncthreads();
            // Restore the previous global FA1 state. Convert normalised O to
            // its unnormalised numerator for the register-only subtile loop.
            // First K/V tile ignores all caller contents, including NaNs.
            const bool old0=key_block_start!=0 && query_global_0<row;
            const bool old1=key_block_start!=0 && query_global_1<row;
            float running_max_0=old0?maxima[state_head_offset+query_global_0]:-CUDART_INF_F;
            float running_max_1=old1?maxima[state_head_offset+query_global_1]:-CUDART_INF_F;
            float running_sum_0=old0?normalizers[state_head_offset+query_global_0]:0.f;
            float running_sum_1=old1?normalizers[state_head_offset+query_global_1]:0.f;
            float output_accumulator[OUTPUT_TILES][4];
            #pragma unroll
            for(int t=0;t<OUTPUT_TILES;++t) {
                const int d=t*MMA_N+2*lane_in_group;
                output_accumulator[t][0]=old0?O[o_head_offset+size_t(query_global_0)*HEAD_DIM+d]*running_sum_0:0.f;
                output_accumulator[t][1]=old0?O[o_head_offset+size_t(query_global_0)*HEAD_DIM+d+1]*running_sum_0:0.f;
                output_accumulator[t][2]=old1?O[o_head_offset+size_t(query_global_1)*HEAD_DIM+d]*running_sum_1:0.f;
                output_accumulator[t][3]=old1?O[o_head_offset+size_t(query_global_1)*HEAD_DIM+d+1]*running_sum_1:0.f;
            }
        // MMA has N=8 columns. We compute TWO 16x8 score tiles to obtain
        // 16 keys, then softmax all 16 keys together. Why 16? PV MMA has
        // K=16 and consumes a complete P[16,16] tile in one instruction.
        for (int key_local_start = 0; key_local_start < BLOCK_KV;
             key_local_start += 16) {
            float score_0[4] = {0.f, 0.f, 0.f, 0.f};
            float score_1[4] = {0.f, 0.f, 0.f, 0.f};

            // Reduction dimension is HEAD_DIM. One m16n8k16 instruction
            // consumes 16 features. Four instructions per score half for H=64.
            for (int d_base = 0; d_base < HEAD_DIM; d_base += MMA_K) {
                // PTX A register order: row0/lowK, row1/lowK, row0/highK, row1/highK.
                uint32_t a[4], b0[2], b1[2];
                const int d0 = d_base + 2 * lane_in_group;
                const int d1 = d0 + 8;
                a[0] = fa1_pack_half2(q_shared[query_row_0][d0], q_shared[query_row_0][d0 + 1]);
                a[2] = fa1_pack_half2(q_shared[query_row_0][d1], q_shared[query_row_0][d1 + 1]);
                a[1] = fa1_pack_half2(q_shared[query_row_1][d0], q_shared[query_row_1][d0 + 1]);
                a[3] = fa1_pack_half2(q_shared[query_row_1][d1], q_shared[query_row_1][d1 + 1]);

                const int k0 = key_local_start + lane_group;
                const int k1 = k0 + 8;
                b0[0] = fa1_pack_half2(k_shared[k0][d0], k_shared[k0][d0 + 1]);
                b0[1] = fa1_pack_half2(k_shared[k0][d1], k_shared[k0][d1 + 1]);
                b1[0] = fa1_pack_half2(k_shared[k1][d0], k_shared[k1][d0 + 1]);
                b1[1] = fa1_pack_half2(k_shared[k1][d1], k_shared[k1][d1 + 1]);
                fa1_mma_m16n8k16(score_0, a, b0);
                fa1_mma_m16n8k16(score_1, a, b1);
            }

            // Each lane has 4 key scores for each of its TWO query rows.
            // C register mapping: score_0[0:2] -> row0, keys 0..7;
            // score_1[0:2] -> row0, keys 8..15; [2:4] -> row1.
            float s0[4] = {score_0[0], score_0[1], score_1[0], score_1[1]};
            float s1[4] = {score_0[2], score_0[3], score_1[2], score_1[3]};
            float partial_max_0 = -CUDART_INF_F;
            float partial_max_1 = -CUDART_INF_F;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key_offset = (j / 2) * 8 + 2 * lane_in_group + (j % 2);
                const int kg = key_block_start + key_local_start + key_offset;
                s0[j] = (query_global_0 < row && kg < col && (!Causal || kg <= query_global_0)) ? s0[j] * softmax_scale : -CUDART_INF_F;
                s1[j] = (query_global_1 < row && kg < col && (!Causal || kg <= query_global_1)) ? s1[j] * softmax_scale : -CUDART_INF_F;
                partial_max_0 = fmaxf(partial_max_0, s0[j]);
                partial_max_1 = fmaxf(partial_max_1, s1[j]);
            }

            // Four-lane reduction, confined to groups of lanes [0..3],
            // [4..7], ... . XOR 1 and XOR 2 never cross group boundaries.
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                partial_max_0 = fmaxf(partial_max_0, __shfl_xor_sync(0xffffffffu, partial_max_0, mask));
                partial_max_1 = fmaxf(partial_max_1, __shfl_xor_sync(0xffffffffu, partial_max_1, mask));
            }
            const float new_max_0 = fmaxf(running_max_0, partial_max_0);
            const float new_max_1 = fmaxf(running_max_1, partial_max_1);
            // All-masked rows: alpha=0 and P=0; avoid exp(-inf - -inf).
            const float alpha_0 = (running_max_0 == -CUDART_INF_F || new_max_0 == -CUDART_INF_F) ? 0.f : expf(running_max_0 - new_max_0);
            const float alpha_1 = (running_max_1 == -CUDART_INF_F || new_max_1 == -CUDART_INF_F) ? 0.f : expf(running_max_1 - new_max_1);
            float partial_sum_0 = 0.f, partial_sum_1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p0 = (s0[j] == -CUDART_INF_F) ? 0.f : expf(s0[j] - new_max_0);
                const float p1 = (s1[j] == -CUDART_INF_F) ? 0.f : expf(s1[j] - new_max_1);
                partial_sum_0 += p0;
                partial_sum_1 += p1;
                const int key_offset = (j / 2) * 8 + 2 * lane_in_group + (j % 2);
                // PV requires FP16 A operands. Rounding P to FP16 is an
                // approximation; m/l remain FP32. No S or P matrix in HBM.
                prob_shared[warp_id][lane_group][key_offset] = __float2half_rn(p0);
                prob_shared[warp_id][lane_group + 8][key_offset] = __float2half_rn(p1);
            }
            #pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                partial_sum_0 += __shfl_xor_sync(0xffffffffu, partial_sum_0, mask);
                partial_sum_1 += __shfl_xor_sync(0xffffffffu, partial_sum_1, mask);
            }
            running_sum_0 = alpha_0 * running_sum_0 + partial_sum_0;
            running_sum_1 = alpha_1 * running_sum_1 + partial_sum_1;
            running_max_0 = new_max_0;
            running_max_1 = new_max_1;
            __syncwarp(); // All 16x16 probabilities are written.

            // PV: P[16,16] @ V[16,8] -> 16x8 FP32. Each lane's output
            // registers are rescaled by the row-specific alpha *before* MMA.
            // This avoids the WMMA implementation's output shared-memory
            // round-trip and row-wise rescaling in a separate loop.
            #pragma unroll
            for (int output_tile_idx = 0; output_tile_idx < OUTPUT_TILES;
                 ++output_tile_idx) {
                const int d_out = output_tile_idx * MMA_N;
                uint32_t a[4], b[2];
                const int k0 = 2 * lane_in_group;
                const int k1 = k0 + 8;
                a[0] = fa1_pack_half2(prob_shared[warp_id][lane_group][k0], prob_shared[warp_id][lane_group][k0 + 1]);
                a[2] = fa1_pack_half2(prob_shared[warp_id][lane_group][k1], prob_shared[warp_id][lane_group][k1 + 1]);
                a[1] = fa1_pack_half2(prob_shared[warp_id][lane_group + 8][k0], prob_shared[warp_id][lane_group + 8][k0 + 1]);
                a[3] = fa1_pack_half2(prob_shared[warp_id][lane_group + 8][k1], prob_shared[warp_id][lane_group + 8][k1 + 1]);
                const int feature = d_out + lane_group;
                b[0] = fa1_pack_half2(v_shared[key_local_start + k0][feature], v_shared[key_local_start + k0 + 1][feature]);
                b[1] = fa1_pack_half2(v_shared[key_local_start + k1][feature], v_shared[key_local_start + k1 + 1][feature]);
                float* c = output_accumulator[output_tile_idx];
                c[0] *= alpha_0; c[1] *= alpha_0;
                c[2] *= alpha_1; c[3] *= alpha_1;
                fa1_mma_m16n8k16(output_accumulator[output_tile_idx], a, b);
            }
            __syncwarp(); // P buffer can be overwritten by the next 16 keys.
        }

    // Output C fragment layout gives each lane 2 adjacent features in each
    // of two rows. We can write FP32 O directly without an output SMEM tile.
    #pragma unroll
    for (int t = 0; t < OUTPUT_TILES; ++t) {
        const int d0 = t * MMA_N + 2 * lane_in_group;
        const int d1 = d0 + 1;
        if (query_global_0 < row) {
            const float inv = running_sum_0 > 0.f ? 1.f / running_sum_0 : 0.f;
            const size_t base = o_head_offset + size_t(query_global_0) * stride_os;
            O[base + size_t(d0) * stride_oh] = output_accumulator[t][0] * inv;
            O[base + size_t(d1) * stride_oh] = output_accumulator[t][1] * inv;
        }
        if (query_global_1 < row) {
            const float inv = running_sum_1 > 0.f ? 1.f / running_sum_1 : 0.f;
            const size_t base = o_head_offset + size_t(query_global_1) * stride_os;
            O[base + size_t(d0) * stride_oh] = output_accumulator[t][2] * inv;
            O[base + size_t(d1) * stride_oh] = output_accumulator[t][3] * inv;
        }
    }
            // Exactly one of the four lanes owning a row publishes m/l.
            if(lane_in_group==0) {
                if(query_global_0<row) {maxima[state_head_offset+query_global_0]=running_max_0;normalizers[state_head_offset+query_global_0]=running_sum_0;}
                if(query_global_1<row) {maxima[state_head_offset+query_global_1]=running_max_1;normalizers[state_head_offset+query_global_1]=running_sum_1;}
            }
            __syncthreads(); // Q/P reuse and O/state publication before next Q tile.
        }
        __syncthreads(); // K/V readers finish before replacing this tile.
    }
}
void fa1_ptx_check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
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



namespace part_c {
namespace {
template<int D,int BQ,int BK,int NW,bool Causal>
LaunchConfig fa1_ptx_launch(Shape s,const DevicePointers& x,FlashAttention1Workspace w,
                   cudaStream_t stream,KernelResources* resources) {
    auto kernel=flash_attention_v1_ptx_mma<BQ,BK,D,NW,Causal>;
    if(resources) {
        cudaFuncAttributes a{};cudaDeviceProp prop{};int device,active;
        fa1_ptx_check(cudaGetDevice(&device));fa1_ptx_check(cudaGetDeviceProperties(&prop,device));
        if(prop.major<8)throw std::invalid_argument("FA1 PTX requires SM80 or later");
        fa1_ptx_check(cudaFuncGetAttributes(&a,kernel));
        fa1_ptx_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,NW*32,0));
        resources->registers_per_thread=a.numRegs;resources->shared_bytes=a.sharedSizeBytes;
        resources->local_bytes_per_thread=a.localSizeBytes;resources->active_blocks_per_sm=active;
        resources->occupancy_estimate_percent=100.0*active*NW*32/prop.maxThreadsPerMultiProcessor;
    }
    kernel<<<dim3(unsigned(s.b*s.h)),NW*32,0,stream>>>(x.hq,x.hk,x.hv,x.o,w.maxima,w.normalizers,s.n,1.f/std::sqrt(float(D)));
    return {BQ,BK,NW,0,0};
}
template<int D,int BQ,int BK,int NW>
LaunchConfig fa1_ptx_mode(Shape s,const DevicePointers& x,FlashAttention1Workspace w,bool causal,cudaStream_t stream,KernelResources* resources) {
    return causal?fa1_ptx_launch<D,BQ,BK,NW,true>(s,x,w,stream,resources):fa1_ptx_launch<D,BQ,BK,NW,false>(s,x,w,stream,resources);
}
template<int D>
LaunchConfig fa1_ptx_configured(Shape s,const DevicePointers& x,FlashAttention1Workspace w,int config,bool causal,cudaStream_t stream,KernelResources* resources) {
    switch(config) {
    case 0:return fa1_ptx_mode<D,32,32,2>(s,x,w,causal,stream,resources);
    case 1:return fa1_ptx_mode<D,16,32,1>(s,x,w,causal,stream,resources);
    case 2:return fa1_ptx_mode<D,32,16,2>(s,x,w,causal,stream,resources);
    case 3:return fa1_ptx_mode<D,16,16,1>(s,x,w,causal,stream,resources);
    default:throw std::invalid_argument("Unknown FA1 PTX configuration");
    }
}
}
LaunchConfig launch_flash_attention_1_ptx(Shape s,const DevicePointers& x,FlashAttention1Workspace w,int config,bool causal,cudaStream_t stream,KernelResources* resources) {
    if(s.b<=0 || s.h<=0 || s.n<=0 || std::size_t(s.b)*s.h>std::size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("FA1 PTX requires positive dimensions");
    if(!x.hq || !x.hk || !x.hv || !x.o || !w.maxima || !w.normalizers)throw std::invalid_argument("FA1 PTX requires half Q/K/V, float O and caller-owned m/l");
    switch(s.d) {
#define PART_C_DIM(D) case D:return fa1_ptx_configured<D>(s,x,w,config,causal,stream,resources);
    PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(64) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default:throw std::invalid_argument("FA1 PTX supports instantiated d divisible by 16");
    }
}
}
