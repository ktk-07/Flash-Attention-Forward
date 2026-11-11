#include "flash_attention/flash_attention_2_tensor_core.h"
#include <stdexcept>
// Explicit alias required by the WMMA section of the original development file.
namespace nvcuda { namespace wmma {} }
namespace wmma = nvcuda::wmma;
// Naming conventions used throughout this file:
//   flash_attention_v1_* / flash_attention_v2_* : algorithm family
//   *_baseline, *_warp_per_query, *_cuda_core_tiled, *_wmma_fused : variant
//   batch_idx, head_idx, warp_id, lane_id : ownership/indexing
//   *_head_offset : base of the selected [sequence, feature] matrix
//   query_block_start / key_block_start : global sequence tile starts
//   running_max, running_sum, output_accumulator : online softmax state
// Some older kernels retain their original parameter spellings (row, col,
// stride_qs, etc.) to keep launch signatures compatible and changes focused.
// NOTE: the WMMA variant uses FP16 input, other variants use FP32.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <math_constants.h>
#include <stdint.h>
#include <vector>
#include <limits>


__device__ __forceinline__ uint32_t fa_pack_half2(half low, half high) {
    return static_cast<uint32_t>(__half_as_ushort(low)) |
           (static_cast<uint32_t>(__half_as_ushort(high)) << 16);
}

// Four FP32 accumulator registers are read/updated in-place by one Tensor
// Core instruction. Unlike WMMA, the register-to-matrix mapping is explicit.
__device__ __forceinline__ void fa_mma_m16n8k16(
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
    // Host pass never executes this device function; architecture guard
    // avoids assembling an unsupported instruction for pre-Ampere GPUs.
#endif
}

// FULLY FUSED WMMA FLASHATTENTION 
// Original FP32 CUDA-core kernels above are preserved as separate variants.
// IMPORTANT: requires BLOCK_Q == 16*NUM_WARPS; use blockDim=32*NUM_WARPS. The original [B,N,S,H] tensor strides are in ELEMENTS (not bytes).
// WMMA ldm refers to the padded SHARED-MEMORY pitch, not the 4D tensor size.
// Q/K/V are half*, which is FP16, unlike the existing FP32 CUDA-core kernel.
// A separate FP32-to-FP16 conversion is needed if reusing FP32 inputs.
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attention_wmma_fused(const half* __restrict__ Q,
                                           const half* __restrict__ K,
                                           const half* __restrict__ V,
                                           float* __restrict__ O,
                                           int B, int N, 
                                           int row, int col,
                                           int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                           int stride_kb, int stride_kn, int stride_ks, int stride_kh,
                                           int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                           int stride_ob, int stride_on, int stride_os, int stride_oh,
                                           float softmax_scale
                                           )
{
    constexpr int WARP = 32;
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int PAD_HEAD = ((HEAD_DIM + 15) / 16) * 16; // Padding is needed cause wmma does 16 x 16 x 16 matrix multiplication
    constexpr int PAD_KV = ((BLOCK_KV + 15) / 16) * 16; // Padding is needed cause wmma does 16 x 16 x 16 matrix multiplication
    static_assert(BLOCK_Q == NUM_WARPS * WM, "Not a production ready kenel, Exactly one 16-row query tile per warp");
    static_assert(BLOCK_KV > 0 && HEAD_DIM > 0 && NUM_WARPS > 0, "Positive dimensions required");

    // All warps in the block cooperate to stage the SAME Q/K/V tiles.
    // One warp then owns exactly 16 query rows for all KV iterations.
    // This makes block-wide __syncthreads() safe: every warp executes the
    // same number of outer KV iterations.
    const int tid = threadIdx.x;
    const int warp = tid / WARP;
    const int lane = tid % WARP;
   
    if (blockIdx.x >= (unsigned)(B * N)) return; // uniform block-wide check

    const int b = blockIdx.x / N;
    const int head = blockIdx.x % N;
    const int q_block_start = blockIdx.y * BLOCK_Q;
    const int q_local_start = warp * WM;
    // Restored from the tested-style WMMA version: two lanes cooperate per row.
    const int row_in_warp = lane / 2;
    const int pair_lane = lane % 2;

    // These offsets select ONE logical [sequence, head_dim] 2D matrix out of the [B,N,S,H] 4D tensor. 
    // They are NOT WMMA leading dimensions.
    const int64_t q_head_offset = (int64_t)b * stride_qb + (int64_t)head * stride_qn;
    const int64_t k_head_offset = (int64_t)b * stride_kb + (int64_t)head * stride_kn;
    const int64_t v_head_offset = (int64_t)b * stride_vb + (int64_t)head * stride_vn;
    const int64_t o_head_offset = (int64_t)b * stride_ob + (int64_t)head * stride_on;

    // SHARED-MEMORY LEADING DIMENSION (ldm - leading dimension of a matrix):
    // We pack [BLOCK_Q, HEAD_DIM] into [BLOCK_Q, PAD_HEAD].
    // For a row-major WMMA load, ldm = PAD_HEAD (distance between rows).
    // It is NOT B*N*S, and is NOT necessarily the global stride_qs.
    // PAD_HEAD is a multiple of 16 FP16 elements, meeting WMMA ldm rules.
    __shared__ __align__(32) half q_s[BLOCK_Q][PAD_HEAD];
    __shared__ __align__(32) half k_s[PAD_KV][PAD_HEAD];
    __shared__ __align__(32) half v_s[PAD_KV][PAD_HEAD];

    // Separate score and probability tiles for EACH warp, so warps do not
    // overwrite one another's intermediate tiles.
    __shared__ __align__(32) float score_s[NUM_WARPS][WM][WN];
    __shared__ __align__(32) half  prob_s[NUM_WARPS][WM][WN];
    __shared__ __align__(32) float pv_s[NUM_WARPS][WM][WN];

    // Online softmax state, persistent across ALL key/value tiles:
    // m_i = running maximum; l_i = running exp sum;
    // out_acc = UNNORMALIZED sum_j exp(score_ij - m_i) * V_j.
    // Store output in shared memory because rescaling is ROW-SPECIFIC and
    // WMMA accumulator fragment elements have opaque lane mapping.
    __shared__ float running_m[BLOCK_Q];
    __shared__ float running_l[BLOCK_Q];
    __shared__ float alpha_s[BLOCK_Q];
    __shared__ float out_acc[BLOCK_Q][PAD_HEAD];

    // Cooperative load Q. Flattened indexing makes all threads participate.
    // Mask invalid query rows and padded head dimensions with zeros.
    for (int idx = tid; idx < BLOCK_Q * PAD_HEAD; idx += blockDim.x) {
        const int qr = idx / PAD_HEAD;
        const int d  = idx % PAD_HEAD;
        const int qg = q_block_start + qr;
        q_s[qr][d] = (qg < row && d < HEAD_DIM)
            ? Q[q_head_offset + (int64_t)qg * stride_qs + (int64_t)d * stride_qh]
            : __float2half(0.0f);
    }
    for (int qr = tid; qr < BLOCK_Q; qr += blockDim.x) {
        running_m[qr] = -CUDART_INF_F;
        running_l[qr] = 0.0f;
        alpha_s[qr] = 0.0f;
    }
    for (int idx = tid; idx < BLOCK_Q * PAD_HEAD; idx += blockDim.x) {
        out_acc[idx / PAD_HEAD][idx % PAD_HEAD] = 0.0f;
    }
    __syncthreads();

    // Causal tile skipping: a query block needs no keys beyond its final row.
    const int last_q = min(row, q_block_start + BLOCK_Q);
    const int key_limit = min(col, last_q);
    const int num_kv_tiles = (key_limit + BLOCK_KV - 1) / BLOCK_KV;

    for (int tile_i = 0; tile_i < num_kv_tiles; ++tile_i) {
        const int kv_start = tile_i * BLOCK_KV;

        // Q stays in shared memory across the entire loop. K/V are replaced
        // for each outer tile. Zero-padding handles both HEAD_DIM and KV tails.
        for (int idx = tid; idx < PAD_KV * PAD_HEAD; idx += blockDim.x) {
            const int kr = idx / PAD_HEAD;
            const int d  = idx % PAD_HEAD;
            const int kg = kv_start + kr;
            const bool valid = kr < BLOCK_KV && kg < col && d < HEAD_DIM;
            k_s[kr][d] = valid ? K[k_head_offset + (int64_t)kg * stride_ks + (int64_t)d * stride_kh] : __float2half(0.0f);
            v_s[kr][d] = valid ? V[v_head_offset + (int64_t)kg * stride_vs + (int64_t)d * stride_vh] : __float2half(0.0f);
        }
        __syncthreads(); // All warps must see completed K/V loads.

        // WMMA always computes a fixed 16x16 OUTPUT tile per warp.
        // For each 16-key chunk, reduce QK^T over HEAD_DIM in steps of 16.
        // Example HEAD_DIM=64 => four 16x16x16 MMA instructions per score tile.
        for (int k_base = 0; k_base < BLOCK_KV; k_base += WN) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> scores;
            wmma::fill_fragment(scores, 0.0f);

            for (int d_base = 0; d_base < PAD_HEAD; d_base += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> kt_frag;

                // Q tile: [16 query rows, 16 feature columns].
                // K^T tile: logically [16 feature rows, 16 key columns].
                // K itself is stored [key, feature] row-major. Interpreting
                // that SAME memory as column-major implements the transpose.
                // Both ldm values are PAD_HEAD because adjacent ORIGINAL
                // query/key rows are PAD_HEAD FP16 elements apart.
                wmma::load_matrix_sync(q_frag, &q_s[q_local_start][d_base], PAD_HEAD);
                wmma::load_matrix_sync(kt_frag, &k_s[k_base][d_base], PAD_HEAD);
                wmma::mma_sync(scores, q_frag, kt_frag, scores);
            }

            // score fragment resides in distributed registers. 
            // We cannot portably index scores as scores[row][key], so stage it in SMEM.
            wmma::store_matrix_sync(&score_s[warp][0][0], scores, WN, wmma::mem_row_major);
            __syncwarp();

            // ALL 32 LANES PARTICIPATE IN SOFTMAX:
            // 16 query rows x 2 lanes per row. Pair (2*r, 2*r+1)
            // owns query row r. Each lane handles alternating key columns:
            //   pair_lane=0 -> keys 0,2,4,...14
            //   pair_lane=1 -> keys 1,3,5,...15
            // Both lanes independently read their partial scores, then use
            // __shfl_xor_sync(..., 1) to combine maxima and denominators.
            // No inter-warp communication is necessary.
            const int qr = q_local_start + row_in_warp;
            const int qg = q_block_start + qr;
            float scores_local[WN / 2];
            float local_max = -CUDART_INF_F;
            #pragma unroll
            for (int j = 0; j < WN / 2; ++j) {
                const int key_in_tile = 2 * j + pair_lane;
                const int kg = kv_start + k_base + key_in_tile;
                const bool valid = qg < row && (k_base + key_in_tile) < BLOCK_KV && kg < col && kg <= qg;
                const float score = valid ? score_s[warp][row_in_warp][key_in_tile] * softmax_scale : -CUDART_INF_F;
                scores_local[j] = score;
                local_max = fmaxf(local_max, score);
            }

            // XOR 1 exchanges values only between adjacent lanes:
            // (0,1), (2,3), ..., (30,31). Both receive the row maximum.
            const float peer_max = __shfl_xor_sync(0xffffffffu, local_max, 1);
            const float tile_max = fmaxf(local_max, peer_max);
            const float old_m = running_m[qr];
            const float new_m = fmaxf(old_m, tile_max);
            const float alpha = (old_m == -CUDART_INF_F || new_m == -CUDART_INF_F) ? 0.0f : expf(old_m - new_m);

            float partial_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WN / 2; ++j) {
                const int key_in_tile = 2 * j + pair_lane;
                const float score = scores_local[j];
                // Guard all-masked rows against exp(-inf - -inf).
                const float p = (score == -CUDART_INF_F) ? 0.0f : expf(score - new_m);
                partial_sum += p;
                // Each element is written by exactly one lane.
                // FP32 -> FP16 here is an approximation for WMMA PV.
                prob_s[warp][row_in_warp][key_in_tile] = __float2half_rn(p);
            }
            const float peer_sum = __shfl_xor_sync(0xffffffffu, partial_sum, 1);
            const float tile_sum = partial_sum + peer_sum;

            // Shuffle collectives do not order shared-memory accesses. Both
            // lanes must finish reading old_m before either updates running_m.
            __syncwarp();

            // Both lanes now have the same row statistics; let the even lane write each row exactly once (avoid write/write races).
            if (pair_lane == 0) {
                alpha_s[qr] = alpha;
                running_m[qr] = new_m;
                running_l[qr] = alpha * running_l[qr] + tile_sum;
            }
            __syncwarp(); // Probabilities and row statistics ready for WMMA PV.

            // PV: [16 query rows, 16 keys] @ [16 keys, 16 features].
            // Loop over HEAD_DIM in 16-column blocks. 
            // This is a DIFFERENT dimension from the d_base reduction loop used in QK^T.
            for (int d_out = 0; d_out < PAD_HEAD; d_out += WN) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> v_frag;
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> pv_frag;
                wmma::fill_fragment(pv_frag, 0.0f);

                // P is a contiguous 16x16 row-major tile => ldm=16.
                // V is stored [key, feature] with padded pitch PAD_HEAD.
                wmma::load_matrix_sync(p_frag, &prob_s[warp][0][0], WN);
                wmma::load_matrix_sync(v_frag, &v_s[k_base][d_out], PAD_HEAD);
                wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                wmma::store_matrix_sync(&pv_s[warp][0][0], pv_frag, WN, wmma::mem_row_major);
                __syncwarp();

                // The running output is row-wise rescaled before adding PV.
                // We cannot do this row-specific operation on an opaque WMMA accumulator fragment, so use an FP32 shared-memory buffer.
                // Two lanes per row also update the FP32 output accumulator.
                // Even lane owns output columns 0,2,...14; odd lane 1,3,...15.
                // Thus all 32 lanes work, and no two lanes write the same cell.
                const float row_alpha = alpha_s[qr];
                #pragma unroll
                for (int j = pair_lane; j < WN; j += 2) {
                    const int d = d_out + j;
                    out_acc[qr][d] = row_alpha * out_acc[qr][d] + pv_s[warp][row_in_warp][j];
                }
                __syncwarp(); // Do not overwrite pv_s until readers finish.
            }
            // The next 16-key chunk updates m, l and O again.
        }
        __syncthreads(); // All warps finished reading K/V before replacement.
    }

    // Final normalization happens ONCE after all KV tiles:
    // O_i = out_acc_i / running_l_i.
    // Here we use all lanes for coalesced output stores; the output strides can differ from the shared-memory pitch.
    for (int idx = tid; idx < BLOCK_Q * PAD_HEAD; idx += blockDim.x) {
        const int qr = idx / PAD_HEAD;
        const int d = idx % PAD_HEAD;
        const int qg = q_block_start + qr;
        if (qg < row && d < HEAD_DIM) {
            const float denom = running_l[qr];
            O[o_head_offset + (int64_t)qg * stride_os + (int64_t)d * stride_oh] = denom > 0.0f ? out_acc[qr][d] / denom : 0.0f;
        }
    }
}

// Fully Fused PTX MMA Optimized FlashAttention
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attention_v2_tensor_core_ptx_mma_fully_fused(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, float* __restrict__ O,
    int B, int N, int row, int col,
    int stride_qb, int stride_qn, int stride_qs, int stride_qh,
    int stride_kb, int stride_kn, int stride_ks, int stride_kh,
    int stride_vb, int stride_vn, int stride_vs, int stride_vh,
    int stride_ob, int stride_on, int stride_os, int stride_oh,
    float softmax_scale) {
    static_assert(BLOCK_Q == 16 * NUM_WARPS, "One warp owns 16 query rows; BLOCK_Q=16*NUM_WARPS");
    static_assert(BLOCK_KV >= 16 && BLOCK_KV % 16 == 0, "KV tiles are processed in 16-key chunks");
    static_assert(HEAD_DIM >= 16 && HEAD_DIM % 16 == 0, "This first PTX version supports H divisible by 16");
    static_assert(NUM_WARPS >= 1 && NUM_WARPS <= 8, "Use 1..8 warps per CTA");

    constexpr int WARP_SIZE = 32;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int OUTPUT_TILES = HEAD_DIM / MMA_N;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_group = lane_id / 4;     // 0..7: two query rows
    const int lane_in_group = lane_id % 4;  // 0..3: two keys per 8-key half

    const int batch_idx = blockIdx.x / N;
    const int head_idx = blockIdx.x % N;
    const int query_block_start = blockIdx.y * BLOCK_Q;
    const int query_local_start = warp_id * MMA_M;
    const int query_row_0 = query_local_start + lane_group;
    const int query_row_1 = query_row_0 + 8;
    const int query_global_0 = query_block_start + query_row_0;
    const int query_global_1 = query_block_start + query_row_1;

    const size_t q_head_offset = size_t(batch_idx) * stride_qb + size_t(head_idx) * stride_qn;
    const size_t k_head_offset = size_t(batch_idx) * stride_kb + size_t(head_idx) * stride_kn;
    const size_t v_head_offset = size_t(batch_idx) * stride_vb + size_t(head_idx) * stride_vn;
    const size_t o_head_offset = size_t(batch_idx) * stride_ob + size_t(head_idx) * stride_on;

    // Shared memory is CTA-wide; each warp has its own P[16][16] to prevent
    // races. Q is loaded once per CTA; K/V are reloaded per KV tile.
    // FP16 shared footprint for 4 warps, BLOCK_Q=64, BLOCK_KV=32, H=64:
    // (64*64 + 2*32*64 + 4*16*16)*2 = 18,432 bytes.
    __shared__ half q_shared[BLOCK_Q][HEAD_DIM];
    __shared__ half k_shared[BLOCK_KV][HEAD_DIM];
    __shared__ half v_shared[BLOCK_KV][HEAD_DIM];
    __shared__ half prob_shared[NUM_WARPS][MMA_M][2 * MMA_N];

    // Each thread loads strided, contiguous-in-feature elements. This is
    // cooperative CTA loading, but NOT vectorized float4/half8 loading.
    for (int idx = threadIdx.x; idx < BLOCK_Q * HEAD_DIM;
         idx += blockDim.x) {
        const int qr = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int qg = query_block_start + qr;
        q_shared[qr][d] = (qg < row) ? Q[q_head_offset + size_t(qg) * stride_qs + size_t(d) * stride_qh] : __float2half(0.0f);
    }
    __syncthreads();

    // Two independent online-softmax states per lane. Each state is
    // replicated across the 4 lanes of a lane_group, which makes a 4-lane
    // max/sum reduction straightforward. NO CTA-wide softmax reduction.
    float running_max_0 = -CUDART_INF_F, running_max_1 = -CUDART_INF_F;
    float running_sum_0 = 0.0f, running_sum_1 = 0.0f;

    // Persistent FP32 output accumulator, indexed by 8-feature MMA tile.
    // For each tile: [row0,col0], [row0,col1], [row1,col0], [row1,col1].
    // This is the crucial improvement over staging WMMA output in SMEM.
    float output_accumulator[OUTPUT_TILES][4];
    #pragma unroll
    for (int t = 0; t < OUTPUT_TILES; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) output_accumulator[t][i] = 0.0f;
    }

    // A query block cannot attend to keys beyond its last query position.
    // All warps use the SAME tile count, so CTA barriers are safe.
    const int valid_query_end = min(row, query_block_start + BLOCK_Q);
    const int valid_key_end = min(col, valid_query_end);
    const int num_key_tiles = (valid_key_end + BLOCK_KV - 1) / BLOCK_KV;

    for (int key_tile_idx = 0; key_tile_idx < num_key_tiles; ++key_tile_idx) {
        const int key_block_start = key_tile_idx * BLOCK_KV;
        for (int idx = threadIdx.x; idx < BLOCK_KV * HEAD_DIM;
             idx += blockDim.x) {
            const int kr = idx / HEAD_DIM;
            const int d = idx % HEAD_DIM;
            const int kg = key_block_start + kr;
            k_shared[kr][d] = (kg < col)
                ? K[k_head_offset + size_t(kg) * stride_ks + size_t(d) * stride_kh] : __float2half(0.0f);
            v_shared[kr][d] = (kg < col)
                ? V[v_head_offset + size_t(kg) * stride_vs + size_t(d) * stride_vh] : __float2half(0.0f);
        }
        __syncthreads();  // Q/K/V ready before Tensor Core loads.

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
                a[0] = fa_pack_half2(q_shared[query_row_0][d0], q_shared[query_row_0][d0 + 1]);
                a[2] = fa_pack_half2(q_shared[query_row_0][d1], q_shared[query_row_0][d1 + 1]);
                a[1] = fa_pack_half2(q_shared[query_row_1][d0], q_shared[query_row_1][d0 + 1]);
                a[3] = fa_pack_half2(q_shared[query_row_1][d1], q_shared[query_row_1][d1 + 1]);

                const int k0 = key_local_start + lane_group;
                const int k1 = k0 + 8;
                b0[0] = fa_pack_half2(k_shared[k0][d0], k_shared[k0][d0 + 1]);
                b0[1] = fa_pack_half2(k_shared[k0][d1], k_shared[k0][d1 + 1]);
                b1[0] = fa_pack_half2(k_shared[k1][d0], k_shared[k1][d0 + 1]);
                b1[1] = fa_pack_half2(k_shared[k1][d1], k_shared[k1][d1 + 1]);
                fa_mma_m16n8k16(score_0, a, b0);
                fa_mma_m16n8k16(score_1, a, b1);
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
                s0[j] = (query_global_0 < row && kg < col && kg <= query_global_0) ? s0[j] * softmax_scale : -CUDART_INF_F;
                s1[j] = (query_global_1 < row && kg < col && kg <= query_global_1) ? s1[j] * softmax_scale : -CUDART_INF_F;
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
                a[0] = fa_pack_half2(prob_shared[warp_id][lane_group][k0], prob_shared[warp_id][lane_group][k0 + 1]);
                a[2] = fa_pack_half2(prob_shared[warp_id][lane_group][k1], prob_shared[warp_id][lane_group][k1 + 1]);
                a[1] = fa_pack_half2(prob_shared[warp_id][lane_group + 8][k0], prob_shared[warp_id][lane_group + 8][k0 + 1]);
                a[3] = fa_pack_half2(prob_shared[warp_id][lane_group + 8][k1], prob_shared[warp_id][lane_group + 8][k1 + 1]);
                const int feature = d_out + lane_group;
                b[0] = fa_pack_half2(v_shared[key_local_start + k0][feature], v_shared[key_local_start + k0 + 1][feature]);
                b[1] = fa_pack_half2(v_shared[key_local_start + k1][feature], v_shared[key_local_start + k1 + 1][feature]);
                float* c = output_accumulator[output_tile_idx];
                c[0] *= alpha_0; c[1] *= alpha_0;
                c[2] *= alpha_1; c[3] *= alpha_1;
                fa_mma_m16n8k16(output_accumulator[output_tile_idx], a, b);
            }
            __syncwarp(); // P buffer can be overwritten by the next 16 keys.
        }
        __syncthreads(); // No warp still reads K/V before the next tile load.
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
}


namespace part_c {
namespace {

template<class Kernel>
void describe_kernel(Kernel kernel, int threads, KernelResources* info) {
    if (!info) return; // Timing calls do not query resources.
    cudaFuncAttributes attr{}; cudaDeviceProp device{}; int index=0, blocks=0;
    auto verify=[](cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); };
    verify(cudaGetDevice(&index)); verify(cudaGetDeviceProperties(&device,index));
    verify(cudaFuncGetAttributes(&attr,kernel));
    verify(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,0));
    info->registers_per_thread=attr.numRegs;info->shared_bytes=attr.sharedSizeBytes;
    info->local_bytes_per_thread=attr.localSizeBytes;info->active_blocks_per_sm=blocks;
    info->occupancy_estimate_percent=100.0*blocks*threads/device.maxThreadsPerMultiProcessor;
}

template<int D, int BQ=32, int BK=32, int NW=2>
LaunchConfig launch_impl(TensorCoreVariant variant, Shape s, const DevicePointers& x, cudaStream_t stream, KernelResources* resources) {
    const int sb=s.h*s.n*D, sh=s.n*D, ss=D;
    const float scale=1.0f/std::sqrt(float(D));
    dim3 grid(s.b*s.h,(s.n+BQ-1)/BQ);
#define QS sb,sh,ss,1
    if (variant==TensorCoreVariant::Wmma) {
        describe_kernel(flash_attention_wmma_fused<BQ,BK,D,NW>,NW*32,resources);
        flash_attention_wmma_fused<BQ,BK,D,NW><<<grid,NW*32,0,stream>>>(x.hq,x.hk,x.hv,x.o,s.b,s.h,s.n,s.n,QS,QS,QS,QS,scale);
    } else if (variant==TensorCoreVariant::PtxMma) {
        if constexpr(D>=16 && D%16==0) {
            describe_kernel(flash_attention_v2_tensor_core_ptx_mma_fully_fused<BQ,BK,D,NW>,NW*32,resources);
        flash_attention_v2_tensor_core_ptx_mma_fully_fused<BQ,BK,D,NW><<<grid,NW*32,0,stream>>>(x.hq,x.hk,x.hv,x.o,s.b,s.h,s.n,s.n,QS,QS,QS,QS,scale);
        } else throw std::invalid_argument("unsupported PTX head dimension");
    } else throw std::invalid_argument("unknown Tensor Core variant");
#undef QS
    return {BQ,BK,NW,0,0};
}
template<int D>
LaunchConfig configured(TensorCoreVariant variant, Shape s, const DevicePointers& x, int config, cudaStream_t stream, KernelResources* info) {
    switch(config) {
    case 0:return launch_impl<D,32,32,2>(variant,s,x,stream,info);
    case 1:return launch_impl<D,16,32,1>(variant,s,x,stream,info);
    case 2:return launch_impl<D,32,16,2>(variant,s,x,stream,info);
    case 3:return launch_impl<D,16,16,1>(variant,s,x,stream,info);
    default:throw std::invalid_argument("unknown Tensor Core configuration");
    }
}
}
LaunchConfig launch_flash_attention_2_tensor_core_configured(TensorCoreVariant variant, Shape s, const DevicePointers& x, int config, cudaStream_t stream, KernelResources* info) {
    switch(s.d) {
#define PART_C_DIM(D) case D:return configured<D>(variant,s,x,config,stream,info);
    PART_C_DIM(7) PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(33) PART_C_DIM(40)
    PART_C_DIM(64) PART_C_DIM(65) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default:throw std::invalid_argument("head dimension not instantiated");
    }
}
LaunchConfig launch_flash_attention_2_tensor_core(TensorCoreVariant variant, Shape s, const DevicePointers& x, cudaStream_t stream) {
    return launch_flash_attention_2_tensor_core_configured(variant,s,x,0,stream);
}
}
