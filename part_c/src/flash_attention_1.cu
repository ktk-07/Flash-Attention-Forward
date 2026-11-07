#include "flash_attention/flash_attention_1.h"
#include <stdexcept>
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


// https://arxiv.org/pdf/2205.14135
// Implementation Similar to this: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu 
// FlashAttention-1-style educational baseline (causal self-attention).
// Q, K, V, O are logically [batch_size, num_heads, sequence_length, HEAD_DIM].
// The tensors may have arbitrary element strides (strides count elements, not bytes).
// K is physically [key_length, HEAD_DIM], not transposed in memory.
//
// One CTA handles BLOCK_Q query rows; one thread handles one query row.
// No shared memory, warp intrinsics, or Tensor Cores. This implements the
// tiled online-softmax recurrence, NOT the I/O-efficient FA1 kernel.
//
// Causal convention: query position q may attend to key position k iff k <= q.
// This assumes query/key positions share the same origin.
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_v1_baseline(const float* Q,
                                            const float* K,
                                            const float* V,
                                            float* O,
                                            int query_length,
                                            int key_length,
                                            int batch_size,
                                            int num_heads,
                                            int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                            int stride_kb, int stride_kn, int stride_ks, int stride_kh,
                                            int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                            int stride_ob, int stride_on, int stride_os, int stride_oh,
                                            float softmax_scale
                                            )
{
    static_assert(BLOCK_Q > 0 && BLOCK_Q <= 1024, "Invalid BLOCK_Q");
    static_assert(BLOCK_KV > 0 && HEAD_DIM > 0, "Invalid tile size");

    // Grid x selects a batch/head pair; grid y selects a query tile.
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int query_block_start = blockIdx.y * BLOCK_Q;
    const int query_idx = query_block_start + threadIdx.x;
    if (batch_idx >= batch_size || query_idx >= query_length) return;

    // Base offsets of the selected 2D matrices, plus the query row for Q/O.
    // Stride along sequence is stride_*s; stride along features is stride_*h.
    const int query_base_offset = batch_idx * stride_qb + head_idx * stride_qn + query_idx * stride_qs;
    const int key_base_offset = batch_idx * stride_kb + head_idx * stride_kn;
    const int value_base_offset = batch_idx * stride_vb + head_idx * stride_vn;
    const int output_base_offset = batch_idx * stride_ob + head_idx * stride_on + query_idx * stride_os;

    // Online-softmax state, local to this thread/query row:
    // running_max = m, running_sum = l,
    // output_acc[d] = sum_k exp(score_k - m) * V[k,d].
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_acc[HEAD_DIM] = {};

    // FA1-style outer loop over K/V tiles. No future-only tiles are visited.
    const int num_key_tiles = (key_length + BLOCK_KV - 1) / BLOCK_KV;
    for (int key_tile_idx = 0; key_tile_idx < num_key_tiles; ++key_tile_idx) {
        const int key_block_start = key_tile_idx * BLOCK_KV;
        if (key_block_start > query_idx) break;

        // QK^T: one query row dotted with up to BLOCK_KV key rows.
        // Scores are scaled and causally masked before online softmax.
        float scores[BLOCK_KV];
        float tile_max = -INFINITY;
        for (int key_local_idx = 0; key_local_idx < BLOCK_KV; ++key_local_idx) {
            const int key_idx = key_block_start + key_local_idx;
            if (key_idx >= key_length || key_idx > query_idx) {
                scores[key_local_idx] = -INFINITY;
                continue;
            }

            float dot_product = 0.0f;
            for (int feature_idx = 0; feature_idx < HEAD_DIM; ++feature_idx) {
                dot_product += Q[query_base_offset + feature_idx * stride_qh] * K[key_base_offset + key_idx * stride_ks + feature_idx * stride_kh];
            }
            scores[key_local_idx] = dot_product * softmax_scale;
            tile_max = fmaxf(tile_max, scores[key_local_idx]);
        }

        // Avoid exp(-inf - -inf) when every key in a tile is masked.
        if (tile_max == -INFINITY) continue;

        // Merge the current tile into the online softmax statistics.
        // alpha rescales previous contributions to the new maximum.
        const float new_max = fmaxf(running_max, tile_max);
        const float alpha = (running_max == -INFINITY) ? 0.0f : expf(running_max - new_max);
        running_sum *= alpha;
        for (int feature_idx = 0; feature_idx < HEAD_DIM; ++feature_idx) {
            output_acc[feature_idx] *= alpha;
        }

        // Unnormalized P = exp(S - new_max), then PV accumulation.
        // We postpone dividing by running_sum until all K/V tiles are done.
        for (int key_local_idx = 0; key_local_idx < BLOCK_KV; ++key_local_idx) {
            if (scores[key_local_idx] == -INFINITY) continue;
            const int key_idx = key_block_start + key_local_idx;
            const float probability = expf(scores[key_local_idx] - new_max);
            running_sum += probability;
            for (int feature_idx = 0; feature_idx < HEAD_DIM; ++feature_idx) {
                output_acc[feature_idx] += probability * V[value_base_offset + key_idx * stride_vs + feature_idx * stride_vh];
            }
        }
        running_max = new_max;
    }

    // Normalize once, after processing all visible keys.
    // If no key is visible, define the output as zero.
    for (int feature_idx = 0; feature_idx < HEAD_DIM; ++feature_idx) {
        O[output_base_offset + feature_idx * stride_oh] = (running_sum > 0.0f) ? output_acc[feature_idx] / running_sum : 0.0f;
    }
}


namespace part_c {
namespace {
template<int D>
LaunchConfig launch_impl(int, Shape s, const DevicePointers& x, cudaStream_t stream) {
    const int sb=s.h*s.n*D, sh=s.n*D, ss=D;
    const float scale=1.0f/std::sqrt(float(D));
    dim3 grid(s.b*s.h,(s.n+31)/32);
    flash_attention_v1_baseline<32,32,D><<<grid,32,0,stream>>>(
        x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,
        sb,sh,ss,1,sb,sh,ss,1,sb,sh,ss,1,sb,sh,ss,1,scale);
    return {32,32,1,1,1};
}
}
LaunchConfig launch_flash_attention_1(Shape s, const DevicePointers& x, cudaStream_t stream) {
    const int variant=0;
    switch (s.d) {
#define PART_C_DIM(D) case D: return launch_impl<D>(variant, s, x, stream);
    PART_C_DIM(7) PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(33) PART_C_DIM(40)
    PART_C_DIM(64) PART_C_DIM(65) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default: throw std::invalid_argument("head dimension not instantiated");
    }
}
}
