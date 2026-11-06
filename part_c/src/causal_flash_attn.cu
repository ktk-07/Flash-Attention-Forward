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

// From this line onwards we are implementing FlashAttention2 progressively https://arxiv.org/pdf/2307.08691
// No Hardware Optimisations Done to this Kernel
// Compile Time Constants
// Take note that  K is mathematically tranposed and not physically tranposed, K is stored in row major format and K_tranposed is not
// In flash attention, it is taken that K is mathematically tranposed and not physically tranposed
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_v2_scalar_baseline(float* Q,
                                                   float* K_transposed,
                                                   float* V,
                                                   float* O,
                                                   int row,
                                                   int col,
                                                   int B, int N,
                                                   int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                                   int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                                   int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                                   int stride_ob, int stride_on, int stride_os, int stride_oh,
                                                   float softmax_scale
                                                   )
{
    // Section 1 Compute the base offsets first
    int batch_idx, head_idx, tile_r_idx, s_idx;
    batch_idx = blockIdx.x / N;
    head_idx = blockIdx.x % N;
    tile_r_idx = blockIdx.y;
    
    // Section 1.1 Computing the row
    // We will loop through the block when computing matmul? so it snot liek triton where we can conveninetly do, , i mean we can create an array but that woudl increase the number of register being used for storign?? idk so i didnt do it?
    // tl.arange(0,BLOCK_Q) + pid1 * BLOCK_Q
    // one thread per query row, each thread independently computes one output row of length HEAD_DIM
    s_idx = tile_r_idx * BLOCK_Q + threadIdx.x; // position of the query token being processed by the current CUDA thread
    if (s_idx >= row) return;

    // Computing the inner dimension
    // No need for this, because we are just going to directly loop over it

    // Ssection 1.2 for the s and h we will do it in the inner loop?
    int q_base_offset, k_base_offset, v_base_offset, o_base_offset;
    q_base_offset = batch_idx * stride_qb + head_idx * stride_qn + s_idx * stride_qs;
    k_base_offset = batch_idx * stride_kb + head_idx * stride_kn; // no dependency on BLOCK Q' Sequence
    v_base_offset = batch_idx * stride_vb + head_idx * stride_vn; // no dependency on BLOCK Q' Sequence
    o_base_offset = batch_idx * stride_ob + head_idx * stride_on + s_idx * stride_os;

    
    // Section 2: Defining the variables needed for Online Softmax Recurrence
    // Do we have to define this outside the cuda kernel or inside is perfectly fine?
    // Each thread owns one query row and its own softmax state
    float running_max, running_sum;
    running_sum = 0.0f;
    running_max = -INFINITY; // In C++ FLT_MIN, in CUDA they already have this pre-built macro
    float output_accumulator[HEAD_DIM] = {};
    for (int i = 0 ; i < col && i <= s_idx ; i += BLOCK_KV){ // Causal attention: query at position s_idx can only attend to keys at positions <= s_idx (including itself).
        // S = QK^T - First Matmul
        int col_start = i;
        float matmul_acc[BLOCK_KV] = {};
        // K/V Loop
        for (int j = 0 ; j < BLOCK_KV; j++){
            int col_idx = col_start + j;
            if (col_idx >= col || col_idx > s_idx){
                matmul_acc[j] = -INFINITY;
                continue;
            }
            
            // Casual Mask add code here later
            int cur_k_base_offset = k_base_offset + col_idx * stride_ks;
            // Head Dimension Loop, which is the same as me passing in the inner_dimension no? is this variable redundant?
            for (int k = 0 ; k < HEAD_DIM ; k++){
                int q_ptr = q_base_offset + k * stride_qh;
                int k_ptr = cur_k_base_offset + k * stride_kh;
                matmul_acc[j] += *(Q + q_ptr) * *(K_transposed + k_ptr);
            }
        }

        // Attention scaling by sqrt
        for (int j = 0 ; j < BLOCK_KV; j++){
            matmul_acc[j] *= softmax_scale;     
        }

        // P = softmax(S) Online Softmax Softmax
        // Defining local running statistics
        float new_max = -INFINITY;        
        float local_max = -INFINITY;
        // Get Local Max
        for (int j = 0 ; j < BLOCK_KV; j++){
            local_max = max(matmul_acc[j], local_max);
        }
        
        // Skip tiles with no valid keys this if or causal attn, we skip this first is fine

        // Get the new_max, dont update first cause we need to rescale
        new_max = max(running_max, local_max);
        
        // Rescaling running statisit
        float rescaling_factor = expf(running_max - new_max);
        for (int j = 0 ; j < HEAD_DIM ; j++){
            output_accumulator[j] *= rescaling_factor;
        }
        running_sum *= rescaling_factor;

        // Update the running_max
        running_max  = new_max;

        // Compute the score = exp(matmul_acc - running_max), which is P
        // We will overwrite the matmul_acc
        // Accumulate the global denominator in the same loop
        for (int j = 0 ; j < BLOCK_KV; j++){
            matmul_acc[j] = std::exp(matmul_acc[j] - running_max);
            running_sum += matmul_acc[j];
        }

        // O = PV - Second Matmul, O is not normalised yet
        for (int j = 0 ; j < HEAD_DIM ; j++){
            for (int k = 0 ; k < BLOCK_KV ; k++){
                int col_idx = i + k;
                if (col_idx >= col || col_idx > s_idx) continue; // No need to compute future tokens
                int cur_v_base_offset = v_base_offset + col_idx * stride_vs;
                int v_ptr = cur_v_base_offset + j * stride_vh;
                output_accumulator[j] +=  matmul_acc[k] * *(V + v_ptr);
            }
        }
    }

    // Divide the weighted accumulator by the denominator
    // Write to the O 
    float inv_denominator = 1.0f / running_sum;
    for (int i = 0 ; i < HEAD_DIM ; i++){
        int o_ptr = o_base_offset + i * stride_oh;
        *(O + o_ptr) = output_accumulator[i] * inv_denominator;
    }

}

// Implementing the General GPU Optimisations
// Global Memory Coalescing - Reviewed not implemented
// Shared Memory Caching - Reviewed not implemented
// 1D Blocktiling - 
// 2D Blocktiling
// Vectorised Memory Access
// References : https://siboehm.com/articles/22/CUDA-MMM


__device__ float warp_max(float val, int warp_size){
    // warp max
    constexpr int FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0 ; offset /= 2){
        float remote_val = __shfl_down_sync(FULL_MASK, val, offset);
        val = fmaxf(val,remote_val);
    }
    int srclane = 0;
    int width = warp_size;
    return __shfl_sync(FULL_MASK, val, srclane, width); // broadcast the final_sum to all other warps, it is sually accumulated into lane 0;
}

__device__ float warp_sum(float val, int warp_size){
    // warp sum
    constexpr int FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0 ; offset /= 2){
        float remote_val = __shfl_down_sync(FULL_MASK, val, offset);
        val += remote_val;
    }
    int srclane = 0;
    int width = warp_size;
    return __shfl_sync(FULL_MASK, val, srclane, width); // broadcast the final_sum to all other warps, it is sually accumulated into lane 0;
}

// 1 warp is responsible for 1 query row
// Example Grid and BlockDimension
// dim3 gridDim(B*N, (Q + BLOCK_Q -1 )/ BLOCK_Q); dim3 blockDim(num_warps & warp_size);
// num_warps = BLOCK_Q right now, but it does not have to be but we would have to adjust the code
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_v2_warp_per_query(float* Q,
                                                  float* K,
                                                  float* V,
                                                  float* O,
                                                  int row,
                                                  int col,
                                                  int B, int N,
                                                  int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                                  int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                                  int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                                  int stride_ob, int stride_on, int stride_os, int stride_oh,
                                                  float softmax_scale
                                                  )
{
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_Q;
    int batch_idx = blockIdx.x / N;
    int head_idx = blockIdx.x % N;
    int tile_r_idx = blockIdx.y;

    //int warp_size = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int s_idx = tile_r_idx * BLOCK_Q + warp_id;

    int q_head_offset = batch_idx * stride_qb + head_idx * stride_qn + s_idx * stride_qs;
    int k_head_offset = batch_idx * stride_kb + head_idx * stride_kn;
    int v_head_offset = batch_idx * stride_vb + head_idx * stride_vn; 
    int o_head_offset = batch_idx * stride_ob + head_idx * stride_on + s_idx * stride_os;
   
    
    float running_sum = 0.0f;
    float running_max = -INFINITY;
    constexpr int NUM_OUTPUT_GROUPS = ( HEAD_DIM + WARP_SIZE - 1 ) / WARP_SIZE;
    // NUM_OUTPUT_GROUP Probably has to be a compile time constant
    float output_accumulator[NUM_OUTPUT_GROUPS] = {}; // 1 query 1 warp, each warp_lane, (HEAD_DIM + WARP_SIZE - 1 ) / warp_size number of output group

    int tile_c = (col + BLOCK_KV - 1 ) / BLOCK_KV;

    // Is this shared amongst each threadblock or each warp?
    __shared__ float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ float V_SHARED[BLOCK_KV][HEAD_DIM];

    for (int i = 0; i < tile_c; i++){
        int key_block_start = i * BLOCK_KV;

        // Loading Loop
        // Cooperative Loading of K and V
        // We want to load K and V in a memory coalescing manner. Since K and V are both in row major format
        // Each warp should be responsible for each col, that why memory coalescing happens 
        // During loading:
        // - warp_id distributes V rows across warps.
        // - lane_id distributes feature columns across lanes.
        constexpr int no_col_iterations = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
        constexpr int no_row_iterations = (BLOCK_KV + NUM_WARPS - 1) / NUM_WARPS;
        for (int j = 0 ; j < no_col_iterations ; j++){
            int col_idx = j * WARP_SIZE + lane_id;
            if (col_idx >= HEAD_DIM) continue;
            for (int k = 0 ; k < no_row_iterations ; k++){
                int row_local = k * NUM_WARPS + warp_id;
                int row_idx = key_block_start + row_local; 

                if (row_local >= BLOCK_KV){
                    continue;
                }

                if (row_idx >= col) {
                    K_SHARED[row_local][col_idx] = 0.0f;
                    V_SHARED[row_local][col_idx] = 0.0f;
                }else{
                    K_SHARED[row_local][col_idx] = *(K + k_head_offset + col_idx * stride_kh + row_idx * stride_ks);
                    V_SHARED[row_local][col_idx] = *(V + v_head_offset + col_idx * stride_vh + row_idx * stride_vs);
                } 
            }
        }
        
        //The first barrier protects consumers from incomplete loading.
        __syncthreads();

        // Computation Loop : QK^T
        // We are computing 1 query row., each warplane assign to 1 element of the K
        // Then we loop throught the inner_dim
        constexpr int NUM_KEY_GROUPS = (BLOCK_KV + WARP_SIZE - 1 ) / WARP_SIZE;
        float matmul_acc[NUM_KEY_GROUPS];
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            matmul_acc[j] = 0.0f;
            int col_local = j * WARP_SIZE + lane_id;
            int col_idx = key_block_start + col_local; 

            if (s_idx >= row || col_idx >= col || col_idx > s_idx || col_local >= BLOCK_KV){
                matmul_acc[j] = -INFINITY;
                continue;
            }

            for (int k = 0 ; k < HEAD_DIM ; k++){
                matmul_acc[j] += *(Q + q_head_offset + k * stride_qh) * K_SHARED[col_local][k];
            }

            matmul_acc[j] *= softmax_scale;
        }

        // Softmax Recurrence
        // Getting the warp lane max
        float new_max = -INFINITY;
        float warp_lane_max = -INFINITY;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            warp_lane_max = max(matmul_acc[j],warp_lane_max);
        }
        // Essentially the local max of that row;
        float local_max = warp_max(warp_lane_max,32);
        new_max = max(running_max, local_max);
        // Rescale Running Statistics
        // Geting the rescaling_factor
        // Rescale output_accumulator
        float rescaling_factor = (new_max == -INFINITY) ? 0.0f : expf(running_max - new_max);
        for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){
            output_accumulator[j] *= rescaling_factor;
        }
        running_sum *=  rescaling_factor;

        running_max = new_max;

        // Sum up the score
        float warp_lane_sum = 0.0f;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            matmul_acc[j] = (running_max == -INFINITY) ? 0.0f : expf(matmul_acc[j] - running_max); // Update matmul_acc with probability
            warp_lane_sum += matmul_acc[j];
        }
        float local_warp_sum = warp_sum(warp_lane_sum,32);
        running_sum += local_warp_sum;

        // Computation Loop : PV
        // We are trying to compute 1 query row
        // We need to distribute the probability of the element to every other row then sum it up
        //  - Pq_warp,k is distributed to Vk_warp_lane_idx,innerdim_i
        // Each warp_lane has an element in the q row of P 
        // Each warp_lane responsible for each column in the warp_id
        // Think of it as we take an element in the q row of P and broadcast it to every other warp_lane which is responsible for their own output element
        // So we take the 
        // Each of warp lane take the broadcasted value and multiplyt it by the V row's indivudial column
        // We then accumulated it into the output_accumulator
        // then we move on to the next element in the q row of P , which shifting the V row to the next row, but each warp indexes the output column they are resopnsible for
        // 1. Broadcasting of probability to each warp manually
        // 2. Accumulated into the output_accumulator
        // During PV computation:
        //  - Every warp can access all V rows in the current tile.
        //  - lane_id still determines which output feature columns each lane owns.
        //  - The loop over k moves sequentially across V rows.
        constexpr unsigned FULL_MASK = 0xffffffffu;
        int width = 32;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++) {
            for (int k = 0 ; k < WARP_SIZE ; k++){
                // Loop to broadcast probability store in every warp to every other lanes
                int srclane = k;
                int k_local = j * WARP_SIZE + k;
                int k_idx = key_block_start + k_local;

                float broadcast_probability_val = __shfl_sync(FULL_MASK, matmul_acc[j], srclane, width);
                for (int l = 0 ; l < NUM_OUTPUT_GROUPS ; l++){
                    // Multiply prbability val with
                    // Each Warp Lane, accumulated into their own output_accumulator[l];
                    int inner_dim_idx = l * WARP_SIZE + lane_id;
                    if (k_local < BLOCK_KV && k_idx < col && inner_dim_idx < HEAD_DIM){
                        output_accumulator[l] += broadcast_probability_val * V_SHARED[k_local][inner_dim_idx];
                    }
                }
            }
        }
        // The second protects consumers from the next iteration's overwrites.
        __syncthreads();

    }
    // Each warp would do 
    // 1. Normalise Accumulation with final global denominator
    // 2. Write output_accumulator to O which is in HBM
    for (int i = 0 ; i < NUM_OUTPUT_GROUPS; i++){
        int col_output_idx = lane_id + WARP_SIZE * i;
        if (s_idx < row && col_output_idx < HEAD_DIM){
            float output_val = (running_sum > 0.0f) ? output_accumulator[i] / running_sum: 0.0f;
            *(O + o_head_offset + col_output_idx * stride_oh) = output_val;
        }
    }
}


// Multiple query rows per warp.
// Multiple warps per CTA.
// 2D register tiling with explicit operand reuse.
// Coalesced scalar global-memory loading (one float per lane/instruction).
// Shared-memory tiling for Q/K/V.
// Online softmax and causal tile skipping.
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, int THREAD_Q, int THREAD_K>
__global__ void flash_attention_v2_cuda_core_tiled(float* Q,
                                                   float* K,
                                                   float* V,
                                                   float* O,
                                                   int row,
                                                   int col,
                                                   int B, int N,
                                                   int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                                   int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                                   int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                                   int stride_ob, int stride_on, int stride_os, int stride_oh,
                                                   float softmax_scale
                                                   )
{
    // We decide how many query rows per warp via this 
    constexpr int query_tile_size = (BLOCK_Q + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_OUTPUT_GROUPS = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int NUM_KEY_GROUPS = (BLOCK_KV + WARP_SIZE - 1) / WARP_SIZE;

    int BN = blockIdx.x;
    int batch_idx = BN / N;
    int head_idx = BN % N;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Computing the base_pointers;
    int q_head_offset = batch_idx * stride_qb + head_idx * stride_qn;
    int k_head_offset = batch_idx * stride_kb + head_idx * stride_kn;
    int v_head_offset = batch_idx * stride_vb + head_idx * stride_vn;
    int o_head_offset = batch_idx * stride_ob + head_idx * stride_on;

    // Running Statistics to keep track of
    float output_accumulator[query_tile_size][NUM_OUTPUT_GROUPS]; // each warp_lane (thread) has this
    float running_sum[query_tile_size]; // each warp_lane (thread) is responsible for this
    float running_max[query_tile_size]; // each warp_lane (thread) is responsible for this

    // Shared Memory
    __shared__ float Q_SHARED[BLOCK_Q][HEAD_DIM];
    __shared__ float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ float V_SHARED[BLOCK_KV][HEAD_DIM];

    int query_block_start = blockIdx.y * BLOCK_Q;
    // Cooperative Loading of Q, we want to reuse Q 
    // Different warps load query_tile_size of rows
    // Each lane loads one float per instruction;
    for (int j = 0 ; j < query_tile_size ; j++){
        running_max[j] = -INFINITY;
        running_sum[j] = 0.0f;
        // Required: the output accumulator is multiplied and incremented later.
        for (int d = 0; d < NUM_OUTPUT_GROUPS; ++d) {
            output_accumulator[j][d] = 0.0f;
        }
        // Loading Loop for 
        // Cooperative Loading of Q, K, V, in the memory coalesced way
        int q_local = warp_id * query_tile_size + j;
        int sq_idx = query_block_start + q_local; // row_idx for q

        // Q Load
        for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
            int h_idx = k * WARP_SIZE + lane_id; // col_idx
            if (q_local >= BLOCK_Q || h_idx >= HEAD_DIM) continue;
            if (sq_idx >= row) {
                Q_SHARED[q_local][h_idx] = 0.0f;
            } else {
                Q_SHARED[q_local][h_idx] = *(Q + q_head_offset + sq_idx * stride_qs + h_idx * stride_qh);
            }
        }
    }
    __syncthreads();

    // Causal tile skipping: a query block cannot attend to keys beyond
    // its last valid query position. Skip those K/V tiles entirely.
    // This assumes aligned causal self-attention (query/key position 0 matches).
    const int last_query_exclusive = min(row, query_block_start + BLOCK_Q);
    const int valid_key_exclusive = min(col, last_query_exclusive);
    const int tile_c = (valid_key_exclusive + BLOCK_KV - 1) / BLOCK_KV;
    for (int i = 0 ; i < tile_c ; i++){
        int key_block_start = i * BLOCK_KV;
        // Loading Loop for 
        // Cooperative Loading of K, V, in the memory coalesced way

        // Basicaly 1 warp load query_tile_size of the row of K and V;
        // K and V are both stored row-major in shared memory.
        // The problem with this is that im afraid shared memory would fill up 
        constexpr int no_of_row_iterations = (BLOCK_KV + NUM_WARPS - 1) / NUM_WARPS;
        for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){       
            int h_idx = j * WARP_SIZE + lane_id;
            if (h_idx >= HEAD_DIM) continue; 
            // K/V Load feels like i might need another loop here that loops over kv_tile_size ?;
            for (int k = 0 ; k < no_of_row_iterations; k++){
                int row_local = k * NUM_WARPS + warp_id;
                int skv_idx = key_block_start + row_local; // row_idx for kv 

                if (row_local >= BLOCK_KV) continue;

                if (skv_idx >= col){
                    K_SHARED[row_local][h_idx] = 0.0f;
                    V_SHARED[row_local][h_idx] = 0.0f;
                }else{
                    K_SHARED[row_local][h_idx] = *(K + k_head_offset + skv_idx * stride_ks + h_idx * stride_kh);
                    V_SHARED[row_local][h_idx] = *(V + v_head_offset + skv_idx * stride_vs + h_idx * stride_vh);                   
                }
            }
        }
        __syncthreads();       

        // Computation Loop
        // QK^T
        // 2D Register tiling for reuse
        // My thinking is this. 
        // We want to 
        // Loading 1 row of Q from shared_memory into registers of each warp
        // Load K columns of K from shared_memory (in shared memory they are store as rows, but for matmul i just called it k column)
        // then go to the next row
        // We are reusing the Q row and using it for differnt columns
        // All warp lane are computing the same thread_q x thread_k output tile
        // However each warp lane are loading a different head_dim
        // Each warp lane owns particular head-dimension indices, not a K row.        

        // float Q_REG[NUM_OUTPUT_GROUPS] = {};
        float Q_REG[THREAD_Q][NUM_OUTPUT_GROUPS];
        float K_REG[THREAD_K][NUM_OUTPUT_GROUPS];
        
        // Loading into Q and K registers
        for (int q_base = 0; q_base < query_tile_size; q_base += THREAD_Q) { // advances to the next register tile.
            // Load tile_q into Q_REG
            for (int j = 0 ; j < THREAD_Q ; j++){ // access each row wihtin tile
                int q_offset = q_base + j;
                int row_q_local_idx = warp_id * query_tile_size + q_offset;
                int q_global = query_block_start + row_q_local_idx;
                for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                    int col_q_local_idx = k * WARP_SIZE + lane_id;
                    if (col_q_local_idx >= HEAD_DIM || row_q_local_idx >= BLOCK_Q || q_global >= row){
                        Q_REG[j][k] = 0.0f;
                    }else{
                        Q_REG[j][k] = Q_SHARED[row_q_local_idx][col_q_local_idx];
                    }
                }
            }

            for (int k_base = 0; k_base < BLOCK_KV; k_base += THREAD_K) {
                // Load K_REG
                float score_acc[THREAD_Q][THREAD_K] = {};
                for (int j = 0 ; j < THREAD_K ; j++){ // access each row wihtin tile
                    int row_k_local_idx = k_base + j;
                    int key_global = key_block_start + row_k_local_idx;
                    for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                        int col_k_local_idx = k * WARP_SIZE + lane_id;
                        if (col_k_local_idx >= HEAD_DIM || row_k_local_idx >= BLOCK_KV || key_global >= col){
                            K_REG[j][k] = 0.0f;
                        }else{
                            K_REG[j][k] = K_SHARED[row_k_local_idx][col_k_local_idx];
                        }
                    }
                }
                // Compute THREAD_Q x THREAD_K partial scores
                for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){
                    for (int q = 0 ; q < THREAD_Q ; q++){
                        for (int k = 0 ; k < THREAD_K ; k++){
                            score_acc[q][k] += Q_REG[q][j] * K_REG[k][j];
                        }
                    }
                }
                // Warp Reduce the partial sums and apply softmax_scale 
                for (int q = 0 ; q < THREAD_Q ; q++){
                    for (int k = 0 ; k < THREAD_K ; k++){
                        int warp_size = WARP_SIZE;
                        score_acc[q][k] = warp_sum(score_acc[q][k], warp_size);
                        score_acc[q][k] *= softmax_scale;
                    }
                }
                // Apply causal mask

                for (int q = 0; q < THREAD_Q; q++) {

                    int q_offset = q_base + q;
                    int q_local = warp_id * query_tile_size + q_offset;
                    int q_global = query_block_start + q_local;

                    for (int k = 0; k < THREAD_K; k++) {

                        int k_local = k_base + k;
                        int key_global = key_block_start + k_local;

                        if (q_offset >= query_tile_size || q_local >= BLOCK_Q || q_global >= row || k_local >= BLOCK_KV || key_global >= col || key_global > q_global) {
                            score_acc[q][k] = -INFINITY;
                        }
                    }
                }

                // Update online softmax
                // No need for warp maxing
                for (int q = 0 ; q < THREAD_Q ; q++){
                    int q_offset = q_base + q;
                    if (q_offset >= query_tile_size) continue;
                    int q_local = warp_id * query_tile_size + q_offset;
                    int q_global = query_block_start + q_local;

                    if (q_local >= BLOCK_Q || q_global >= row) continue;

                    // 1. Maximum of current score tile
                    float local_max = -INFINITY;
                    for (int k = 0 ; k < THREAD_K ; k++){
                        local_max = max(local_max, score_acc[q][k]);
                    }
                    float new_max = max(local_max, running_max[q_offset]);

                    if (new_max == -INFINITY) continue;
                    // Rescaling of global denominaotr and accumulator
                    float rescaling_factor = (running_max[q_offset] == -INFINITY) ? 0.0f : expf(running_max[q_offset] - new_max);

                    running_sum[q_offset] *= rescaling_factor;
                    for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){ // We do this because every warp_lane contains it, so its easier to do PV
                        output_accumulator[q_offset][j] *= rescaling_factor; 
                    }
                    
                    // Update scores to turn it into probabilites
                    for (int k = 0 ; k < THREAD_K ; k++){
                        score_acc[q][k] = (score_acc[q][k] == -INFINITY) ? 0.0f : expf(score_acc[q][k] - new_max);
                        running_sum[q_offset] += score_acc[q][k]; 

                        int k_local = k_base + k;
                        // Update PV
                        for (int l = 0 ; l < NUM_OUTPUT_GROUPS; l++){
                            int h_idx = l * WARP_SIZE + lane_id;
                            if (k_local >= BLOCK_KV || h_idx >= HEAD_DIM) continue;
                            output_accumulator[q_offset][l] += score_acc[q][k] * V_SHARED[k_local][h_idx];
                        }
                    }
                    running_max[q_offset] = new_max;
                }
            }
        }
        __syncthreads();
    }
    // Write O here, after ALL K/V tiles
    for (int j = 0; j < query_tile_size; j++) {
        int row_q_local_idx = warp_id * query_tile_size + j;
        int q_global = query_block_start + row_q_local_idx;
        if (row_q_local_idx >= BLOCK_Q || q_global >= row) continue;
        for (int d = 0; d < NUM_OUTPUT_GROUPS; d++) {
            int h = d * WARP_SIZE + lane_id;
            if (h >= HEAD_DIM) continue;
            float output = (running_sum[j] > 0.0f) ? output_accumulator[j][d] / running_sum[j] : 0.0f;
            *(O + o_head_offset + q_global * stride_os + h * stride_oh) = output;
        }
    }
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
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void casual_flash_attention_forward_gpu_optimised_for_tensor_core(float* Q,
                                                                             float* K,
                                                                             float* V,
                                                                             float* O
){

}


int main(){
    
    std::cout << "Hello! Writing Cuda Kernels Now!" << std::endl;
    float *Q,*K_transposed,*V,*O;
    // B x N x S x H
    // B is batch_size
    // N is number of heads
    // S is the sequence length
    // H is the head_dimension

    int B, N, S, H;
    int stride_qb, stride_qn, stride_qs, stride_qh;
    int stride_kb, stride_kn, stride_kh, stride_ks;
    int stride_vb, stride_vn, stride_vs, stride_vh;
    int stride_ob, stride_on, stride_oqs, stride_oh;

    int row, col, inner_dim;

    // Copy from data from host to device
    // Determining the Grid Size
    // Determining Block Size
    // 1 Warp 32 threads
    // 1 thread block 1024 threads


    // BLOCK_Q - B_r in the paper
    // BLOCK_KV - B_c in the paper
    // BLOCK_INNER_DIM - I would just set it to the min (head dimension)

    int BLOCK_Q, BLOCK_KV, BLOCK_INNER_DIM;
    BLOCK_Q = 32;
    BLOCK_KV = 32;
    BLOCK_INNER_DIM = H; // 

    // Determining grid and block dimension size
    // Can be done in the same way as triton
    // 1 element in the grid dimension is 1 block

    int TILE_R, TILE_C;
    TILE_R = (S + BLOCK_Q + 1) % S; // Ceil of Block
    dim3 gridDimension(B*N, TILE_R);

    // 1 thread block handles 
    //int num_of_threads_per_block_by_user = 128;
    //int num_of_threads_per_block = std::min(1024,num_of_threads_per_block_by_user);
    //dim3 blockDimension(num_of_threads_per_block);

    dim3 blockDimension(BLOCK_Q);



    return 0;
}