#include "flash_attention/flash_attention_2_cuda_core.h"
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


namespace tile_skip {
// Multiple query rows per warp.
// Multiple warps per CTA.
// 2D register tiling with explicit operand reuse.
// Coalesced scalar global-memory loading (one float per lane/instruction).
// Shared-memory tiling for Q/K/V.
// Online softmax and causal tile skipping.
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, int THREAD_Q, int THREAD_K>
__global__ void casual_flash_attention_forward_gpu_optimised_for_cuda_core(float* Q,
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
    int b_idx = BN / N;
    int n_idx = BN % N;

    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_lane_idx = threadIdx.x % WARP_SIZE;

    // Computing the base_pointers;
    int q_base_ptr = b_idx * stride_qb + n_idx * stride_qn;
    int k_base_ptr = b_idx * stride_kb + n_idx * stride_kn;
    int v_base_ptr = b_idx * stride_vb + n_idx * stride_vn;
    int o_base_ptr = b_idx * stride_ob + n_idx * stride_on;

    // Running Statistics to keep track of
    float final_acc[query_tile_size][NUM_OUTPUT_GROUPS]; // each warp_lane (thread) has this
    float global_denominator[query_tile_size]; // each warp_lane (thread) is responsible for this
    float global_max[query_tile_size]; // each warp_lane (thread) is responsible for this

    // Shared Memory
    __shared__ float Q_SHARED[BLOCK_Q][HEAD_DIM];
    __shared__ float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ float V_SHARED[BLOCK_KV][HEAD_DIM];

    int tile_r_start = blockIdx.y * BLOCK_Q;
    // Cooperative Loading of Q, we want to reuse Q 
    // Different warps load query_tile_size of rows
    // Each lane loads one float per instruction;
    for (int j = 0 ; j < query_tile_size ; j++){
        global_max[j] = -INFINITY;
        global_denominator[j] = 0.0f;
        // Required: the output accumulator is multiplied and incremented later.
        for (int d = 0; d < NUM_OUTPUT_GROUPS; ++d) {
            final_acc[j][d] = 0.0f;
        }
        // Loading Loop for 
        // Cooperative Loading of Q, K, V, in the memory coalesced way
        int q_local = warp_idx * query_tile_size + j;
        int sq_idx = tile_r_start + q_local; // row_idx for q

        // Q Load
        for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
            int h_idx = k * WARP_SIZE + warp_lane_idx; // col_idx
            if (q_local >= BLOCK_Q || h_idx >= HEAD_DIM) continue;
            if (sq_idx >= row) {
                Q_SHARED[q_local][h_idx] = 0.0f;
            } else {
                Q_SHARED[q_local][h_idx] = *(Q + q_base_ptr + sq_idx * stride_qs + h_idx * stride_qh);
            }
        }
    }
    __syncthreads();

    // Causal tile skipping: a query block cannot attend to keys beyond
    // its last valid query position. Skip those K/V tiles entirely.
    // This assumes aligned causal self-attention (query/key position 0 matches).
    const int last_query_exclusive = min(row, tile_r_start + BLOCK_Q);
    const int valid_key_exclusive = min(col, last_query_exclusive);
    const int tile_c = (valid_key_exclusive + BLOCK_KV - 1) / BLOCK_KV;
    for (int i = 0 ; i < tile_c ; i++){
        int tile_c_start = i * BLOCK_KV;
        // Loading Loop for 
        // Cooperative Loading of K, V, in the memory coalesced way

        // Basicaly 1 warp load query_tile_size of the row of K and V;
        // K and V are both stored row-major in shared memory.
        // The problem with this is that im afraid shared memory would fill up 
        constexpr int no_of_row_iterations = (BLOCK_KV + NUM_WARPS - 1) / NUM_WARPS;
        for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){       
            int h_idx = j * WARP_SIZE + warp_lane_idx;
            if (h_idx >= HEAD_DIM) continue; 
            // K/V Load feels like i might need another loop here that loops over kv_tile_size ?;
            for (int k = 0 ; k < no_of_row_iterations; k++){
                int row_local = k * NUM_WARPS + warp_idx;
                int skv_idx = tile_c_start + row_local; // row_idx for kv 

                if (row_local >= BLOCK_KV) continue;

                if (skv_idx >= col){
                    K_SHARED[row_local][h_idx] = 0.0f;
                    V_SHARED[row_local][h_idx] = 0.0f;
                }else{
                    K_SHARED[row_local][h_idx] = *(K + k_base_ptr + skv_idx * stride_ks + h_idx * stride_kh);
                    V_SHARED[row_local][h_idx] = *(V + v_base_ptr + skv_idx * stride_vs + h_idx * stride_vh);                   
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
                int row_q_local_idx = warp_idx * query_tile_size + q_offset;
                int q_global = tile_r_start + row_q_local_idx;
                for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                    int col_q_local_idx = k * WARP_SIZE + warp_lane_idx;
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
                    int key_global = tile_c_start + row_k_local_idx;
                    for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                        int col_k_local_idx = k * WARP_SIZE + warp_lane_idx;
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
                    int q_local = warp_idx * query_tile_size + q_offset;
                    int q_global = tile_r_start + q_local;

                    for (int k = 0; k < THREAD_K; k++) {

                        int k_local = k_base + k;
                        int key_global = tile_c_start + k_local;

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
                    int q_local = warp_idx * query_tile_size + q_offset;
                    int q_global = tile_r_start + q_local;

                    if (q_local >= BLOCK_Q || q_global >= row) continue;

                    // 1. Maximum of current score tile
                    float local_max = -INFINITY;
                    for (int k = 0 ; k < THREAD_K ; k++){
                        local_max = max(local_max, score_acc[q][k]);
                    }
                    float new_max = max(local_max, global_max[q_offset]);

                    if (new_max == -INFINITY) continue;
                    // Rescaling of global denominaotr and accumulator
                    float rescaling_factor = (global_max[q_offset] == -INFINITY) ? 0.0f : expf(global_max[q_offset] - new_max);

                    global_denominator[q_offset] *= rescaling_factor;
                    for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){ // We do this because every warp_lane contains it, so its easier to do PV
                        final_acc[q_offset][j] *= rescaling_factor; 
                    }
                    
                    // Update scores to turn it into probabilites
                    for (int k = 0 ; k < THREAD_K ; k++){
                        score_acc[q][k] = (score_acc[q][k] == -INFINITY) ? 0.0f : expf(score_acc[q][k] - new_max);
                        global_denominator[q_offset] += score_acc[q][k]; 

                        int k_local = k_base + k;
                        // Update PV
                        for (int l = 0 ; l < NUM_OUTPUT_GROUPS; l++){
                            int h_idx = l * WARP_SIZE + warp_lane_idx;
                            if (k_local >= BLOCK_KV || h_idx >= HEAD_DIM) continue;
                            final_acc[q_offset][l] += score_acc[q][k] * V_SHARED[k_local][h_idx];
                        }
                    }
                    global_max[q_offset] = new_max;
                }
            }
        }
        __syncthreads();
    }
    // Write O here, after ALL K/V tiles
    for (int j = 0; j < query_tile_size; j++) {
        int row_q_local_idx = warp_idx * query_tile_size + j;
        int q_global = tile_r_start + row_q_local_idx;
        if (row_q_local_idx >= BLOCK_Q || q_global >= row) continue;
        for (int d = 0; d < NUM_OUTPUT_GROUPS; d++) {
            int h = d * WARP_SIZE + warp_lane_idx;
            if (h >= HEAD_DIM) continue;
            float output = (global_denominator[j] > 0.0f) ? final_acc[j][d] / global_denominator[j] : 0.0f;
            *(O + o_base_ptr + q_global * stride_os + h * stride_oh) = output;
        }
    }
}
}

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


namespace vector_load {
// Vectorized Q/K/V global-memory loads, with scalar fallback.
// Original coalesced scalar output stores are retained.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Multiple query rows per warp.
// Multiple warps per CTA.
// 2D register tiling with explicit operand reuse.
// Coalesced float4 global-memory loading where aligned; scalar fallback otherwise.
// Shared-memory tiling for Q/K/V.
// Online softmax and causal tile skipping.
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int NUM_WARPS, int THREAD_Q, int THREAD_K>
__global__ void casual_flash_attention_forward_gpu_optimised_for_cuda_core(float* Q,
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
    // Head features remain distributed across lanes during computation.

    int BN = blockIdx.x;
    int b_idx = BN / N;
    int n_idx = BN % N;

    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_lane_idx = threadIdx.x % WARP_SIZE;

    // Computing the base_pointers;
    int q_base_ptr = b_idx * stride_qb + n_idx * stride_qn;
    int k_base_ptr = b_idx * stride_kb + n_idx * stride_kn;
    int v_base_ptr = b_idx * stride_vb + n_idx * stride_vn;
    int o_base_ptr = b_idx * stride_ob + n_idx * stride_on;

    // Running Statistics to keep track of
    float final_acc[query_tile_size][NUM_OUTPUT_GROUPS]; // each warp_lane (thread) has this
    float global_denominator[query_tile_size]; // each warp_lane (thread) is responsible for this
    float global_max[query_tile_size]; // each warp_lane (thread) is responsible for this

    // Shared Memory
    __shared__ __align__(16) float Q_SHARED[BLOCK_Q][HEAD_DIM];
    __shared__ __align__(16) float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ __align__(16) float V_SHARED[BLOCK_KV][HEAD_DIM];

    int tile_r_start = blockIdx.y * BLOCK_Q;
    // Cooperative Loading of Q, we want to reuse Q 
    // Different warps load query_tile_size of rows
    // Each lane loads up to four adjacent floats per vector instruction;
    for (int j = 0 ; j < query_tile_size ; j++){
        global_max[j] = -INFINITY;
        global_denominator[j] = 0.0f;
        // Required: the output accumulator is multiplied and incremented later.
        for (int d = 0; d < NUM_OUTPUT_GROUPS; ++d) {
            final_acc[j][d] = 0.0f;
        }
        // Loading Loop for 
        // Cooperative Loading of Q, K, V, in the memory coalesced way
        int q_local = warp_idx * query_tile_size + j;
        int sq_idx = tile_r_start + q_local; // row_idx for q

        // Vectorized cooperative Q loading: each lane handles 4 adjacent
        // features per iteration. The shared-memory layout stays row-major,
        // so the existing 2D register-tiled compute phase is unchanged.
        // A float4 access requires a 16-byte aligned source AND destination.
        if (q_local >= BLOCK_Q) continue; // No shared-memory row allocated.
        const bool q_valid = sq_idx < row;
        const float* q_src = Q + q_base_ptr + (q_valid ? sq_idx * stride_qs : 0);
        for (int h0 = 4 * warp_lane_idx; h0 < HEAD_DIM; h0 += 4 * WARP_SIZE) {
            if (h0 + 3 < HEAD_DIM && stride_qh == 1 &&
                (HEAD_DIM % 4 == 0) &&
                ((reinterpret_cast<uintptr_t>(q_src + h0) & 15u) == 0u)) {
                float4 vec = q_valid ? *reinterpret_cast<const float4*>(q_src + h0) : make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&Q_SHARED[q_local][h0]) = vec;
            } else {
                for (int t = 0; t < 4 && h0 + t < HEAD_DIM; ++t) {
                    Q_SHARED[q_local][h0 + t] = q_valid ? Q[q_base_ptr + sq_idx * stride_qs + (h0 + t) * stride_qh] : 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // Causal tile skipping: a query block cannot attend to keys beyond
    // its last valid query position. Skip those K/V tiles entirely.
    // This assumes aligned causal self-attention (query/key position 0 matches).
    const int last_query_exclusive = min(row, tile_r_start + BLOCK_Q);
    const int valid_key_exclusive = min(col, last_query_exclusive);
    const int tile_c = (valid_key_exclusive + BLOCK_KV - 1) / BLOCK_KV;
    for (int i = 0 ; i < tile_c ; i++){
        int tile_c_start = i * BLOCK_KV;
        // Loading Loop for 
        // Cooperative Loading of K, V, in the memory coalesced way

        // Basicaly 1 warp load query_tile_size of the row of K and V;
        // K and V are both stored row-major in shared memory.
        // The problem with this is that im afraid shared memory would fill up 
        constexpr int no_of_row_iterations = (BLOCK_KV + NUM_WARPS - 1) / NUM_WARPS;
        // Interleaved warp ownership of K/V rows. Each lane loads a
        // contiguous four-float vector when alignment and strides permit.
        // Keep K_SHARED and V_SHARED row-major (NOT transposed).
        for (int k = 0; k < no_of_row_iterations; ++k) {
            int row_local = k * NUM_WARPS + warp_idx;
            if (row_local >= BLOCK_KV) continue;
            int skv_idx = tile_c_start + row_local;
            bool kv_valid = skv_idx < col;
            const float* k_src = K + k_base_ptr + (kv_valid ? skv_idx * stride_ks : 0);
            const float* v_src = V + v_base_ptr + (kv_valid ? skv_idx * stride_vs : 0);
            for (int h0 = 4 * warp_lane_idx; h0 < HEAD_DIM; h0 += 4 * WARP_SIZE) {
                bool full_vec = (h0 + 3 < HEAD_DIM) && (HEAD_DIM % 4 == 0);
                bool k_vec = full_vec && stride_kh == 1 &&
                    ((reinterpret_cast<uintptr_t>(k_src + h0) & 15u) == 0u);
                bool v_vec = full_vec && stride_vh == 1 &&
                    ((reinterpret_cast<uintptr_t>(v_src + h0) & 15u) == 0u);
                if (k_vec) {
                    float4 x = kv_valid ? *reinterpret_cast<const float4*>(k_src + h0) : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<float4*>(&K_SHARED[row_local][h0]) = x;
                } else {
                    for (int t = 0; t < 4 && h0 + t < HEAD_DIM; ++t)
                        K_SHARED[row_local][h0 + t] = kv_valid ? K[k_base_ptr + skv_idx * stride_ks + (h0 + t) * stride_kh] : 0.f;
                }
                if (v_vec) {
                    float4 x = kv_valid ? *reinterpret_cast<const float4*>(v_src + h0)
                                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    *reinterpret_cast<float4*>(&V_SHARED[row_local][h0]) = x;
                } else {
                    for (int t = 0; t < 4 && h0 + t < HEAD_DIM; ++t)
                        V_SHARED[row_local][h0 + t] = kv_valid
                            ? V[v_base_ptr + skv_idx * stride_vs + (h0 + t) * stride_vh]
                            : 0.f;
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
                int row_q_local_idx = warp_idx * query_tile_size + q_offset;
                int q_global = tile_r_start + row_q_local_idx;
                for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                    int col_q_local_idx = k * WARP_SIZE + warp_lane_idx;
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
                    int key_global = tile_c_start + row_k_local_idx;
                    for (int k = 0 ; k < NUM_OUTPUT_GROUPS ; k++){
                        int col_k_local_idx = k * WARP_SIZE + warp_lane_idx;
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
                    int q_local = warp_idx * query_tile_size + q_offset;
                    int q_global = tile_r_start + q_local;

                    for (int k = 0; k < THREAD_K; k++) {

                        int k_local = k_base + k;
                        int key_global = tile_c_start + k_local;

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
                    int q_local = warp_idx * query_tile_size + q_offset;
                    int q_global = tile_r_start + q_local;

                    if (q_local >= BLOCK_Q || q_global >= row) continue;

                    // 1. Maximum of current score tile
                    float local_max = -INFINITY;
                    for (int k = 0 ; k < THREAD_K ; k++){
                        local_max = max(local_max, score_acc[q][k]);
                    }
                    float new_max = max(local_max, global_max[q_offset]);

                    if (new_max == -INFINITY) continue;
                    // Rescaling of global denominaotr and accumulator
                    float rescaling_factor = (global_max[q_offset] == -INFINITY) ? 0.0f : expf(global_max[q_offset] - new_max);

                    global_denominator[q_offset] *= rescaling_factor;
                    for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){ // We do this because every warp_lane contains it, so its easier to do PV
                        final_acc[q_offset][j] *= rescaling_factor; 
                    }
                    
                    // Update scores to turn it into probabilites
                    for (int k = 0 ; k < THREAD_K ; k++){
                        score_acc[q][k] = (score_acc[q][k] == -INFINITY) ? 0.0f : expf(score_acc[q][k] - new_max);
                        global_denominator[q_offset] += score_acc[q][k]; 

                        int k_local = k_base + k;
                        // Update PV
                        for (int l = 0 ; l < NUM_OUTPUT_GROUPS; l++){
                            int h_idx = l * WARP_SIZE + warp_lane_idx;
                            if (k_local >= BLOCK_KV || h_idx >= HEAD_DIM) continue;
                            final_acc[q_offset][l] += score_acc[q][k] * V_SHARED[k_local][h_idx];
                        }
                    }
                    global_max[q_offset] = new_max;
                }
            }
        }
        __syncthreads();
    }
    // Normalize only after all K/V tiles. Each lane owns head features
    // h = 32*d + lane, which are not contiguous within a lane.
    // Keep coalesced scalar output stores; vectorizing them would require
    // redistributing results between lanes (e.g. via shared memory).
    for (int j = 0; j < query_tile_size; j++) {
        int row_q_local_idx = warp_idx * query_tile_size + j;
        int q_global = tile_r_start + row_q_local_idx;
        if (row_q_local_idx >= BLOCK_Q || q_global >= row) continue;
        for (int d = 0; d < NUM_OUTPUT_GROUPS; d++) {
            int h = d * WARP_SIZE + warp_lane_idx;
            if (h >= HEAD_DIM) continue;
            float output = (global_denominator[j] > 0.0f)
                ? final_acc[j][d] / global_denominator[j] : 0.0f;
            O[o_base_ptr + q_global * stride_os + h * stride_oh] = output;
        }
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

template<int D, int BQ=32, int BK=32, int NW=4, int TQ=3, int TK=5>
LaunchConfig launch_impl(CudaCoreVariant variant, Shape s, const DevicePointers& x, cudaStream_t stream, KernelResources* resources) {
    const int sb=s.h*s.n*D, sh=s.n*D, ss=D;
    const float scale=1.0f/std::sqrt(float(D));
    dim3 grid(s.b*s.h,(s.n+BQ-1)/BQ);
    // Older functions take K strides as (feature, sequence).
#define QS sb,sh,ss,1
#define KS sb,sh,1,ss
    switch (variant) {
    case CudaCoreVariant::Scalar:
        flash_attention_v2_scalar_baseline<32,32,D><<<grid,32,0,stream>>>(x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,QS,KS,QS,QS,scale);
        return {32,32,1,1,1};
    case CudaCoreVariant::WarpPerQuery: {
        dim3 warp_grid(s.b*s.h,(s.n+3)/4);
        flash_attention_v2_warp_per_query<4,32,D><<<warp_grid,128,0,stream>>>(x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,QS,KS,QS,QS,scale);
        return {4,32,4,1,1};
    }
    case CudaCoreVariant::Tiled:
        describe_kernel(flash_attention_v2_cuda_core_tiled<BQ,BK,D,NW,TQ,TK>,NW*32,resources);
        flash_attention_v2_cuda_core_tiled<BQ,BK,D,NW,TQ,TK><<<grid,NW*32,0,stream>>>(x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,QS,KS,QS,QS,scale);
        break;
    case CudaCoreVariant::TileSkipping:
        describe_kernel(tile_skip::casual_flash_attention_forward_gpu_optimised_for_cuda_core<BQ,BK,D,NW,TQ,TK>,NW*32,resources);
        tile_skip::casual_flash_attention_forward_gpu_optimised_for_cuda_core<BQ,BK,D,NW,TQ,TK><<<grid,NW*32,0,stream>>>(x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,QS,KS,QS,QS,scale);
        break;
    case CudaCoreVariant::Vectorized:
        describe_kernel(vector_load::casual_flash_attention_forward_gpu_optimised_for_cuda_core<BQ,BK,D,NW,TQ,TK>,NW*32,resources);
        vector_load::casual_flash_attention_forward_gpu_optimised_for_cuda_core<BQ,BK,D,NW,TQ,TK><<<grid,NW*32,0,stream>>>(x.q,x.k,x.v,x.o,s.n,s.n,s.b,s.h,QS,KS,QS,QS,scale);
        break;
    default: throw std::invalid_argument("unknown CUDA-core variant");
    }
#undef QS
#undef KS
    return {BQ,BK,NW,TQ,TK};
}
template<int D>
LaunchConfig configured(CudaCoreVariant variant, Shape s, const DevicePointers& x, int config, cudaStream_t stream, KernelResources* info) {
    switch(config) {
    case 0:return launch_impl<D,32,32,4,3,5>(variant,s,x,stream,info);
    case 1:return launch_impl<D,32,32,4,2,4>(variant,s,x,stream,info);
    case 2:return launch_impl<D,16,32,4,2,4>(variant,s,x,stream,info);
    case 3:return launch_impl<D,32,16,4,2,4>(variant,s,x,stream,info);
    case 4:return launch_impl<D,32,32,8,2,4>(variant,s,x,stream,info);
    default:throw std::invalid_argument("unknown CUDA-core configuration");
    }
}
}
LaunchConfig launch_flash_attention_2_cuda_core_configured(CudaCoreVariant variant, Shape s, const DevicePointers& x, int config, cudaStream_t stream, KernelResources* info) {
    if(config!=0 && (variant==CudaCoreVariant::Scalar || variant==CudaCoreVariant::WarpPerQuery))
        throw std::invalid_argument("scalar/warp baselines only support config 0");
    switch(s.d) {
#define PART_C_DIM(D) case D:return configured<D>(variant,s,x,config,stream,info);
    PART_C_DIM(7) PART_C_DIM(16) PART_C_DIM(32) PART_C_DIM(33) PART_C_DIM(40)
    PART_C_DIM(64) PART_C_DIM(65) PART_C_DIM(96) PART_C_DIM(128)
#undef PART_C_DIM
    default:throw std::invalid_argument("head dimension not instantiated");
    }
}
LaunchConfig launch_flash_attention_2_cuda_core(CudaCoreVariant variant, Shape s, const DevicePointers& x, cudaStream_t stream) {
    return launch_flash_attention_2_cuda_core_configured(variant,s,x,0,stream);
}
}
