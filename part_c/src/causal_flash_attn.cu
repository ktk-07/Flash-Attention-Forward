#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>

// No Hardware Optimisations Done to this Kernel
// Compile Time Constants
// Take note that  K is mathematically tranposed and not physically tranposed, K is stored in row major format and K_tranposed is not
// In flash attention, it is taken that K is mathematically tranposed and not physically tranposed
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void correct_casual_flash_attention_forward(float* Q,
                                                       float* K_transposed,
                                                       float* V,
                                                       float* O,
                                                       int row,
                                                       int col,
                                                       int inner_dimension, // not too sure if this is needed cause it is essentially HEAD_DIM
                                                       int B, int N,
                                                       int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                                       int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                                       int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                                       int stride_ob, int stride_on, int stride_os, int stride_oh,
                                                       float softmax_scale
                                                       )
{
    // Section 1 Compute the base offsets first
    int b_idx, n_idx, tile_r_idx, s_idx;
    b_idx = blockIdx.x / N;
    n_idx = blockIdx.x % N;
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
    q_base_offset = b_idx * stride_qb + n_idx * stride_qn + s_idx * stride_qs;
    k_base_offset = b_idx * stride_kb + n_idx * stride_kn; // no dependency on BLOCK Q' Sequence
    v_base_offset = b_idx * stride_vb + n_idx * stride_vn; // no dependency on BLOCK Q' Sequence
    o_base_offset = b_idx * stride_ob + n_idx * stride_on + s_idx * stride_os;

    
    // Section 2: Defining the variables needed for Online Softmax Recurrence
    // Do we have to define this outside the cuda kernel or inside is perfectly fine?
    // Each thread owns one query row and its own softmax state
    float global_max, global_denominator;
    global_denominator = 0.0f;
    global_max = -INFINITY; // In C++ FLT_MIN, in CUDA they already have this pre-built macro
    float final_acc[HEAD_DIM] = {};
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
        new_max = max(global_max, local_max);
        
        // Rescaling running statisit
        float rescaling_factor = expf(global_max - new_max);
        for (int j = 0 ; j < HEAD_DIM ; j++){
            final_acc[j] *= rescaling_factor;
        }
        global_denominator *= rescaling_factor;

        // Update the global_max
        global_max  = new_max;

        // Compute the score = exp(matmul_acc - global_max), which is P
        // We will overwrite the matmul_acc
        // Accumulate the global denominator in the same loop
        for (int j = 0 ; j < BLOCK_KV; j++){
            matmul_acc[j] = std::exp(matmul_acc[j] - global_max);
            global_denominator += matmul_acc[j];
        }

        // O = PV - Second Matmul, O is not normalised yet
        for (int j = 0 ; j < HEAD_DIM ; j++){
            for (int k = 0 ; k < BLOCK_KV ; k++){
                int col_idx = i + k;
                if (col_idx >= col || col_idx > s_idx) continue; // No need to compute future tokens
                int cur_v_base_offset = v_base_offset + col_idx * stride_vs;
                int v_ptr = cur_v_base_offset + j * stride_vh;
                final_acc[j] +=  matmul_acc[k] * *(V + v_ptr);
            }
        }
    }

    // Divide the weighted accumulator by the denominator
    // Write to the O 
    float inv_denominator = 1.0f / global_denominator;
    for (int i = 0 ; i < HEAD_DIM ; i++){
        int o_ptr = o_base_offset + i * stride_oh;
        *(O + o_ptr) = final_acc[i] * inv_denominator;
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
__global__ void casual_flash_attention_forward_gpu_optimised_for_cuda_core(float* Q,
                                                                           float* K_transposed,
                                                                           float* V,
                                                                           float* O,
                                                                           int row,
                                                                           int col,
                                                                           int inner_dimension, // not too sure if this is needed cause it is essentially HEAD_DIM
                                                                           int B, int N,
                                                                           int num_warps, int warp_size;
                                                                           int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                                                           int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                                                           int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                                                           int stride_ob, int stride_on, int stride_os, int stride_oh,
                                                                           float softmax_scale
                                                                           )
{
    int b_idx = blockIdx.x / N;
    int n_idx = blockIdx.x % N;
    int tile_r_idx = blockIdx.y;

    //int warp_size = 32;
    int warp_idx = threadIdx.x / warp_size;
    int warp_lane_idx = threadIdx.x % warp_size;
    int s_idx = tile_r_idx * BLOCK_Q + warp_idx;

    int q_base_ptr = b_idx * stride_qb + n_idx * stride_qn + s_idx * stride_qs;
    int k_base_ptr = b_idx * stride_kb + n_idx * stride_kn;
    int v_base_ptr = b_idx * stride_vb + n_idx * stride_vn; 
    int o_base_ptr = b_idx * stride_ob + n_idx * stride_on + s_idx * stride_os;
    
    // Defining the running statistics;
    float global_denominator = 0.0f;
    float global_max = -INFINITY;
    constexpr int NUM_OUTPUT_GROUPS = ( HEAD_DIM + warp_size - 1 ) / warp_size;
    // NUM_OUTPUT_GROUP Probably has to be a compile time constant
    float final_acc[NUM_OUTPUT_GROUPS] = {}; // 1 query 1 warp, each warp_lane, (HEAD_DIM + WARP_SIZE - 1 ) / warp_size number of output group

    int tile_c = (col + BLOCK_KV - 1 ) / BLOCK_KV;

    // Is this shared amongst each threadblock or each warp?
    __shared__ float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ float V_SHARED[BLOCK_KV][HEAD_DIM];

    for (int i ; i < tile_c; i++){
        int tile_c_start = i * BLOCK_KV;

        // Loading Loop
        // Cooperative Loading of K and V
        // We want to load K and V in a memory coalescing manner. Since K and V are both in row major format
        // Each warp should be responsible for each col, that why memory coalescing happens
        constexpr int no_col_iterations = (HEAD_DIM + warp_size - 1) / warp_size;
        constexpr int no_row_iterations = (BLOCK_KV + num_warps - 1) / num_warps;
        for (int j = 0 ; j < no_col_iterations ; j++){
            int col_idx = j * warp_size + warp_lane_idx;
            for (int k = 0 ; k < no_row_iterations ; k++){
                int row_local = k * num_warps + warp_idx;
                int row_idx = tile_c_start + row_local; 

                if (row_local >= BLOCK_KV)
                    continue;

                if (row_idx >= col) {
                    K_shared[row_local][inner_dim_idx] = 0.0f;
                    V_shared[row_local][inner_dim_idx] = 0.0f;
                    continue;
                } 

                K_SHARED[row_local][col_idx] = *(K + k_base_ptr + col_idx * stride_kh + row_idx * stride_ks);
                V_SHARED[row_local][col_idx] = *(V + v_base_ptr + col_idx * stride_vh + row_idx * stride_vs);
            }
        }

        __syncthreads();

         // Computation Loop : QK^T
        // We are computing 1 query row., each warplane assign to 1 element of the K
        // Then we loop throught the inner_dim
        constexpr int NUM_KEY_GROUPS = (BLOCK_KV + warp_size - 1 ) / warp_size;
        float matmul_acc[NUM_KEY_GROUPS];
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            int col_local = j * warp_size + warp_lane_idx;
            int col_idx = tile_c_start + col_local; 

            if (col_idx >= col || col_idx > s_idx || col_local >= BLOCK_KV){
                matmul_acc[j] = -INFINITY;
                continue;
            }

            for (int k = 0 ; k < HEAD_DIM ; k++){
                matmul_acc[j] += *(Q + q_base_ptr + k * stride_qh) * K_SHARED[col_local][k];
            }

            matmul_acc[j] *= softmax_scale;
        }

        __syncthreads();

        // Softmax Recurrence
        // Getting the warp lane max
        float new_max = -INFINITY;
        float warp_lane_max = -INFINITY;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            warp_lane_max = max(matmul_acc[j],warp_lane_max);
        }
        // Essentially the local max of that row;
        float local_max = warp_max(warp_lane_max,32);
        new_max = max(global_max, local_max);
        // Rescale Running Statistics
        // Geting the rescaling_factor
        // Rescale final_acc
        float rescaling_factor = (new_max == -INFINITY) ? 0.0f : expf(global_max - new_max);
        for (int j = 0 ; j < NUM_OUTPUT_GROUPS ; j++){
            final_acc[j] *= rescaling_factor;
        }
        global_denominator *=  rescaling_factor;

        global_max = new_max;

        // Sum up the score
        float warp_lane_sum = 0.0f;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            matmul_acc[j] = (global_max == -INFINITY) ? 0.0f : expf(matmul_acc[j] - global_max); // Update matmul_acc with probability
            warp_lane_sum += matmul_acc[j];
        }
        float local_warp_sum = warp_sum(warp_lane_sum,32);
        global_denominator += local_warp_sum;

        // Computation Loop : PV
        // We are trying to compute 1 query row
        // We need to distribute the probabilyt of the element to every other row then sum it up
        // 1. Broadcasting of probability to each warp
        // 2. Accumulated into the final_acc
        constexpr int FULLMASK = 0xffffffffu;
        int width = 32;
        for (int j = 0 ; j < NUM_KEY_GROUPS ; j++){
            for(int k = 0 ; k < warp_size ; k++){
                // We want to broadcast this probability value
                int p_idx = warp_idx * warp_size + k;
                int srcLane = k;
                float broadcast_probability_value = __shfl_sync(FULL_MASK, matmul[j], srcLane, width);
                for(int l = 0 ; l < NUM_OUTPUT_GROUPS; l++){
                    int row_idx = l * warp_lane_idx +  ;
                    int col_idx = tile_c_start + l * warp_lane_idx;
                    final_acc[l] += broadcast_probability_value * V[][];
                }
            }
        }

    }
    // Normalise Accumulation then Write to O which is in HBM
    for (int i = 0 ; i < NUM_OUTPUT_GROUPS; i++){
        final_acc[i] /= global_denominator;
        int col_output_idx = warp_lane_idx + warp_size * i;
        *(O + o_base_ptr + col_output_idx * stride_oh) = final_acc[i];
    }
}

// Implementing the TensorCore Optimised Kernels
template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void casual_flash_attention_forward_gpu_optimised_for_tensor_core(float* Q,
                                                                             float* K
                                                                             float* V
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