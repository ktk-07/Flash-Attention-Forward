#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>

// Why is this a template parameter?
// Compile Time Constants
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
    s_idx = tile_r_idx * BLOCK_Q + threadIdx.x;
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
    for (int i = 0 ; i < col; i += BLOCK_KV){
        // S = QK^T - First Matmul
        int col_start = i;
        float matmul_acc[BLOCK_KV] = {};
        // K/V Loop
        for (int j = 0 ; j < BLOCK_KV; j++){
            int col_idx = col_start + j;
            if (col_idx >= col) break;
            
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
        for (int i = 0 ; i < BLOCK_KV; i++){
            matmul_acc[i] /= softmax_scale;     
        }

        // P = softmax(S) Online Softmax Softmax
        // Defining local running statistics
        float new_max = -INFINITY;        
        float local_max = -INFINITY;
        // Get Local Max
        for (int i = 0 ; i < BLOCK_KV; i++){
            local_max = max(matmul_acc[i], local_max);
        }
        
        // Skip tiles with no valid keys this if or causal attn, we skip this first is fine

        // Get the new_max, dont update first cause we need to rescale
        new_max = max(global_max, local_max);
        
        // Rescaling running statisit
        float rescaling_factor = expf(global_max - new_max);
        for (int i = 0 ; i < HEAD_DIM ; i++){
            final_acc[i] *= rescaling_factor;
        }
        global_denominator *= rescaling_factor;

        // Update the global_max
        global_max  = new_max;

        // Compute the score = exp(matmul_acc - global_max), which is P
        // We will overwrite the matmul_acc
        // Can we combine it in the loop on line 83? i think we can? 
        // after local_max = max(matmul_acc[i], local_max);
        for (int i = 0 ; i < BLOCK_KV; i++){
            matmul_acc[i] = std::exp(matmul_acc[i] - global_max);
            // we can probably sum here global_denominator += matmul_acc[i]; basically kernel fusion
        }

        for (int i = 0 ; i < BLOCK_KV; i++){
            global_denominator += matmul_acc[i];
        }

        // O = PV - Second Matmul, O is not normalised yet
        for (int j = 0 ; j < HEAD_DIM ; j++){
            for (int k = 0 ; k < BLOCK_KV ; k++){
                int col_idx = i + k;
                if (col_idx >= col) return; 
                int cur_v_base_offset = v_base_offset + col_idx * stride_vs;
                int v_ptr = cur_v_base_offset + j * stride_vh;
                final_acc[j] +=  matmul_acc[k] * *(V + v_ptr);
            }
        }
    }

    // divide the weighted accumulator by the denominator
    for (int i = 0 ; i < HEAD_DIM ; i++){
        final_acc[i] /= global_denominator;
    }

    // Write to the O 
    for (int i = 0 ; i < HEAD_DIM ; i++){

    }

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