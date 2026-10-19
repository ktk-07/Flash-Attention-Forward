#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void casual_flash_attention_forward(float* Q,
                                               float* K_transposed,
                                               float* V,
                                               float* O,
                                               int row,
                                               int col,
                                               int inner_dimension,
                                               int stride_qb, int stride_qn, int stride_qs, int stride_qh,
                                               int stride_kb, int stride_kn, int stride_kh, int stride_ks,
                                               int stride_vb, int stride_vn, int stride_vs, int stride_vh,
                                               int stride_ob, int stride_on, int stride_os, int stride_oh,
                                               float softmax_scale
                                            )
{
    // Compute the base offsets first

    // Defining the variables needed for Online Softmax Recurrence

    // QK^T

    // S = softmax(QK^T) Online Softmax Softmax


    //

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
    int num_of_threads_per_block_by_user = 128;
    int num_of_threads_per_block = std::min(1024,num_of_threads_per_block_by_user);
    dim3 blockDimension(num_of_threads_per_block);




    return 0;
}