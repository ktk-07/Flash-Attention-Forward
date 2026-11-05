#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <stdexcept>
#include <iomanip>

// No Hardware Optimisations Done to this Kernel
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

// 1 warp is responsible for 1 query row
// Example Grid and BlockDimension
// dim3 gridDim(B*N, (Q + BLOCK_Q -1 )/ BLOCK_Q); dim3 blockDim(num_warps * warp_size);
// num_warps = BLOCK_Q right now, but it does not have to be but we would have to adjust the code
__device__ float warp_max(float val, int warp_size) {
    constexpr unsigned FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return __shfl_sync(FULL_MASK, val, 0, warp_size);
}
__device__ float warp_sum(float val, int warp_size) {
    constexpr unsigned FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return __shfl_sync(FULL_MASK, val, 0, warp_size);
}

template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void casual_flash_attention_forward_gpu_optimised_warp_per_query(
    float* Q, float* K, float* V, float* O,
    int row, int col, int B, int N,
    int stride_qb, int stride_qn, int stride_qs, int stride_qh,
    int stride_kb, int stride_kn, int stride_kh, int stride_ks,
    int stride_vb, int stride_vn, int stride_vs, int stride_vh,
    int stride_ob, int stride_on, int stride_os, int stride_oh,
    float softmax_scale)
{
    // One warp per query row, and all warps in a CTA share the K/V tile.
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_Q;
    int b_idx = blockIdx.x / N;
    int n_idx = blockIdx.x % N;
    int tile_r_idx = blockIdx.y;
    // warp_idx selects the query row; warp_lane_idx selects a lane within that warp.
    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_lane_idx = threadIdx.x % WARP_SIZE;
    int s_idx = tile_r_idx * BLOCK_Q + warp_idx;

    int q_base_ptr = b_idx * stride_qb + n_idx * stride_qn + s_idx * stride_qs;
    int k_base_ptr = b_idx * stride_kb + n_idx * stride_kn;
    int v_base_ptr = b_idx * stride_vb + n_idx * stride_vn;
    int o_base_ptr = b_idx * stride_ob + n_idx * stride_on + s_idx * stride_os;

    // Section 2: Online softmax running statistics, independently owned by each warp.
    float global_denominator = 0.0f;
    float global_max = -INFINITY;
    constexpr int NUM_OUTPUT_GROUPS = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
    // One query per warp: each lane owns (HEAD_DIM + WARP_SIZE - 1)/WARP_SIZE
    // output features, accumulated across all K/V tiles.
    float final_acc[NUM_OUTPUT_GROUPS] = {};
    int tile_c = (col + BLOCK_KV - 1) / BLOCK_KV;

    // Shared among all threads in the CTA, not allocated separately per warp.
    __shared__ float K_SHARED[BLOCK_KV][HEAD_DIM];
    __shared__ float V_SHARED[BLOCK_KV][HEAD_DIM];

    for (int i = 0; i < tile_c; i++) {
        int tile_c_start = i * BLOCK_KV;
        // Loading Loop: cooperative K/V loading.
        // We want to load K and V in a memory coalescing manner.
        // During loading:
        // - warp_idx distributes K/V rows across warps.
        // - warp_lane_idx distributes feature columns across lanes.
        constexpr int no_col_iterations = (HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;
        constexpr int no_row_iterations = (BLOCK_KV + NUM_WARPS - 1) / NUM_WARPS;
        for (int j = 0; j < no_col_iterations; j++) {
            int col_idx = j * WARP_SIZE + warp_lane_idx;
            if (col_idx >= HEAD_DIM) continue;
            for (int k = 0; k < no_row_iterations; k++) {
                int row_local = k * NUM_WARPS + warp_idx;
                int row_idx = tile_c_start + row_local;
                if (row_local >= BLOCK_KV) continue;
                if (row_idx >= col) {
                    K_SHARED[row_local][col_idx] = 0.0f;
                    V_SHARED[row_local][col_idx] = 0.0f;
                } else {
                    K_SHARED[row_local][col_idx] = *(K + k_base_ptr + col_idx * stride_kh + row_idx * stride_ks);
                    V_SHARED[row_local][col_idx] = *(V + v_base_ptr + col_idx * stride_vh + row_idx * stride_vs);
                }
            }
        }
        // The first barrier protects consumers from incomplete loading.
        __syncthreads();

        // Computation Loop: QK^T.
        // One warp computes one query row; each lane handles a key position
        // within each group of up to 32 keys, looping over HEAD_DIM.
        constexpr int NUM_KEY_GROUPS = (BLOCK_KV + WARP_SIZE - 1) / WARP_SIZE;
        float matmul_acc[NUM_KEY_GROUPS];
        for (int j = 0; j < NUM_KEY_GROUPS; j++) {
            matmul_acc[j] = 0.0f;
            int col_local = j * WARP_SIZE + warp_lane_idx;
            int col_idx = tile_c_start + col_local;
            if (s_idx >= row || col_idx >= col || col_idx > s_idx || col_local >= BLOCK_KV) {
                matmul_acc[j] = -INFINITY;
                continue;
            }
            for (int k = 0; k < HEAD_DIM; k++)
                matmul_acc[j] += *(Q + q_base_ptr + k * stride_qh) * K_SHARED[col_local][k];
            matmul_acc[j] *= softmax_scale;
        }

        // Softmax recurrence:
        // 1. Reduce the current tile's maximum across the warp.
        // 2. Rescale the previous numerator and denominator.
        // 3. Add the current tile's unnormalised exponential weights.
        float warp_lane_max = -INFINITY;
        for (int j = 0; j < NUM_KEY_GROUPS; j++)
            warp_lane_max = fmaxf(matmul_acc[j], warp_lane_max);
        float local_max = warp_max(warp_lane_max, WARP_SIZE);
        float new_max = fmaxf(global_max, local_max);
        float rescaling_factor = (new_max == -INFINITY) ? 0.0f : expf(global_max - new_max);
        for (int j = 0; j < NUM_OUTPUT_GROUPS; j++) final_acc[j] *= rescaling_factor;
        global_denominator *= rescaling_factor;
        global_max = new_max;

        float warp_lane_sum = 0.0f;
        for (int j = 0; j < NUM_KEY_GROUPS; j++) {
            matmul_acc[j] = (global_max == -INFINITY) ? 0.0f : expf(matmul_acc[j] - global_max);
            warp_lane_sum += matmul_acc[j];
        }
        global_denominator += warp_sum(warp_lane_sum, WARP_SIZE);

        // Computation Loop: PV.
        // Each lane holds the weight for one key in matmul_acc[j].
        // Broadcast that weight across the warp using __shfl_sync.
        // Every lane multiplies it by the V feature columns it owns and
        // accumulates the result into final_acc.
        // Every warp can access all V rows in the current shared-memory tile.
        constexpr unsigned FULL_MASK = 0xffffffffu;
        int width = 32;
        for (int j = 0; j < NUM_KEY_GROUPS; j++) {
            for (int k = 0; k < WARP_SIZE; k++) {
                int k_local = j * WARP_SIZE + k;
                int k_idx = tile_c_start + k_local;
                float broadcast_probability_val = __shfl_sync(FULL_MASK, matmul_acc[j], k, width);
                for (int l = 0; l < NUM_OUTPUT_GROUPS; l++) {
                    int inner_dim_idx = l * WARP_SIZE + warp_lane_idx;
                    if (k_local < BLOCK_KV && k_idx < col && inner_dim_idx < HEAD_DIM)
                        final_acc[l] += broadcast_probability_val * V_SHARED[k_local][inner_dim_idx];
                }
            }
        }
        // The second barrier prevents the next K/V load from overwriting
        // shared memory while another warp is still consuming this tile.
        __syncthreads();
    }
    // Normalise the final weighted accumulator by the online-softmax
    // denominator, then write the output row back to global memory.
    for (int i = 0; i < NUM_OUTPUT_GROUPS; i++) {
        int col_output_idx = warp_lane_idx + WARP_SIZE * i;
        if (s_idx < row && col_output_idx < HEAD_DIM) {
            float output_val = (global_denominator > 0.0f) ? final_acc[i] / global_denominator : 0.0f;
            *(O + o_base_ptr + col_output_idx * stride_oh) = output_val;
        }
    }
}

// Naive causal attention reference: one thread per output element.
// Unlike FlashAttention, this computes each softmax row using three full passes
// over all visible keys, with no online-softmax recurrence or K/V tiling.
// Tensors are contiguous [B, N, S, H] in this test harness.
__global__ void naive_causal_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int B, int N, int S, int H, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * S * H;
    if (idx >= total) return;

    int h = idx % H;
    int query = (idx / H) % S;
    int head = (idx / (H * S)) % N;
    int batch = idx / (H * S * N);
    int base = ((batch * N + head) * S) * H;

    float max_score = -INFINITY;
    for (int key = 0; key <= query; ++key) {
        float score = 0.0f;
        for (int d = 0; d < H; ++d)
            score += Q[base + query * H + d] * K[base + key * H + d];
        max_score = fmaxf(max_score, score * scale);
    }

    float denom = 0.0f;
    float numerator = 0.0f;
    for (int key = 0; key <= query; ++key) {
        float score = 0.0f;
        for (int d = 0; d < H; ++d)
            score += Q[base + query * H + d] * K[base + key * H + d];
        float weight = expf(score * scale - max_score);
        denom += weight;
        numerator += weight * V[base + key * H + h];
    }
    O[idx] = numerator / denom;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

// Independent host-side mathematical reference (double precision accumulation).
static std::vector<float> cpu_reference(
    const std::vector<float>& Q, const std::vector<float>& K,
    const std::vector<float>& V, int B, int N, int S, int H)
{
    std::vector<float> O(Q.size());
    const double scale = 1.0 / std::sqrt(static_cast<double>(H));
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n) {
            int base = (b * N + n) * S * H;
            for (int q = 0; q < S; ++q) {
                std::vector<double> scores(q + 1);
                double mx = -INFINITY;
                for (int k = 0; k <= q; ++k) {
                    double dot = 0.0;
                    for (int d = 0; d < H; ++d)
                        dot += static_cast<double>(Q[base + q * H + d]) * K[base + k * H + d];
                    scores[k] = dot * scale;
                    mx = std::max(mx, scores[k]);
                }
                double denom = 0.0;
                for (double& x : scores) {
                    x = std::exp(x - mx);
                    denom += x;
                }
                for (int d = 0; d < H; ++d) {
                    double acc = 0.0;
                    for (int k = 0; k <= q; ++k)
                        acc += scores[k] * V[base + k * H + d];
                    O[base + q * H + d] = static_cast<float>(acc / denom);
                }
            }
        }
    return O;
}

static void compare(const char* label, const std::vector<float>& actual,
                    const std::vector<float>& expected, float atol = 2e-4f,
                    float rtol = 2e-4f)
{
    float max_error = 0.0f;
    int failures = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float error = std::fabs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual[i]) || error > atol + rtol * std::fabs(expected[i])) {
            if (failures < 3)
                std::cerr << "  mismatch at " << i << ": got " << actual[i]
                          << ", expected " << expected[i] << "\n";
            ++failures;
        }
    }
    std::cout << "  " << label << ": " << (failures ? "FAIL" : "PASS")
              << " (max abs error=" << max_error << ")\n";
    if (failures) throw std::runtime_error(std::string(label) + " failed");
}

template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
static void run_test(int B, int N, int S, unsigned seed)
{
    constexpr int H = HEAD_DIM;
    const size_t count = static_cast<size_t>(B) * N * S * H;
    const size_t bytes = count * sizeof(float);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> q(count), k(count), v(count);
    for (size_t i = 0; i < count; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
        v[i] = dist(rng);
    }
    const auto expected = cpu_reference(q, k, v, B, N, S, H);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    float *dNaive = nullptr, *dFlash = nullptr, *dWarp = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dQ), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dK), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dV), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNaive), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFlash), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dWarp), bytes));
        CUDA_CHECK(cudaMemcpy(dQ, q.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, k.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, v.data(), bytes, cudaMemcpyHostToDevice));

        const float scale = 1.0f / std::sqrt(static_cast<float>(H));
        int total = static_cast<int>(count);
        naive_causal_attention_forward<<<(total + 255) / 256, 256>>>(
            dQ, dK, dV, dNaive, B, N, S, H, scale);
        CUDA_CHECK(cudaGetLastError());

        const int stride_h = 1, stride_s = H, stride_n = S * H, stride_b = N * S * H;
        dim3 grid(B * N, (S + BLOCK_Q - 1) / BLOCK_Q);
        dim3 block(BLOCK_Q);
        correct_casual_flash_attention_forward<BLOCK_Q, BLOCK_KV, HEAD_DIM>
            <<<grid, block>>>(dQ, dK, dV, dFlash,
                S, S, H, B, N,
                stride_b, stride_n, stride_s, stride_h,
                stride_b, stride_n, stride_h, stride_s, // K uses (h, s) stride order
                stride_b, stride_n, stride_s, stride_h,
                stride_b, stride_n, stride_s, stride_h,
                scale);
        CUDA_CHECK(cudaGetLastError());
        dim3 warp_block(BLOCK_Q * 32);
        casual_flash_attention_forward_gpu_optimised_warp_per_query<BLOCK_Q, BLOCK_KV, HEAD_DIM>
            <<<grid, warp_block>>>(dQ, dK, dV, dWarp,
                S, S, B, N,
                stride_b, stride_n, stride_s, stride_h,
                stride_b, stride_n, stride_h, stride_s,
                stride_b, stride_n, stride_s, stride_h,
                stride_b, stride_n, stride_s, stride_h,
                scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> naive(count), flash(count), warp(count);
        CUDA_CHECK(cudaMemcpy(naive.data(), dNaive, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flash.data(), dFlash, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(warp.data(), dWarp, bytes, cudaMemcpyDeviceToHost));
        std::cout << "Test B=" << B << " N=" << N << " S=" << S
                  << " H=" << H << " BLOCK_Q=" << BLOCK_Q
                  << " BLOCK_KV=" << BLOCK_KV << "\n";
        compare("Naive CUDA vs CPU", naive, expected);
        compare("Flash CUDA vs CPU", flash, expected);
        compare("Flash CUDA vs naive CUDA", flash, naive);
        compare("Warp-per-query vs CPU", warp, expected);
        compare("Warp-per-query vs baseline Flash", warp, flash);
    } catch (...) {
        cudaFree(dQ); cudaFree(dK); cudaFree(dV);
        cudaFree(dNaive); cudaFree(dFlash); cudaFree(dWarp);
        throw;
    }
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dNaive); cudaFree(dFlash); cudaFree(dWarp);
}

int main()
{
    try {
        // Includes short sequences, partial Q/KV tiles, and multiple heads/batches.
        run_test<4, 32, 64>(1, 1, 1, 1);
        run_test<4, 32, 64>(1, 1, 17, 2);
        run_test<4, 32, 64>(2, 3, 65, 3);
        run_test<4, 16, 64>(1, 2, 79, 4);
        run_test<4, 32, 128>(1, 2, 33, 5);
        run_test<4, 48, 64>(1, 2, 97, 6); // non-multiple-of-warp key tile
        run_test<4, 20, 96>(2, 1, 37, 7); // partial Q/KV tiles, non-64 head
        std::cout << "All correctness tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
