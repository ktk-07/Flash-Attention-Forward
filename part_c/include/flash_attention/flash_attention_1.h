#pragma once
#include "types.h"

namespace part_c {
// FA1 Algorithm-1 schedule: one CTA per batch/head, K/V outer and Q inner.
// FP32 CUDA-core arithmetic only. Q/K/V/O are contiguous [B,H,N,d].
// The caller owns two B*H*N-element FP32 arrays, allocated outside timing.
// Each forward overwrites its workspace and O; no separate reset is required.
struct FlashAttention1Workspace {
    float* maxima;
    float* normalizers;
};
std::size_t flash_attention_1_workspace_elements(Shape shape);
LaunchConfig launch_flash_attention_1(Shape shape, const DevicePointers& buffers,
                                     FlashAttention1Workspace workspace,
                                     bool causal = true,
                                     cudaStream_t stream = nullptr,
                                     KernelResources* resources = nullptr);
}
