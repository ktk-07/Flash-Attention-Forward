#pragma once
#include "flash_attention_1.h"

namespace part_c {
// Optional FA1 WMMA pilot, not an official FlashAttention implementation.
// K/V outer and Q inner; global normalised O/m/l between K/V tiles.
// FP16 Q/K/V and P operands; FP32 accumulators, softmax, O and workspace.
// BQ=32, BK=32, two warps; one CTA per batch/head. Workspace is caller-owned.
LaunchConfig launch_flash_attention_1_tensor_core(
    Shape shape, const DevicePointers& buffers, FlashAttention1Workspace workspace,
    bool causal = true, cudaStream_t stream = nullptr,
    KernelResources* resources = nullptr);

// SM80+: K/V outer, Q inner; normalised O and FP32 m/l in global memory.
// FP16 Q/K/V/P, FP32 accumulation/softmax/output, same MMA mapping as PTX FA2.
// Configs 0..3 match FA2 PTX: (BQ,BK,warps)=(32,32,2),(16,32,1),
// (32,16,2),(16,16,1). One CTA per head. Caller owns workspace and stream.
LaunchConfig launch_flash_attention_1_ptx(
    Shape shape,const DevicePointers& buffers,FlashAttention1Workspace workspace,
    int config=0,bool causal=true,cudaStream_t stream=nullptr,
    KernelResources* resources=nullptr);
}
