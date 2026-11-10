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
}
