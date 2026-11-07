#pragma once
#include "types.h"

namespace part_c {
enum class TensorCoreVariant { Wmma, PtxMma };
// FP16 Q/K/V, FP32 output; FP32 accumulation and online-softmax state.
LaunchConfig launch_flash_attention_2_tensor_core(TensorCoreVariant variant, Shape shape,
                                                 const DevicePointers& buffers,
                                                 cudaStream_t stream = nullptr);
LaunchConfig launch_flash_attention_2_tensor_core_configured(
    TensorCoreVariant variant, Shape shape, const DevicePointers& buffers, int config,
    cudaStream_t stream = nullptr, KernelResources* resources = nullptr);
}
