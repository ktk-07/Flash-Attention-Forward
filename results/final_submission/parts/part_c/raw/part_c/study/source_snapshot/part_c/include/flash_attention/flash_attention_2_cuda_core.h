#pragma once
#include "types.h"

namespace part_c {
enum class CudaCoreVariant { Scalar, WarpPerQuery, Tiled, TileSkipping, Vectorized };
// Launch an FP32 causal kernel; definitions and template dispatch are in src/.
LaunchConfig launch_flash_attention_2_cuda_core(CudaCoreVariant variant, Shape shape,
                                               const DevicePointers& buffers,
                                               cudaStream_t stream = nullptr);
// Bounded, predefined configurations. Index 0 retains the historical launch.
LaunchConfig launch_flash_attention_2_cuda_core_configured(
    CudaCoreVariant variant, Shape shape, const DevicePointers& buffers, int config,
    cudaStream_t stream = nullptr, KernelResources* resources = nullptr);
}
