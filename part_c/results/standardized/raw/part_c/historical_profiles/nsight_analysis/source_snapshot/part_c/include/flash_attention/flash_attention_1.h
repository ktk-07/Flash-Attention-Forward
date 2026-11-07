#pragma once
#include "types.h"

namespace part_c {
// Launch the FA1-style scalar causal kernel. Check CUDA launch errors afterward.
LaunchConfig launch_flash_attention_1(Shape shape, const DevicePointers& buffers,
                                     cudaStream_t stream = nullptr);
}
