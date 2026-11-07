#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstddef>

namespace part_c {
// Contiguous logical [batch, head, sequence, feature] tensors.
struct Shape { int b, h, n, d; };
struct DevicePointers {
    float *q, *k, *v, *o;
    half *hq, *hk, *hv;
};
struct LaunchConfig { int bq, bk, warps, tq, tk; };
// Queried outside timed regions; runtime occupancy is an estimate, not a counter.
struct KernelResources {
    int registers_per_thread = 0;
    std::size_t shared_bytes = 0, local_bytes_per_thread = 0;
    int active_blocks_per_sm = 0;
    double occupancy_estimate_percent = 0;
};
}
