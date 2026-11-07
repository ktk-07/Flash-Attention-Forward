#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace part_c {
// Contiguous logical [batch, head, sequence, feature] tensors.
struct Shape { int b, h, n, d; };
struct DevicePointers {
    float *q, *k, *v, *o;
    half *hq, *hk, *hv;
};
struct LaunchConfig { int bq, bk, warps, tq, tk; };
}
