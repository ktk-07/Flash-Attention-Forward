#pragma once
#include <cstddef>
#include <vector>
namespace attention {
struct Shape { std::size_t B, H, N, d; };
// Contiguous FP32 layout [B,H,N,d]
struct Input { Shape shape; std::vector<float> q, k, v; };
}
