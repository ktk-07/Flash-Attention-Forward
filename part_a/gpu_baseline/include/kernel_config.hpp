#pragma once
#include <cstddef>
#include <string>
namespace attention {
struct KernelConfig { unsigned tile = 16, softmax_threads = 256; };
struct KernelResources {
    std::string stage;
    unsigned blocks, threads;
    int registers_per_thread, active_blocks_per_sm;
    std::size_t shared_bytes, local_bytes_per_thread;
    double theoretical_occupancy_percent;
};
}
