#pragma once
#include "attention_baseline.hpp"
#include "kernel_config.hpp"
#include <memory>
#include <stdexcept>
namespace attention {
class WorkspaceMemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct TimingSample { double forward_ms, qk_ms, softmax_ms, pv_ms; };
struct GpuEnvironment {
    std::string name;
    int major, minor, runtime_version, driver_api_version;
    std::size_t total_bytes, free_bytes;
};
GpuEnvironment gpu_environment();
// Preallocated Q/K/V/S/P/O on the default stream
class BaselineWorkspace {
public:
    explicit BaselineWorkspace(const Input& input, double max_free_fraction = 0.8);
    ~BaselineWorkspace();
    BaselineWorkspace(const BaselineWorkspace&) = delete;
    BaselineWorkspace& operator=(const BaselineWorkspace&) = delete;
    void run(bool causal, BaselineVariant variant, KernelConfig config = {});
    // Requires a successful run or measure
    GpuResult download(bool intermediates = true);
    void warmup(bool causal, BaselineVariant variant, KernelConfig config, unsigned calls);
    // Separate forward and stage timing
    TimingSample measure(bool causal, BaselineVariant variant, KernelConfig config);
    std::vector<KernelResources> resources(BaselineVariant variant, KernelConfig config = {}) const;
    std::size_t allocated_bytes() const;
    double setup_ms() const; // Host setup time
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
