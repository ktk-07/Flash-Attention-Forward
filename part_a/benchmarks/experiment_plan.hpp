#pragma once
#include "attention_baseline.hpp"
#include "kernel_config.hpp"
#include <filesystem>
#include <cstdint>
#include <stdexcept>

namespace attention::experiment {
struct Options {
    std::string grid="representative", mode="both", variants="main", configs="default";
    std::filesystem::path output;
    std::uint64_t seed=5206;
    unsigned warmup=5,repeats=20;
    attention::Shape shape{};
    KernelConfig fixed{};
    bool custom=false;
    bool config_override=false;
    bool profile_only=false;
    bool validate_only=false;
    bool variants_set=false;
};
struct Variant {std::string name; BaselineVariant value;};
inline std::vector<Variant> variants(const Options& o) {
    std::vector<Variant> out;
    for(auto v:{Variant{"naive",BaselineVariant::Naive},Variant{"tiled",BaselineVariant::TiledMatmul},
                Variant{"parallel",BaselineVariant::ParallelSoftmax},Variant{"optimized",BaselineVariant::Optimized}})
        if(o.variants=="all" || o.variants==v.name ||
           (o.variants=="main" && (v.value==BaselineVariant::Naive || v.value==BaselineVariant::Optimized))) out.push_back(v);
    if(out.empty()) throw std::invalid_argument("Invalid variants");
    return out;
}
inline std::vector<KernelConfig> configs(const Options& o) {
    if(o.configs=="study" && o.config_override) throw std::invalid_argument("Study cannot use fixed configuration flags");
    return o.configs=="study" ? std::vector<KernelConfig>{{8,256},{16,256},{32,256},{16,128}} : std::vector<KernelConfig>{o.fixed};
}
inline std::vector<attention::Shape> validation_shapes() {
    return {{1,1,17,7},{2,2,33,33},{1,1,129,64},{1,1,257,128}};
}
inline std::vector<attention::Shape> shapes(const Options& o) {
    if(o.validate_only) return validation_shapes();
    if(o.custom) return {o.shape};
    if(o.grid=="representative") return {{1,1,33,32},{4,8,256,64}};
    if(o.grid=="tuning" || o.grid=="ablation") return {{4,8,256,64},{4,8,1024,64}};
    std::vector<attention::Shape> out;
    for(std::size_t n:{256,512,1024,2048,4096})
        for(std::size_t d:{32,64,128}) out.push_back({4,8,n,d});
    return out;
}
inline std::vector<bool> modes(const Options& o) {
    if(o.mode=="dense") return {false};
    if(o.mode=="causal") return {true};
    return {false,true};
}
}
