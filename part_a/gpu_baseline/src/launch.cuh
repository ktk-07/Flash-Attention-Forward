#pragma once
#include "attention_baseline.hpp"
#include "kernels.cuh"
namespace attention::detail {
inline bool uses_tiles(BaselineVariant v) {
    return v==BaselineVariant::TiledMatmul || v==BaselineVariant::Optimized;
}
inline bool uses_parallel(BaselineVariant v) {
    return v==BaselineVariant::ParallelSoftmax || v==BaselineVariant::Optimized;
}
inline void validate_configuration(BaselineVariant v,KernelConfig c={}) {
    if(v!=BaselineVariant::Naive && v!=BaselineVariant::TiledMatmul &&
       v!=BaselineVariant::ParallelSoftmax && v!=BaselineVariant::Optimized)
        throw std::invalid_argument("Unknown variant");
    if(c.tile!=8 && c.tile!=16 && c.tile!=32) throw std::invalid_argument("Supported tiles: 8,16,32");
    if(c.softmax_threads!=128 && c.softmax_threads!=256)
        throw std::invalid_argument("Supported softmax blocks: 128,256");
}
inline void qk_stage(const float* q,const float* k,float* s,Shape shape,bool causal,BaselineVariant v,KernelConfig c={}) {
    if(uses_tiles(v)) launch_qk_tiled(q,k,s,shape,causal,c.tile);
    else launch_qk(q,k,s,shape,causal);
    cuda_check(cudaGetLastError(),"QK launch");
}
inline void softmax_stage(const float* s,float* p,Shape shape,bool causal,BaselineVariant v,KernelConfig c={}) {
    if(uses_parallel(v)) launch_softmax_parallel(s,p,shape,causal,c.softmax_threads);
    else launch_softmax(s,p,shape,causal);
    cuda_check(cudaGetLastError(),"softmax launch");
}
inline void pv_stage(const float* p,const float* v,float* o,Shape shape,BaselineVariant variant,KernelConfig c={}) {
    if(uses_tiles(variant)) launch_pv_tiled(p,v,o,shape,c.tile);
    else launch_pv(p,v,o,shape);
    cuda_check(cudaGetLastError(),"PV launch");
}
}
