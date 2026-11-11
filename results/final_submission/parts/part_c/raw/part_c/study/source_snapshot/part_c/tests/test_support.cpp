#include "test_support.h"
#include "input_generation.hpp"
#include "flash_attention/flash_attention_1.h"
#include "flash_attention/flash_attention_1_tensor_core.h"
#include "flash_attention/flash_attention_2_cuda_core.h"
#include "flash_attention/flash_attention_2_tensor_core.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
std::size_t Shape::size() const { return std::size_t(b)*h*n*d; }
const std::vector<std::string> kernels={"fa1","fa1_ptx","scalar","warp","tiled","skip","vector","wmma","ptx"};
bool tensor(const std::string& id) { return id=="wmma" || id=="ptx" || id=="fa1_wmma" || id=="fa1_ptx"; }
bool needs_fa1_workspace(const std::string& id) {return id=="fa1" || id=="fa1_wmma" || id=="fa1_ptx";}
Buffers::Buffers(std::size_t n,bool tc,bool minimal,std::size_t fa1_query_rows):fa1_rows(fa1_query_rows) {
    try {
        if(!minimal || !tc) { check(cudaMalloc(&q,n*4));allocated_bytes+=n*4;check(cudaMalloc(&k,n*4));allocated_bytes+=n*4;check(cudaMalloc(&v,n*4));allocated_bytes+=n*4; }
        check(cudaMalloc(&o,n*4));allocated_bytes+=n*4;
        if(fa1_rows) {check(cudaMalloc(&fa1_workspace,2*fa1_rows*sizeof(float)));allocated_bytes+=2*fa1_rows*sizeof(float);}
        if(!minimal || tc) { check(cudaMalloc(&hq,n*2));allocated_bytes+=n*2;check(cudaMalloc(&hk,n*2));allocated_bytes+=n*2;check(cudaMalloc(&hv,n*2));allocated_bytes+=n*2; }
    } catch(...) { release();throw; }
}
void Buffers::release() noexcept { cudaFree(fa1_workspace);fa1_workspace=nullptr;cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(hq);cudaFree(hk);cudaFree(hv);q=k=v=o=nullptr;hq=hk=hv=nullptr; }
Buffers::~Buffers() { release(); }
Input::Input(Shape s,const std::string& pattern,bool canonical):q(s.size()),k(s.size()),v(s.size()),rq(s.size()),rk(s.size()),rv(s.size()),hq(s.size()),hk(s.size()),hv(s.size()) {
    if(canonical) {
        attention::fill_synthetic_input({std::size_t(s.b),std::size_t(s.h),std::size_t(s.n),std::size_t(s.d)},q,k,v,5206);
    } else {
        // Preserve the historical standalone input stream.
        std::mt19937 gen(5206);std::uniform_real_distribution<float> dist(-1.f,1.f);
        for(auto* values:{&q,&k,&v}) for(auto& value:*values) value=dist(gen);
    }
    attention::apply_pattern({std::size_t(s.b),std::size_t(s.h),std::size_t(s.n),std::size_t(s.d)},q,k,v,pattern);
    for(std::size_t i=0;i<s.size();++i) {
        hq[i]=__float2half(q[i]);hk[i]=__float2half(k[i]);hv[i]=__float2half(v[i]);
        rq[i]=__half2float(hq[i]);rk[i]=__half2float(hk[i]);rv[i]=__half2float(hv[i]);
    }
}
void Input::upload(Buffers& b) const {
    const auto n=q.size();
    if(b.q) {check(cudaMemcpy(b.q,q.data(),n*4,cudaMemcpyHostToDevice));check(cudaMemcpy(b.k,k.data(),n*4,cudaMemcpyHostToDevice));check(cudaMemcpy(b.v,v.data(),n*4,cudaMemcpyHostToDevice));}
    if(b.hq) {check(cudaMemcpy(b.hq,hq.data(),n*2,cudaMemcpyHostToDevice));check(cudaMemcpy(b.hk,hk.data(),n*2,cudaMemcpyHostToDevice));check(cudaMemcpy(b.hv,hv.data(),n*2,cudaMemcpyHostToDevice));}
}
std::string input_digest(const Input& in,bool rounded) {
    return rounded?attention::input_fingerprint(in.rq,in.rk,in.rv):attention::input_fingerprint(in.q,in.k,in.v);
}
// FP64 dot products, stable softmax and output accumulation; no GPU reference.
// Full output for correctness; three query rows per head for benchmark checks.
std::vector<double> reference(Shape s,const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,bool sampled) {
    return attention::reference_output({std::size_t(s.b),std::size_t(s.h),std::size_t(s.n),std::size_t(s.d)},q,k,v,true,sampled);
}

Errors compare(const std::vector<float>& a,const std::vector<double>& r,double atol,double rtol) {
    Errors e;
    for(size_t i=0;i<a.size();++i) {
        // Check all device outputs for finiteness, including unsampled rows.
        if(!std::isfinite(a[i])) {++e.nonfinite;++e.failures; e.absolute=e.relative=INFINITY;continue;}
        if(std::isnan(r[i])) continue;
        ++e.checked;double diff=std::abs(double(a[i])-r[i]);
        e.absolute=std::max(e.absolute,diff);e.relative=std::max(e.relative,diff/std::max(1e-12,std::abs(r[i])));
        if(diff>atol+rtol*std::abs(r[i])) ++e.failures;
    }
    return e;
}

Launch dispatch(const std::string& id, Shape s, Buffers& x,int config,part_c::KernelResources* resources) {
    const part_c::Shape shape{s.b,s.h,s.n,s.d};
    const part_c::DevicePointers buffers{x.q,x.k,x.v,x.o,x.hq,x.hk,x.hv};
    Launch result{};
    using C = part_c::CudaCoreVariant;
    using T = part_c::TensorCoreVariant;
    if((id=="fa1" || id=="fa1_wmma") && config!=0) throw std::invalid_argument("FA1 supports config 0 only");
    if(needs_fa1_workspace(id)) {
        if(!x.fa1_workspace || x.fa1_rows!=std::size_t(s.b)*s.h*s.n)throw std::invalid_argument("FA1 workspace must be allocated outside timing");
        auto workspace=part_c::FlashAttention1Workspace{x.fa1_workspace,x.fa1_workspace+x.fa1_rows};
        result=id=="fa1"?part_c::launch_flash_attention_1(shape,buffers,workspace,true,nullptr,resources)
            :id=="fa1_ptx"?part_c::launch_flash_attention_1_ptx(shape,buffers,workspace,config,true,nullptr,resources)
            :part_c::launch_flash_attention_1_tensor_core(shape,buffers,workspace,true,nullptr,resources);
    }
    else if(id=="scalar") result=part_c::launch_flash_attention_2_cuda_core_configured(C::Scalar,shape,buffers,config,nullptr,resources);
    else if(id=="warp") result=part_c::launch_flash_attention_2_cuda_core_configured(C::WarpPerQuery,shape,buffers,config,nullptr,resources);
    else if(id=="tiled") result=part_c::launch_flash_attention_2_cuda_core_configured(C::Tiled,shape,buffers,config,nullptr,resources);
    else if(id=="skip") result=part_c::launch_flash_attention_2_cuda_core_configured(C::TileSkipping,shape,buffers,config,nullptr,resources);
    else if(id=="vector") result=part_c::launch_flash_attention_2_cuda_core_configured(C::Vectorized,shape,buffers,config,nullptr,resources);
    else if(id=="wmma") result=part_c::launch_flash_attention_2_tensor_core_configured(T::Wmma,shape,buffers,config,nullptr,resources);
    else if(id=="ptx") result=part_c::launch_flash_attention_2_tensor_core_configured(T::PtxMma,shape,buffers,config,nullptr,resources);
    else throw std::invalid_argument("unknown kernel");
    check(cudaGetLastError());
    return result;
}
double quantile(const std::vector<double>& x,double p) { double z=p*(x.size()-1);size_t l=size_t(z),h=std::min(l+1,x.size()-1);return x[l]+(z-l)*(x[h]-x[l]); }
