#include "test_support.h"
#include "flash_attention/flash_attention_1.h"
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
const std::vector<std::string> kernels={"fa1","scalar","warp","tiled","skip","vector","wmma","ptx"};
bool tensor(const std::string& id) { return id=="wmma" || id=="ptx"; }
Buffers::Buffers(std::size_t n,bool tc,bool minimal) {
    try {
        if(!minimal || !tc) { check(cudaMalloc(&q,n*4));allocated_bytes+=n*4;check(cudaMalloc(&k,n*4));allocated_bytes+=n*4;check(cudaMalloc(&v,n*4));allocated_bytes+=n*4; }
        check(cudaMalloc(&o,n*4));allocated_bytes+=n*4;
        if(!minimal || tc) { check(cudaMalloc(&hq,n*2));allocated_bytes+=n*2;check(cudaMalloc(&hk,n*2));allocated_bytes+=n*2;check(cudaMalloc(&hv,n*2));allocated_bytes+=n*2; }
    } catch(...) { release();throw; }
}
void Buffers::release() noexcept { cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(hq);cudaFree(hk);cudaFree(hv);q=k=v=o=nullptr;hq=hk=hv=nullptr; }
Buffers::~Buffers() { release(); }
Input::Input(Shape s,const std::string& pattern,bool canonical):q(s.size()),k(s.size()),v(s.size()),rq(s.size()),rk(s.size()),rv(s.size()),hq(s.size()),hk(s.size()),hv(s.size()) {
    if(canonical) {
        // Bit-for-bit equivalent to Part A synthetic_input, including Q/K/V order.
        std::uint64_t state=5206;
        for(auto* data:{&q,&k,&v}) for(float& value:*data) {
            std::uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
            z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
            z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);z^=z>>31;
            const auto integer=static_cast<std::int32_t>(z>>40)-8388608;
            value=static_cast<float>(integer)/8388608.0f;
        }
    } else {
        // Preserve the historical standalone input stream.
        std::mt19937 gen(5206);std::uniform_real_distribution<float> dist(-1.f,1.f);
        for(std::size_t i=0;i<s.size();++i) { q[i]=dist(gen);k[i]=dist(gen);v[i]=dist(gen); }
    }
    for(std::size_t i=0;i<s.size();++i) {
        if(pattern=="uniform") q[i]=k[i]=0;
        if(pattern=="constant_v") v[i]=0.375f;
        if(pattern=="rising") { q[i]=1.f;k[i]=float((i/s.d)%s.n)*0.125f; }
        if(pattern=="stress") { q[i]*=16;k[i]*=16; }
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
    std::uint64_t hash=UINT64_C(14695981039346656037);
    for(const auto* data:(rounded?std::vector<const std::vector<float>*>{&in.rq,&in.rk,&in.rv}:std::vector<const std::vector<float>*>{&in.q,&in.k,&in.v}))
        for(float value:*data) {std::uint32_t bits;std::memcpy(&bits,&value,4);for(int i=0;i<4;++i){hash^=(bits>>(8*i))&255u;hash*=UINT64_C(1099511628211);}}
    std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
// FP64 dot products, stable softmax and output accumulation; no GPU reference.
// Full output for correctness; three query rows per head for benchmark checks.
std::vector<double> reference(Shape s,const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,bool sampled) {
    std::vector<double> out(s.size(),std::numeric_limits<double>::quiet_NaN());
    for(int bh=0;bh<s.b*s.h;++bh) for(int i=0;i<s.n;++i) {
        if(sampled && i!=0 && i!=s.n/2 && i!=s.n-1) continue;
        size_t base=size_t(bh)*s.n*s.d; std::vector<double> scores(i+1); double m=-INFINITY;
        for(int j=0;j<=i;++j) {
            double dot=0;for(int d=0;d<s.d;++d) dot+=double(q[base+size_t(i)*s.d+d])*k[base+size_t(j)*s.d+d];
            scores[j]=dot/std::sqrt(double(s.d));m=std::max(m,scores[j]);
        }
        double sum=0; for(auto& x:scores) {x=std::exp(x-m);sum+=x;}
        for(int d=0;d<s.d;++d) { double acc=0;for(int j=0;j<=i;++j) acc+=scores[j]*double(v[base+size_t(j)*s.d+d]);out[base+size_t(i)*s.d+d]=acc/sum; }
    }
    return out;
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
    if(id=="fa1" && config!=0) throw std::invalid_argument("FA1 supports config 0 only");
    if(id=="fa1") result=part_c::launch_flash_attention_1(shape,buffers);
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
