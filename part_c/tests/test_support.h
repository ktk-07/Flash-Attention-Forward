#pragma once
#include "flash_attention/types.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>
void check(cudaError_t error);
struct Shape { int b,h,n,d; std::size_t size() const; };
extern const std::vector<std::string> kernels;
bool tensor(const std::string& id);
struct Buffers {
    float *q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr;
    half *hq=nullptr,*hk=nullptr,*hv=nullptr;
    std::size_t allocated_bytes=0;
    Buffers(std::size_t count,bool tensor_inputs=false,bool minimal=false);
    ~Buffers();
    Buffers(const Buffers&)=delete;
    void release() noexcept;
};
struct Input {
    std::vector<float> q,k,v,rq,rk,rv;
    std::vector<half> hq,hk,hv;
    Input(Shape shape,const std::string& pattern,bool canonical=false);
    void upload(Buffers& buffers) const;
};
std::string input_digest(const Input& input,bool rounded);
std::vector<double> reference(Shape shape,const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,bool sampled);
struct Errors { double absolute=0,relative=0;std::size_t failures=0,checked=0,nonfinite=0; };
Errors compare(const std::vector<float>& actual,const std::vector<double>& expected,double atol,double rtol);
using Launch=part_c::LaunchConfig;
Launch dispatch(const std::string& id,Shape shape,Buffers& buffers,int config=0,part_c::KernelResources* resources=nullptr);
double quantile(const std::vector<double>& values,double probability);
