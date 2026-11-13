#include "attention_types.hpp"
#include "input_generation.hpp"
#include "flash_attention/dense_flash_attention.h"
#include "flash_attention/flash_attention_2_tensor_core.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocations {
    std::vector<void*> pointers;size_t bytes=0;
    template<class T>T* get(size_t n){T* p=nullptr;check(cudaMalloc((void**)&p,n*sizeof(T)));pointers.push_back(p);bytes+=n*sizeof(T);return p;}
    ~Allocations(){for(void* p:pointers)cudaFree(p);}
};
struct Record {
    std::string name;size_t allocated=0,checked=0,failures=0,nonfinite=0;
    double maximum=0;std::vector<float> times,output;
};
template<class T>void upload(T* p,const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class F>void execute(Record& r,F forward,float* output,size_t count,int repeats,const std::vector<double>& expected,double atol,double rtol){
    check(cudaMemset(output,0xff,count*sizeof(float)));
    forward();check(cudaGetLastError());check(cudaDeviceSynchronize());
    r.output.resize(count);check(cudaMemcpy(r.output.data(),output,count*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<count;++i){
        if(!std::isfinite(r.output[i])){++r.nonfinite;continue;}
        if(std::isnan(expected[i]))continue;
        ++r.checked;double error=std::abs(double(r.output[i])-expected[i]);
        r.maximum=std::max(r.maximum,error);
        if(error>atol+rtol*std::abs(expected[i]))++r.failures;
    }
    if(r.nonfinite||r.failures||!r.checked)return;
    for(int i=0;i<5;++i)forward();check(cudaDeviceSynchronize());
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    for(int i=0;i<repeats;++i){check(cudaEventRecord(begin));forward();check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));r.times.push_back(ms);}
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
}
void array(std::ostream& o,const std::vector<float>& v,size_t begin,size_t count){
    o<<'[';for(size_t i=0;i<count;++i){if(i)o<<',';float x=v.at(begin+i);if(std::isfinite(x))o<<x;else o<<"null";}o<<']';
}
int main(int argc,char** argv){try{
    if((argc==2&&std::string(argv[1])=="--device")||(argc==3&&std::string(argv[1])=="--device-json")){
        cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));int driver=0,runtime=0;check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
        std::ofstream file;
        if(argc==3){file.open(argv[2]);if(!file)throw std::runtime_error("cannot write device JSON file");}
        std::ostream& out=argc==3?static_cast<std::ostream&>(file):std::cout;
        out<<"{\"name\":\""<<p.name<<"\",\"sm\":"<<p.major*10+p.minor<<",\"total_bytes\":"<<p.totalGlobalMem<<",\"driver\":"<<driver<<",\"runtime\":"<<runtime<<"}\n";out.flush();if(!out)throw std::runtime_error("device JSON write failed");return 0;
    }
    if(argc!=9)throw std::invalid_argument("usage: demo_runner ptx|dense B H N d SEED REPEATS OUTPUT.json");
    std::string mode=argv[1];if(mode!="ptx"&&mode!="dense")throw std::invalid_argument("unknown mode");
    attention::Shape s{size_t(std::stoi(argv[2])),size_t(std::stoi(argv[3])),size_t(std::stoi(argv[4])),size_t(std::stoi(argv[5]))};
    if(s.B<1||s.B>4||s.H<1||s.H>8||s.N<16||s.N>2048||s.d!=64)throw std::invalid_argument("demo limits: B 1..4, H 1..8, N 16..2048, d=64");
    auto seed=std::stoull(argv[6]);int repeats=std::stoi(argv[7]);if(repeats<5||repeats>100)throw std::invalid_argument("repeats 5..100");
    const bool tc=mode=="ptx",causal=tc,sampled=s.N>256;const size_t count=attention::input_elements(s),scores=s.B*s.H*s.N*s.N;
    auto input=attention::synthetic_input(s,seed);auto q=input.q,k=input.k,v=input.v;
    std::vector<half> hq,hk,hv;
    if(tc){hq.resize(count);hk.resize(count);hv.resize(count);for(size_t i=0;i<count;++i){hq[i]=__float2half_rn(q[i]);hk[i]=__float2half_rn(k[i]);hv[i]=__float2half_rn(v[i]);q[i]=__half2float(hq[i]);k[i]=__half2float(hk[i]);v[i]=__half2float(hv[i]);}}
    auto expected=attention::reference_output(s,q,k,v,causal,sampled);
    const double atol=tc?0.002:0.0001,rtol=tc?0.02:0.0001;
    std::vector<Record> records;
    for(int implementation=0;implementation<2;++implementation){
        Allocations mem;Record r;r.name=implementation==0?(tc?"A materialised PTX":"A optimized FP32"):(tc?"C fused PTX FA2":"B fused FP32");
        float* out=mem.get<float>(count);
        if(tc){
            half* dq=mem.get<half>(count),*dk=mem.get<half>(count),*dv=mem.get<half>(count);upload(dq,hq);upload(dk,hk);upload(dv,hv);
            if(implementation==0){
                float* ds=mem.get<float>(scores);half* dp=mem.get<half>(scores);
                auto f=[&](){attention::detail::launch_qk_tensorcore(dq,dk,ds,s,true,true);attention::detail::launch_softmax_mixed(ds,dp,s,true,256);attention::detail::launch_pv_tensorcore(dp,dv,out,s,true);};
                execute(r,f,out,count,repeats,expected,atol,rtol);
            }else{
                part_c::Shape cs{int(s.B),int(s.H),int(s.N),int(s.d)};part_c::DevicePointers buffers{nullptr,nullptr,nullptr,out,dq,dk,dv};
                auto f=[&](){part_c::launch_flash_attention_2_tensor_core_configured(part_c::TensorCoreVariant::PtxMma,cs,buffers,0);};
                execute(r,f,out,count,repeats,expected,atol,rtol);
            }
        }else{
            float* dq=mem.get<float>(count),*dk=mem.get<float>(count),*dv=mem.get<float>(count);upload(dq,q);upload(dk,k);upload(dv,v);
            if(implementation==0){
                float* ds=mem.get<float>(scores),*dp=mem.get<float>(scores);
                auto f=[&](){attention::detail::launch_qk_tiled(dq,dk,ds,s,false,16);attention::detail::launch_softmax_parallel(ds,dp,s,false,256);attention::detail::launch_pv_tiled(dp,dv,out,s,16);};
                execute(r,f,out,count,repeats,expected,atol,rtol);
            }else{
                dense_attention::Config cfg;cfg.variant="vector";cfg.warps=4;cfg.key_tile=32;
                auto f=[&](){dense_attention::forward(dq,dk,dv,out,s,cfg);};
                execute(r,f,out,count,repeats,expected,atol,rtol);
            }
        }
        r.allocated=mem.bytes;records.push_back(std::move(r));
    }
    cudaDeviceProp gpu{};check(cudaGetDeviceProperties(&gpu,0));
    size_t preview=(s.N/2)*s.d;
    std::ofstream o(argv[8]);if(!o)throw std::runtime_error("cannot create output JSON");o<<std::setprecision(10);
    o<<"{\"gpu\":\""<<gpu.name<<"\",\"sm\":"<<gpu.major*10+gpu.minor<<",\"mode\":\""<<(causal?"causal":"dense")<<"\",\"precision\":\""<<(tc?"FP16 inputs / FP32 accumulation and output":"FP32")<<"\",\"B\":"<<s.B<<",\"H\":"<<s.H<<",\"N\":"<<s.N<<",\"d\":"<<s.d<<",\"seed\":"<<seed<<",\"input_digest\":\""<<attention::input_fingerprint(q,k,v)<<"\",\"reference_scope\":\""<<(sampled?"sampled query rows 0,N/2,N-1 in every head":"full output")<<"\",\"atol\":"<<atol<<",\"rtol\":"<<rtol<<",\"warmup\":5,\"repeats\":"<<repeats<<",\"preview_query\":"<<s.N/2<<",\"q_preview\":";array(o,q,preview,8);
    o<<",\"reference_preview\":[";for(size_t i=0;i<8;++i){if(i)o<<',';o<<expected[preview+i];}o<<"],\"configuration\":\"A tile16 softmax256 / A PTX softmax256; B vector warps4 key32; C PTX config0; fixed demo choices, no tuning\",\"records\":[";
    bool passed=true;
    for(size_t i=0;i<records.size();++i){if(i)o<<',';const auto& r=records[i];bool ok=!r.failures&&!r.nonfinite&&r.checked;passed&=ok;
        o<<"{\"implementation\":\""<<r.name<<"\",\"passed\":"<<(ok?"true":"false")<<",\"checked_elements\":"<<r.checked<<",\"failures\":"<<r.failures<<",\"nonfinite\":"<<r.nonfinite<<",\"max_absolute_error\":"<<r.maximum<<",\"allocated_bytes\":"<<r.allocated<<",\"output_preview\":";array(o,r.output,preview,8);o<<",\"samples_ms\":";array(o,r.times,0,r.times.size());o<<'}';}
    o<<"],\"passed\":"<<(passed?"true":"false")<<"}\n";o.close();
    std::cout<<(passed?"PASS":"FAIL")<<" "<<mode<<" N="<<s.N<<" seed="<<seed<<'\n';return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
