// Numerical/state regression for the FA1 Algorithm-1 implementation.
// Dense checks use the shared dense oracle; the Part C runner remains causal.
#include "test_support.h"
#include "input_generation.hpp"
#include "flash_attention/flash_attention_1.h"
#include "flash_attention/flash_attention_1_tensor_core.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
void poison(Buffers& x,cudaStream_t stream) {
    check(cudaMemsetAsync(x.fa1_workspace,0xff,2*x.fa1_rows*sizeof(float),stream));
}
}
int main(int argc,char** argv) {
    try {
        bool smoke=false,tc=false,ptx=false;int config=0;std::filesystem::path output="fa1-validation";
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--smoke")smoke=true;
            else if(arg=="--tensorcore")tc=true;
            else if(arg=="--ptx") {tc=true;ptx=true;}
            else if(arg=="--config" && i+1<argc)config=std::stoi(argv[++i]);
            else if(arg=="--output" && i+1<argc)output=argv[++i];
            else throw std::invalid_argument("--smoke --tensorcore|--ptx [--config 0..3] --output NEW_DIR");
        }
        if(config<0 || config>3 || (!ptx && config))throw std::invalid_argument("Only PTX accepts configs 0..3");
        if(std::filesystem::exists(output))throw std::invalid_argument("Choose a fresh validation directory");
        std::filesystem::create_directories(output);
        std::ofstream csv(output/"correctness.csv");csv<<std::setprecision(17);
        csv<<"B,H,N,d,mode,pattern,checked_elements,max_absolute,max_relative,state_max_absolute,registers_per_thread,shared_bytes,active_blocks_per_sm,repeat_identical,causal_isolation,status\n";
        std::vector<Shape> cases;
        if(smoke)cases={{1,1,1,7},{1,1,33,64},{2,3,65,33},{1,1,65,128}};
        else {
            for(int d:{7,16,32,33,40,64,65,96,128})
                for(int n:{1,15,16,17,31,32,33,63,64,65,97,129})cases.push_back({1,1,n,d});
            for(int d:{7,32,33,64,65,128})cases.push_back({2,3,65,d});
        }
        if(ptx) {
            cases.erase(std::remove_if(cases.begin(),cases.end(),[](Shape s){return s.d<16 || s.d%16;}),cases.end());
        }
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::size_t tested=0;
        auto launch=[&](part_c::Shape s,const part_c::DevicePointers& x,part_c::FlashAttention1Workspace w,bool causal,cudaStream_t stream,part_c::KernelResources* r) {
            return ptx?part_c::launch_flash_attention_1_ptx(s,x,w,config,causal,stream,r)
                :(tc?part_c::launch_flash_attention_1_tensor_core:part_c::launch_flash_attention_1)(s,x,w,causal,stream,r);
        };
        for(auto shape:cases) {
            std::vector<std::string> patterns={"random"};
            if(shape.n==65 && shape.d==64 && shape.b==2)patterns={"random","uniform","constant_v","rising","stress"};
            if(smoke && shape.d==128)patterns={"random","rising","stress"};
            for(const auto& pattern:patterns)for(bool causal:{false,true}) {
                Input input(shape,pattern,true);
                const auto rows=std::size_t(shape.b)*shape.h*shape.n;
                Buffers x(shape.size(),tc,true,rows);input.upload(x);
                const part_c::Shape s{shape.b,shape.h,shape.n,shape.d};
                const part_c::DevicePointers pointers{x.q,x.k,x.v,x.o,x.hq,x.hk,x.hv};
                const part_c::FlashAttention1Workspace w{x.fa1_workspace,x.fa1_workspace+rows};
                const auto expected=attention::reference({{std::size_t(shape.b),std::size_t(shape.h),std::size_t(shape.n),std::size_t(shape.d)},tc?input.rq:input.q,tc?input.rk:input.k,tc?input.rv:input.v},causal);
                check(cudaMemsetAsync(x.o,0xff,shape.size()*sizeof(float),stream));poison(x,stream);
                part_c::KernelResources resources;
                launch(s,pointers,w,causal,stream,&resources);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
                std::vector<float> actual(shape.size()),maximum(rows),normalizer(rows);
                check(cudaMemcpy(actual.data(),x.o,actual.size()*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(maximum.data(),w.maxima,rows*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(normalizer.data(),w.normalizers,rows*4,cudaMemcpyDeviceToHost));
                auto error=compare(actual,expected.output,tc?.002:1e-4,tc?.02:1e-4);require(error.failures==0,"FA1 output differs from FP64 oracle");
                double state_error=0;
                for(std::size_t row=0;row<rows;++row) {
                    const auto start=row*shape.n;
                    double m=-std::numeric_limits<double>::infinity();
                    for(int j=0;j<shape.n;++j)m=std::max(m,expected.scores[start+j]);
                    double l=0;for(int j=0;j<shape.n;++j)l+=std::exp(expected.scores[start+j]-m);
                    require(std::isfinite(maximum[row]) && std::isfinite(normalizer[row]),"FA1 state is not finite");
                    require(std::abs(maximum[row]-m)<=1e-4+1e-4*std::abs(m),"FA1 running maximum invariant failed");
                    require(std::abs(normalizer[row]-l)<=1e-4+1e-4*l,"FA1 normalizer invariant failed");
                    state_error=std::max({state_error,std::abs(maximum[row]-m),std::abs(normalizer[row]-l)});
                }
                // A new forward must start fresh despite previous O/m/l contents.
                launch(s,pointers,w,causal,stream,nullptr);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
                std::vector<float> repeat(actual.size());check(cudaMemcpy(repeat.data(),x.o,repeat.size()*4,cudaMemcpyDeviceToHost));
                require(actual==repeat,"FA1 carries stale state across forward calls");
                std::string isolation="not_applicable";
                if(causal && shape.n>1) {
                    const int cutoff=shape.n/2;
                    for(int bh=0;bh<shape.b*shape.h;++bh)for(int row=cutoff;row<shape.n;++row)for(int d=0;d<shape.d;++d) {
                        const auto index=(std::size_t(bh)*shape.n+row)*shape.d+d;
                        input.k[index]+=11;input.v[index]-=7;
                        if(tc) {input.hk[index]=__float2half_rn(input.k[index]);input.hv[index]=__float2half_rn(input.v[index]);}
                    }
                    input.upload(x);
                    launch(s,pointers,w,true,stream,nullptr);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
                    check(cudaMemcpy(repeat.data(),x.o,repeat.size()*4,cudaMemcpyDeviceToHost));
                    for(int bh=0;bh<shape.b*shape.h;++bh)for(int row=0;row<cutoff;++row)for(int d=0;d<shape.d;++d) {
                        const auto index=(std::size_t(bh)*shape.n+row)*shape.d+d;
                        require(actual[index]==repeat[index],"FA1 leaks future K/V into earlier queries");
                    }
                    isolation="PASS";
                }
                const int pad=(shape.d+15)/16*16;
                const int bq=(config==1 || config==3)?16:32,bk=config>=2?16:32,nw=bq/16;
                const std::size_t shared=ptx?2*(bq+2*bk)*shape.d+2*nw*16*16:tc?2*(32+2*32)*pad+4*32*pad+2*16*16*(4+4+2)+2*32*4:4*(2*32*shape.d+2*32*shape.d+32*32);
                require(resources.shared_bytes==shared,"FA1 shared-memory ledger mismatch");
                require(x.allocated_bytes==(tc?10:16)*shape.size()+8*rows,"FA1 workspace ledger mismatch");
                require(part_c::flash_attention_1_workspace_elements(s)==2*rows,"FA1 workspace API extent mismatch");
                csv<<shape.b<<','<<shape.h<<','<<shape.n<<','<<shape.d<<','<<(causal?"causal":"dense")<<','<<pattern<<','<<error.checked<<','<<error.absolute<<','<<error.relative<<','<<state_error<<','<<resources.registers_per_thread<<','<<resources.shared_bytes<<','<<resources.active_blocks_per_sm<<",true,"<<isolation<<",PASS\n";
                ++tested;
            }
        }
        check(cudaStreamDestroy(stream));
        std::cout<<"PASS: "<<tested<<(ptx?" PTX Tensor Core":tc?" WMMA Tensor Core":" CUDA-core")<<" dense/causal FA1 cases; full output, m/l invariants, poisoned and repeat state, tails, nondefault stream and causal isolation\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
