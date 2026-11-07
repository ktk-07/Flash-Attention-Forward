// One isolated study case. The Python driver freezes tuning before final testing.
#include "test_support.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=10) throw std::invalid_argument("usage: evaluation_runner KERNEL CONFIG B H N D PATTERN validate|benchmark|memory OUTPUT.csv");
        std::string id=argv[1],pattern=argv[7],mode=argv[8];int config=std::stoi(argv[2]);
        Shape s{std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6])};
        if(s.b<1||s.h<1||s.n<1||s.d<1) throw std::invalid_argument("positive shape required");
        if(mode!="validate"&&mode!="benchmark"&&mode!="memory") throw std::invalid_argument("unknown mode");
        const bool tc=tensor(id),sampled=mode!="validate";
        check(cudaFree(nullptr)); // Context setup precedes memory baseline.
        // Eagerly load/query this instantiation before measuring explicit allocations.
        { Buffers tiny(s.d,tc,true);Input warm({1,1,1,s.d},"random",true);warm.upload(tiny);dispatch(id,{1,1,1,s.d},tiny,config);check(cudaDeviceSynchronize()); }
        Input input(s,pattern,true);
        auto expected=reference(s,tc?input.rq:input.q,tc?input.rk:input.k,tc?input.rv:input.v,sampled);
        std::size_t free0,total,free1,free2;check(cudaMemGetInfo(&free0,&total));
        Buffers buffers(s.size(),tc,true);check(cudaMemGetInfo(&free1,&total));input.upload(buffers);
        check(cudaMemset(buffers.o,0xff,s.size()*4));part_c::KernelResources resource;
        auto cfg=dispatch(id,s,buffers,config,id=="fa1"?nullptr:&resource);check(cudaDeviceSynchronize());
        check(cudaMemGetInfo(&free2,&total));std::vector<float> actual(s.size());
        check(cudaMemcpy(actual.data(),buffers.o,s.size()*4,cudaMemcpyDeviceToHost));
        double atol=tc?.002:.0001,rtol=tc?.02:.0001;
        auto error=compare(actual,expected,atol,rtol);
        std::vector<double> times;
        if(!error.failures && mode=="benchmark") {
            for(int i=0;i<5;++i)dispatch(id,s,buffers,config);check(cudaDeviceSynchronize());
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            for(int i=0;i<20;++i){check(cudaEventRecord(begin));dispatch(id,s,buffers,config);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms);}
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
        }
        cudaDeviceProp gpu{};check(cudaGetDeviceProperties(&gpu,0));int driver,runtime;check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
        std::filesystem::path path=argv[9];std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);out<<std::setprecision(17);
        out<<"kernel,config,B,H,N,d,pattern,mode,input_precision,output_precision,accumulator_precision,input_digest,rounded_input_digest,reference_scope,atol,rtol,max_absolute,max_relative,checked_elements,nonfinite,failures,status,block_q,block_kv,warps,thread_q,thread_k,registers_per_thread,shared_bytes,local_bytes_per_thread,active_blocks_per_sm,occupancy_estimate_percent,explicit_live_device_bytes,observed_device_delta_bytes,driver,runtime,gpu,repeat,forward_ms\n";
        auto row=[&](int rep,double ms){out<<id<<','<<config<<','<<s.b<<','<<s.h<<','<<s.n<<','<<s.d<<','<<pattern<<','<<mode<<','<<(tc?"FP16":"FP32")<<",FP32,FP32,"<<input_digest(input,false)<<','<<input_digest(input,true)<<','<<(sampled?"sampled_queries":"full")<<','<<atol<<','<<rtol<<','<<error.absolute<<','<<error.relative<<','<<error.checked<<','<<error.nonfinite<<','<<error.failures<<','<<(error.failures?"FAIL":"PASS")<<','<<cfg.bq<<','<<cfg.bk<<','<<cfg.warps<<','<<cfg.tq<<','<<cfg.tk<<',';
            if(id!="fa1")out<<resource.registers_per_thread<<','<<resource.shared_bytes<<','<<resource.local_bytes_per_thread<<','<<resource.active_blocks_per_sm<<','<<resource.occupancy_estimate_percent;
            else out<<",,,,";
            auto observed=free0>std::min(free1,free2)?free0-std::min(free1,free2):0;
            out<<','<<buffers.allocated_bytes<<','<<observed<<','<<driver<<','<<runtime<<','<<gpu.name<<','<<rep<<','<<ms<<'\n';};
        if(times.empty()) row(-1,0);else for(std::size_t i=0;i<times.size();++i)row(int(i),times[i]);
        std::cout<<id<<" cfg="<<config<<" "<<s.n<<"x"<<s.d<<" "<<mode<<" "<<(error.failures?"FAIL":"PASS")<<std::endl;
        return error.failures?2:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
