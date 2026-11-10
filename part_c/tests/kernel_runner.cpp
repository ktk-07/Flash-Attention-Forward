// Standalone Part C validation and measurement. Shared CPU reference; no PyTorch dependency.
#include "flash_attention/flash_attention_1.h"
#include "flash_attention/flash_attention_2_cuda_core.h"
#include "flash_attention/flash_attention_2_tensor_core.h"
#include <cuda_runtime.h>
#include "test_support.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <random>
#include <chrono>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main(int argc,char** argv) {
    std::string suite="correctness",only="all",grid="full";fs::path output="part_c/results/local";
    Shape profile_shape{4,8,256,64};
    bool custom_shape=false;
    try {
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--help") {std::cout<<"--suite correctness|benchmark|smoke|profile --kernel all|fa1|fa1_wmma|scalar|warp|tiled|skip|vector|wmma|ptx --grid full|representative --output DIR [--shape B H N d (profile only)]\n";return 0;}
            if(arg=="--shape") {
                if(i+4>=argc) throw std::invalid_argument("--shape requires B H N d");
                profile_shape={std::stoi(argv[i+1]),std::stoi(argv[i+2]),std::stoi(argv[i+3]),std::stoi(argv[i+4])};
                if(profile_shape.b<=0 || profile_shape.h<=0 || profile_shape.n<=0 || profile_shape.d<=0)
                    throw std::invalid_argument("shape dimensions must be positive");
                i+=4;custom_shape=true;continue;
            }
            if(i+1==argc) throw std::runtime_error("missing option value");std::string val=argv[++i];
            if(arg=="--suite")suite=val;else if(arg=="--kernel")only=val;else if(arg=="--output")output=val;else if(arg=="--grid")grid=val;else throw std::runtime_error("unknown option");
        }
        if(suite!="correctness" && suite!="benchmark" && suite!="smoke" && suite!="profile") throw std::runtime_error("invalid suite");
        if(custom_shape && suite!="profile") throw std::invalid_argument("--shape is supported only for profile");
        if(suite=="profile" && only=="all") throw std::invalid_argument("profile requires one --kernel");
        if(only!="all" && std::find(kernels.begin(),kernels.end(),only)==kernels.end()) throw std::runtime_error("invalid kernel");
        if(grid!="full" && grid!="representative") throw std::runtime_error("invalid grid");
        fs::create_directories(output);
        std::ofstream env(output/"environment.txt");
        int device_count=0;auto device_status=cudaGetDeviceCount(&device_count);
        env<<"cuda_device_status="<<cudaGetErrorString(device_status)<<"\ndevice_count="<<device_count<<'\n';env.flush();
        check(device_status);if(!device_count)throw std::runtime_error("no CUDA device");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));int runtime,driver;check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
        env<<"gpu="<<prop.name<<"\ncompute_capability="<<prop.major<<'.'<<prop.minor<<"\nruntime="<<runtime<<"\ndriver_api="<<driver<<"\ncompiler="<<PART_C_CUDA_COMPILER_VERSION<<"\nseed=5206\ninput_generator=shared splitmix64-fp32-v1\nfa1_implementation=kv_outer_shared_fp32_v2\nfa1_wmma_implementation=kv_outer_wmma_v1\nfa1_grid=one_CTA_per_batch_head\nfa1_workspace=two_FP32_arrays_BHN;reset_in_first_KV_tile\nmode=causal\nwarmups=5\nrepeats="<<(suite=="profile"?1:20)<<"\nsuite="<<suite<<"\ngrid="<<grid<<"\nconfiguration=fixed educational defaults; no tuning/winner selection performed\nbenchmark_reference=queries 0,N/2,N-1 in every batch/head; all outputs checked finite\nrelative_error_denominator=max(abs(reference),1e-12)\n";
        std::ofstream results(output/"correctness.csv"),samples(output/"samples.csv"),summary(output/"summary.csv");
        for(auto* f:{&results,&samples,&summary}) *f<<std::setprecision(17);
        std::string header="kernel,precision,output_precision,accumulator_precision,B,H,N,d,pattern,block_q,block_kv,warps,thread_q,thread_k";
        results<<header<<",reference_scope,atol,rtol,max_absolute,max_relative,checked_elements,nonfinite,failures,status\n";
        samples<<header<<",repeat,forward_ms\n";
        summary<<header<<",warmups,repeats,median_ms,q1_ms,q3_ms,iqr_ms,sequences_per_second,logical_device_bytes,harness_allocated_bytes,reference_scope,status\n";
        std::vector<Shape> shapes;
        if(suite=="profile") shapes.push_back(profile_shape);
        else if(suite=="benchmark") {
            for(int d:{32,64,128}) for(int n:{256,512,1024,2048,4096}) {
                if(grid=="representative" && (d!=64 || (n!=256 && n!=1024 && n!=4096)))continue;
                shapes.push_back({4,8,n,d});
            }
        } else if(suite=="smoke") shapes={{1,1,33,64}};
        else {
            for(int d:{32,64,128}) for(int n:{1,15,16,17,31,32,33,63,64,65,97}) shapes.push_back({1,1,n,d});
            for(int d:{7,16,33,40,65,96}) shapes.push_back({2,3,65,d});
            shapes.push_back({2,3,97,64});
        }
        std::ofstream run_manifest(output/"run_manifest.json");
        run_manifest<<"{\"suite\":\""<<suite<<"\",\"grid\":\""<<grid<<"\",\"kernels\":[";
        bool first_kernel=true;
        for(const auto& id:kernels)if(only=="all" || only==id) {if(!first_kernel)run_manifest<<',';run_manifest<<'"'<<id<<'"';first_kernel=false;}
        run_manifest<<"],\"fa1_baselines\":[\"fa1\",\"fa1_wmma\"]}\n";
        int bad=0,passed=0,skipped=0;
        for(Shape s:shapes) {
            std::vector<std::string> patterns={"random"};
            if(suite=="correctness" && s.d==64 && s.n==97)patterns={"random","uniform","constant_v","rising","stress"};
            for(const auto& pattern:patterns) {
                Input input(s,pattern,true);Buffers buffers(s.size(),false,false,(only=="all" || needs_fa1_workspace(only))?std::size_t(s.b)*s.h*s.n:0);input.upload(buffers);
                const bool sampled=suite=="benchmark" || suite=="profile";
                auto fp=reference(s,input.q,input.k,input.v,sampled),hp=reference(s,input.rq,input.rk,input.rv,sampled);
                for(const auto& id:kernels) {
                    if(only!="all" && only!=id)continue;
                    bool tc=tensor(id);double atol=tc?0.002:0.0001,rtol=tc?0.02:0.0001;
                    if(id=="ptx" && (s.d<16 || s.d%16)) {
                        results<<id<<",FP16,FP32,FP32,"<<s.b<<','<<s.h<<','<<s.n<<','<<s.d<<','<<pattern<<",32,32,2,0,0,"<<(sampled?"sampled_queries":"full")<<','<<atol<<','<<rtol<<",,,0,0,0,UNSUPPORTED\n";++skipped;continue;
                    }
                    check(cudaMemset(buffers.o,0xff,s.size()*4)); // Missing stores must not pass as allocator leftovers.
                    auto cfg=dispatch(id,s,buffers);check(cudaDeviceSynchronize());
                    std::vector<float> actual(s.size());check(cudaMemcpy(actual.data(),buffers.o,s.size()*4,cudaMemcpyDeviceToHost));
                    Errors e=compare(actual,tc?hp:fp,atol,rtol);
                    auto identity=[&](std::ostream& f) {f<<id<<','<<(tc?"FP16":"FP32")<<",FP32,FP32,"<<s.b<<','<<s.h<<','<<s.n<<','<<s.d<<','<<pattern<<','<<cfg.bq<<','<<cfg.bk<<','<<cfg.warps<<','<<cfg.tq<<','<<cfg.tk;};
                    identity(results);results<<','<<(sampled?"sampled_queries":"full")<<','<<atol<<','<<rtol<<','<<e.absolute<<','<<e.relative<<','<<e.checked<<','<<e.nonfinite<<','<<e.failures<<','<<(e.failures?"FAIL":"PASS")<<'\n';results.flush();
                    std::cout<<id<<" B="<<s.b<<" H="<<s.h<<" N="<<s.n<<" d="<<s.d<<" "<<pattern<<" "<<(e.failures?"FAIL":"PASS")<<" max_abs="<<e.absolute<<std::endl;
                    if(e.failures) {++bad;continue;} ++passed;
                    if(suite=="profile") {
                        // One validation launch above, five warmups, then one target.
                        // ncu --launch-skip 6 --launch-count 1 selects only the target.
                        for(int w=0;w<5;++w) dispatch(id,s,buffers);
                        check(cudaDeviceSynchronize());
                        dispatch(id,s,buffers);check(cudaDeviceSynchronize());
                        continue;
                    }
                    if(suite!="benchmark")continue;
                    for(int w=0;w<5;++w)dispatch(id,s,buffers);check(cudaDeviceSynchronize());
                    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));std::vector<double> times;
                    for(int rep=0;rep<20;++rep) {
                        check(cudaEventRecord(begin));dispatch(id,s,buffers);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));
                        times.push_back(ms);identity(samples);samples<<','<<rep<<','<<ms<<'\n';
                    }
                    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));std::sort(times.begin(),times.end());
                    double med=quantile(times,.5),q1=quantile(times,.25),q3=quantile(times,.75);
                    identity(summary);summary<<",5,20,"<<med<<','<<q1<<','<<q3<<','<<q3-q1<<','<<1000*s.b/med<<','<<(s.size()*(tc?10:16)+(needs_fa1_workspace(id)?8*std::size_t(s.b)*s.h*s.n:0))<<','<<buffers.allocated_bytes<<",sampled_queries,PASS\n";summary.flush();samples.flush();
                }
            }
        }
        std::cout<<"PASS="<<passed<<" FAIL="<<bad<<" UNSUPPORTED="<<skipped<<std::endl;
        return bad?2:0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
