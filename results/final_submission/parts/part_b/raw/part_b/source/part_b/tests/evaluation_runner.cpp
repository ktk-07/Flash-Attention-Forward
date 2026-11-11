#include "flash_attention/dense_flash_attention.h"
#include "input_generation.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <map>
using Clock=std::chrono::steady_clock;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct DeviceBuffer{float* ptr=nullptr;explicit DeviceBuffer(size_t n){check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(float)));}~DeviceBuffer(){cudaFree(ptr);}DeviceBuffer(const DeviceBuffer&)=delete;};
struct Event{cudaEvent_t e;Event(){check(cudaEventCreate(&e));}~Event(){cudaEventDestroy(e);}};
int main(int argc,char** argv){try{
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("Each option needs a value");args[argv[i]]=argv[i+1];}
    const std::vector<std::string> allowed={"--B","--H","--N","--d","--variant","--warps","--key-tile","--pattern","--sampled","--timing","--output","--profile"};
    for(const auto& a:args)if(std::find(allowed.begin(),allowed.end(),a.first)==allowed.end())throw std::invalid_argument("Unknown option: "+a.first);
    auto get=[&](const std::string& k,const std::string& fallback){return args.count(k)?args[k]:fallback;};
    auto dim=[&](const std::string& k,const std::string& fallback){auto text=get(k,fallback);if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("Positive integer required: "+k);size_t value=std::stoull(text);if(!value)throw std::invalid_argument("Zero dimension");return value;};
    attention::Shape s{dim("--B","1"),dim("--H","1"),dim("--N","33"),dim("--d","64")};
    dense_attention::Config c{get("--variant","tiled"),int(dim("--warps","4")),int(dim("--key-tile","32"))};
    bool sampled=get("--sampled","0")=="1",timing=get("--timing","1")=="1",profile=get("--profile","0")=="1";
    const auto begin=Clock::now();auto input=attention::synthetic_input(s,5206);auto pattern=get("--pattern","random");attention::apply_pattern(input,pattern);
    size_t count=attention::input_elements(s),free_before,total,free_after;
    check(cudaFree(nullptr));check(cudaMemGetInfo(&free_before,&total));
    DeviceBuffer q(count),k(count),v(count),o(count);
    check(cudaMemGetInfo(&free_after,&total));
    for(auto pair:{std::make_pair(q.ptr,&input.q),std::make_pair(k.ptr,&input.k),std::make_pair(v.ptr,&input.v)})check(cudaMemcpy(pair.first,pair.second->data(),count*sizeof(float),cudaMemcpyHostToDevice));
    // NaN sentinel detects unwritten outputs. It is outside all timed intervals.
    check(cudaMemset(o.ptr,0xff,count*sizeof(float)));
    auto launch=[&](){dense_attention::forward(q.ptr,k.ptr,v.ptr,o.ptr,s,c);};
    if(profile){launch();check(cudaDeviceSynchronize());return 0;}
    auto res=dense_attention::resources(s,c);launch();check(cudaDeviceSynchronize());
    std::vector<float> actual(count);check(cudaMemcpy(actual.data(),o.ptr,count*sizeof(float),cudaMemcpyDeviceToHost));
    auto expected=attention::reference_output(input,false,sampled);
    size_t failures=0,nonfinite=0,checked=0;double max_abs=0,max_rel=0;
    for(size_t i=0;i<count;++i){if(!std::isfinite(actual[i]))++nonfinite;if(!std::isfinite(expected[i]))continue;++checked;
        double error=std::abs(double(actual[i])-expected[i]);max_abs=std::max(max_abs,error);max_rel=std::max(max_rel,error/std::max(std::abs(expected[i]),1e-30));
        if(!std::isfinite(actual[i])||error>1e-4+1e-4*std::abs(expected[i]))++failures;
    }
    double materialised_error=0;size_t materialised_failures=0;
    if(s.N<=65){
        // Supplementary A/B equivalence check. These extra allocations are freed
        // before timing and excluded from B's four-buffer allocation ledger.
        DeviceBuffer scores(s.B*s.H*s.N*s.N),probabilities(s.B*s.H*s.N*s.N),aout(count);
        attention::detail::launch_qk(q.ptr,k.ptr,scores.ptr,s,false);
        attention::detail::launch_softmax(scores.ptr,probabilities.ptr,s,false);
        attention::detail::launch_pv(probabilities.ptr,v.ptr,aout.ptr,s);
        check(cudaDeviceSynchronize());std::vector<float> a(count);check(cudaMemcpy(a.data(),aout.ptr,count*sizeof(float),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<count;++i){double error=std::abs(double(actual[i])-a[i]);materialised_error=std::max(materialised_error,error);
            if(!std::isfinite(a[i])||error>1e-4+1e-4*std::abs(double(a[i])))++materialised_failures;}
        failures+=materialised_failures;
    }
    std::vector<float> samples;double setup=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    if(timing&&!failures&&!nonfinite){for(int i=0;i<5;++i)launch();check(cudaDeviceSynchronize());Event start,stop;
        for(int i=0;i<20;++i){check(cudaEventRecord(start.e));launch();check(cudaEventRecord(stop.e));check(cudaEventSynchronize(stop.e));float ms;check(cudaEventElapsedTime(&ms,start.e,stop.e));samples.push_back(ms);}
    }
    std::ofstream file;std::ostream* stream=&std::cout;auto path=get("--output","");if(!path.empty()){file.open(path);if(!file)throw std::runtime_error("Cannot write output");stream=&file;}
    auto& out=*stream;out<<std::setprecision(17);
    out<<"{\"B\":"<<s.B<<",\"H\":"<<s.H<<",\"N\":"<<s.N<<",\"d\":"<<s.d
       <<",\"variant\":\""<<c.variant<<"\",\"warps\":"<<c.warps<<",\"key_tile\":"<<c.key_tile<<",\"pattern\":\""<<pattern
       <<"\",\"input_digest\":\""<<attention::input_fingerprint(input)<<"\",\"status\":\""<<((failures||nonfinite)?"FAIL":"PASS")
       <<"\",\"failures\":"<<failures<<",\"nonfinite\":"<<nonfinite<<",\"max_absolute\":"<<max_abs<<",\"max_relative\":"<<max_rel
       <<",\"reference_scope\":\""<<(sampled?"sampled":"full")<<"\",\"checked_elements\":"<<checked
       <<",\"checked_query_rows\":"<<(sampled?attention::reference_query_rows(s.N).size():s.N)*s.B*s.H
       <<",\"explicit_live_device_bytes\":"<<4*count*sizeof(float)<<",\"auxiliary_global_bytes\":0,\"observed_device_delta_bytes\":"<<(free_before>=free_after?free_before-free_after:0)
       <<",\"setup_host_ms\":"<<setup<<",\"kernel\":\""<<dense_attention::symbol(c)<<"\",\"blocks\":"<<res.blocks<<",\"threads\":"<<res.threads
       <<",\"registers_per_thread\":"<<res.registers_per_thread<<",\"shared_bytes\":"<<res.shared_bytes<<",\"local_bytes_per_thread\":"<<res.local_bytes_per_thread
       <<",\"active_blocks_per_sm\":"<<res.active_blocks_per_sm<<",\"theoretical_occupancy_percent\":"<<res.theoretical_occupancy_percent<<",\"samples_ms\":[";
    for(size_t i=0;i<samples.size();++i){if(i)out<<',';out<<samples[i];}out<<"],\"materialised_compared\":"<<(s.N<=65?"true":"false")<<",\"materialised_max_absolute\":"<<materialised_error<<",\"materialised_failures\":"<<materialised_failures<<"}\n";
    return failures||nonfinite?2:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
