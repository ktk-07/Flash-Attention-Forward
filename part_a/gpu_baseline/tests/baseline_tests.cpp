#include "attention_baseline.hpp"
#include "benchmark_baseline.hpp"
#include "attention_reference.hpp"
#include "compare.hpp"
#include "input_generation.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <utility>
using attention::Input;
attention::BaselineVariant selected_variant = attention::BaselineVariant::Naive;
attention::GpuResult verify(const Input& input,bool causal) {
    const auto cpu=attention::reference(input,causal);
    const auto gpu=attention::baseline(input,causal,selected_variant);
    const auto s=attention::compare(gpu.scores,cpu.scores,1e-4,1e-4,true);
    const auto p=attention::compare(gpu.probabilities,cpu.probabilities);
    const auto o=attention::compare(gpu.output,cpu.output);
    std::cout<<"Shape "<<input.shape.B<<','<<input.shape.H<<','<<input.shape.N<<','<<input.shape.d
             <<(causal?" causal\n":" dense\n")<<std::scientific;
    for(const auto& pair : {std::make_pair("S",s),std::make_pair("P",p),std::make_pair("O",o)})
        std::cout<<pair.first<<": max_abs="<<pair.second.max_absolute
                 <<" max_rel="<<pair.second.max_relative<<" failures="<<pair.second.failures<<'\n';
    if(!s.passed() || !p.passed() || !o.passed()) throw std::runtime_error("CPU/GPU mismatch");
    if(input.shape.N==2 && input.shape.d==2 && input.shape.B==1 && input.shape.H==1)
        for(std::size_t i=0;i<2;++i)
            std::cout<<"O["<<i<<"] = "<<gpu.output[i*2]<<", "<<gpu.output[i*2+1]<<'\n';
    return gpu;
}
Input generated(attention::Shape shape) {
    return attention::synthetic_input(shape);
}
void check_workspace_state(const Input& input) {
    attention::BaselineWorkspace workspace(input);
    auto require_invalid = [&] {
        try { (void)workspace.download(); }
        catch(const std::logic_error&) { return; }
        throw std::runtime_error("Download accepted an invalid result");
    };
    require_invalid();
    workspace.run(false,selected_variant);
    const auto expected=attention::reference(input,false);
    if(!attention::compare(workspace.download().output,expected.output).passed())
        throw std::runtime_error("Workspace result mismatch");
    try { workspace.run(false,selected_variant,{0,0}); }
    catch(const std::invalid_argument&) {}
    require_invalid();
    workspace.run(true,selected_variant);
    if(!attention::compare(workspace.download(false).output,attention::reference(input,true).output).passed())
        throw std::runtime_error("Workspace recovery mismatch");
    (void)workspace.measure(false,selected_variant,{});
    if(!attention::compare(workspace.download(false).output,expected.output).passed())
        throw std::runtime_error("Measured workspace result mismatch");
}
void run_case(const std::string& name) {
        std::cout<<"Case: "<<name<<'\n';
        Input x;
        if(name=="tiny") x={{1,1,2,2},{1,0,0,1},{1,0,0,1},{10,0,0,20}};
        else if(name=="singleton") x=generated({2,3,1,7});
        else if(name=="uniform") {x=generated({2,2,17,7}); for(auto& q:x.q) q=0;}
        else if(name=="stability") x={{1,1,2,1},{1000,-1000},{1000,-1000},{2,-3}};
        else if(name=="isolation") x=generated({2,3,17,7});
        else if(name=="row_lengths") {
            for(auto shape:{attention::Shape{1,1,255,15},attention::Shape{1,1,256,16},attention::Shape{1,1,257,17}})
                for(bool causal:{false,true}) verify(generated(shape),causal);
            std::cout<<"PASS row_lengths\n"; return;
        } else if(name=="causal_prefix") {
            x=generated({2,2,33,17});
            const auto before=verify(x,true);
            // Causal prefix isolation
            for(std::size_t bh=0;bh<4;++bh)
                for(std::size_t r=0;r<17;++r) {
                    const auto t=(bh*33+32)*17+r;
                    x.k[t]+=0.5f; x.v[t]-=0.75f;
                }
            const auto after=verify(x,true);
            for(std::size_t bh=0;bh<4;++bh)
                for(std::size_t i=0;i<32;++i)
                    for(std::size_t r=0;r<17;++r) {
                        const auto t=(bh*33+i)*17+r;
                        if(!std::isfinite(before.output[t]) || !std::isfinite(after.output[t]) ||
                           std::abs(before.output[t]-after.output[t])>1e-6f)
                            throw std::runtime_error("Causal prefix changed");
                    }
            std::cout<<"PASS causal_prefix\n"; return;
        }
        else if(name=="irregular") {
            for(auto shape:{attention::Shape{1,1,33,7},attention::Shape{2,2,65,33},attention::Shape{1,2,129,65}})
                for(bool causal:{false,true}) verify(generated(shape),causal);
            std::cout<<"PASS irregular\n"; return;
        } else throw std::invalid_argument("Unknown selector");
        if(name=="singleton") check_workspace_state(x);
        for(bool causal:{false,true}) verify(x,causal);
        std::cout<<"PASS "<<name<<'\n';
}
int main(int argc,char** argv) {
    try {
        if(argc!=2 && argc!=3) throw std::invalid_argument("Usage: gpu_baseline_tests <selector> [naive|tiled|parallel|optimized]");
        const std::string variant = argc==3 ? argv[2] : "naive";
        if(variant=="naive") selected_variant=attention::BaselineVariant::Naive;
        else if(variant=="tiled") selected_variant=attention::BaselineVariant::TiledMatmul;
        else if(variant=="parallel") selected_variant=attention::BaselineVariant::ParallelSoftmax;
        else if(variant=="optimized") selected_variant=attention::BaselineVariant::Optimized;
        else throw std::invalid_argument("Unknown variant");
        std::cout<<"Variant: "<<variant<<'\n';
        std::cout<<attention::gpu_description()<<'\n';
        const std::string name=argv[1];
        if(name=="all") {
            for(const char* selector:{"tiny","singleton","uniform","stability","isolation","irregular","row_lengths","causal_prefix"})
                run_case(selector);
        } else run_case(name);
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;}
}
