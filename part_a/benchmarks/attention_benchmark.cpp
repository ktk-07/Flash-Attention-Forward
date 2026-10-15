#include "benchmark_baseline.hpp"
#include "attention_reference.hpp"
#include "compare.hpp"
#include "input_generation.hpp"
#include "statistics.hpp"
#include "experiment_plan.hpp"
#include "benchmark_build_info.hpp"
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace {
using attention::BaselineVariant;
using attention::KernelConfig;
using attention::experiment::Options;
using attention::experiment::Variant;
using attention::experiment::variants;
using attention::experiment::configs;
using attention::experiment::shapes;
using attention::experiment::modes;
using attention::experiment::validation_shapes;
std::uint64_t integer(const std::string& text) {
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("Expected unsigned decimal integer");
    std::size_t used;
    const auto value=std::stoull(text,&used);
    if(used!=text.size()) throw std::invalid_argument("Invalid integer");
    return value;
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--help") {
            std::cout << "attention_benchmark --output NEW_DIRECTORY [--grid representative|tuning|ablation|proposal] "
                "[--shape B H N d] [--mode dense|causal|both] "
                "[--variants main|all|naive|tiled|parallel|optimized] [--configs default|study] "
                "[--tile 8|16|32] [--softmax-threads 128|256] [--seed INTEGER] "
                "[--warmup 5..10] [--repeats 10..30] [--profile-only|--validate-only]\n";
            std::exit(0);
        }
        auto next=[&]() -> std::string {
            if(++i>=argc) throw std::invalid_argument("Missing value for "+arg);
            return argv[i];
        };
        if(arg=="--profile-only") o.profile_only=true;
        else if(arg=="--validate-only") o.validate_only=true;
        else if(arg=="--output") o.output=next();
        else if(arg=="--grid") o.grid=next();
        else if(arg=="--mode") o.mode=next();
        else if(arg=="--variants") {o.variants=next(); o.variants_set=true;}
        else if(arg=="--configs") o.configs=next();
        else if(arg=="--seed") o.seed=integer(next());
        else if(arg=="--warmup") {
            auto n=integer(next()); if(n<5 || n>10) throw std::invalid_argument("Warmup must be 5..10");
            o.warmup=static_cast<unsigned>(n);
        } else if(arg=="--repeats") {
            auto n=integer(next()); if(n<10 || n>30) throw std::invalid_argument("Repeats must be 10..30");
            o.repeats=static_cast<unsigned>(n);
        } else if(arg=="--shape") {
            auto dim=[&]() {const auto n=integer(next());
                if(!n || n>std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("Invalid shape");
                return static_cast<std::size_t>(n);};
            o.shape={dim(),dim(),dim(),dim()}; o.custom=true;
        } else if(arg=="--tile" || arg=="--softmax-threads") {
            const auto n=integer(next());
            if(arg=="--tile" && n!=8 && n!=16 && n!=32) throw std::invalid_argument("Invalid tile");
            if(arg=="--softmax-threads" && n!=128 && n!=256) throw std::invalid_argument("Invalid softmax block");
            if(arg=="--tile") o.fixed.tile=static_cast<unsigned>(n);
            else o.fixed.softmax_threads=static_cast<unsigned>(n);
            o.config_override=true;
        } else throw std::invalid_argument("Unknown option: "+arg);
    }
    if(o.output.empty()) throw std::invalid_argument("--output NEW_DIRECTORY is required");
    if(o.grid!="representative" && o.grid!="tuning" && o.grid!="ablation" && o.grid!="proposal") throw std::invalid_argument("Invalid grid");
    if(o.mode!="dense" && o.mode!="causal" && o.mode!="both") throw std::invalid_argument("Invalid mode");
    if(o.configs!="default" && o.configs!="study") throw std::invalid_argument("Invalid configs");
    if(o.profile_only && o.validate_only) throw std::invalid_argument("Choose profile-only or validate-only");
    if(!o.variants_set && o.grid=="tuning") o.variants="optimized";
    if(!o.variants_set && o.grid=="ablation") o.variants="all";
    return o;
}
std::string mode(bool causal) {return causal?"causal":"dense";}
std::string prefix(attention::Shape s,bool causal,const Variant& v,KernelConfig c) {
    return std::to_string(s.B)+","+std::to_string(s.H)+","+std::to_string(s.N)+","+
        std::to_string(s.d)+","+mode(causal)+","+v.name+","+std::to_string(c.tile)+","+
        std::to_string(c.softmax_threads);
}
std::ofstream file(const std::filesystem::path& p) {
    std::ofstream f(p); if(!f) throw std::runtime_error("Cannot create "+p.string());
    f.exceptions(std::ios::badbit | std::ios::failbit); f<<std::setprecision(17); return f;
}
void report(std::ofstream& out,const attention::Input& input,bool causal,const Variant& v,KernelConfig c,
            const attention::Result& reference,const attention::GpuResult& actual,const std::string& scope) {
    const auto s=attention::compare(actual.scores,reference.scores,1e-4,1e-4,true);
    const auto p=attention::compare(actual.probabilities,reference.probabilities);
    const auto o=attention::compare(actual.output,reference.output);
    for(const auto& stage:{std::make_pair("S",s),std::make_pair("P",p),std::make_pair("O",o)})
        out << prefix(input.shape,causal,v,c)<<','<<attention::input_fingerprint(input)<<','<<scope<<','
            <<stage.first<<','<<stage.second.max_absolute<<','<<stage.second.max_relative<<','
            <<stage.second.failures<<','<<(stage.second.passed()?"PASS":"FAIL")<<'\n';
    out.flush();
    if(!s.passed() || !p.passed() || !o.passed()) throw std::runtime_error("CPU/GPU validation failed");
}
}
int main(int argc,char** argv) {
    std::filesystem::path destination;
    std::string current_case;
    try {
        const auto options=parse(argc,argv);
        const auto selected=variants(options); const auto configurations=configs(options);
        const auto selected_shapes=shapes(options); const auto selected_modes=modes(options);
        if(std::filesystem::exists(options.output)) throw std::invalid_argument("Output directory already exists; choose a new run ID");
        std::filesystem::create_directories(options.output); destination=options.output;
        auto meta=file(destination/"metadata.txt");
        meta << "source_commit=" << PART_A_SOURCE_COMMIT
             << "\nconfigure_worktree=" << PART_A_SOURCE_STATE << "\nbuild=" << PART_A_BUILD_INFO
             << "\nbuild_configuration=" << PART_A_BUILD_CONFIG
             << "\nstarted_unix_seconds="<<std::time(nullptr)<<"\ncommand=";
        for(int i=0;i<argc;++i) meta<<argv[i]<<' ';
        meta << "\ngenerator=splitmix64-fp32-v1\nseed="<<options.seed
             <<"\nlayout=QKV/O[B,H,N,d];SP[B,H,N,N]\nprecision=FP32; oracle FP64"
             <<"\natol=0.0001\nrtol=0.0001\nquantiles=linear interpolation at (n-1)*p"
             <<"\nlarge_shape_validation=representative gate; full CPU S/P/O only when N<=257"
             <<"\ntiming=default-stream CUDA events; separate forward and breakdown passes"
             <<"\nmemory=peak simultaneous requested Q/K/V/S/P/O bytes; not HBM traffic"
             <<"\nworkspace_free_memory_budget=0.8\nwarmup="<<options.warmup<<"\nrepeats="<<options.repeats<<'\n';
        meta << "profile_only="<<options.profile_only<<'\n';
        meta << "validate_only="<<options.validate_only<<'\n';
        auto status=file(destination/"status.txt"); status<<"RUNNING\n"; status.close();
        meta << "planned_configurations="<<selected_shapes.size()*selected_modes.size()*selected.size()*configurations.size()
             <<"\nnaive_launch=256 threads; grid=min(ceil(elements/256),65535); softmax one thread per row"
             <<"\ntiled_qk_launch=tile*tile threads; grid=min(B*H*ceil(N/tile)^2,65535)"
             <<"\ntiled_pv_launch=tile*tile threads; grid=min(B*H*ceil(N/tile)*ceil(d/tile),65535)"
             <<"\nparallel_softmax_launch=softmax_threads per block; grid=min(B*H*N,65535)"
             <<"\nconfig_inactive_fields=tile ignored by naive/parallel; softmax_threads ignored by naive/tiled\n";
        meta.flush();
        const auto device=attention::gpu_environment();
        meta<<"gpu="<<device.name<<"\ncompute_capability="<<device.major<<'.'<<device.minor
            <<"\nreported_vram_bytes="<<device.total_bytes<<"\ninitial_free_bytes="<<device.free_bytes
            <<"\ncuda_runtime_api="<<device.runtime_version<<"\ncuda_driver_api="<<device.driver_api_version
            <<"\nclock_power_settings=collect nvidia-smi separately; runner does not change settings\n";
        meta.flush();
        auto correctness=file(destination/"correctness.csv");
        const std::string head="B,H,N,d,mode,variant,tile,softmax_threads";
        auto plan=file(destination/"plan.csv");
        plan<<head<<",status\n";
        for(auto shape:selected_shapes) for(bool causal:selected_modes)
            for(const auto& variant:selected) for(auto config:configurations)
                plan<<prefix(shape,causal,variant,config)<<",PLANNED\n";
        plan.flush();
        correctness<<head<<",input_fingerprint,scope,stage,max_absolute,max_relative,failures,status\n";
        // Validate before timing
        for(auto shape:validation_shapes()) {
            const auto input=attention::synthetic_input(shape,options.seed);
            attention::BaselineWorkspace workspace(input);
            for(bool causal:selected_modes) {
                const auto cpu=attention::reference(input,causal);
                for(const auto& variant:selected) for(auto config:configurations) {
                    workspace.run(causal,variant.value,config);
                    report(correctness,input,causal,variant,config,cpu,workspace.download(),"representative_gate");
                }
            }
        }
        std::cout<<"PASS representative CPU/GPU gate\n";
        if(options.validate_only) {
            meta<<"completed_configurations="<<selected_shapes.size()*selected_modes.size()*selected.size()*configurations.size()<<'\n';
            file(destination/"status.txt")<<"PASS\n";
            std::cout<<"PASS configuration validation\n"; return 0;
        }
        auto raw=file(destination/"samples.csv"), summary=file(destination/"summary.csv");
        auto resources=file(destination/"resources.csv");
        auto cases=file(destination/"case_status.csv");
        cases<<head<<",status\n";
        resources<<head<<",stage,blocks,threads,registers_per_thread,shared_bytes,local_bytes_per_thread,active_blocks_per_sm,theoretical_occupancy_percent\n";
        raw<<head<<",input_fingerprint,repeat,forward_ms,qk_ms,softmax_ms,pv_ms\n";
        summary<<head<<",input_fingerprint,validation,warmup,repeats,forward_median_ms,forward_q1_ms,forward_q3_ms,forward_iqr_ms,"
            "qk_median_ms,qk_iqr_ms,softmax_median_ms,softmax_iqr_ms,pv_median_ms,pv_iqr_ms,"
            "sequences_per_second,tokens_per_second,speedup_vs_naive,peak_workspace_allocated_bytes,setup_host_ms\n";
        std::size_t completed=0,not_run=0;
        for(auto shape:selected_shapes) {
            current_case="shape="+std::to_string(shape.B)+","+std::to_string(shape.H)+","+
                std::to_string(shape.N)+","+std::to_string(shape.d)+"; allocation/setup";
            // Shared input and device workspace
            const auto input=attention::synthetic_input(shape,options.seed);
            const auto fingerprint=attention::input_fingerprint(input);
            std::unique_ptr<attention::BaselineWorkspace> owner;
            try { owner=std::make_unique<attention::BaselineWorkspace>(input); }
            catch(const attention::WorkspaceMemoryError& error) {
                for(bool causal:selected_modes) for(const auto& variant:selected) for(auto config:configurations) {
                    cases<<prefix(shape,causal,variant,config)<<",NOT_RUN_MEMORY\n"; ++not_run;
                }
                cases.flush();
                std::ofstream errors(destination/"memory_errors.txt",std::ios::app);
                errors<<current_case<<": "<<error.what()<<'\n';
                continue;
            }
            auto& workspace=*owner;
            for(bool causal:selected_modes) {
                attention::Result cpu;
                const bool full_validation=shape.N<=257;
                if(full_validation) cpu=attention::reference(input,causal);
                std::map<std::string,attention::Distribution> forward;
                for(const auto& variant:selected) for(auto config:configurations) {
                    current_case=prefix(shape,causal,variant,config);
                    for(const auto& resource:workspace.resources(variant.value,config))
                        resources<<current_case<<','<<resource.stage<<','<<resource.blocks<<','<<resource.threads<<','
                                 <<resource.registers_per_thread<<','<<resource.shared_bytes<<','<<resource.local_bytes_per_thread<<','
                                 <<resource.active_blocks_per_sm<<','<<resource.theoretical_occupancy_percent<<'\n';
                    resources.flush();
                    workspace.run(causal,variant.value,config);
                    if(full_validation) report(correctness,input,causal,variant,config,cpu,workspace.download(),"exact_timed_shape");
                    else for(float value:workspace.download(false).output)
                        if(!std::isfinite(value)) throw std::runtime_error("Nonfinite large-shape output");
                    workspace.warmup(causal,variant.value,config,options.warmup);
                    if(options.profile_only) {
                        workspace.run(causal,variant.value,config);
                        cases<<current_case<<",PROFILED\n"; cases.flush();
                        ++completed;
                        std::cout<<"PROFILE forward "<<prefix(shape,causal,variant,config)<<'\n';
                        continue;
                    }
                    std::vector<double> total,qk,softmax,pv;
                    std::vector<attention::TimingSample> samples;
                    for(unsigned r=0;r<options.repeats;++r) {
                        const auto t=workspace.measure(causal,variant.value,config);
                        samples.push_back(t);
                        total.push_back(t.forward_ms); qk.push_back(t.qk_ms);
                        softmax.push_back(t.softmax_ms); pv.push_back(t.pv_ms);
                    }
                    for(unsigned r=0;r<samples.size();++r) {
                        const auto& t=samples[r];
                        raw<<prefix(shape,causal,variant,config)<<','<<fingerprint<<','<<r<<','
                           <<t.forward_ms<<','<<t.qk_ms<<','<<t.softmax_ms<<','<<t.pv_ms<<'\n';
                    }
                    raw.flush();
                    // Check output outside timing
                    if(full_validation) report(correctness,input,causal,variant,config,cpu,workspace.download(),"after_timing");
                    else for(float value:workspace.download(false).output)
                        if(!std::isfinite(value)) throw std::runtime_error("Nonfinite output after timing");
                    const auto ft=attention::distribution(total),qt=attention::distribution(qk),
                               st=attention::distribution(softmax),pt=attention::distribution(pv);
                    // Matched naive latency
                    const auto key=std::to_string(config.tile)+":"+std::to_string(config.softmax_threads);
                    if(variant.value==BaselineVariant::Naive) forward[key]=ft;
                    summary<<prefix(shape,causal,variant,config)<<','<<fingerprint<<','
                        <<(full_validation?"exact_CPU_SPO":"representative_CPU_gate+finite_O")<<','
                        <<options.warmup<<','<<options.repeats<<','<<ft.median<<','<<ft.q1<<','<<ft.q3<<','<<ft.iqr()<<','
                        <<qt.median<<','<<qt.iqr()<<','<<st.median<<','<<st.iqr()<<','<<pt.median<<','<<pt.iqr()<<','
                        <<1000.0*shape.B/ft.median<<','<<1000.0*shape.B*shape.N/ft.median<<',';
                    if(forward.count(key)) summary<<forward.at(key).median/ft.median;
                    summary<<','<<workspace.allocated_bytes()<<','<<workspace.setup_ms()<<'\n';
                    summary.flush(); ++completed;
                    cases<<current_case<<",PASS\n"; cases.flush();
                    std::cout<<prefix(shape,causal,variant,config)<<" median_ms="<<ft.median<<" IQR_ms="<<ft.iqr()<<'\n';
                }
            }
        }
        meta<<"completed_configurations="<<completed<<"\nnot_run_configurations="<<not_run<<'\n';
        if(not_run) {
            file(destination/"status.txt")<<"PARTIAL\n";
            std::cerr<<"PARTIAL: "<<not_run<<" configurations could not allocate workspace\n";
            return 2;
        }
        file(destination/"status.txt")<<"PASS\n";
        std::cout<<"PASS benchmark; results="<<destination.string()<<'\n'; return 0;
    } catch(const std::exception& e) {
        if(!destination.empty()) {
            std::ofstream status(destination/"status.txt"); status<<"INCOMPLETE\n";
            std::ofstream failed(destination/"failure.txt");
            failed<<"status=INCOMPLETE\ncurrent_case="<<current_case<<"\nerror="<<e.what()
                  <<"\nPartial CSV rows are not a complete grid. Remaining plan.csv rows without summary results were NOT RUN.\n";
        }
        std::cerr<<"FAIL / INCOMPLETE: "<<e.what()<<'\n'; return 1;
    }
}
