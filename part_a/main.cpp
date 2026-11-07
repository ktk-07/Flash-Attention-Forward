#include "attention_types.hpp"
#include "input_generation.hpp"
#include "compare.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#ifdef PART_A_ENABLE_CUDA
#include "build_flags.hpp"
#include <cuda_runtime.h>
#endif

using namespace attention;
namespace {
struct Options {
    Shape shape{1,1,2,2};
    std::string grid="representative", mode="both", variants, pattern="random";
    std::filesystem::path output;
    KernelConfig config{};
    // seed and timing settings
    static constexpr std::uint64_t seed=5206;
    static constexpr unsigned warmup=5, repeats=20;
    bool cpu=false, custom=false, fixed=false, profile=false;
};

struct Variant { const char* name; BaselineVariant value; };
struct Stats { double q1, median, q3; double iqr() const { return q3-q1; } };
std::uint64_t integer(const std::string& text) {
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer");
    return std::stoull(text);
}
// command line arguments
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        auto next=[&]() { if(++i>=argc) throw std::invalid_argument("Missing value for "+arg); return std::string(argv[i]); };
        if(arg=="--help") {
            std::cout<<"attention [--cpu] [--shape B H N d] [--mode dense|causal|both]\n"
                "  --output NEW_DIRECTORY --grid correctness|representative|tuning|ablation|proposal\n"
                "  --variants main|all|naive|tiled|parallel|optimized|tensorcore\n"
                "  --tile 8|16|32 --softmax-threads 128|256\n"
                "  --profile-only\n";
            std::exit(0);
        } else if(arg=="--cpu") o.cpu=true;
        else if(arg=="--shape") {
            auto dim=[&]() { auto n=integer(next()); if(!n) throw std::invalid_argument("Invalid dimension"); return static_cast<std::size_t>(n); };
            o.shape={dim(),dim(),dim(),dim()}; o.custom=true;
        } else if(arg=="--grid") o.grid=next();
        else if(arg=="--mode") o.mode=next();
        else if(arg=="--variants") o.variants=next();
        else if(arg=="--pattern") o.pattern=next();
        else if(arg=="--output") o.output=next();
        else if(arg=="--profile-only") o.profile=true;
        else if(arg=="--tile" || arg=="--softmax-threads") {
            auto n=integer(next()); o.fixed=true;
            if(arg=="--tile") { if(n!=8 && n!=16 && n!=32) throw std::invalid_argument("Invalid tile"); o.config.tile=static_cast<unsigned>(n); }
            else { if(n!=128 && n!=256) throw std::invalid_argument("Invalid softmax threads"); o.config.softmax_threads=static_cast<unsigned>(n); }
        } else throw std::invalid_argument("Unknown option: "+arg);
    }
    if(o.mode!="dense" && o.mode!="causal" && o.mode!="both") throw std::invalid_argument("Invalid mode");
    if(o.grid!="correctness" && o.grid!="representative" && o.grid!="tuning" && o.grid!="ablation" && o.grid!="proposal") throw std::invalid_argument("Invalid grid");
    if(o.grid=="tuning" && o.fixed) throw std::invalid_argument("Tuning uses the predefined configurations");
    if(o.cpu && (o.profile || !o.output.empty() || o.grid!="representative"))
        throw std::invalid_argument("CPU mode cannot run GPU experiments");
    if(o.grid=="correctness") {
        if(o.profile || o.custom || o.fixed) throw std::invalid_argument("Correctness uses fixed small shapes and all study configurations");
    }
    if(o.variants.empty()) {
        if(o.grid=="correctness" || o.grid=="ablation") o.variants="all";
        else if(o.grid=="tuning") o.variants="optimized";
        else o.variants="main";
    }
    if(o.variants=="tensorcore" && (o.config.tile!=16 || o.grid=="ablation"))
        throw std::invalid_argument("Tensor Core uses tile 16 and a separate FP16 experiment");
    return o;
}

std::vector<bool> modes(const Options& o) {
    if(o.mode=="dense") return {false};
    if(o.mode=="causal") return {true};
    return {false,true};
}

std::vector<Variant> variants(const Options& o) {
    if(o.variants=="tensorcore") return {{"tensorcore",BaselineVariant::TensorCore}};
    std::vector<Variant> selected;
    for(auto v:{Variant{"naive",BaselineVariant::Naive},Variant{"tiled",BaselineVariant::TiledMatmul},
                Variant{"parallel",BaselineVariant::ParallelSoftmax},Variant{"optimized",BaselineVariant::Optimized}})
        if(o.variants=="all" || o.variants==v.name || (o.variants=="main" && (v.value==BaselineVariant::Naive || v.value==BaselineVariant::Optimized))) selected.push_back(v);
    if(selected.empty()) throw std::invalid_argument("Invalid variants");
    return selected;
}

std::vector<KernelConfig> configurations(const Options& o) {
    if(o.variants=="tensorcore" && (o.grid=="correctness" || o.grid=="tuning")) return {{16,128},{16,256}};
    if(o.variants!="tensorcore" && (o.grid=="correctness" || o.grid=="tuning"))
        return {{8,256},{16,256},{32,256},{16,128}};
    return {o.config};
}

std::vector<Shape> shapes(const Options& o) {
    if(o.custom) return {o.shape};
    if(o.grid=="correctness") return {{1,1,1,1},{2,3,15,7},{1,2,16,16},{2,1,17,33},{1,1,33,65}};
    if(o.grid=="representative") return {{1,1,33,32},{4,8,256,64}};
    if(o.grid=="tuning" || o.grid=="ablation") return {{4,8,256,64},{4,8,1024,64}};
    std::vector<Shape> result;
    for(std::size_t n:{256,512,1024,2048,4096}) for(std::size_t d:{32,64,128}) result.push_back({4,8,n,d});
    return result;
}
Stats statistics(std::vector<double> values) {
    if(values.empty()) throw std::invalid_argument("Empty timing samples");
    for(double x:values) if(!std::isfinite(x) || x<=0) throw std::runtime_error("Invalid timing sample");
    std::sort(values.begin(),values.end());
    auto q=[&](double p) { double position=(values.size()-1)*p; auto lo=static_cast<std::size_t>(position),hi=std::min(lo+1,values.size()-1); return values[lo]+(position-lo)*(values[hi]-values[lo]); };
    return {q(.25),q(.5),q(.75)};
}

std::string identity(Shape s,bool causal,Variant v,KernelConfig c) {
    return std::to_string(s.B)+","+std::to_string(s.H)+","+std::to_string(s.N)+","+std::to_string(s.d)+","+
        (causal?"causal":"dense")+","+v.name+","+std::to_string(c.tile)+","+std::to_string(c.softmax_threads);
}

std::ofstream file(const std::filesystem::path& path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit|std::ios::failbit); out<<std::setprecision(17); return out;
}
// CPU execution
int cpu_run(const Options& o) {
    const auto input=synthetic_input(o.shape,o.seed);
    for(bool causal:modes(o)) {
        const auto result=reference(input,causal);
        std::cout<<(causal?"Causal":"Dense")<<" CPU: output_elements="<<result.output.size()<<'\n';
    }
    return 0;
}

#ifdef PART_A_ENABLE_CUDA
void checked(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
// GPU buffers
template<class T>
struct DeviceBuffers {
    T *q=nullptr,*k=nullptr,*v=nullptr,*s=nullptr,*p=nullptr,*o=nullptr;
    cudaEvent_t events[4]{};
    std::size_t elements=0, matrices=0, bytes=0;
    double setup_ms=0;
    DeviceBuffers()=default;
    DeviceBuffers(const DeviceBuffers&)=delete;
    DeviceBuffers& operator=(const DeviceBuffers&)=delete;
    ~DeviceBuffers() {
        for(auto event:events) if(event) (void)cudaEventDestroy(event);
        for(T* ptr:{q,k,v,s,p,o}) if(ptr) (void)cudaFree(ptr);
    }
};

template<class T>
void prepare(DeviceBuffers<T>& device,Input& input) {
    device.elements=input_elements(input.shape);
    device.matrices=checked_product(checked_product(input.shape.B,input.shape.H),checked_product(input.shape.N,input.shape.N));
    const auto tb=checked_product(device.elements,sizeof(T)),mb=checked_product(device.matrices,sizeof(T));
    auto tensors=checked_product(tb,4), matrices=checked_product(mb,2);
    if(tensors>std::numeric_limits<std::size_t>::max()-matrices) throw std::invalid_argument("Workspace size overflow");
    device.bytes=tensors+matrices;
    const auto start=std::chrono::steady_clock::now();
    for(T** ptr:{&device.q,&device.k,&device.v,&device.o}) checked(cudaMalloc(reinterpret_cast<void**>(ptr),tb),"allocate tensor");
    for(T** ptr:{&device.s,&device.p}) checked(cudaMalloc(reinterpret_cast<void**>(ptr),mb),"allocate matrix");

    // prepare CPU and GPU inputs
    T* destinations[]={device.q,device.k,device.v};
    std::vector<float>* sources[]={&input.q,&input.k,&input.v};
    for(int i=0;i<3;++i) {
        if constexpr(std::is_same_v<T,float>) {
            checked(cudaMemcpy(destinations[i],sources[i]->data(),tb,cudaMemcpyHostToDevice),"upload input");
        } else {
            std::vector<__half> host(device.elements);
            // CPU reference uses these rounded values
            for(std::size_t j=0;j<device.elements;++j) {
                host[j]=__float2half_rn((*sources[i])[j]);
                (*sources[i])[j]=__half2float(host[j]);
            }
            checked(cudaMemcpy(destinations[i],host.data(),tb,cudaMemcpyHostToDevice),"upload input");
        }
    }

    for(auto& event:device.events) checked(cudaEventCreate(&event),"create event");
    checked(cudaDeviceSynchronize(),"finish upload");
    device.setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
// Kernel dispatch
template<class T>
void stage(DeviceBuffers<T>& b,Shape shape,bool causal,Variant v,KernelConfig c,int index) {
    // choose the stage kernel
    if constexpr(std::is_same_v<T,__half>) {
        if(index==0) detail::launch_qk_tensorcore(b.q,b.k,b.s,shape,causal);
        else if(index==1) detail::launch_softmax_fp16(b.s,b.p,shape,causal,c.softmax_threads);
        else detail::launch_pv_tensorcore(b.p,b.v,b.o,shape);
    } else {
        bool tiled=v.value==BaselineVariant::TiledMatmul || v.value==BaselineVariant::Optimized;
        bool parallel=v.value==BaselineVariant::ParallelSoftmax || v.value==BaselineVariant::Optimized;
        if(index==0) {
            if(tiled) detail::launch_qk_tiled(b.q,b.k,b.s,shape,causal,c.tile);
            else detail::launch_qk(b.q,b.k,b.s,shape,causal);
        } else if(index==1) {
            if(parallel) detail::launch_softmax_parallel(b.s,b.p,shape,causal,c.softmax_threads);
            else detail::launch_softmax(b.s,b.p,shape,causal);
        } else {
            if(tiled) detail::launch_pv_tiled(b.p,b.v,b.o,shape,c.tile);
            else detail::launch_pv(b.p,b.v,b.o,shape);
        }
    }
    checked(cudaGetLastError(),"kernel launch");
}

template<class T>
void forward(DeviceBuffers<T>& b,Shape shape,bool causal,Variant v,KernelConfig c) {
    for(int i=0;i<3;++i) stage(b,shape,causal,v,c,i);
    checked(cudaDeviceSynchronize(),"forward");
}

template<class T>
std::vector<float> download(const T* source,std::size_t count) {
    std::vector<T> host(count);
    checked(cudaMemcpy(host.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost),"download result");

    // host conversion for comparison
    if constexpr(std::is_same_v<T,float>) return host;
    else {
        std::vector<float> values(count);
        for(std::size_t i=0;i<count;++i) values[i]=__half2float(host[i]);
        return values;
    }
}

struct Tolerance { double absolute,relative; };
Tolerance tolerance(const Options& o) {
    return o.variants=="tensorcore" ? Tolerance{0.002,0.02} : Tolerance{0.0001,0.0001};
}
// compare with CPU
template<class T>
void correctness(const Options& o) {
    const auto tol=tolerance(o);
    auto out=file(o.output/"correctness.csv");
    const std::string prefix=o.grid+","+(o.variants=="tensorcore"?"FP16,":"FP32,");
    out<<"experiment,precision,B,H,N,d,mode,variant,tile,softmax_threads,stage,atol,rtol,max_absolute,max_relative,failures,status,checked_elements,nonfinite,input_digest,effective_input_digest,pattern,reference_scope,reference_backend\n";
    for(auto shape:shapes(o)) {
        auto input=synthetic_input(shape,o.seed);apply_pattern(input,o.pattern);
        const auto original_digest=input_fingerprint(input);DeviceBuffers<T> b; prepare(b,input);
        const auto effective_digest=input_fingerprint(input);
        for(bool causal:modes(o)) {
            const auto expected=reference(input,causal);
            for(auto v:variants(o)) for(auto c:configurations(o)) {
                if(v.value==BaselineVariant::Naive && (c.tile!=16 || c.softmax_threads!=256)) continue;
                if(v.value==BaselineVariant::TiledMatmul && c.softmax_threads!=256) continue;
                if(v.value==BaselineVariant::ParallelSoftmax && c.tile!=16) continue;
                forward(b,shape,causal,v,c);
                const T* device[]={b.s,b.p,b.o};
                const std::vector<double>* host[]={&expected.scores,&expected.probabilities,&expected.output};
                const char* names[]={"S","P","O"};
                for(int i=0;i<3;++i) {
                    const auto r=compare(download(device[i],host[i]->size()),*host[i],tol.absolute,tol.relative,i==0);
                    out<<prefix<<identity(shape,causal,v,c)<<','<<names[i]<<','<<tol.absolute<<','<<tol.relative<<','<<r.max_absolute<<','<<r.max_relative<<','<<r.failures<<','<<(r.passed()?"PASS":"FAIL")<<','<<r.checked<<','<<r.nonfinite<<','<<original_digest<<','<<effective_digest<<','<<o.pattern<<",full,part_a_shared_fp64\n";
                    out.flush();
                    if(!r.passed()) throw std::runtime_error("CPU/GPU mismatch in "+std::string(names[i]));
                }
            }
        }
    }
    std::cout<<"Small-shape S/P/O comparisons passed\n";
}

void environment(const Options& o) {
    int index,runtime,driver; cudaDeviceProp prop{};
    checked(cudaGetDevice(&index),"device"); checked(cudaGetDeviceProperties(&prop,index),"device properties");
    checked(cudaRuntimeGetVersion(&runtime),"runtime"); checked(cudaDriverGetVersion(&driver),"driver");
    const auto tol=tolerance(o);
    auto out=file(o.output/"environment.txt");
    out<<"gpu="<<prop.name<<"\ncompute_capability="<<prop.major<<'.'<<prop.minor<<"\nvram_bytes="<<prop.totalGlobalMem
       <<"\nCUDA_toolkit="<<PART_A_TOOLKIT<<"\nhost_compiler="<<PART_A_HOST_COMPILER<<"\nbuild="<<PART_A_BUILD_CONFIG
       <<"\nCUDA_architectures="<<PART_A_ARCHITECTURES<<"\nCUDA_runtime_API="<<runtime<<"\nCUDA_driver_API="<<driver
       <<"\nCXX_standard=17\nCUDA_standard=17\nprofile_info="<<PART_A_PROFILE_ENABLED<<"\nCXX_flags="<<PART_A_CXX_FLAGS<<"\nCUDA_flags="<<PART_A_CUDA_FLAGS
       <<"\ngenerator=splitmix64-fp32-v1\nseed="<<o.seed<<"\ngrid="<<o.grid<<"\nvariants="<<o.variants
       <<"\nmode="<<o.mode<<"\nprecision="<<(o.variants=="tensorcore"?"FP16 storage, WMMA accumulator and softmax":"FP32")
       <<"\nCPU_accumulation=FP64\nCPU_input="<<(o.variants=="tensorcore"?"FP16-rounded Q/K/V":"original FP32 Q/K/V")
       <<"\natol="<<tol.absolute<<"\nrtol="<<tol.relative
       <<"\nrelative_error_denominator=max(abs(reference),1e-12)\nwarmup="<<o.warmup<<"\nrepeats="<<o.repeats
       <<"\ntiming=CUDA events; separate forward and stage passes; setup and copies excluded"
       <<"\nworkspace=Q/K/V/S/P/O allocations only; not device-wide usage or memory traffic\n";
    for(auto c:configurations(o)) out<<"configuration="<<c.tile<<','<<c.softmax_threads<<'\n';
    for(auto shape:shapes(o)) out<<"shape="<<shape.B<<','<<shape.H<<','<<shape.N<<','<<shape.d<<'\n';
}
// CUDA event timing
template<class T>
std::array<double,4> measure(DeviceBuffers<T>& b,Shape shape,bool causal,Variant v,KernelConfig c) {
    auto elapsed=[&](int a,int z) { float ms; checked(cudaEventElapsedTime(&ms,b.events[a],b.events[z]),"elapsed time"); return static_cast<double>(ms); };

    // complete forward timing
    checked(cudaEventRecord(b.events[0]),"start forward");
    for(int i=0;i<3;++i) stage(b,shape,causal,v,c,i);
    checked(cudaEventRecord(b.events[3]),"stop forward"); checked(cudaEventSynchronize(b.events[3]),"finish forward");
    std::array<double,4> times{elapsed(0,3),0,0,0};
    // Separate stage pass
    checked(cudaEventRecord(b.events[0]),"start stages");
    for(int i=0;i<3;++i) { stage(b,shape,causal,v,c,i); checked(cudaEventRecord(b.events[i+1]),"stop stage"); }
    checked(cudaEventSynchronize(b.events[3]),"finish stages");
    for(int i=0;i<3;++i) times[i+1]=elapsed(i,i+1);
    return times;
}

template<class T>
void resources(std::ofstream& out,Shape shape,bool causal,Variant v,KernelConfig c,const std::string& prefix) {
    auto write=[&](KernelResources r) {
        out<<prefix<<identity(shape,causal,v,c)<<','<<r.stage<<','<<r.blocks<<','<<r.threads<<','<<r.registers_per_thread<<','<<r.shared_bytes<<','<<r.local_bytes_per_thread<<','<<r.active_blocks_per_sm<<','<<r.theoretical_occupancy_percent<<'\n';
    };
    if constexpr(std::is_same_v<T,__half>) {
        write(detail::qk_tensorcore_resources(shape));
        write(detail::softmax_fp16_resources(shape,c.softmax_threads));
        write(detail::pv_tensorcore_resources(shape));
    } else {
        bool tiled=v.value==BaselineVariant::TiledMatmul || v.value==BaselineVariant::Optimized;
        bool parallel=v.value==BaselineVariant::ParallelSoftmax || v.value==BaselineVariant::Optimized;
        write(tiled?detail::qk_tiled_resources(shape,c.tile):detail::qk_resources(shape));
        write(parallel?detail::softmax_parallel_resources(shape,c.softmax_threads):detail::softmax_resources(shape));
        write(tiled?detail::pv_tiled_resources(shape,c.tile):detail::pv_resources(shape));
    }
}

template<class T>
void gpu_run(const Options& o) {
    const auto tol=tolerance(o);
    const auto selected=variants(o); const auto configs=configurations(o); const auto selected_modes=modes(o);
    const std::string prefix=o.grid+","+(o.variants=="tensorcore"?"FP16,":"FP32,");
    const std::string head="experiment,precision,B,H,N,d,mode,variant,tile,softmax_threads";
    std::ofstream summary,resource,samples;
    if(o.grid!="ablation") resource=file(o.output/"resources.csv");
    if(!o.profile) {
        summary=file(o.output/"summary.csv");
        samples=file(o.output/"samples.csv");
    }
    if(!o.profile) samples<<head<<",repeat,forward_ms,qk_ms,softmax_ms,pv_ms,input_digest,effective_input_digest,pattern\n";
    if(!o.profile) summary<<head<<",atol,rtol,max_absolute,max_relative,warmup,repeats,forward_median_ms,forward_q1_ms,forward_q3_ms,forward_iqr_ms,qk_median_ms,qk_iqr_ms,softmax_median_ms,softmax_iqr_ms,pv_median_ms,pv_iqr_ms,sequences_per_second,speedup_vs_naive,peak_workspace_allocated_bytes,setup_host_ms,input_digest,effective_input_digest,pattern,reference_scope,checked_elements,nonfinite,failures,status,reference_backend\n";
    if(resource.is_open()) resource<<head<<",stage,blocks,threads,registers_per_thread,shared_bytes,local_bytes_per_thread,active_blocks_per_sm,theoretical_occupancy_percent\n";
    for(auto shape:shapes(o)) {
        auto input=synthetic_input(shape,o.seed);apply_pattern(input,o.pattern);
        const auto original_digest=input_fingerprint(input);DeviceBuffers<T> b; prepare(b,input);
        const auto effective_digest=input_fingerprint(input);
        const bool validate=!o.profile;
        const bool sampled=shape.N>97;
        for(bool causal:selected_modes) {
            std::vector<double> expected;
            if(validate) expected=reference_output(input,causal,sampled);

            std::map<std::string,double> naive;
            for(auto v:selected) for(auto c:configs) {
                if(resource.is_open()) resources<T>(resource,shape,causal,v,c,prefix);
                forward(b,shape,causal,v,c);
                ErrorReport error;
                if(validate) {
                    error=compare(download(b.o,b.elements),expected,tol.absolute,tol.relative,false,sampled);
                    if(!error.passed()) throw std::runtime_error("Output mismatch: "+identity(shape,causal,v,c));
                }

                // warmup before timing
                for(unsigned i=0;i<o.warmup;++i) forward(b,shape,causal,v,c);
                if(o.profile) { forward(b,shape,causal,v,c); continue; }

                // forward and stage samples
                std::array<std::vector<double>,4> times;
                for(unsigned repeat=0;repeat<o.repeats;++repeat) {
                    auto t=measure(b,shape,causal,v,c); for(int i=0;i<4;++i) times[i].push_back(t[i]);
                    samples<<prefix<<identity(shape,causal,v,c)<<','<<repeat;
                    for(double ms:t) samples<<','<<ms;
                    samples<<','<<original_digest<<','<<effective_digest<<','<<o.pattern<<'\n';
                }

                // median and IQR
                const auto f=statistics(times[0]),q=statistics(times[1]),s=statistics(times[2]),p=statistics(times[3]);
                const auto key=std::to_string(c.tile)+":"+std::to_string(c.softmax_threads);
                if(v.value==BaselineVariant::Naive) naive[key]=f.median;
                summary<<prefix<<identity(shape,causal,v,c)<<','<<tol.absolute<<','<<tol.relative<<',';
                if(validate) summary<<error.max_absolute;
                summary<<',';
                if(validate) summary<<error.max_relative;
                summary<<','<<o.warmup<<','<<o.repeats<<','
                    <<f.median<<','<<f.q1<<','<<f.q3<<','<<f.iqr()<<','<<q.median<<','<<q.iqr()<<','<<s.median<<','<<s.iqr()<<','<<p.median<<','<<p.iqr()<<','
                    <<1000.0*shape.B/f.median<<',';
                if(naive.count(key)) summary<<naive[key]/f.median;
                summary<<','<<b.bytes<<','<<b.setup_ms<<','<<original_digest<<','<<effective_digest<<','<<o.pattern<<','<<(sampled?"sampled_queries":"full")<<','<<error.checked<<','<<error.nonfinite<<','<<error.failures<<",PASS,part_a_shared_fp64\n"; summary.flush(); samples.flush();
                std::cout<<identity(shape,causal,v,c)<<" median_ms="<<f.median<<" IQR_ms="<<f.iqr()<<'\n';
            }
        }
    }
}
#endif
}

int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv); (void)variants(options);
#ifndef PART_A_ENABLE_CUDA
        if(options.profile || options.grid!="representative" || !options.output.empty())
            throw std::invalid_argument("This build has no CUDA support");
        return cpu_run(options);
#else
        if(options.cpu) return cpu_run(options);
        if(options.output.empty()) throw std::invalid_argument("--output NEW_DIRECTORY is required for GPU runs");
        if(!std::filesystem::create_directories(options.output)) throw std::invalid_argument("Output directory already exists");
        environment(options);
        if(options.variants=="tensorcore") {
            if(options.grid=="correctness") correctness<__half>(options); else gpu_run<__half>(options);
        } else {
            if(options.grid=="correctness") correctness<float>(options); else gpu_run<float>(options);
        }
        return 0;
#endif
    } catch(const std::exception& e) {
        std::cerr<<"Error: "<<e.what()<<'\n'; return 1;
    }
}
