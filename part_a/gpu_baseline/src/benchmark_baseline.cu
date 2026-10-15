#include "benchmark_baseline.hpp"
#include "input_generation.hpp"
#include "launch.cuh"
#include <chrono>
#include <stdexcept>
namespace attention {
namespace {
void checked(cudaError_t result, const char* op) {
    if(result==cudaErrorMemoryAllocation)
        throw WorkspaceMemoryError(std::string(op)+": "+cudaGetErrorString(result));
    if(result != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(result));
}

}
GpuEnvironment gpu_environment() {
    int device; cudaDeviceProp p{}; GpuEnvironment out{};
    checked(cudaGetDevice(&device),"device");
    checked(cudaGetDeviceProperties(&p,device),"properties");
    out.name=p.name; out.major=p.major; out.minor=p.minor; out.total_bytes=p.totalGlobalMem;
    std::size_t total;
    checked(cudaMemGetInfo(&out.free_bytes,&total),"free memory");
    checked(cudaRuntimeGetVersion(&out.runtime_version),"runtime version");
    checked(cudaDriverGetVersion(&out.driver_api_version),"driver API version");
    return out;
}
struct BaselineWorkspace::Impl {
    Shape shape;
    std::size_t elements, matrices, tensor_bytes, matrix_bytes, bytes;
    float *q=nullptr,*k=nullptr,*v=nullptr,*s=nullptr,*p=nullptr,*o=nullptr;
    cudaEvent_t events[4]{};
    double setup=0;
    bool result_valid=false;
    explicit Impl(Shape x):shape(x) {
        elements=input_elements(x);
        matrices=checked_product(checked_product(x.B,x.H),checked_product(x.N,x.N));
        tensor_bytes=checked_product(elements,sizeof(float));
        matrix_bytes=checked_product(matrices,sizeof(float));
        const auto tensors=checked_product(tensor_bytes,4), scores=checked_product(matrix_bytes,2);
        if(tensors>std::numeric_limits<std::size_t>::max()-scores) throw std::invalid_argument("Byte sum overflow");
        bytes=tensors+scores;
    }
    ~Impl() {
        // Release owned CUDA resources
        for(auto event:events) if(event) (void)cudaEventDestroy(event);
        for(float* ptr:{q,k,v,s,p,o}) if(ptr) (void)cudaFree(ptr);
    }
    void qk(bool causal,BaselineVariant variant,KernelConfig config) {
        detail::qk_stage(q,k,s,shape,causal,variant,config);
    }
    void softmax(bool causal,BaselineVariant variant,KernelConfig config) {
        detail::softmax_stage(s,p,shape,causal,variant,config);
    }
    void pv(BaselineVariant variant,KernelConfig config) {
        detail::pv_stage(p,v,o,shape,variant,config);
    }
    double elapsed(unsigned first,unsigned last) {
        float ms; checked(cudaEventElapsedTime(&ms,events[first],events[last]),"elapsed time");
        if(!std::isfinite(ms) || ms<=0) throw std::runtime_error("Nonpositive/invalid event time");
        return ms;
    }
};
BaselineWorkspace::BaselineWorkspace(const Input& input,double max_free_fraction):impl_(new Impl(input.shape)) {
    if(!std::isfinite(max_free_fraction) || max_free_fraction<=0 || max_free_fraction>1)
        throw std::invalid_argument("Memory budget fraction must be in (0,1]");
    for(const auto* tensor:{&input.q,&input.k,&input.v}) {
        if(tensor->size()!=impl_->elements) throw std::invalid_argument("Input size mismatch");
        for(float x:*tensor) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite input");
    }
    std::size_t free,total; checked(cudaMemGetInfo(&free,&total),"memory budget");
    if(static_cast<double>(impl_->bytes)>static_cast<double>(free)*max_free_fraction)
        throw WorkspaceMemoryError("Workspace needs "+std::to_string(impl_->bytes)+
                                   " bytes; free VRAM="+std::to_string(free)+" bytes; budget fraction="+
                                   std::to_string(max_free_fraction));
    const auto start=std::chrono::steady_clock::now();
    for(float** ptr:{&impl_->q,&impl_->k,&impl_->v,&impl_->o})
        checked(cudaMalloc(reinterpret_cast<void**>(ptr),impl_->tensor_bytes),"allocate tensor");
    for(float** ptr:{&impl_->s,&impl_->p})
        checked(cudaMalloc(reinterpret_cast<void**>(ptr),impl_->matrix_bytes),"allocate matrix");
    checked(cudaMemcpy(impl_->q,input.q.data(),impl_->tensor_bytes,cudaMemcpyHostToDevice),"upload Q");
    checked(cudaMemcpy(impl_->k,input.k.data(),impl_->tensor_bytes,cudaMemcpyHostToDevice),"upload K");
    checked(cudaMemcpy(impl_->v,input.v.data(),impl_->tensor_bytes,cudaMemcpyHostToDevice),"upload V");
    for(auto& event:impl_->events) checked(cudaEventCreate(&event),"create event");
    checked(cudaDeviceSynchronize(),"finish setup");
    impl_->setup=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
BaselineWorkspace::~BaselineWorkspace()=default;
void BaselineWorkspace::run(bool causal,BaselineVariant variant,KernelConfig config) {
    impl_->result_valid=false;
    detail::validate_configuration(variant,config);
    impl_->qk(causal,variant,config); impl_->softmax(causal,variant,config); impl_->pv(variant,config);
    checked(cudaDeviceSynchronize(),"execute forward");
    impl_->result_valid=true;
}
GpuResult BaselineWorkspace::download(bool intermediates) {
    if(!impl_->result_valid) throw std::logic_error("Run a successful forward before download");
    checked(cudaDeviceSynchronize(),"before download");
    GpuResult result;
    result.output.resize(impl_->elements);
    checked(cudaMemcpy(result.output.data(),impl_->o,impl_->tensor_bytes,cudaMemcpyDeviceToHost),"download O");
    if(intermediates) {
        result.scores.resize(impl_->matrices); result.probabilities.resize(impl_->matrices);
        checked(cudaMemcpy(result.scores.data(),impl_->s,impl_->matrix_bytes,cudaMemcpyDeviceToHost),"download S");
        checked(cudaMemcpy(result.probabilities.data(),impl_->p,impl_->matrix_bytes,cudaMemcpyDeviceToHost),"download P");
    }
    return result;
}
void BaselineWorkspace::warmup(bool causal,BaselineVariant variant,KernelConfig config,unsigned calls) {
    for(unsigned i=0;i<calls;++i) run(causal,variant,config);
}
TimingSample BaselineWorkspace::measure(bool causal,BaselineVariant variant,KernelConfig config) {
    impl_->result_valid=false;
    detail::validate_configuration(variant,config);
    // Complete forward timing
    checked(cudaEventRecord(impl_->events[0]),"start forward");
    impl_->qk(causal,variant,config); impl_->softmax(causal,variant,config); impl_->pv(variant,config);
    checked(cudaEventRecord(impl_->events[3]),"stop forward");
    checked(cudaEventSynchronize(impl_->events[3]),"finish forward");
    TimingSample out{impl_->elapsed(0,3),0,0,0};
    // Separate stage timing
    checked(cudaEventRecord(impl_->events[0]),"start QK");
    impl_->qk(causal,variant,config);
    checked(cudaEventRecord(impl_->events[1]),"stop QK");
    impl_->softmax(causal,variant,config);
    checked(cudaEventRecord(impl_->events[2]),"stop softmax");
    impl_->pv(variant,config);
    checked(cudaEventRecord(impl_->events[3]),"stop PV");
    checked(cudaEventSynchronize(impl_->events[3]),"finish breakdown");
    out.qk_ms=impl_->elapsed(0,1); out.softmax_ms=impl_->elapsed(1,2); out.pv_ms=impl_->elapsed(2,3);
    impl_->result_valid=true;
    return out;
}
std::size_t BaselineWorkspace::allocated_bytes() const {return impl_->bytes;}
std::vector<KernelResources> BaselineWorkspace::resources(BaselineVariant variant,KernelConfig config) const {
    detail::validate_configuration(variant,config);
    const auto shape=impl_->shape;
    return {detail::uses_tiles(variant)?detail::qk_tiled_resources(shape,config.tile):detail::qk_resources(shape),
            detail::uses_parallel(variant)?detail::softmax_parallel_resources(shape,config.softmax_threads):detail::softmax_resources(shape),
            detail::uses_tiles(variant)?detail::pv_tiled_resources(shape,config.tile):detail::pv_resources(shape)};
}
double BaselineWorkspace::setup_ms() const {return impl_->setup;}
}
