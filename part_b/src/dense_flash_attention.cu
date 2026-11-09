#include "flash_attention/dense_flash_attention.h"
#include <cmath>
#include <stdexcept>
namespace dense_attention {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ float sum(float x){for(int s=16;s;s/=2)x+=__shfl_down_sync(0xffffffff,x,s);return __shfl_sync(0xffffffff,x,0);}
__device__ float maximum(float x){for(int s=16;s;s/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,s));return __shfl_sync(0xffffffff,x,0);}
// One warp owns a query row. Cooperative loads are contiguous over [key,feature].
// All warps, including inactive tail queries, participate in both tile barriers.
template<int W,int BK,bool Vector>
__global__ void tiled_dense(const float* __restrict__ q,const float* __restrict__ k,
 const float* __restrict__ v,float* __restrict__ o,int n,int d){
    extern __shared__ float shared[];
    float* kt=shared;float* vt=kt+BK*d;float* p=vt+BK*d;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*W+warp;
    size_t base=size_t(blockIdx.y)*n*d;
    float qr[8]={},acc[8]={};
    for(int t=0;t<8;++t){int r=lane+t*32;if(row<n && r<d)qr[t]=q[base+size_t(row)*d+r];}
    float m=-CUDART_INF_F,l=0,scale=rsqrtf(float(d));
    for(int start=0;start<n;start+=BK){
        if constexpr(Vector){
            for(int x=threadIdx.x;x<BK*(d/4);x+=W*32){int j=x/(d/4),r=4*(x%(d/4));
                float4 a=make_float4(0,0,0,0),b=a;
                if(start+j<n){a=*reinterpret_cast<const float4*>(k+base+size_t(start+j)*d+r);b=*reinterpret_cast<const float4*>(v+base+size_t(start+j)*d+r);}
                *reinterpret_cast<float4*>(kt+j*d+r)=a;*reinterpret_cast<float4*>(vt+j*d+r)=b;
            }
        }else{
            for(int x=threadIdx.x;x<BK*d;x+=W*32){int j=x/d,r=x%d;kt[x]=start+j<n?k[base+size_t(start+j)*d+r]:0;vt[x]=start+j<n?v[base+size_t(start+j)*d+r]:0;}
        }
        __syncthreads();
        if(row<n){
            for(int j=0;j<BK;++j){float dot=0;
                #pragma unroll
                for(int t=0;t<8;++t){int r=lane+t*32;if(r<d)dot=fmaf(qr[t],kt[j*d+r],dot);}
                dot=sum(dot)*scale;if(lane==0)p[warp*BK+j]=start+j<n?dot:-CUDART_INF_F;
            }
            __syncwarp();
            float local=-CUDART_INF_F;
            for(int j=lane;j<BK;j+=32)local=fmaxf(local,p[warp*BK+j]);
            float next=fmaxf(m,maximum(local)),alpha=expf(m-next),partial=0;
            for(int j=lane;j<BK;j+=32){float prob=expf(p[warp*BK+j]-next);p[warp*BK+j]=prob;partial+=prob;}
            float tile_sum=sum(partial);__syncwarp();
            #pragma unroll
            for(int t=0;t<8;++t){int r=lane+t*32;if(r<d){float z=0;for(int j=0;j<BK;++j)z=fmaf(p[warp*BK+j],vt[j*d+r],z);acc[t]=alpha*acc[t]+z;}}
            l=alpha*l+tile_sum;m=next;
        }
        __syncthreads();
    }
    if(row<n)for(int t=0;t<8;++t){int r=lane+t*32;if(r<d)o[base+size_t(row)*d+r]=acc[t]/l;}
}
// Controlled starting point: same warp mapping, per-key online update, global K/V.
template<int W>
__global__ void streaming_dense(const float* q,const float* k,const float* v,float* o,int n,int d){
    int lane=threadIdx.x%32,row=blockIdx.x*W+threadIdx.x/32;
    if(row>=n)return;size_t base=size_t(blockIdx.y)*n*d;
    float qr[8]={},acc[8]={},m=-CUDART_INF_F,l=0;
    for(int t=0;t<8;++t)if(lane+t*32<d)qr[t]=q[base+size_t(row)*d+lane+t*32];
    for(int j=0;j<n;++j){float dot=0;for(int t=0;t<8;++t)if(lane+t*32<d)dot=fmaf(qr[t],k[base+size_t(j)*d+lane+t*32],dot);
        float s=sum(dot)*rsqrtf(float(d)),next=fmaxf(m,s),a=expf(m-next),b=expf(s-next);
        for(int t=0;t<8;++t)if(lane+t*32<d)acc[t]=a*acc[t]+b*v[base+size_t(j)*d+lane+t*32];l=a*l+b;m=next;
    }
    for(int t=0;t<8;++t)if(lane+t*32<d)o[base+size_t(row)*d+lane+t*32]=acc[t]/l;
}
void validate(attention::Shape s,const Config& c){
    if(!s.B||!s.H||!s.N||!s.d||s.d>256||s.N>2147483647||s.B*s.H>65535)throw std::invalid_argument("Unsupported shape (d=1..256, positive dimensions, grid.y<=65535)");
    if(c.warps!=4&&c.warps!=8)throw std::invalid_argument("warps must be 4 or 8");
    if(c.key_tile!=32&&c.key_tile!=64)throw std::invalid_argument("key tile must be 32 or 64");
    if(c.variant!="streaming"&&c.variant!="tiled"&&c.variant!="vector")throw std::invalid_argument("Unknown variant");
    if(c.variant=="vector"&&s.d%4)throw std::invalid_argument("float4 requires d divisible by four");
}
template<class F> void dispatch(const Config& c,F f){
    if(c.variant=="streaming"){if(c.warps==4)f(streaming_dense<4>,0);else f(streaming_dense<8>,0);return;}
    #define PICK(W,B) if(c.warps==W&&c.key_tile==B){if(c.variant=="vector")f(tiled_dense<W,B,true>,1);else f(tiled_dense<W,B,false>,1);return;}
    PICK(4,32) PICK(4,64) PICK(8,32) PICK(8,64)
    #undef PICK
}
}
void forward(const float* q,const float* k,const float* v,float* o,attention::Shape s,const Config& c,cudaStream_t stream){
    validate(s,c);dim3 grid(unsigned((s.N+c.warps-1)/c.warps),unsigned(s.B*s.H));
    dispatch(c,[&](auto kernel,int tiled){size_t bytes=tiled?(2*c.key_tile*s.d+c.warps*c.key_tile)*sizeof(float):0;
        if(bytes>48*1024)check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
        kernel<<<grid,c.warps*32,bytes,stream>>>(q,k,v,o,int(s.N),int(s.d));});check(cudaGetLastError());
}
attention::KernelResources resources(attention::Shape s,const Config& c){
    validate(s,c);attention::KernelResources result{};
    dispatch(c,[&](auto kernel,int tiled){cudaFuncAttributes a{};cudaDeviceProp prop{};int dev,active;
        size_t bytes=tiled?(2*c.key_tile*s.d+c.warps*c.key_tile)*sizeof(float):0;
        if(bytes>48*1024)check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
        check(cudaFuncGetAttributes(&a,kernel));check(cudaGetDevice(&dev));check(cudaGetDeviceProperties(&prop,dev));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,c.warps*32,bytes));
        result={"forward",unsigned((s.N+c.warps-1)/c.warps*s.B*s.H),unsigned(c.warps*32),a.numRegs,active,bytes+a.sharedSizeBytes,a.localSizeBytes,100.0*active*c.warps*32/prop.maxThreadsPerMultiProcessor};});return result;
}
std::string symbol(const Config& c){return c.variant=="streaming"?"streaming_dense":"tiled_dense";}
}
