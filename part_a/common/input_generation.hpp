#pragma once
#include "attention_types.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace attention {
inline std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Shape/byte count overflow");
    return a * b;
}

inline std::size_t input_elements(Shape shape) {
    if (!shape.B || !shape.H || !shape.N || !shape.d)
        throw std::invalid_argument("Dimensions must be positive");
    return checked_product(checked_product(checked_product(shape.B, shape.H), shape.N), shape.d);
}

// Shared deterministic stream; retain the original Q-then-K-then-V ordering.
inline void fill_synthetic_input(Shape shape,std::vector<float>& q,std::vector<float>& k,
                                std::vector<float>& v,std::uint64_t seed=5206) {
    const auto count=input_elements(shape);
    if(q.size()!=count || k.size()!=count || v.size()!=count) throw std::invalid_argument("Input size mismatch");
    std::uint64_t state=seed;
    for(auto* tensor:{&q,&k,&v}) for(float& value:*tensor) {
        std::uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);z^=z>>31;
        value=static_cast<float>(static_cast<std::int32_t>(z>>40)-8388608)/8388608.0f;
    }
}
inline Input synthetic_input(Shape shape,std::uint64_t seed=5206) {
    const auto count=input_elements(shape);
    Input input{shape,std::vector<float>(count),std::vector<float>(count),std::vector<float>(count)};
    fill_synthetic_input(shape,input.q,input.k,input.v,seed);return input;
}
inline void apply_pattern(Shape shape,std::vector<float>& q,std::vector<float>& k,
                          std::vector<float>& v,const std::string& pattern) {
    if(pattern!="random" && pattern!="uniform" && pattern!="constant_v" && pattern!="rising" && pattern!="stress")
        throw std::invalid_argument("Unknown input pattern");
    const auto count=input_elements(shape);
    if(q.size()!=count || k.size()!=count || v.size()!=count)throw std::invalid_argument("Input size mismatch");
    for(std::size_t i=0;i<count;++i) {
        if(pattern=="uniform")q[i]=k[i]=0;
        if(pattern=="constant_v")v[i]=0.375f;
        if(pattern=="rising"){q[i]=1.f;k[i]=float((i/shape.d)%shape.N)*0.125f;}
        if(pattern=="stress"){q[i]*=16;k[i]*=16;}
    }
}
inline void apply_pattern(Input& input,const std::string& pattern) {
    apply_pattern(input.shape,input.q,input.k,input.v,pattern);
}
inline std::string input_fingerprint(const std::vector<float>& q,const std::vector<float>& k,
                                     const std::vector<float>& v) {
    std::uint64_t hash=UINT64_C(14695981039346656037);
    for(const auto* tensor:{&q,&k,&v})for(float value:*tensor){
        std::uint32_t bits;std::memcpy(&bits,&value,4);
        for(int i=0;i<4;++i){hash^=(bits>>(8*i))&255u;hash*=UINT64_C(1099511628211);}
    }
    std::ostringstream result;result<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return result.str();
}
inline std::string input_fingerprint(const Input& input) {
    return input_fingerprint(input.q,input.k,input.v);
}
}
