#include "input_generation.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace attention;
void require(bool condition){if(!condition)throw std::runtime_error("Shared reference contract failed");}
int main(){
    for(Shape s:std::vector<Shape>{{1,1,1,1},{2,3,17,7},{1,2,33,32}})
    for(const std::string pattern:{"random","uniform","constant_v","rising","stress"}){
        auto input=synthetic_input(s);apply_pattern(input,pattern);
        for(bool causal:{false,true}){
            auto full=reference(input,causal).output;
            auto all=reference_output(input,causal,false),sampled=reference_output(input,causal,true);
            auto selected=reference_query_rows(s.N);
            for(std::size_t i=0;i<full.size();++i){
                require(std::abs(full[i]-all[i])<1e-12);
                auto row=(i/s.d)%s.N;bool checked=std::binary_search(selected.begin(),selected.end(),row);
                require(checked?std::abs(full[i]-sampled[i])<1e-12:std::isnan(sampled[i]));
            }
        }
        auto before=reference_output(input,true,false);
        const auto cutoff=s.N/2;
        for(std::size_t bh=0;bh<s.B*s.H;++bh)for(std::size_t i=cutoff;i<s.N;++i)for(std::size_t d=0;d<s.d;++d){auto index=(bh*s.N+i)*s.d+d;input.k[index]+=10;input.v[index]-=5;}
        auto after=reference_output(input,true,false);
        for(std::size_t bh=0;bh<s.B*s.H;++bh)for(std::size_t i=0;i<cutoff;++i)for(std::size_t d=0;d<s.d;++d){auto index=(bh*s.N+i)*s.d+d;require(before[index]==after[index]);}
    }
    require(input_fingerprint(synthetic_input({1,1,1,32}))=="b77d9233a247d288");
    std::cout<<"PASS: shared generator fingerprint, full/sampled dense/causal oracle, stress/tails and causal isolation\n";
}
