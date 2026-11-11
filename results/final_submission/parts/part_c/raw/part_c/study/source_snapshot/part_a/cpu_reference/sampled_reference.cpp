#include "attention_types.hpp"
#include "input_generation.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace attention {
std::vector<std::size_t> reference_query_rows(std::size_t n) {
    if(!n)throw std::invalid_argument("Empty sequence");
    std::vector<std::size_t> rows{0,n/2,n-1};
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());return rows;
}
std::vector<double> reference_output(Shape s,const std::vector<float>& q,
    const std::vector<float>& k,const std::vector<float>& v,bool causal,bool sampled) {
    const auto count=input_elements(s);
    if(q.size()!=count||k.size()!=count||v.size()!=count)throw std::invalid_argument("Reference input size mismatch");
    std::vector<double> output(count,std::numeric_limits<double>::quiet_NaN());
    const double scale=1.0/std::sqrt(double(s.d));
    const auto selected=reference_query_rows(s.N);
    for(std::size_t bh=0;bh<s.B*s.H;++bh)for(std::size_t i=0;i<s.N;++i){
        if(sampled && !std::binary_search(selected.begin(),selected.end(),i))continue;
        const auto base=bh*s.N*s.d, keys=causal?i+1:s.N;
        std::vector<double> scores(keys);double maximum=-std::numeric_limits<double>::infinity();
        for(std::size_t j=0;j<keys;++j){
            double score=0;for(std::size_t r=0;r<s.d;++r)score+=double(q[base+i*s.d+r])*double(k[base+j*s.d+r]);
            scores[j]=score*scale;maximum=std::max(maximum,scores[j]);
        }
        double sum=0;for(double& value:scores){value=std::exp(value-maximum);sum+=value;}
        for(double& value:scores)value/=sum;
        for(std::size_t r=0;r<s.d;++r){double value=0;for(std::size_t j=0;j<keys;++j)value+=scores[j]*double(v[base+j*s.d+r]);output[base+i*s.d+r]=value;}
    }
    return output;
}
}
