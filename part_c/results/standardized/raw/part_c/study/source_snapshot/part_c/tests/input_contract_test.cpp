#include "test_support.h"
#include "../../part_a/common/input_generation.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
int main(){
    for(Shape s:std::vector<Shape>{{1,1,17,7},{2,3,33,64},{1,1,65,128}}){
        Input c(s,"random",true);auto a=attention::synthetic_input({std::size_t(s.b),std::size_t(s.h),std::size_t(s.n),std::size_t(s.d)});
        for(auto pair:{std::make_pair(&a.q,&c.q),std::make_pair(&a.k,&c.k),std::make_pair(&a.v,&c.v)})
            if(std::memcmp(pair.first->data(),pair.second->data(),s.size()*4))throw std::runtime_error("Part A/C input mismatch");
    }
    std::cout<<"PASS: Part A/C Q,K,V match bit for bit (3 shapes)\n";
}
