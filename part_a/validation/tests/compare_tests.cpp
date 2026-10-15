#include "compare.hpp"
#include <iostream>
#include <string>
using attention::compare;
void require(bool ok) { if(!ok) throw std::runtime_error("Assertion failed"); }
int main(int argc,char** argv) {
    try {
        require(argc==2); const std::string name=argv[1];
        if(name=="tolerance") {
            require(compare({0.00005f,100.005f},{0,100}).passed());
            require(!compare({0.0002f,100.02f},{0,100}).passed());
        } else if(name=="nonfinite") {
            const auto inf=std::numeric_limits<float>::infinity();
            require(!compare({std::numeric_limits<float>::quiet_NaN()},{0}).passed());
            require(!compare({inf},{inf},1e-4,1e-4,true).passed());
            require(!compare({-inf},{-inf}).passed());
            require(compare({-inf},{-inf},1e-4,1e-4,true).passed());
            require(!compare({0},{-inf},1e-4,1e-4,true).passed());
        } else if(name=="metrics") {
            const auto r=compare({0.25f,2.5f},{0,2},1,0);
            require(r.passed() && r.max_absolute==0.5 && r.max_relative==2.5e11);
        } else if(name=="validation") {
            bool rejected=false;
            try {(void)compare({1},{},1e-4,1e-4);} catch(const std::invalid_argument&) {rejected=true;}
            require(rejected); rejected=false;
            try {(void)compare({1},{1},-1,0);} catch(const std::invalid_argument&) {rejected=true;}
            require(rejected);
        } else throw std::runtime_error("Unknown selector");
        std::cout<<"PASS comparator "<<name<<'\n'; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
