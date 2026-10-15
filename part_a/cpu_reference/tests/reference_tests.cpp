#include "attention_reference.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using attention::Input;
void require(bool condition) { if (!condition) throw std::runtime_error("Assertion failed"); }
void near(double actual, double expected, double tolerance=1e-12) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance)
        throw std::runtime_error("Numerical mismatch: " + std::to_string(actual));
}
Input zeros(attention::Shape s) {
    const auto n=s.B*s.H*s.N*s.d;
    return {s,std::vector<float>(n),std::vector<float>(n),std::vector<float>(n)};
}
void test(const std::string& name) {
    if(name=="tiny") {
        Input x{{1,1,2,2},{1,0,0,1},{1,0,0,1},{10,0,0,20}};
        auto y=attention::reference(x);
        near(y.scores[0],1/std::sqrt(2.0)); near(y.scores[1],0);
        near(y.probabilities[0],0.6697615493266569);
        near(y.output[0],6.697615493266569); near(y.output[1],6.604769013466862);
        near(y.output[2],3.302384506733431); near(y.output[3],13.395230986533138);
        y=attention::reference(x,true);
        near(y.output[0],10); near(y.output[1],0); near(y.probabilities[1],0);
        require(std::isinf(y.scores[1]) && y.scores[1]<0);
    } else if(name=="singleton") {
        Input x{{1,1,1,3},{2,-3,4},{-1,2,5},{3,-7,9}};
        for(bool causal:{false,true}) {
            auto y=attention::reference(x,causal); near(y.probabilities[0],1);
            for(int r=0;r<3;++r) near(y.output[r],x.v[r]);
        }
    } else if(name=="uniform") {
        Input x{{1,1,3,1},{0,0,0},{5,-2,7},{3,6,12}};
        auto y=attention::reference(x);
        for(double p:y.probabilities) near(p,1.0/3);
        for(double v:y.output) near(v,7);
        y=attention::reference(x,true);
        near(y.output[0],3); near(y.output[1],4.5); near(y.output[2],7);
    } else if(name=="stability") {
        Input x{{1,1,2,1},{1000,-1000},{1000,-1000},{2,-3}};
        auto y=attention::reference(x); near(y.output[0],2); near(y.output[1],-3);
        // Check FP64 cast before multiplication
        x.q={1e20f,1e20f}; x.k={1e20f,1e20f};
        y=attention::reference(x); near(y.probabilities[0],0.5); near(y.output[0],-0.5);
    } else if(name=="isolation") {
        auto x=zeros({2,3,3,2});
        for(std::size_t h=0;h<6;++h)
            for(std::size_t i=0;i<3;++i) {
                x.v[h*6+i*2]=static_cast<float>(10*h+1);
                x.v[h*6+i*2+1]=static_cast<float>(-static_cast<int>(h)-2);
            }
        for(bool causal:{false,true}) {
            auto y=attention::reference(x,causal);
            for(std::size_t k=0;k<x.v.size();++k) near(y.output[k],x.v[k]);
        }
    } else if(name=="causal_prefix") {
        Input x{{1,1,3,2},{1,-2,3,1,-1,4},{-2,1,2,3,1,-3},{3,1,-2,4,5,-1}};
        auto before=attention::reference(x,true);
        x.k[4]=100; x.k[5]=-100; x.v[4]=999; x.v[5]=-999;
        auto after=attention::reference(x,true);
        for(int k=0;k<4;++k) near(after.output[k],before.output[k]);
    } else if(name=="irregular") {
        for(auto s:{attention::Shape{1,1,33,7},attention::Shape{2,2,65,33},attention::Shape{1,2,129,65}}) {
            auto x=zeros(s);
            for(std::size_t k=0;k<x.q.size();++k) {
                x.q[k]=static_cast<float>(static_cast<int>(k%13)-6)/7;
                x.k[k]=static_cast<float>(static_cast<int>(k%17)-8)/9;
                // Constant-V output
                x.v[k]=static_cast<float>(k%s.d)-3;
            }
            for(bool causal:{false,true}) {
                auto y=attention::reference(x,causal);
                for(std::size_t k=0;k<y.output.size();++k) near(y.output[k],x.v[k],1e-10);
                for(std::size_t row=0;row<s.B*s.H*s.N;++row) {
                    double sum=0;
                    for(std::size_t j=0;j<s.N;++j) {
                        const auto p=y.probabilities[row*s.N+j];
                        require(std::isfinite(p) && p>=0 && p<=1); sum+=p;
                        if(causal && j>row%s.N) near(p,0);
                    }
                    near(sum,1,1e-12);
                }
            }
        }
    } else if(name=="validation") {
        auto rejects=[](const Input& x) {
            bool caught=false;
            try { (void)attention::reference(x); } catch(const std::invalid_argument&) { caught=true; }
            require(caught);
        };
        rejects(zeros({1,1,0,1}));
        auto x=zeros({1,1,1,1}); x.q.clear(); rejects(x);
        x=zeros({1,1,1,1}); x.k[0]=std::numeric_limits<float>::quiet_NaN(); rejects(x);
        x=zeros({1,1,1,1}); x.v[0]=std::numeric_limits<float>::infinity(); rejects(x);
        rejects({{std::numeric_limits<std::size_t>::max(),2,1,1},{},{},{}});
    } else throw std::runtime_error("Unknown test selector");
}
int main(int argc,char** argv) {
    try { require(argc==2); test(argv[1]); std::cout<<"PASS "<<argv[1]<<'\n'; return 0; }
    catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
