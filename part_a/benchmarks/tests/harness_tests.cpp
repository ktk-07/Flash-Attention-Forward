#include "input_generation.hpp"
#include "statistics.hpp"
#include "experiment_plan.hpp"
#include <iostream>
#include <limits>
void require(bool ok) { if (!ok) throw std::runtime_error("Harness invariant failed"); }
int main() {
    try {
        const auto a = attention::synthetic_input({2,3,17,7},5206);
        const auto b = attention::synthetic_input({2,3,17,7},5206);
        const auto c = attention::synthetic_input({2,3,17,7},5207);
        require(a.q == b.q && a.k == b.k && a.v == b.v);
        require(a.q != a.k && a.k != a.v && a.q != c.q);
        const auto golden=attention::synthetic_input({1,1,2,2},5206);
        require(golden.q[0]==0.7879422903060913f && golden.q[1]==0.8089205026626587f);
        require(attention::input_fingerprint(golden)==UINT64_C(12197107921315666494));
        require(attention::input_fingerprint(a) == attention::input_fingerprint(b));
        require(attention::input_fingerprint(a) != attention::input_fingerprint(c));
        for (const auto* tensor : {&a.q,&a.k,&a.v})
            for (float value : *tensor) require(value >= -1 && value < 1 && std::isfinite(value));
        const auto stats=attention::distribution({4,1,3,2});
        require(stats.q1==1.75 && stats.median==2.5 && stats.q3==3.25 && stats.iqr()==1.5);
        require(attention::distribution({2}).median==2);
        attention::experiment::Options options;
        options.grid="proposal";
        require(attention::experiment::shapes(options).size()==15);
        require(attention::experiment::variants(options).size()==2);
        require(attention::experiment::modes(options).size()==2);
        options.configs="study";
        const auto candidates=attention::experiment::configs(options);
        require(candidates.size()==4);
        require(candidates[0].tile==8 && candidates[0].softmax_threads==256);
        require(candidates[3].tile==16 && candidates[3].softmax_threads==128);
        options.config_override=true;
        bool conflict=false;
        try { (void)attention::experiment::configs(options); } catch(const std::invalid_argument&) {conflict=true;}
        require(conflict);
        options.validate_only=true;
        require(attention::experiment::shapes(options).size()==4);
        bool rejected=false;
        try { (void)attention::synthetic_input({0,1,1,1}); } catch(const std::invalid_argument&) {rejected=true;}
        require(rejected); rejected=false;
        try { (void)attention::input_elements({std::numeric_limits<std::size_t>::max(),2,1,1}); }
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected);
        std::cout << "PASS shared inputs, shape overflow and quantiles\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
