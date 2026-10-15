#include "attention_reference.hpp"
#include <iomanip>
#include <iostream>
int main() {
    attention::Input input{{1,1,2,2}, {1,0,0,1}, {1,0,0,1}, {10,0,0,20}};
    for (bool causal : {false, true}) {
        const auto result = attention::reference(input, causal);
        std::cout << (causal ? "Causal" : "Dense") << " attention\n" << std::setprecision(12);
        const char* labels[] = {"S: scaled scores (after masking)", "P: row-wise probabilities", "O: weighted values"};
        int stage = 0;
        for (const auto* matrix : {&result.scores, &result.probabilities, &result.output}) {
            std::cout << labels[stage++] << '\n';
            for (std::size_t i=0; i<2; ++i)
                std::cout << (*matrix)[i*2] << "  " << (*matrix)[i*2+1] << '\n';
            std::cout << '\n';
        }
    }
}
