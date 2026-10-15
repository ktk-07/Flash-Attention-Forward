#include "attention_reference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace attention {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Shape size overflows size_t");
    return a * b;
}
}
Result reference(const Input& input, bool causal) {
    const auto [B, H, N, d] = input.shape;
    if (!B || !H || !N || !d) throw std::invalid_argument("Dimensions must be positive");
    const auto heads = multiply(B, H);
    const auto elements = multiply(multiply(heads, N), d);
    const auto matrix_elements = multiply(multiply(heads, N), N);
    for (const auto* tensor : {&input.q, &input.k, &input.v}) {
        if (tensor->size() != elements) throw std::invalid_argument("Input size mismatch");
        for (float x : *tensor)
            if (!std::isfinite(x)) throw std::invalid_argument("Inputs must be finite");
    }
    Result result{std::vector<double>(matrix_elements),
                  std::vector<double>(matrix_elements), std::vector<double>(elements)};
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    // Independent batch/head indexing
    for (std::size_t bh = 0; bh < heads; ++bh) {
        const auto tensor_base = bh * N * d;
        const auto matrix_base = bh * N * N;
        // Scaled QK^T reduction over r
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                double score = 0.0;
                for (std::size_t r = 0; r < d; ++r)
                    score += static_cast<double>(input.q[tensor_base + i*d + r])
                           * static_cast<double>(input.k[tensor_base + j*d + r]);
                result.scores[matrix_base + i*N + j] = causal && j > i
                    ? -std::numeric_limits<double>::infinity() : score * scale;
            }
        }
        // Stable row-wise softmax
        for (std::size_t i = 0; i < N; ++i) {
            const auto row = matrix_base + i*N;
            const auto valid_keys = causal ? i + 1 : N;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < valid_keys; ++j)
                maximum = std::max(maximum, result.scores[row+j]);
            double sum = 0.0;
            for (std::size_t j = 0; j < valid_keys; ++j) {
                result.probabilities[row+j] = std::exp(result.scores[row+j] - maximum);
                sum += result.probabilities[row+j];
            }
            for (std::size_t j = 0; j < valid_keys; ++j)
                result.probabilities[row+j] /= sum;
            // Masked P remains zero
        }
        // PV reduction over j
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t r = 0; r < d; ++r) {
                double value = 0.0;
                for (std::size_t j = 0; j < N; ++j)
                    value += result.probabilities[matrix_base+i*N+j]
                           * static_cast<double>(input.v[tensor_base+j*d+r]);
                result.output[tensor_base+i*d+r] = value;
            }
    }
    return result;
}
} // namespace attention
