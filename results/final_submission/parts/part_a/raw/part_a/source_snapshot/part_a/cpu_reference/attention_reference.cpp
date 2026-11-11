#include "attention_types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace attention {
Result reference(const Input& input, bool causal) {
    const auto [B, H, N, d] = input.shape;
    const auto heads = B * H;
    const auto elements = heads * N * d;
    const auto matrix_elements = heads * N * N;

    Result result{std::vector<double>(matrix_elements),
                  std::vector<double>(matrix_elements), std::vector<double>(elements)};
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));

    // batch and head offsets
    for (std::size_t bh = 0; bh < heads; ++bh) {
        const auto tensor_base = bh * N * d;
        const auto matrix_base = bh * N * N;

        // QK^T over r
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

        // Row softmax
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

        // PV over j
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
