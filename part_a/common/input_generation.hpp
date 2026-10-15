#pragma once
#include "attention_types.hpp"
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

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

// Deterministic Q/K/V generation
inline Input synthetic_input(Shape shape, std::uint64_t seed = 5206) {
    const auto count = input_elements(shape);
    Input input{shape, std::vector<float>(count), std::vector<float>(count), std::vector<float>(count)};
    std::uint64_t state = seed;
    for (auto* tensor : {&input.q, &input.k, &input.v}) {
        for (float& value : *tensor) {
            std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
            z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
            z ^= z >> 31;
            const auto integer = static_cast<std::int32_t>(z >> 40) - 8388608;
            value = static_cast<float>(integer) / 8388608.0f; // [-1,1),step=2^-23
        }
    }
    return input;
}

// FNV-1a input fingerprint
inline std::uint64_t input_fingerprint(const Input& input) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "The shared contract requires IEEE-754 FP32");
    const auto count = input_elements(input.shape);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    auto feed = [&](std::uint64_t word, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) {
            hash ^= (word >> (8 * i)) & 255;
            hash *= UINT64_C(1099511628211);
        }
    };
    for (auto dim : {input.shape.B, input.shape.H, input.shape.N, input.shape.d}) feed(dim, 8);
    for (const auto* tensor : {&input.q, &input.k, &input.v}) {
        if (tensor->size() != count) throw std::invalid_argument("Input size mismatch");
        for (float value : *tensor) {
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            feed(bits, 4);
        }
    }
    return hash;
}
}
