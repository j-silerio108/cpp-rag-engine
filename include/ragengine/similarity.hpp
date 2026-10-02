#pragma once

#include <cmath>
#include <concepts>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace ragengine {

// Constrains similarity functions to any random-access range of
// floating-point values, not just std::vector<float> specifically.
template <typename Vec>
concept FloatVector =
    std::ranges::random_access_range<Vec> &&
    std::floating_point<std::ranges::range_value_t<Vec>>;

// Cosine similarity in [-1, 1]; 1 means identical direction (same meaning),
// 0 means unrelated, -1 means opposite. Vectors of mismatched length or a
// zero vector are a caller error, not a value to silently coerce to 0.
template <FloatVector Vec>
double cosine_similarity(const Vec& a, const Vec& b) {
    if (std::ranges::size(a) != std::ranges::size(b)) {
        throw std::invalid_argument("cosine_similarity: vector size mismatch");
    }

    double dot = std::inner_product(std::ranges::begin(a), std::ranges::end(a),
                                     std::ranges::begin(b), 0.0);
    double norm_a = std::sqrt(std::inner_product(std::ranges::begin(a), std::ranges::end(a),
                                                   std::ranges::begin(a), 0.0));
    double norm_b = std::sqrt(std::inner_product(std::ranges::begin(b), std::ranges::end(b),
                                                   std::ranges::begin(b), 0.0));

    if (norm_a == 0.0 || norm_b == 0.0) {
        throw std::invalid_argument("cosine_similarity: zero-magnitude vector");
    }

    return dot / (norm_a * norm_b);
}

} // namespace ragengine
