#include "ragengine/similarity.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using ragengine::cosine_similarity;
using Catch::Approx;

TEST_CASE("identical vectors have similarity 1", "[similarity]") {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    REQUIRE(cosine_similarity(a, a) == Approx(1.0));
}

TEST_CASE("orthogonal vectors have similarity 0", "[similarity]") {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f};
    REQUIRE(cosine_similarity(a, b) == Approx(0.0).margin(1e-9));
}

TEST_CASE("opposite vectors have similarity -1", "[similarity]") {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{-1.0f, 0.0f};
    REQUIRE(cosine_similarity(a, b) == Approx(-1.0));
}

TEST_CASE("scale does not affect cosine similarity", "[similarity]") {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{2.0f, 4.0f, 6.0f};
    REQUIRE(cosine_similarity(a, b) == Approx(1.0));
}

TEST_CASE("mismatched sizes throw", "[similarity]") {
    std::vector<float> a{1.0f, 2.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};
    REQUIRE_THROWS_AS(cosine_similarity(a, b), std::invalid_argument);
}

TEST_CASE("zero-magnitude vector throws", "[similarity]") {
    std::vector<float> a{0.0f, 0.0f};
    std::vector<float> b{1.0f, 1.0f};
    REQUIRE_THROWS_AS(cosine_similarity(a, b), std::invalid_argument);
}
