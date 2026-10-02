#include "ragengine/brute_force_index.hpp"

#include <catch2/catch_test_macros.hpp>

using ragengine::BruteForceIndex;
using ragengine::Document;

namespace {

BruteForceIndex make_index_with_three_topics() {
    BruteForceIndex index;
    index.add(Document(1, "animal doc", {1.0f, 0.0f, 0.0f}));
    index.add(Document(2, "programming doc", {0.0f, 1.0f, 0.0f}));
    index.add(Document(3, "cooking doc", {0.0f, 0.0f, 1.0f}));
    return index;
}

} // namespace

TEST_CASE("size reflects number of added documents", "[brute_force_index]") {
    BruteForceIndex index;
    REQUIRE(index.size() == 0);
    index.add(Document(1, "doc", {1.0f}));
    REQUIRE(index.size() == 1);
}

TEST_CASE("search returns the closest document first", "[brute_force_index]") {
    BruteForceIndex index = make_index_with_three_topics();

    auto results = index.search({0.0f, 1.0f, 0.0f}, 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].document->id() == 2);
    REQUIRE(results[0].score > 0.99);
}

TEST_CASE("search respects top_k", "[brute_force_index]") {
    BruteForceIndex index = make_index_with_three_topics();

    auto results = index.search({1.0f, 0.0f, 0.0f}, 2);

    REQUIRE(results.size() == 2);
}

TEST_CASE("top_k larger than the index returns every document", "[brute_force_index]") {
    BruteForceIndex index = make_index_with_three_topics();

    auto results = index.search({1.0f, 0.0f, 0.0f}, 100);

    REQUIRE(results.size() == 3);
}

TEST_CASE("search on an empty index returns no results", "[brute_force_index]") {
    BruteForceIndex index;

    auto results = index.search({1.0f, 0.0f}, 5);

    REQUIRE(results.empty());
}

TEST_CASE("results are sorted by descending score", "[brute_force_index]") {
    BruteForceIndex index = make_index_with_three_topics();

    auto results = index.search({0.6f, 0.5f, 0.0f}, 3);

    REQUIRE(results.size() == 3);
    REQUIRE(results[0].score >= results[1].score);
    REQUIRE(results[1].score >= results[2].score);
}
