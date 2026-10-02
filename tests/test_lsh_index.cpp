#include "ragengine/lsh_index.hpp"

#include <catch2/catch_test_macros.hpp>

using ragengine::Document;
using ragengine::LshIndex;

namespace {

LshIndex make_index_with_three_topics(std::size_t num_tables = 4, std::size_t num_hyperplanes = 2,
                                       std::uint64_t seed = 42) {
    LshIndex index(num_tables, num_hyperplanes, seed);
    index.add(Document(1, "animal doc", {1.0f, 0.0f, 0.0f}));
    index.add(Document(2, "programming doc", {0.0f, 1.0f, 0.0f}));
    index.add(Document(3, "cooking doc", {0.0f, 0.0f, 1.0f}));
    return index;
}

} // namespace

TEST_CASE("size reflects number of added documents", "[lsh_index]") {
    LshIndex index;
    REQUIRE(index.size() == 0);
    index.add(Document(1, "doc", {1.0f}));
    REQUIRE(index.size() == 1);
}

TEST_CASE("search on an empty index returns no results", "[lsh_index]") {
    LshIndex index;
    REQUIRE(index.search({1.0f, 0.0f}, 5).empty());
}

TEST_CASE("search finds the closest document, with a fixed seed", "[lsh_index]") {
    LshIndex index = make_index_with_three_topics();

    auto results = index.search({0.0f, 1.0f, 0.0f}, 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].document->id() == 2);
    REQUIRE(results[0].score > 0.99);
}

TEST_CASE("falls back to exact search when a bucket doesn't have enough candidates", "[lsh_index]") {
    // 1 table, many hyperplanes relative to 3 documents: buckets are so
    // selective that the matching bucket is extremely unlikely to hold all
    // 3 documents, forcing the exact fallback for top_k == size().
    LshIndex index = make_index_with_three_topics(/*num_tables=*/1, /*num_hyperplanes=*/16);

    auto results = index.search({0.6f, 0.5f, 0.0f}, 3);

    REQUIRE(results.size() == 3);
    REQUIRE(results[0].score >= results[1].score);
    REQUIRE(results[1].score >= results[2].score);
}

TEST_CASE("rejects a second document with a different embedding size", "[lsh_index]") {
    LshIndex index;
    index.add(Document(1, "first", {1.0f, 0.0f}));
    REQUIRE_THROWS_AS(index.add(Document(2, "second", {1.0f, 0.0f, 0.0f})), std::invalid_argument);
}

TEST_CASE("rejects a query with the wrong embedding size", "[lsh_index]") {
    LshIndex index;
    index.add(Document(1, "first", {1.0f, 0.0f}));
    REQUIRE_THROWS_AS(index.search({1.0f, 0.0f, 0.0f}, 1), std::invalid_argument);
}

TEST_CASE("rejects invalid construction parameters", "[lsh_index]") {
    REQUIRE_THROWS_AS(LshIndex(0, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(LshIndex(4, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(LshIndex(4, 65), std::invalid_argument);
}
