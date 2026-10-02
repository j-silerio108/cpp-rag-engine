#include "ragengine/csv_loader.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using ragengine::load_documents_csv;

namespace {
std::filesystem::path test_data(const char* filename) {
    return std::filesystem::path(RAGENGINE_TEST_DATA_DIR) / filename;
}
} // namespace

TEST_CASE("loads a well-formed csv, including a quoted comma", "[csv_loader]") {
    auto docs = load_documents_csv(test_data("sample.csv"));

    REQUIRE(docs.size() == 2);

    REQUIRE(docs[0].id() == 1);
    REQUIRE(docs[0].text() == "first document");
    REQUIRE(docs[0].embedding() == std::vector<float>{1.0f, 0.0f, 0.0f});

    REQUIRE(docs[1].id() == 2);
    REQUIRE(docs[1].text() == "second document, with a comma");
    REQUIRE(docs[1].embedding() == std::vector<float>{0.0f, 1.0f, 0.0f});
}

TEST_CASE("missing file throws", "[csv_loader]") {
    REQUIRE_THROWS_AS(load_documents_csv(test_data("does_not_exist.csv")), std::runtime_error);
}
