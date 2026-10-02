#include "ragengine/csv_loader.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ragengine {

namespace {

// Splits one CSV line into fields, honoring double-quoted fields that may
// contain commas and "" as an escaped quote.
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                field += c;
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

std::vector<float> parse_embedding(const std::string& field) {
    std::vector<float> embedding;
    std::stringstream ss(field);
    std::string token;
    while (std::getline(ss, token, '|')) {
        float value{};
        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || ptr != token.data() + token.size()) {
            throw std::runtime_error("load_documents_csv: invalid embedding value '" + token + "'");
        }
        embedding.push_back(value);
    }
    return embedding;
}

} // namespace

std::vector<Document> load_documents_csv(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("load_documents_csv: cannot open '" + path.string() + "'");
    }

    std::vector<Document> documents;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 3) {
            throw std::runtime_error("load_documents_csv: line " + std::to_string(line_number) +
                                      " expected 3 fields, got " + std::to_string(fields.size()));
        }

        std::size_t id{};
        auto [ptr, ec] = std::from_chars(fields[0].data(), fields[0].data() + fields[0].size(), id);
        if (ec != std::errc{} || ptr != fields[0].data() + fields[0].size()) {
            throw std::runtime_error("load_documents_csv: line " + std::to_string(line_number) +
                                      " has a non-numeric id");
        }

        documents.emplace_back(id, fields[1], parse_embedding(fields[2]));
    }

    return documents;
}

} // namespace ragengine
