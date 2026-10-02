#include "ragengine/brute_force_index.hpp"
#include "ragengine/csv_loader.hpp"
#include "ragengine/index.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> parse_query(const std::string& arg) {
    std::vector<float> query;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, '|')) {
        query.push_back(std::stof(token));
    }
    return query;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " <documents.csv> <query_embedding e.g. 0.1|0.2|0.3> [top_k=3] [--json]\n";
}

// Escapes a string for embedding in a JSON string literal. Document text is
// free-form user content, so quotes, backslashes, and control characters all
// need handling, not just the common cases.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void print_results_json(const std::vector<ragengine::ScoredDocument>& results) {
    std::cout << '[';
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << "{\"id\":" << results[i].document->id() << ",\"score\":" << results[i].score
                   << ",\"text\":\"" << json_escape(results[i].document->text()) << "\"}";
    }
    std::cout << "]\n";
}

void print_results_human(const std::vector<ragengine::ScoredDocument>& results) {
    std::cout << "Top " << results.size() << " results:\n";
    for (const ragengine::ScoredDocument& result : results) {
        std::cout << "  [" << result.score << "] (id=" << result.document->id() << ") "
                  << result.document->text() << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    bool json_output = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            json_output = true;
        } else {
            positional.push_back(std::move(arg));
        }
    }

    if (positional.size() < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string& csv_path = positional[0];
    const std::vector<float> query = parse_query(positional[1]);
    const std::size_t top_k = positional.size() >= 3 ? static_cast<std::size_t>(std::stoul(positional[2])) : 3;

    std::unique_ptr<ragengine::Index> index = std::make_unique<ragengine::BruteForceIndex>();

    try {
        for (ragengine::Document& doc : ragengine::load_documents_csv(csv_path)) {
            index->add(std::move(doc));
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load documents: " << e.what() << '\n';
        return 1;
    }

    if (!json_output) {
        std::cout << "Loaded " << index->size() << " documents from " << csv_path << "\n\n";
    }

    std::vector<ragengine::ScoredDocument> results;
    try {
        results = index->search(query, top_k);
    } catch (const std::exception& e) {
        std::cerr << "Search failed: " << e.what() << '\n';
        return 1;
    }

    if (json_output) {
        print_results_json(results);
    } else {
        print_results_human(results);
    }

    return 0;
}
