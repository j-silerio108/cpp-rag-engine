#include "ragengine/brute_force_index.hpp"
#include "ragengine/csv_loader.hpp"
#include "ragengine/index.hpp"

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
              << " <documents.csv> <query_embedding e.g. 0.1|0.2|0.3> [top_k=3]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string csv_path = argv[1];
    const std::vector<float> query = parse_query(argv[2]);
    const std::size_t top_k = argc >= 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 3;

    std::unique_ptr<ragengine::Index> index = std::make_unique<ragengine::BruteForceIndex>();

    try {
        for (ragengine::Document& doc : ragengine::load_documents_csv(csv_path)) {
            index->add(std::move(doc));
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load documents: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Loaded " << index->size() << " documents from " << csv_path << "\n\n";

    std::vector<ragengine::ScoredDocument> results;
    try {
        results = index->search(query, top_k);
    } catch (const std::exception& e) {
        std::cerr << "Search failed: " << e.what() << '\n';
        return 1;
    }

    std::cout << "Top " << results.size() << " results:\n";
    for (const ragengine::ScoredDocument& result : results) {
        std::cout << "  [" << result.score << "] (id=" << result.document->id() << ") "
                  << result.document->text() << '\n';
    }

    return 0;
}
