#include "ragengine/lsh_index.hpp"
#include "ragengine/similarity.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace ragengine {

LshIndex::LshIndex(std::size_t num_tables, std::size_t num_hyperplanes_per_table, std::uint64_t seed)
    : num_hyperplanes_per_table_(num_hyperplanes_per_table), rng_(seed), tables_(num_tables) {
    if (num_hyperplanes_per_table_ == 0 || num_hyperplanes_per_table_ > 64) {
        throw std::invalid_argument("LshIndex: num_hyperplanes_per_table must be in [1, 64]");
    }
    if (num_tables == 0) {
        throw std::invalid_argument("LshIndex: num_tables must be at least 1");
    }
}

void LshIndex::add(Document document) {
    if (dimensions_ == 0) {
        dimensions_ = document.embedding().size();
        std::normal_distribution<float> gaussian(0.0f, 1.0f);
        for (HashTable& table : tables_) {
            table.hyperplanes.reserve(num_hyperplanes_per_table_);
            for (std::size_t i = 0; i < num_hyperplanes_per_table_; ++i) {
                std::vector<float> plane(dimensions_);
                for (float& component : plane) {
                    component = gaussian(rng_);
                }
                table.hyperplanes.push_back(std::move(plane));
            }
        }
    } else if (document.embedding().size() != dimensions_) {
        throw std::invalid_argument("LshIndex: embedding dimension mismatch");
    }

    const std::size_t doc_id = documents_.size();
    for (HashTable& table : tables_) {
        table.buckets[hash(table, document.embedding())].push_back(doc_id);
    }
    documents_.push_back(std::move(document));
}

std::uint64_t LshIndex::hash(const HashTable& table, const std::vector<float>& embedding) const {
    std::uint64_t signature = 0;
    for (std::size_t i = 0; i < table.hyperplanes.size(); ++i) {
        float dot = std::inner_product(embedding.begin(), embedding.end(), table.hyperplanes[i].begin(), 0.0f);
        if (dot >= 0.0f) {
            signature |= (std::uint64_t{1} << i);
        }
    }
    return signature;
}

std::vector<ScoredDocument> LshIndex::score_candidates(const std::vector<float>& query,
                                                         const std::vector<std::size_t>& candidate_ids,
                                                         std::size_t top_k) const {
    std::vector<ScoredDocument> scored;
    scored.reserve(candidate_ids.size());
    for (std::size_t id : candidate_ids) {
        scored.push_back(
            ScoredDocument{&documents_[id], cosine_similarity(query, documents_[id].embedding())});
    }
    const std::size_t k = std::min(top_k, scored.size());
    std::ranges::partial_sort(scored, scored.begin() + static_cast<std::ptrdiff_t>(k),
                               std::ranges::greater{}, &ScoredDocument::score);
    scored.resize(k);
    return scored;
}

std::vector<ScoredDocument> LshIndex::search(const std::vector<float>& query, std::size_t top_k) const {
    if (documents_.empty()) {
        return {};
    }
    if (query.size() != dimensions_) {
        throw std::invalid_argument("LshIndex: query dimension mismatch");
    }

    std::unordered_set<std::size_t> candidates;
    for (const HashTable& table : tables_) {
        std::uint64_t key = hash(table, query);
        auto it = table.buckets.find(key);
        if (it != table.buckets.end()) {
            candidates.insert(it->second.begin(), it->second.end());
        }
    }

    if (candidates.size() < top_k) {
        // Not enough approximate candidates to satisfy the request -- fall
        // back to scoring every document exactly rather than silently
        // returning a truncated or lower-quality result.
        std::vector<std::size_t> all_ids(documents_.size());
        std::iota(all_ids.begin(), all_ids.end(), 0);
        return score_candidates(query, all_ids, top_k);
    }

    std::vector<std::size_t> candidate_ids(candidates.begin(), candidates.end());
    return score_candidates(query, candidate_ids, top_k);
}

std::size_t LshIndex::size() const noexcept { return documents_.size(); }

} // namespace ragengine
