#pragma once

#include "ragengine/index.hpp"

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace ragengine {

// Approximate index using random-projection locality-sensitive hashing
// (SimHash-style): each of num_tables hash tables projects an embedding
// onto num_hyperplanes_per_table random hyperplanes and records which side
// of each one it falls on, packed into a bitmask. Vectors that are close in
// cosine-similarity terms tend to land in the same bucket, so a query only
// has to be scored exactly against the (usually much smaller) union of
// candidates from matching buckets across all tables, instead of every
// stored document.
//
// More hyperplanes per table -> more selective buckets (fewer candidates,
// higher chance of missing a true neighbor). More tables -> more chances to
// catch a true neighbor (higher recall, more memory/hashing work). If a
// query's candidate set is smaller than top_k, this falls back to scoring
// every document exactly, so a result is never truncated just because the
// hashing got unlucky -- the same tradeoff real ANN libraries make.
class LshIndex final : public Index {
public:
    explicit LshIndex(std::size_t num_tables = 8, std::size_t num_hyperplanes_per_table = 8,
                       std::uint64_t seed = std::random_device{}());

    void add(Document document) override;
    std::vector<ScoredDocument> search(const std::vector<float>& query,
                                        std::size_t top_k) const override;
    std::size_t size() const noexcept override;

private:
    struct HashTable {
        std::vector<std::vector<float>> hyperplanes; // num_hyperplanes_per_table x dimensions
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
    };

    std::uint64_t hash(const HashTable& table, const std::vector<float>& embedding) const;
    std::vector<ScoredDocument> score_candidates(const std::vector<float>& query,
                                                  const std::vector<std::size_t>& candidate_ids,
                                                  std::size_t top_k) const;

    std::size_t num_hyperplanes_per_table_;
    std::size_t dimensions_ = 0;
    std::mt19937_64 rng_;
    std::vector<HashTable> tables_;
    std::vector<Document> documents_;
};

} // namespace ragengine
