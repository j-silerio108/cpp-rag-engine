#pragma once

#include "ragengine/index.hpp"

#include <vector>

namespace ragengine {

// Exhaustive O(n) search: scores every stored document against the query
// and returns the top-k by cosine similarity. Correct and simple; the
// baseline a smarter index would need to beat.
class BruteForceIndex final : public Index {
public:
    void add(Document document) override;
    std::vector<ScoredDocument> search(const std::vector<float>& query,
                                        std::size_t top_k) const override;
    std::size_t size() const noexcept override;

private:
    std::vector<Document> documents_;
};

} // namespace ragengine
