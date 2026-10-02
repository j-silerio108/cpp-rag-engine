#include "ragengine/brute_force_index.hpp"
#include "ragengine/similarity.hpp"

#include <algorithm>
#include <utility>

namespace ragengine {

void BruteForceIndex::add(Document document) {
    documents_.push_back(std::move(document));
}

std::vector<ScoredDocument> BruteForceIndex::search(const std::vector<float>& query,
                                                      std::size_t top_k) const {
    std::vector<ScoredDocument> scored;
    scored.reserve(documents_.size());

    for (const Document& doc : documents_) {
        scored.push_back(ScoredDocument{&doc, cosine_similarity(query, doc.embedding())});
    }

    const std::size_t k = std::min(top_k, scored.size());
    std::ranges::partial_sort(scored, scored.begin() + static_cast<std::ptrdiff_t>(k),
                               std::ranges::greater{}, &ScoredDocument::score);
    scored.resize(k);
    return scored;
}

std::size_t BruteForceIndex::size() const noexcept { return documents_.size(); }

} // namespace ragengine
