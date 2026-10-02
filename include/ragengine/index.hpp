#pragma once

#include "ragengine/document.hpp"

#include <cstddef>
#include <vector>

namespace ragengine {

struct ScoredDocument {
    const Document* document;
    double score;
};

// Interface for anything that can store documents and retrieve the most
// similar ones to a query embedding. BruteForceIndex is the only
// implementation today; the interface exists so a smarter index (e.g. an
// approximate-nearest-neighbor structure) can be swapped in later without
// touching callers.
class Index {
public:
    virtual ~Index() = default;

    virtual void add(Document document) = 0;
    virtual std::vector<ScoredDocument> search(const std::vector<float>& query,
                                                std::size_t top_k) const = 0;
    virtual std::size_t size() const noexcept = 0;
};

} // namespace ragengine
