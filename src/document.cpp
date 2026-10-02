#include "ragengine/document.hpp"

#include <utility>

namespace ragengine {

Document::Document(std::size_t id, std::string text, std::vector<float> embedding)
    : id_(id), text_(std::move(text)), embedding_(std::move(embedding)) {}

std::size_t Document::id() const noexcept { return id_; }

const std::string& Document::text() const noexcept { return text_; }

const std::vector<float>& Document::embedding() const noexcept { return embedding_; }

} // namespace ragengine
