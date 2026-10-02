#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ragengine {

// A single retrievable item: its source text plus the embedding vector
// that represents it in similarity space.
class Document {
public:
    Document(std::size_t id, std::string text, std::vector<float> embedding);

    std::size_t id() const noexcept;
    const std::string& text() const noexcept;
    const std::vector<float>& embedding() const noexcept;

private:
    std::size_t id_;
    std::string text_;
    std::vector<float> embedding_;
};

} // namespace ragengine
