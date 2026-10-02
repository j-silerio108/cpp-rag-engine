#pragma once

#include "ragengine/document.hpp"

#include <filesystem>
#include <vector>

namespace ragengine {

// Loads documents from a CSV with columns: id,text,embedding
// - id: integer
// - text: the document body; quote it ("...") if it contains commas
// - embedding: pipe-separated floats, e.g. 0.12|0.87|-0.4
//
// Throws std::runtime_error on a malformed file (missing file, bad row
// shape, non-numeric embedding value) rather than skipping bad rows
// silently — a corrupt dataset should fail loudly, not degrade search
// quality invisibly.
std::vector<Document> load_documents_csv(const std::filesystem::path& path);

} // namespace ragengine
