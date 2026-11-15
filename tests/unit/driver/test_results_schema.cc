// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The `results.json` schema table of design/16 section 2.10 against the keys a
// real run writes (He4, emax 2: cheap), and the schema version.
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

#ifndef IMSRG_SOURCE_DIR
#error "IMSRG_SOURCE_DIR must be defined by the build"
#endif

namespace imsrg::it {
namespace {

/// One row of the documented table.
struct Row {
  std::string key;
  std::string present;
};

/// The rows of the table under "#### `results.json` schema" in design/16.
std::vector<Row> DocumentedRows() {
  std::ifstream in(std::string(IMSRG_SOURCE_DIR) +
                   "/design/16-config-and-cli.md");
  std::vector<Row> rows;
  std::string line;
  bool inside = false;
  while (std::getline(in, line)) {
    if (line.rfind("#### `results.json` schema", 0) == 0) {
      inside = true;
      continue;
    }
    if (!inside) continue;
    if (line.rfind("Changelog", 0) == 0) break;
    if (line.rfind("| `", 0) != 0) continue;
    // | `key` | type | present | meaning |
    std::vector<std::string> cells;
    std::istringstream ss(line);
    std::string cell;
    std::getline(ss, cell, '|');  // text before the first bar (empty)
    while (std::getline(ss, cell, '|')) cells.push_back(cell);
    if (cells.size() < 4) continue;
    const auto trim = [](std::string s) {
      const auto b = s.find_first_not_of(' ');
      const auto e = s.find_last_not_of(' ');
      return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
    };
    std::string key = trim(cells[0]);
    if (key.size() > 2 && key.front() == '`' && key.back() == '`')
      key = key.substr(1, key.size() - 2);
    rows.push_back({key, trim(cells[2])});
  }
  return rows;
}

/// Collects the key paths of `v`: object members joined with '.', array
/// elements as `[]`, the members of `operators` as `<name>`.
void Flatten(const Json& v, const std::string& path,
             std::set<std::string>& out) {
  if (!path.empty()) out.insert(path);
  if (v.kind == Json::Kind::kObject) {
    for (const auto& [name, child] : v.fields) {
      const std::string key = path == "operators" ? "<name>" : name;
      Flatten(child, path.empty() ? key : path + "." + key, out);
    }
  } else if (v.kind == Json::Kind::kArray) {
    for (const Json& child : v.items) {
      // The element itself is not a key; only its members are.
      if (child.kind == Json::Kind::kObject) {
        for (const auto& [name, grand] : child.fields)
          Flatten(grand, path + "[]." + name, out);
      }
    }
  }
}

std::string HeTomlText(const std::filesystem::path& dir, bool hf_only,
                       bool operators) {
  std::ostringstream t;
  t << "[run]\ntask = \"" << (hf_only ? "hf" : "imsrg") << "\"\n\n"
    << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n";
  if (operators) t << "[operators]\nlist = [\"Rp2\"]\n\n";
  t << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  return t.str();
}

TEST(ResultsSchema, WrittenKeysMatchTheDocumentedTable) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  std::set<std::string> documented;
  std::set<std::string> always;
  for (const Row& r : DocumentedRows()) {
    documented.insert(r.key);
    if (r.present == "always") always.insert(r.key);
  }
  ASSERT_GT(documented.size(), 20u) << "table not found in design/16";

  // Full run with an operator: every documented key is written.
  const auto dir = FreshDir("schema_full");
  driver::ConfigSources src;
  src.config_text = HeTomlText(dir, false, true);
  const RunResult full = Execute(src, dir);
  ASSERT_FALSE(full.planned) << full.note;
  std::set<std::string> written;
  Flatten(full.results, "", written);
  for (const std::string& k : written)
    EXPECT_TRUE(documented.count(k)) << "written but undocumented: " << k;
  for (const std::string& k : documented)
    EXPECT_TRUE(written.count(k)) << "documented but not written: " << k;

  // The version.
  ASSERT_TRUE(full.results.has("schema_version"));
  EXPECT_EQ(full.results["schema_version"].number, 1.0);

  // HF-only run: a subset of the table that includes every "always" key.
  const auto dir_hf = FreshDir("schema_hf");
  driver::ConfigSources hf_src;
  hf_src.config_text = HeTomlText(dir_hf, true, false);
  const RunResult hf = Execute(hf_src, dir_hf);
  std::set<std::string> written_hf;
  Flatten(hf.results, "", written_hf);
  for (const std::string& k : written_hf)
    EXPECT_TRUE(documented.count(k)) << "(hf run) undocumented: " << k;
  for (const std::string& k : always)
    EXPECT_TRUE(written_hf.count(k)) << "(hf run) missing 'always' key: " << k;
}

}  // namespace
}  // namespace imsrg::it
