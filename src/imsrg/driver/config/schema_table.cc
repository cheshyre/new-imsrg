// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The key and table lists derived from visit_schema (the single schema table).

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/driver/config/schema_keys.h"

namespace imsrg::driver {
namespace {

/// Collects the KeyInfo of every key, optional tables included.
class KeyCollector {
 public:
  KeyCollector() = default;
  KeyCollector(const KeyCollector&) = default;
  KeyCollector& operator=(const KeyCollector&) = default;
  KeyCollector(KeyCollector&&) = default;
  KeyCollector& operator=(KeyCollector&&) = default;

  /// Moves the collected keys out (visiting order).
  std::vector<KeyInfo> TakeKeys() { return std::move(keys_); }

  /// Visitor hook: records one schema field.
  template <class T>
  void Field(const KeyInfo& key, const T& /*value*/) {
    keys_.push_back(key);
  }

  /// Visitor hook: visits the body of an optional table (default-constructed).
  template <class Section, class Body>
  void OptionalTable(std::string_view /*table*/,
                     const std::optional<Section>& /*section*/, Body&& body) {
    const Section tmp{};
    body(tmp);
  }

 private:
  std::vector<KeyInfo> keys_;
};

/// The immutable key and table lists, built once from the schema.
class SchemaLists {
 public:
  SchemaLists() {
    KeyCollector collector;
    const RunConfig defaults{};
    VisitSchema(defaults, collector);
    keys_ = collector.TakeKeys();
    for (const KeyInfo& key : keys_) {
      // Every proper prefix of the path is a table ("imsrg",
      // "imsrg.generator").
      std::string_view path = key.path;
      for (std::size_t pos = path.find('.'); pos != std::string_view::npos;
           pos = path.find('.', pos + 1)) {
        std::string table(path.substr(0, pos));
        if (std::find(table_storage_.begin(), table_storage_.end(), table) ==
            table_storage_.end()) {
          table_storage_.push_back(std::move(table));
        }
      }
    }
    tables_.assign(table_storage_.begin(), table_storage_.end());
  }

  // `tables_` views `table_storage_`; copying or moving would dangle.
  SchemaLists(const SchemaLists&) = delete;
  SchemaLists& operator=(const SchemaLists&) = delete;
  SchemaLists(SchemaLists&&) = delete;
  SchemaLists& operator=(SchemaLists&&) = delete;

  /// Every schema key, optional tables included.
  const std::vector<KeyInfo>& Keys() const noexcept { return keys_; }
  /// Every table path ("imsrg", "imsrg.generator", ...).
  const std::vector<std::string_view>& Tables() const noexcept {
    return tables_;
  }

 private:
  std::vector<KeyInfo> keys_;
  std::vector<std::string> table_storage_;
  std::vector<std::string_view> tables_;
};

const SchemaLists& Lists() {
  static const auto* const kInstance = new SchemaLists();
  return *kInstance;
}

}  // namespace

std::span<const KeyInfo> SchemaKeys() { return Lists().Keys(); }

const KeyInfo* FindSchemaKey(std::string_view path) {
  for (const KeyInfo& key : Lists().Keys()) {
    if (key.path == path) return &key;
  }
  return nullptr;
}

std::span<const std::string_view> SchemaTables() { return Lists().Tables(); }

std::vector<std::string_view> SchemaChildren(std::string_view table) {
  std::vector<std::string_view> children;
  const std::string prefix =
      table.empty() ? std::string() : std::string(table) + ".";
  auto add_child = [&](std::string_view path) {
    if (!path.starts_with(prefix)) return;
    std::string_view rest = path.substr(prefix.size());
    if (rest.empty()) return;
    rest = rest.substr(0, rest.find('.'));
    if (std::find(children.begin(), children.end(), rest) == children.end()) {
      children.push_back(rest);
    }
  };
  for (const KeyInfo& key : Lists().Keys()) add_child(key.path);
  return children;
}

}  // namespace imsrg::driver
