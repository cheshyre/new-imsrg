// The key and table lists derived from visit_schema (the single schema table).

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {
namespace {

/// Collects the KeyInfo of every key, optional tables included.
struct KeyCollector {
  std::vector<KeyInfo> keys;

  template <class T>
  void field(const KeyInfo& key, const T& /*value*/) {
    keys.push_back(key);
  }

  template <class Section, class Body>
  void optional_table(std::string_view /*table*/, const std::optional<Section>& /*section*/,
                      Body&& body) {
    const Section tmp{};
    body(tmp);
  }
};

struct SchemaLists {
  std::vector<KeyInfo> keys;
  std::vector<std::string> table_storage;
  std::vector<std::string_view> tables;

  SchemaLists() {
    KeyCollector collector;
    const RunConfig defaults{};
    visit_schema(defaults, collector);
    keys = std::move(collector.keys);
    for (const KeyInfo& key : keys) {
      // Every proper prefix of the path is a table ("imsrg", "imsrg.generator").
      std::string_view path = key.path;
      for (std::size_t pos = path.find('.'); pos != std::string_view::npos;
           pos = path.find('.', pos + 1)) {
        std::string table(path.substr(0, pos));
        if (std::find(table_storage.begin(), table_storage.end(), table) == table_storage.end()) {
          table_storage.push_back(std::move(table));
        }
      }
    }
    tables.assign(table_storage.begin(), table_storage.end());
  }
};

const SchemaLists& lists() {
  static const SchemaLists instance;
  return instance;
}

}  // namespace

std::span<const KeyInfo> schema_keys() { return lists().keys; }

const KeyInfo* find_schema_key(std::string_view path) {
  for (const KeyInfo& key : lists().keys) {
    if (key.path == path) return &key;
  }
  return nullptr;
}

std::span<const std::string_view> schema_tables() { return lists().tables; }

std::vector<std::string_view> schema_children(std::string_view table) {
  std::vector<std::string_view> children;
  const std::string prefix = table.empty() ? std::string() : std::string(table) + ".";
  auto add_child = [&](std::string_view path) {
    if (!path.starts_with(prefix)) return;
    std::string_view rest = path.substr(prefix.size());
    if (rest.empty()) return;
    rest = rest.substr(0, rest.find('.'));
    if (std::find(children.begin(), children.end(), rest) == children.end()) {
      children.push_back(rest);
    }
  };
  for (const KeyInfo& key : lists().keys) add_child(key.path);
  return children;
}

}  // namespace imsrg::driver
