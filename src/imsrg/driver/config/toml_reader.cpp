// TOML -> raw tree (design/16 sections 2 and 6.1). The only file that includes toml++.
#include "imsrg/driver/config/toml_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <toml++/toml.hpp>

#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {
namespace {

/// Origin of the text being read: the kind and name of every `SourceRef` made from it.
struct Origin {
  SourceRef::Kind kind;
  std::string name;

  SourceRef at(const toml::node& node) const {
    SourceRef ref{kind, name, 0, 0};
    const toml::source_region& src = node.source();
    ref.line = static_cast<int>(src.begin.line);
    ref.column = static_cast<int>(src.begin.column);
    return ref;
  }
};

/// Why a node could not be converted.
struct ConvertError {
  std::string message;
  SourceRef where;
};

/// Converts a TOML node to a `RawValue`. Arrays recurse; inline tables become `RawTable`. If
/// `allow_table` is false a table at this level is an error (a schema leaf never is a table).
/// Dates and times are errors at any depth. Returns nullopt and fills `error` on failure.
std::optional<RawValue> convert(const toml::node& node, const Origin& origin, bool allow_table,
                                ConvertError& error) {
  RawValue out;
  out.where = origin.at(node);
  switch (node.type()) {
    case toml::node_type::boolean:
      out.data = node.as_boolean()->get();
      return out;
    case toml::node_type::integer:
      out.data = static_cast<std::int64_t>(node.as_integer()->get());
      return out;
    case toml::node_type::floating_point:
      out.data = node.as_floating_point()->get();
      return out;
    case toml::node_type::string:
      out.data = node.as_string()->get();
      return out;
    case toml::node_type::array: {
      RawArray items;
      for (const toml::node& element : *node.as_array()) {
        auto value = convert(element, origin, true, error);
        if (!value) return std::nullopt;
        items.push_back(std::move(*value));
      }
      out.data = std::move(items);
      return out;
    }
    case toml::node_type::table: {
      if (!allow_table) {
        error = {"expected a value, got a table", out.where};
        return std::nullopt;
      }
      RawTable items;
      for (auto&& [key, child] : *node.as_table()) {
        auto value = convert(child, origin, true, error);
        if (!value) return std::nullopt;
        items.emplace_back(std::string(key.str()), std::move(*value));
      }
      out.data = std::move(items);
      return out;
    }
    default:
      error = {"dates and times are not supported", out.where};
      return std::nullopt;
  }
}

std::string join_path(std::string_view table, std::string_view key) {
  return table.empty() ? std::string(key) : std::string(table) + "." + std::string(key);
}

bool is_schema_table(std::string_view path) {
  const auto tables = schema_tables();
  return std::find(tables.begin(), tables.end(), path) != tables.end();
}

/// Walks table `tbl` at dotted path `path`, storing known leaves and reporting everything else.
void walk(const toml::table& tbl, const std::string& path, const Origin& origin, RawTree& tree,
          ConfigIssues& issues) {
  for (auto&& [key, node] : tbl) {
    const std::string k(key.str());
    if (path.empty() && k == "provenance") continue;
    const std::string child = join_path(path, k);
    if (find_schema_key(child) != nullptr) {
      ConvertError error;
      auto value = convert(node, origin, false, error);
      if (!value) {
        issues.push_back(make_error("E-TYPE", child, error.message, error.where));
        continue;
      }
      tree.set(child, std::move(*value));
    } else if (is_schema_table(child)) {
      if (const toml::table* sub = node.as_table()) {
        walk(*sub, child, origin, tree, issues);
      } else {
        issues.push_back(
            make_error("E-TYPE", child, "'" + child + "' is a table", origin.at(node)));
      }
    } else {
      std::optional<std::string> suggestion;
      if (auto best = did_you_mean(k, schema_children(path))) {
        suggestion = join_path(path, *best);
      }
      const std::string message =
          node.is_table() ? "unknown table '[" + child + "]'" : "unknown key '" + child + "'";
      issues.push_back(
          make_error("E-KEY-UNKNOWN", child, message, origin.at(node), std::move(suggestion)));
    }
  }
}

ConfigIssues read_text(std::string_view text, const Origin& origin, RawTree& tree) {
  ConfigIssues issues;
  toml::table root;
  try {
    root = toml::parse(text, origin.name);
  } catch (const toml::parse_error& e) {
    SourceRef where{origin.kind, origin.name, static_cast<int>(e.source().begin.line),
                    static_cast<int>(e.source().begin.column)};
    issues.push_back(make_error("E-TOML-SYNTAX", "", std::string(e.description()), where));
    return issues;
  }
  walk(root, "", origin, tree, issues);
  return issues;
}

std::string_view trim(std::string_view s) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

}  // namespace

ConfigIssues read_toml_file(const std::filesystem::path& file, RawTree& tree) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream buffer;
  if (in) buffer << in.rdbuf();
  if (!in || in.bad()) {
    ConfigIssues issues;
    issues.push_back(make_error("E-FILE", "", "cannot read config file '" + file.string() + "'",
                                SourceRef{SourceRef::Kind::TomlFile, file.string(), 0, 0}));
    return issues;
  }
  return read_text(buffer.str(), Origin{SourceRef::Kind::TomlFile, file.string()}, tree);
}

ConfigIssues read_toml_string(std::string_view text, std::string_view source_name, RawTree& tree) {
  return read_text(text, Origin{SourceRef::Kind::TomlString, std::string(source_name)}, tree);
}

ConfigIssues apply_set_override(std::string_view assignment, RawTree& tree) {
  ConfigIssues issues;
  const SourceRef where{SourceRef::Kind::SetOption, std::string(assignment), 0, 0};
  const std::size_t eq = assignment.find('=');
  const std::string path(eq == std::string_view::npos ? std::string_view{}
                                                      : trim(assignment.substr(0, eq)));
  if (eq == std::string_view::npos || path.empty()) {
    issues.push_back(
        make_error("E-SET-SYNTAX", path,
                   "expected 'section.key=value', got '" + std::string(assignment) + "'", where));
    return issues;
  }
  if (find_schema_key(path) == nullptr) {
    std::vector<std::string> all;
    for (const KeyInfo& key : schema_keys()) all.emplace_back(key.path);
    issues.push_back(make_error("E-KEY-UNKNOWN", path, "unknown key '" + path + "'", where,
                                did_you_mean(path, all)));
    return issues;
  }
  const std::string_view text = trim(assignment.substr(eq + 1));
  std::optional<RawValue> value;
  try {
    const toml::table parsed = toml::parse("v = " + std::string(text));
    if (parsed.size() == 1) {
      if (const toml::node* node = parsed.get("v")) {
        ConvertError error;
        const Origin origin{SourceRef::Kind::SetOption, std::string(assignment)};
        value = convert(*node, origin, true, error);
        if (value) value->where = where;
      }
    }
  } catch (const toml::parse_error&) {
    value.reset();
  }
  if (!value) {
    RawValue verbatim;
    verbatim.data = std::string(text);
    verbatim.where = where;
    value = std::move(verbatim);
  }
  tree.set(path, std::move(*value));
  return issues;
}

}  // namespace imsrg::driver
