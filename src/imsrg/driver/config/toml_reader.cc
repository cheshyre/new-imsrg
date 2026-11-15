// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// TOML -> raw tree (design/16 sections 2 and 6.1). The only file that includes
// toml++.
#include "imsrg/driver/config/toml_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <toml++/toml.hpp>

#include "imsrg/driver/config/schema_keys.h"

namespace imsrg::driver {
namespace {

/// Origin of the text being read: the kind and name of every `SourceRef` made
/// from it.
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

/// Converts a TOML node to a `RawValue`. Arrays recurse; inline tables become
/// `RawTable`. If `allow_table` is false a table at this level is an error (a
/// schema leaf never is a table). Dates and times are errors at any depth.
/// Returns nullopt and fills `error` on failure.
std::optional<RawValue> Convert(const toml::node& node, const Origin& origin,
                                bool allow_table, ConvertError& error) {
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
        auto value = Convert(element, origin, true, error);
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
        auto value = Convert(child, origin, true, error);
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

std::string JoinPath(std::string_view table, std::string_view key) {
  return table.empty() ? std::string(key)
                       : std::string(table) + "." + std::string(key);
}

bool IsSchemaTable(std::string_view path) {
  const auto tables = SchemaTables();
  return std::find(tables.begin(), tables.end(), path) != tables.end();
}

/// Walks table `tbl` at dotted path `path`, storing known leaves and reporting
/// everything else.
void Walk(const toml::table& tbl, const std::string& path, const Origin& origin,
          RawTree& tree, ConfigIssues& issues) {
  for (auto&& [key, node] : tbl) {
    const std::string k(key.str());
    if (path.empty() && k == "provenance") continue;
    const std::string child = JoinPath(path, k);
    if (FindSchemaKey(child) != nullptr) {
      ConvertError error;
      auto value = Convert(node, origin, false, error);
      if (!value) {
        issues.push_back(
            MakeError("E-TYPE", child, error.message, error.where));
        continue;
      }
      tree.set(child, std::move(*value));
    } else if (IsSchemaTable(child)) {
      if (const toml::table* sub = node.as_table()) {
        Walk(*sub, child, origin, tree, issues);
      } else {
        issues.push_back(MakeError(
            "E-TYPE", child, "'" + child + "' is a table", origin.at(node)));
      }
    } else {
      std::optional<std::string> suggestion;
      if (auto best = DidYouMean(k, SchemaChildren(path))) {
        suggestion = JoinPath(path, *best);
      }
      const std::string message = node.is_table()
                                      ? "unknown table '[" + child + "]'"
                                      : "unknown key '" + child + "'";
      issues.push_back(MakeError("E-KEY-UNKNOWN", child, message,
                                 origin.at(node), std::move(suggestion)));
    }
  }
}

ConfigIssues ReadText(std::string_view text, const Origin& origin,
                      RawTree& tree) {
  ConfigIssues issues;
  toml::table root;
  try {
    root = toml::parse(text, origin.name);
  } catch (const toml::parse_error& e) {
    SourceRef where{origin.kind, origin.name,
                    static_cast<int>(e.source().begin.line),
                    static_cast<int>(e.source().begin.column)};
    issues.push_back(
        MakeError("E-TOML-SYNTAX", "", std::string(e.description()), where));
    return issues;
  }
  Walk(root, "", origin, tree, issues);
  return issues;
}

std::string_view Trim(std::string_view s) {
  const auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
  return s;
}

}  // namespace

ConfigIssues ReadTomlFile(const std::filesystem::path& file, RawTree& tree) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream buffer;
  if (in) buffer << in.rdbuf();
  if (!in || in.bad()) {
    ConfigIssues issues;
    issues.push_back(MakeError(
        "E-FILE", "", "cannot read config file '" + file.string() + "'",
        SourceRef{SourceRef::Kind::kTomlFile, file.string(), 0, 0}));
    return issues;
  }
  return ReadText(buffer.str(),
                  Origin{SourceRef::Kind::kTomlFile, file.string()}, tree);
}

ConfigIssues ReadTomlString(std::string_view text, std::string_view source_name,
                            RawTree& tree) {
  return ReadText(
      text, Origin{SourceRef::Kind::kTomlString, std::string(source_name)},
      tree);
}

ConfigIssues ApplySetOverride(std::string_view assignment, RawTree& tree) {
  ConfigIssues issues;
  const SourceRef where{SourceRef::Kind::kSetOption, std::string(assignment), 0,
                        0};
  const std::size_t eq = assignment.find('=');
  const std::string path(eq == std::string_view::npos
                             ? std::string_view{}
                             : Trim(assignment.substr(0, eq)));
  if (eq == std::string_view::npos || path.empty()) {
    issues.push_back(MakeError(
        "E-SET-SYNTAX", path,
        "expected 'section.key=value', got '" + std::string(assignment) + "'",
        where));
    return issues;
  }
  if (FindSchemaKey(path) == nullptr) {
    std::vector<std::string> all;
    for (const KeyInfo& key : SchemaKeys()) all.emplace_back(key.path);
    issues.push_back(MakeError("E-KEY-UNKNOWN", path,
                               "unknown key '" + path + "'", where,
                               DidYouMean(path, all)));
    return issues;
  }
  const std::string_view text = Trim(assignment.substr(eq + 1));
  std::optional<RawValue> value;
  try {
    const toml::table parsed = toml::parse("v = " + std::string(text));
    if (parsed.size() == 1) {
      if (const toml::node* node = parsed.get("v")) {
        ConvertError error;
        const Origin origin{SourceRef::Kind::kSetOption,
                            std::string(assignment)};
        value = Convert(*node, origin, true, error);
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
