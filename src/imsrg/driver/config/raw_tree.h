// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_CONFIG_RAW_TREE_H_
#define IMSRG_DRIVER_CONFIG_RAW_TREE_H_
/// @file
/// The raw configuration tree (design/16 section 1.3): what the TOML reader,
/// the legacy translator and `--set` produce, before type conversion, defaults
/// and validation (`resolve`).
///
/// The tree is flat: a map from the dotted key path of a schema leaf
/// ("basis.emax", "interaction.two_body.file") to its value with the source it
/// came from. Arrays and inline tables (operator specs, `operators.from_file`
/// entries) are values, not subtrees.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "imsrg/driver/config/config_issue.h"

namespace imsrg::driver {

struct RawValue;

/// A TOML array.
using RawArray = std::vector<RawValue>;
/// A TOML (inline) table, keys in source order.
using RawTable = std::vector<std::pair<std::string, RawValue>>;

/// One untyped value with its origin. Integers and floats stay distinct (TOML
/// semantics), so the resolver can reject `emax = 4.5` but accept `hw = 16`.
struct RawValue {
  /// The value: boolean, integer, floating point, string, array or inline
  /// table.
  using Data =
      std::variant<bool, std::int64_t, double, std::string, RawArray, RawTable>;
  Data data;
  SourceRef where;

  /// "boolean", "integer", "float", "string", "array" or "table" (for
  /// messages).
  std::string_view type_name() const noexcept;

  /// The alternative `T` if held, else nullptr.
  template <class T>
  const T* get_if() const noexcept {
    return std::get_if<T>(&data);
  }
};

/// The flat raw tree: dotted path -> value. Setting a path twice keeps the last
/// value.
class RawTree {
 public:
  RawTree() = default;

  RawTree(const RawTree&) = default;
  RawTree& operator=(const RawTree&) = default;
  RawTree(RawTree&&) = default;
  RawTree& operator=(RawTree&&) = default;

  /// The underlying ordered map (paths in lexicographic order).
  using Map = std::map<std::string, RawValue, std::less<>>;

  /// Sets (or replaces) the value at `path`.
  void set(std::string path, RawValue value);
  /// The value at `path`, or nullptr.
  const RawValue* find(std::string_view path) const;
  /// True if `path` is set.
  bool contains(std::string_view path) const;
  /// Removes `path`; returns true if it was set.
  bool erase(std::string_view path);
  /// True if any path starts with `table` + "." (e.g.
  /// has_table("interaction.three_body")).
  bool has_table(std::string_view table) const;
  /// All entries.
  const Map& entries() const noexcept { return entries_; }
  /// True if nothing is set.
  bool empty() const noexcept { return entries_.empty(); }
  /// Copies every entry of `overlay` into this tree; `overlay` wins on
  /// conflicts.
  void Merge(const RawTree& overlay);

 private:
  Map entries_;
};

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_CONFIG_RAW_TREE_H_
