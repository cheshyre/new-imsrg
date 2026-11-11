#pragma once
/// @file
/// TOML -> raw tree (design/16 sections 2 and 6.1). toml++ is used only in toml_reader.cpp.
///
/// Every leaf is checked against the schema (`schema_keys.hpp`): unknown keys and tables are
/// `E-KEY-UNKNOWN` with a did-you-mean suggestion among the siblings in that table; values keep
/// their TOML type and source position (line, column). The `[provenance]` table of a
/// `resolved_config.toml` is skipped. Dates and times are `E-TYPE`. All issues are collected.

#include <filesystem>
#include <string_view>

#include "imsrg/driver/config/config_issue.hpp"
#include "imsrg/driver/config/raw_tree.hpp"

namespace imsrg::driver {

/// Reads the TOML file `file` into `tree` (entries replace existing ones). Returns the issues:
/// `E-FILE` if the file cannot be read, `E-TOML-SYNTAX` (with line and column) on a parse error,
/// `E-KEY-UNKNOWN` / `E-TYPE` per offending key (valid keys are still stored).
ConfigIssues read_toml_file(const std::filesystem::path& file, RawTree& tree);

/// As `read_toml_file`, for TOML text; `source_name` labels it in `SourceRef` (Kind::TomlString).
ConfigIssues read_toml_string(std::string_view text, std::string_view source_name, RawTree& tree);

/// Applies one `--set` assignment "a.b.c=value" to `tree` (design/16 section 2.12): the text after
/// the first '=' is parsed as a TOML value; if that fails, it is taken as a bare string. The path
/// must be a schema key (`E-KEY-UNKNOWN` with suggestion otherwise); a missing '=' or an empty
/// path is `E-SET-SYNTAX`. Repeated assignments: the last wins.
ConfigIssues apply_set_override(std::string_view assignment, RawTree& tree);

}  // namespace imsrg::driver
