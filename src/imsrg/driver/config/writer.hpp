#pragma once
/// @file
/// `RunConfig` -> TOML (`resolved_config.toml`, design/16 sections 1.3, 1.7 and 6.1).
///
/// Deterministic: keys in `visit_schema` order, one `[table]` header per table that has at least
/// one written key, absent optionals omitted, doubles with 17 significant digits ("%.17g", a ".0"
/// appended if the text would otherwise read as a TOML integer), strings as TOML basic strings
/// with escapes, arrays on one line, operator requests as their canonical names, from-file
/// entries as inline tables. `imsrg.generator.valence` is left out when `imsrg.steps == 1` (it is
/// inert then, and an explicit value with a valence space is V-41). Reading the output back with
/// `load_config` gives an equal config.

#include <filesystem>
#include <string>

#include "imsrg/driver/config/schema.hpp"

namespace imsrg::driver {

/// The resolved configuration as TOML text; with `include_provenance`, a `[provenance]` table
/// (version, git_hash, date, command_line, legacy_args, warnings, applied_rules) is appended last.
std::string to_toml(const RunConfig& config, bool include_provenance = true);

/// Writes `to_toml(config)` to `file`, creating missing parent directories.
/// @throws InputError if the directory cannot be created or the file cannot be written.
void write_resolved_config(const RunConfig& config, const std::filesystem::path& file);

}  // namespace imsrg::driver
