#pragma once
/// @file
/// The configuration entry point (design/16 sections 1.3 and 6.1): merge the sources (TOML file
/// or text, or legacy key=value tokens; then `--set` overrides) into one raw tree, resolve and
/// validate it, and return the frozen `RunConfig` or every issue at once.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "imsrg/driver/config/config_issue.hpp"
#include "imsrg/driver/config/expected.hpp"
#include "imsrg/driver/config/schema.hpp"

namespace imsrg::driver {

/// The inputs of one configuration load. At most one of `config_file`, `config_text` and
/// `legacy_args` may be used (`E-CLI-MIX` otherwise); `set_overrides` apply last, in order.
struct ConfigSources {
  std::optional<std::filesystem::path> config_file;  ///< `--config FILE`
  std::optional<std::string> config_text;            ///< TOML text (Python bindings, tests)
  std::string config_text_name = "<string>";         ///< label of `config_text` in messages
  std::vector<std::string> legacy_args;              ///< legacy `key=value` tokens, verbatim
  std::vector<std::string> set_overrides;            ///< `--set a.b=v` assignments, in order
  std::string command_line;                          ///< recorded in the provenance
};

/// Options of `load_config`.
struct LoadOptions {
  bool check_files = true;  ///< V-14 preflight of input files and output directories
};

/// Loads, resolves and validates a configuration. On success the config's `provenance` is filled
/// (version, git hash, UTC date, command line, legacy args, warnings, applied rules) and the
/// warnings are in `provenance.warnings`. On failure the error holds all issues found (errors
/// and warnings). Never throws for bad input.
Expected<RunConfig, ConfigIssues> load_config(const ConfigSources& sources,
                                              const LoadOptions& options = {});

}  // namespace imsrg::driver
