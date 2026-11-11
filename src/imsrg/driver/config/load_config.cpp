// Config loading entry point: sources -> raw tree -> resolved, validated RunConfig.
#include "imsrg/driver/config/load_config.hpp"

#include <algorithm>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/driver/config/resolve.hpp"
#include "imsrg/driver/config/toml_reader.hpp"
#include "imsrg/driver/legacy/legacy_parameters.hpp"

#ifndef IMSRG_VERSION_STRING
#define IMSRG_VERSION_STRING "unknown"
#endif
#ifndef IMSRG_GIT_HASH
#define IMSRG_GIT_HASH "unknown"
#endif

namespace imsrg::driver {

namespace {

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

void append(ConfigIssues& dst, ConfigIssues&& src) {
  for (auto& i : src) dst.push_back(std::move(i));
}

}  // namespace

Expected<RunConfig, ConfigIssues> load_config(const ConfigSources& src, const LoadOptions& opt) {
  const int n_sources =
      (src.config_file ? 1 : 0) + (src.config_text ? 1 : 0) + (src.legacy_args.empty() ? 0 : 1);
  if (n_sources > 1) {
    return Unexpected<ConfigIssues>{ConfigIssues{
        make_error("E-CLI-MIX", "",
                   "use exactly one of a config file, config text or legacy arguments; got " +
                       std::to_string(n_sources))}};
  }

  ConfigIssues issues;
  RawTree tree;
  std::vector<std::string> applied_rules;
  const bool legacy = !src.legacy_args.empty();

  if (src.config_file || src.config_text) {
    append(issues, src.config_file
                       ? read_toml_file(*src.config_file, tree)
                       : read_toml_string(*src.config_text, src.config_text_name, tree));
    // An unreadable or unparsable document yields no tree: resolving it would only add
    // consequential errors (missing required keys) to the one that matters.
    const bool unreadable = std::any_of(issues.begin(), issues.end(), [](const ConfigIssue& i) {
      return i.code == "E-FILE" || i.code == "E-TOML-SYNTAX";
    });
    if (unreadable) return Unexpected<ConfigIssues>{std::move(issues)};
  } else if (legacy) {
    auto t = translate_legacy(std::span<const std::string>(src.legacy_args));
    tree = std::move(t.tree);
    append(issues, std::move(t.issues));
    applied_rules = std::move(t.applied_rules);
  }

  for (const auto& assignment : src.set_overrides) {
    append(issues, apply_set_override(assignment, tree));
  }

  auto resolved = resolve(tree, ResolveOptions{legacy, opt.check_files});
  append(issues, std::move(resolved.issues));

  if (has_errors(issues)) return Unexpected<ConfigIssues>{std::move(issues)};

  RunConfig cfg = std::move(resolved.config);
  cfg.provenance.version = IMSRG_VERSION_STRING;
  cfg.provenance.git_hash = IMSRG_GIT_HASH;
  cfg.provenance.date = utc_now();
  cfg.provenance.command_line = src.command_line;
  cfg.provenance.legacy_args = src.legacy_args;
  for (const auto& i : issues) {
    if (i.severity == Severity::Warning) cfg.provenance.warnings.push_back(i);
  }
  cfg.provenance.applied_rules = std::move(applied_rules);
  return cfg;
}

}  // namespace imsrg::driver
