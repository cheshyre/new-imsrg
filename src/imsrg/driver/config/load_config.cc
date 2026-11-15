// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Config loading entry point: sources -> raw tree -> resolved, validated
// RunConfig.
#include "imsrg/driver/config/load_config.h"

#include <algorithm>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/driver/config/resolve.h"
#include "imsrg/driver/config/toml_reader.h"
#include "imsrg/driver/legacy/legacy_parameters.h"

#ifndef IMSRG_VERSION_STRING
#define IMSRG_VERSION_STRING "unknown"
#endif
#ifndef IMSRG_GIT_HASH
#define IMSRG_GIT_HASH "unknown"
#endif

namespace imsrg::driver {

namespace {

std::string UtcNow() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

void Append(ConfigIssues& dst, ConfigIssues&& src) {
  for (auto& i : src) dst.push_back(std::move(i));
}

}  // namespace

Expected<RunConfig, ConfigIssues> LoadConfig(const ConfigSources& src,
                                             const LoadOptions& opt) {
  const int n_sources = (src.config_file ? 1 : 0) + (src.config_text ? 1 : 0) +
                        (src.legacy_args.empty() ? 0 : 1);
  if (n_sources > 1) {
    return Unexpected<ConfigIssues>{
        ConfigIssues{MakeError("E-CLI-MIX", "",
                               "use exactly one of a config file, config text "
                               "or legacy arguments; got " +
                                   std::to_string(n_sources))}};
  }

  ConfigIssues issues;
  RawTree tree;
  std::vector<std::string> applied_rules;
  const bool legacy = !src.legacy_args.empty();

  if (src.config_file || src.config_text) {
    Append(issues,
           src.config_file
               ? ReadTomlFile(*src.config_file, tree)
               : ReadTomlString(*src.config_text, src.config_text_name, tree));
    // An unreadable or unparsable document yields no tree: resolving it would
    // only add consequential errors (missing required keys) to the one that
    // matters.
    const bool unreadable =
        std::any_of(issues.begin(), issues.end(), [](const ConfigIssue& i) {
          return i.code == "E-FILE" || i.code == "E-TOML-SYNTAX";
        });
    if (unreadable) return Unexpected<ConfigIssues>{std::move(issues)};
  } else if (legacy) {
    auto t = TranslateLegacy(std::span<const std::string>(src.legacy_args));
    tree = std::move(t.tree);
    Append(issues, std::move(t.issues));
    applied_rules = std::move(t.applied_rules);
  }

  for (const auto& assignment : src.set_overrides) {
    Append(issues, ApplySetOverride(assignment, tree));
  }

  auto resolved = Resolve(tree, ResolveOptions{legacy, opt.check_files});
  Append(issues, std::move(resolved.issues));

  if (HasErrors(issues)) return Unexpected<ConfigIssues>{std::move(issues)};

  RunConfig cfg = std::move(resolved.config);
  cfg.provenance.version = IMSRG_VERSION_STRING;
  cfg.provenance.git_hash = IMSRG_GIT_HASH;
  cfg.provenance.date = UtcNow();
  cfg.provenance.command_line = src.command_line;
  cfg.provenance.legacy_args = src.legacy_args;
  for (const auto& i : issues) {
    if (i.severity == Severity::kWarning) cfg.provenance.warnings.push_back(i);
  }
  cfg.provenance.applied_rules = std::move(applied_rules);
  return cfg;
}

}  // namespace imsrg::driver
