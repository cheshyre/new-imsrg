// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Config loading entry point: sources -> raw tree -> resolved, validated
// RunConfig.
#include "imsrg/driver/config/load_config.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
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

namespace fs = std::filesystem;

/// `path` made absolute against `base` and lexically normalized; empty and
/// already absolute paths are returned unchanged.
std::string Absolute(const std::string& path, const fs::path& base) {
  if (path.empty() || fs::path(path).is_absolute()) return path;
  std::string out = (base / path).lexically_normal().string();
  // "dir/." normalizes to "dir/": drop the separator again.
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  return out;
}

/// The current working directory (empty if it cannot be determined, which
/// leaves relative paths as they are).
fs::path WorkingDirectory() {
  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  return ec ? fs::path() : cwd;
}

/// The schema keys whose value is a file or directory path (a string, or an
/// array of strings for `interaction.two_body.file`); the entries of
/// `operators.from_file` are handled separately.
const std::vector<std::string_view>& PathKeys() {
  static const auto* const kKeys = new std::vector<std::string_view>{
      "interaction.two_body.file",  "interaction.three_body.file",
      "reference.occupations_file", "run.scratch_dir",
      "output.directory",           "output.flow_file"};
  return *kKeys;
}

/// Makes every relative path value of `tree` absolute against `base`
/// (D37/M8a-Q5: the directory of the config file for the keys of a TOML file).
void AbsolutizeTree(RawTree& tree, const fs::path& base) {
  const auto fix = [&base](RawValue& v) {
    if (auto* str = std::get_if<std::string>(&v.data)) {
      *str = Absolute(*str, base);
    }
  };
  for (const std::string_view key : PathKeys()) {
    const RawValue* found = tree.find(key);
    if (found == nullptr) continue;
    RawValue v = *found;
    if (auto* arr = std::get_if<RawArray>(&v.data)) {
      for (RawValue& e : *arr) fix(e);
    } else {
      fix(v);
    }
    tree.set(std::string(key), std::move(v));
  }
  if (const RawValue* found = tree.find("operators.from_file")) {
    RawValue v = *found;
    if (auto* arr = std::get_if<RawArray>(&v.data)) {
      for (RawValue& e : *arr) {
        auto* tbl = std::get_if<RawTable>(&e.data);
        if (tbl == nullptr) continue;
        for (auto& [key, val] : *tbl) {
          if (key == "file_2b" || key == "file_3b") fix(val);
        }
      }
    }
    tree.set("operators.from_file", std::move(v));
  }
}

/// Makes the remaining relative paths of the resolved config absolute against
/// the working directory: legacy and command-line values, and defaults.
void AbsolutizeConfig(RunConfig& c, const fs::path& base) {
  if (base.empty()) return;
  for (std::string& f : c.interaction.two_body.file) f = Absolute(f, base);
  if (c.interaction.three_body) {
    c.interaction.three_body->file =
        Absolute(c.interaction.three_body->file, base);
  }
  if (c.reference.occupations_file) {
    c.reference.occupations_file =
        Absolute(*c.reference.occupations_file, base);
  }
  c.run.scratch_dir = Absolute(c.run.scratch_dir, base);
  c.output.directory = Absolute(c.output.directory, base);
  c.output.flow_file = Absolute(c.output.flow_file, base);
  for (OperatorFromFile& o : c.operators.from_file) {
    o.file_2b = Absolute(o.file_2b, base);
    if (o.file_3b) o.file_3b = Absolute(*o.file_3b, base);
  }
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
    // Relative paths of a TOML *file* resolve against its directory; the
    // overrides applied below and legacy values resolve against the working
    // directory (after `Resolve`).
    if (src.config_file) {
      std::error_code ec;
      const fs::path abs = fs::absolute(*src.config_file, ec);
      if (!ec) AbsolutizeTree(tree, abs.parent_path());
    }
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
  AbsolutizeConfig(cfg, WorkingDirectory());
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
