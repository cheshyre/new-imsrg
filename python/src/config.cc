// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the driver config (design/18 section 6): `RunConfig` as an
/// opaque resolved value with `to_toml()` / `to_dict()`, and `load_config()`.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/config_issue.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/config/writer.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;

namespace {

using driver::RunConfig;

/// Joins the issues one per line.
std::string JoinIssues(const driver::ConfigIssues& issues) {
  std::string out;
  for (const auto& i : issues) {
    if (!out.empty()) out += '\n';
    out += i.ToString();
  }
  return out;
}

}  // namespace

void BindConfig(nb::module_& m) {
  nb::class_<RunConfig>(
      m, "RunConfig",
      "A resolved, validated run configuration (immutable from Python).")
      .def(
          "to_toml",
          [](const RunConfig& c, bool include_provenance) {
            return driver::ToToml(c, include_provenance);
          },
          nb::arg("include_provenance") = true,
          "Canonical TOML text of the resolved config.")
      .def(
          "to_dict",
          [](const RunConfig& c) {
            nb::object tomllib = nb::module_::import_("tomllib");
            return tomllib.attr("loads")(driver::ToToml(c, false));
          },
          "The resolved config as nested dicts (parsed from its canonical "
          "TOML).")
      .def_prop_ro("warnings", [](const RunConfig& c) {
        std::vector<std::string> w;
        for (const auto& i : c.provenance.warnings) w.push_back(i.ToString());
        return w;
      });

  m.def(
      "load_config",
      [](std::optional<std::string> text, std::optional<std::string> file,
         std::vector<std::string> set, bool check_files) {
        driver::ConfigSources src;
        if (text) src.config_text = *text;
        if (file) src.config_file = std::filesystem::path(*file);
        src.set_overrides = std::move(set);
        src.command_line = "python imsrg.load_config";
        driver::LoadOptions opt;
        opt.check_files = check_files;
        auto r = driver::LoadConfig(src, opt);
        if (!r) throw imsrg::ConfigError(JoinIssues(r.error()));
        return std::move(r).value();
      },
      nb::arg("text") = nb::none(), nb::arg("file") = nb::none(),
      nb::arg("set") = std::vector<std::string>{},
      nb::arg("check_files") = false,
      "Loads, resolves and validates a config from TOML `text` and/or a "
      "`file`, then applies the\n"
      "`set` overrides (\"a.b=v\"). Raises ConfigError listing all issues.");
}

}  // namespace imsrg::py
