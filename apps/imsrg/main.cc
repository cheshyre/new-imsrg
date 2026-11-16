// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The `imsrg` executable: CLI11 command line -> imsrg::driver (design/16
// section 2.12). Exit codes: 0 ok, 1 runtime failure, 2 configuration error, 3
// input file error.
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "imsrg/base/error.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/comm/environment.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/driver/legacy/legacy_parameters.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/run.h"
#include "imsrg/exec/runtime.h"

namespace {

constexpr const char* kLegacyHint =
    "Legacy key=value mode: imsrg key=value ... (old keys and their new names: "
    "imsrg "
    "--legacy-keys)";

/// The command line as recorded in the provenance: arguments joined by spaces;
/// those containing whitespace or quotes are single-quoted.
std::string JoinCommandLine(int argc, char** argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a.find_first_of(" \t\"'") != std::string::npos) {
      std::string q = "'";
      for (char c : a) {
        if (c == '\'') {
          q += "'\\''";
        } else {
          q += c;
        }
      }
      a = q + "'";
    }
    if (i > 0) out += ' ';
    out += a;
  }
  return out;
}

/// Prints the issues of a failed `LoadConfig` to stderr and returns the exit
/// code: 3 if only input files are at fault (unreadable config file, missing
/// input files), else 2.
int ReportLoadFailure(const imsrg::driver::ConfigIssues& issues) {
  bool all_input = true;
  for (const auto& issue : issues) {
    std::cerr << "imsrg: " << issue.ToString() << "\n";
    if (issue.severity == imsrg::driver::Severity::kError &&
        issue.code != "V-14" && issue.code != "E-FILE") {
      all_input = false;
    }
  }
  return all_input ? 3 : 2;
}

/// Reports a failure of the pipeline and returns the exit code `code`. On more
/// than one rank the message carries the rank prefix and the whole job is
/// aborted at once (`Environment::Abort`; only `main` may do that), so a
/// failure on one rank cannot leave the others blocked in a collective.
int Fail(const imsrg::comm::Communicator& world, int code, const char* what) {
  if (world.size() > 1) {
    std::cerr << "imsrg[rank " << world.rank() << "]: " << what << "\n";
    std::cerr.flush();
    imsrg::comm::Environment::Abort(code);
  }
  std::cerr << "imsrg: " << what << "\n";
  return code;
}

/// Runs the pipeline for `config` and returns the exit code (0 ok, 1 runtime
/// failure, 2 configuration error, 3 input file error). With several MPI ranks
/// a failure on any rank aborts the job (see `Fail`).
int RunPipeline(int argc, char** argv, const imsrg::driver::RunConfig& config) {
  // MPI first, Kokkos second (design/10 section 4). The Environment outlives
  // the handlers, so that MPI is still initialized when they abort the job.
  std::unique_ptr<imsrg::comm::Environment> environment;
  imsrg::comm::Communicator world;
  try {
    environment = std::make_unique<imsrg::comm::Environment>(argc, argv);
    world = environment->world();
    imsrg::exec::Runtime runtime(argc, argv);
    imsrg::driver::Run(config, world);
  } catch (const imsrg::ConfigError& e) {
    return Fail(world, 2, e.what());
  } catch (const imsrg::InputError& e) {
    return Fail(world, 3, e.what());
  } catch (const std::exception& e) {
    return Fail(world, 1, e.what());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"imsrg: in-medium similarity renormalization group", "imsrg"};
  app.footer(kLegacyHint);

  std::string config_file;
  std::vector<std::string> set_overrides;
  std::vector<std::string> tokens;
  bool validate = false;
  bool print_config = false;
  bool list_operators = false;
  bool legacy_keys = false;

  auto* config_opt =
      app.add_option("--config", config_file, "TOML configuration file");
  app.add_option("--set", set_overrides,
                 "Override one value: section.key=value (repeatable)")
      ->expected(1)
      ->multi_option_policy(CLI::MultiOptionPolicy::TakeAll)
      ->type_name("KEY=VALUE");
  app.add_flag("--validate", validate,
               "Resolve and validate the configuration, write nothing");
  app.add_flag("--print-config", print_config,
               "Print the resolved configuration as TOML");
  app.add_flag("--list-operators", list_operators,
               "List the operator names and exit");
  app.add_flag("--legacy-keys", legacy_keys,
               "List the legacy key=value keys and exit");
  app.add_option("tokens", tokens, "Legacy key=value arguments")
      ->type_name("KEY=VALUE");

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "help" || a == "-help") {
      std::cout << app.help() << kLegacyHint << "\n";
      return 0;
    }
  }

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    const int code = app.exit(e);
    return code == 0 ? 0 : 2;
  }

  if (legacy_keys || list_operators) {
    if (legacy_keys) std::cout << imsrg::driver::LegacyKeysText();
    if (list_operators) std::cout << imsrg::driver::OperatorCatalogText();
    return 0;
  }

  const bool have_config = config_opt->count() > 0;
  if (!have_config && tokens.empty()) {
    std::cerr << app.help() << kLegacyHint << "\n";
    return 2;
  }

  imsrg::driver::ConfigSources sources;
  if (have_config) sources.config_file = std::filesystem::path(config_file);
  sources.legacy_args = tokens;
  sources.set_overrides = set_overrides;
  sources.command_line = JoinCommandLine(argc, argv);

  auto loaded = imsrg::driver::LoadConfig(sources);
  if (!loaded) return ReportLoadFailure(loaded.error());
  const imsrg::driver::RunConfig& config = loaded.value();
  for (const auto& w : config.provenance.warnings) {
    std::cerr << "imsrg: " << w.ToString() << "\n";
  }

  if (print_config) {
    std::cout << imsrg::driver::ToToml(config);
    return 0;
  }
  if (validate) {
    std::cout << "imsrg: configuration is valid\n";
    return 0;
  }

  return RunPipeline(argc, argv, config);
}
