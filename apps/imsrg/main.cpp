// The `imsrg` executable: CLI11 command line -> imsrg::driver (design/16 section 2.12).
// Exit codes: 0 ok, 1 runtime failure, 2 configuration error, 3 input file error.
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "imsrg/base/error.hpp"
#include "imsrg/comm/environment.hpp"
#include "imsrg/driver/config/load_config.hpp"
#include "imsrg/driver/config/writer.hpp"
#include "imsrg/driver/legacy/legacy_parameters.hpp"
#include "imsrg/driver/operators/operator_request.hpp"
#include "imsrg/driver/run.hpp"
#include "imsrg/exec/runtime.hpp"

namespace {

constexpr const char* kLegacyHint =
    "Legacy key=value mode: imsrg key=value ... (old keys and their new names: imsrg "
    "--legacy-keys)";

/// The command line as recorded in the provenance: arguments joined by spaces; those containing
/// whitespace or quotes are single-quoted.
std::string join_command_line(int argc, char** argv) {
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

  auto* config_opt = app.add_option("--config", config_file, "TOML configuration file");
  app.add_option("--set", set_overrides, "Override one value: section.key=value (repeatable)")
      ->expected(1)
      ->multi_option_policy(CLI::MultiOptionPolicy::TakeAll)
      ->type_name("KEY=VALUE");
  app.add_flag("--validate", validate, "Resolve and validate the configuration, write nothing");
  app.add_flag("--print-config", print_config, "Print the resolved configuration as TOML");
  app.add_flag("--list-operators", list_operators, "List the operator names and exit");
  app.add_flag("--legacy-keys", legacy_keys, "List the legacy key=value keys and exit");
  app.add_option("tokens", tokens, "Legacy key=value arguments")->type_name("KEY=VALUE");

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
    if (legacy_keys) std::cout << imsrg::driver::legacy_keys_text();
    if (list_operators) std::cout << imsrg::driver::operator_catalog_text();
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
  sources.command_line = join_command_line(argc, argv);

  auto loaded = imsrg::driver::load_config(sources);
  if (!loaded) {
    // Exit 3 if only input files are at fault (unreadable config file, missing input files).
    bool all_input = true;
    for (const auto& issue : loaded.error()) {
      std::cerr << "imsrg: " << issue.to_string() << "\n";
      if (issue.severity == imsrg::driver::Severity::Error && issue.code != "V-14" &&
          issue.code != "E-FILE") {
        all_input = false;
      }
    }
    return all_input ? 3 : 2;
  }
  const imsrg::driver::RunConfig& config = loaded.value();
  for (const auto& w : config.provenance.warnings) {
    std::cerr << "imsrg: " << w.to_string() << "\n";
  }

  if (print_config) {
    std::cout << imsrg::driver::to_toml(config);
    return 0;
  }
  if (validate) {
    std::cout << "imsrg: configuration is valid\n";
    return 0;
  }

  try {
    // MPI first, Kokkos second (design/10 section 4); both live only around the pipeline, which
    // destroys all operators before they finalize.
    imsrg::comm::Environment environment(argc, argv);
    imsrg::exec::Runtime runtime(argc, argv);
    imsrg::driver::run(config);
  } catch (const imsrg::ConfigError& e) {
    std::cerr << "imsrg: " << e.what() << "\n";
    return 2;
  } catch (const imsrg::InputError& e) {
    std::cerr << "imsrg: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "imsrg: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
