// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_DRIVER_INTEGRATION_RUN_SUPPORT_H_
#define TESTS_UNIT_DRIVER_INTEGRATION_RUN_SUPPORT_H_
/// @file
/// Helpers shared by the driver integration tests: temporary output
/// directories, the flow-CSV reader, and `Execute()`, which loads a config,
/// runs the pipeline and collects the `results.json` / flow CSV outputs of the
/// documented output contract.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/run.h"
#include "tests/unit/driver/integration/mini_json.h"

namespace imsrg::it {

/// A flow CSV: column names and numeric rows (one per step, istep 0 of every
/// stage included).
struct FlowTable {
  std::vector<std::string> columns;
  std::vector<std::vector<double>> rows;

  /// Index of column `name`. @throws std::runtime_error if absent.
  std::size_t col(const std::string& name) const {
    for (std::size_t i = 0; i < columns.size(); ++i)
      if (columns[i] == name) return i;
    throw std::runtime_error("flow CSV has no column " + name);
  }
  /// Value of column `name` in row `r`.
  double at(std::size_t r, const std::string& name) const {
    return rows.at(r)[col(name)];
  }
};

/// Reads a comma separated flow file with a header line. @throws
/// std::runtime_error if missing.
inline FlowTable ReadFlow(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open flow file " + path.string());
  FlowTable t;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string cell;
    if (header) {
      while (std::getline(ss, cell, ',')) t.columns.push_back(cell);
      header = false;
    } else {
      std::vector<double> row;
      while (std::getline(ss, cell, ','))
        row.push_back(std::strtod(cell.c_str(), nullptr));
      t.rows.push_back(std::move(row));
    }
  }
  return t;
}

/// Reads a whole text file. @throws std::runtime_error if missing.
inline std::string Slurp(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

/// A fresh, empty directory `<tmp>/imsrg_it_<tag>_<pid-independent counter>`;
/// any earlier directory of the same name is removed first.
inline std::filesystem::path FreshDir(const std::string& tag) {
  // CTest runs every test case in its own process, in parallel, and the
  // IT-2/IT-6/IT-12 cases each repeat the shared run: the process id keeps
  // their output directories apart.
  const std::filesystem::path d =
      std::filesystem::temp_directory_path() /
      ("imsrg_it_" + tag + "_" + std::to_string(::getpid()));
  std::error_code ec;
  std::filesystem::remove_all(d, ec);
  std::filesystem::create_directories(d);
  return d;
}

/// Absolute path of an input file below `IMSRG_TEST_DATA_DIR`.
inline std::filesystem::path DataFile(const std::string& name) {
  return std::filesystem::path(IMSRG_TEST_DATA_DIR) / name;
}

/// Absolute path of a golden file below `IMSRG_GOLDEN_DIR/e2e`.
inline std::filesystem::path GoldenFile(const std::string& rel) {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / "e2e" / rel;
}

/// The 2N interaction file used by every integration test.
inline const char* TwoBodyName() {
  return "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
}
/// The NO2B 3N file used by IT-4.
inline const char* ThreeBodyName() {
  return "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";
}

/// Everything a finished run left behind.
struct RunResult {
  driver::RunConfig config;
  std::filesystem::path directory;
  Json results;          ///< parsed `results.json`
  FlowTable flow;        ///< parsed flow CSV (empty if the run wrote none)
  bool planned = false;  ///< `Run()` is not implemented yet (E-PLANNED)
  std::string note;      ///< diagnostic for `planned`
};

/// Loads `sources`, runs the pipeline and parses the outputs. `Run()` throwing
/// an error whose message contains "E-PLANNED" sets `planned` instead of
/// throwing. Other failures throw `std::runtime_error` carrying every config
/// issue or the run's message.
inline RunResult Execute(const driver::ConfigSources& sources,
                         const std::filesystem::path& directory) {
  RunResult r;
  r.directory = directory;
  auto loaded = driver::LoadConfig(sources);
  if (!loaded.has_value()) {
    std::string msg = "config rejected:";
    for (const auto& issue : loaded.error()) msg += "\n  " + issue.ToString();
    throw std::runtime_error(msg);
  }
  r.config = loaded.value();
  try {
    driver::Run(r.config);
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("E-PLANNED") != std::string::npos) {
      r.planned = true;
      r.note = e.what();
      return r;
    }
    throw;
  }
  r.results = ParseJson(Slurp(directory / "results.json"));
  const std::filesystem::path flow_path =
      r.config.output.flow_file.empty()
          ? directory / (r.config.output.stem + ".flow")
          : std::filesystem::path(r.config.output.flow_file);
  if (std::filesystem::exists(flow_path)) r.flow = ReadFlow(flow_path);
  return r;
}

/// TOML text for an O16/Ne20 run at hw = 16, emax = 4 with the 2N file, writing
/// into `dir`. `extra` is appended verbatim (further tables such as `[valence]`
/// or `[operators]`); keys of tables already open at the end must not be
/// repeated there.
inline std::string TomlText(const std::filesystem::path& dir,
                            const std::string& nucleus,
                            const std::string& extra, bool three_body = false,
                            const std::string& smax = "") {
  std::ostringstream t;
  t << "[basis]\nemax = 4\nhw = 16.0\n\n"
    << "[reference]\nnucleus = \"" << nucleus << "\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\n"
    << "e1max = 14\ne2max = 28\nlmax = 14\n\n";
  if (three_body)
    t << "[interaction.three_body]\nfile = \""
      << DataFile(ThreeBodyName()).string() << "\"\n"
      << "type = \"no2b\"\ne1max = 6\ne2max = 12\ne3max = 18\n\n";
  // The integration tests reproduce imsrg++ step counts and trajectories, so
  // they use its linear HF mixing; the TOML default (diis, D37) reaches the
  // same solution only to the HF tolerance.
  t << "[hartree_fock]\nmixing = \"linear\"\n\n";
  t << "[imsrg]\nmethod = \"magnus\"\n\n";
  if (!smax.empty()) t << "[imsrg.step]\nsmax = " << smax << "\n\n";
  t << extra << "\n[output]\ndirectory = \"" << dir.string() << "\"\n";
  return t.str();
}

}  // namespace imsrg::it

#endif  // TESTS_UNIT_DRIVER_INTEGRATION_RUN_SUPPORT_H_
