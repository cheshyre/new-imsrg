// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/driver/integration/run_support.h"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace imsrg::it {

FlowTable ReadFlow(const std::filesystem::path& path) {
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

std::filesystem::path FreshDir(const std::string& tag) {
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

RunResult Execute(const driver::ConfigSources& sources,
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

std::string TomlText(const std::filesystem::path& dir,
                     const std::string& nucleus, const std::string& extra,
                     bool three_body, const std::string& smax,
                     const std::string& basis_kind) {
  std::ostringstream t;
  t << "[basis]\nemax = 4\nhw = 16.0\n"
    << (basis_kind.empty() ? "" : "kind = \"" + basis_kind + "\"\n") << "\n"
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
