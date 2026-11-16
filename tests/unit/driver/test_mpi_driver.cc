// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// M12b phase 1, task D4: `driver::Run(config, comm)` on several ranks gives
// the serial results, and only rank 0 writes files. He4 at emax 2 (the
// smallest driver cases): single reference with Rp2 (so the internal Rso is
// broadcast and transformed too), and the two-stage p-shell valence-space
// case. Run as `driver_mpi_2ranks` and `driver_mpi_4ranks` (shared main of
// tests/support/mpi_test_main.cc). Skips without the 2N input file.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/comm/communicator.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/run.h"
#include "tests/support/mpi_test_main.h"
#include "tests/support/scratch_dir.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace fs = std::filesystem;
using comm::Communicator;
using test_support::ScratchDir;
using test_support::TestWorld;

/// The quantities that must agree between the MPI and the serial run.
struct Summary {
  double e0 = 0.0;
  double e_hf = 0.0;
  std::vector<std::int64_t> steps;  ///< per stage
  std::size_t flow_lines = 0;
  double rp2 = 0.0;  ///< transformed Rp2 (0 if not requested)
};

std::string He4Toml(const fs::path& dir, const std::string& extra) {
  std::ostringstream t;
  t << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[hartree_fock]\nmixing = \"linear\"\n\n"
    << "[imsrg]\nmethod = \"magnus\"\n\n"
    << extra << "\n[output]\ndirectory = \"" << dir.string() << "\"\n";
  return t.str();
}

driver::RunConfig Load(const fs::path& dir, const std::string& extra) {
  driver::ConfigSources src;
  src.config_text = He4Toml(dir, extra);
  auto loaded = driver::LoadConfig(src);
  if (!loaded.has_value()) {
    std::string msg = "config rejected:";
    for (const auto& issue : loaded.error()) msg += "\n  " + issue.ToString();
    throw std::runtime_error(msg);
  }
  return loaded.value();
}

fs::path FlowPath(const driver::RunConfig& c) {
  return c.output.flow_file.empty() ? fs::path(c.output.directory) / "flow.csv"
                                    : fs::path(c.output.flow_file);
}

std::size_t CountLines(const fs::path& path) {
  const std::string text = Slurp(path);
  return static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n'));
}

Summary Summarize(const driver::RunConfig& c) {
  const fs::path dir(c.output.directory);
  const Json res = ParseJson(Slurp(dir / "results.json"));
  Summary s;
  s.e0 = res["imsrg"]["e0"].number;
  s.e_hf = res["hf"]["e_hf"].number;
  for (const auto& st : res["imsrg"]["stages"].items) {
    s.steps.push_back(static_cast<std::int64_t>(st["steps"].number));
  }
  s.flow_lines = CountLines(FlowPath(c));
  if (res["operators"].has("Rp2"))
    s.rp2 = res["operators"]["Rp2"]["imsrg"].number;
  return s;
}

/// Number of regular files below `dir` (0 if it does not exist).
std::size_t CountFiles(const fs::path& dir) {
  std::size_t n = 0;
  if (!fs::exists(dir)) return 0;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    if (e.is_regular_file()) ++n;
  }
  return n;
}

/// Checks |a - b| <= 1e-12 max(1, |b|) and prints the deviation.
void ExpectClose(const char* what, double a, double b) {
  const double d = a - b;
  std::printf("[mpi-driver] %-6s mpi %.15g serial %.15g  d = %.3e\n", what, a,
              b, d);
  EXPECT_LE(std::abs(d), 1.0e-12 * std::max(1.0, std::abs(b))) << what;
}

/// The shared base directory: rank 0 creates it, and broadcasts its path.
std::string BroadcastBase(const Communicator& world,
                          const std::unique_ptr<ScratchDir>& scratch) {
  std::string base = scratch ? scratch->path().string() : std::string();
  world.Broadcast(base, 0);
  return base;
}

/// Runs the pipeline on every rank (output below <base>/mpi/rank<r>), then on
/// rank 0 alone serially, and compares. Failures of `Run` are caught and
/// agreed on, so no rank is left in a collective.
void CompareWithSerial(const std::string& extra, bool expect_rp2) {
  const Communicator& world = TestWorld();
  if (!fs::exists(DataFile(TwoBodyName()))) {
    GTEST_SKIP() << "missing 2N input file";
  }
  std::unique_ptr<ScratchDir> scratch;
  if (world.is_root()) scratch = std::make_unique<ScratchDir>();
  const fs::path base = BroadcastBase(world, scratch);

  const driver::RunConfig cfg =
      Load(base / "mpi" / ("rank" + std::to_string(world.rank())), extra);
  std::int64_t failed = 0;
  try {
    driver::Run(cfg, world);
  } catch (const std::exception& e) {
    ADD_FAILURE() << "rank " << world.rank() << ": " << e.what();
    failed = 1;
  }
  if (world.AllreduceMax(failed) != 0) return;  // all ranks agree: leave
  world.Barrier();

  // Rank 0 compares; exceptions are caught and agreed on, so that every rank
  // reaches the closing barrier.
  std::int64_t root_failed = 0;
  if (world.is_root()) {
    try {
      EXPECT_TRUE(fs::exists(fs::path(cfg.output.directory) / "results.json"));
      EXPECT_TRUE(fs::exists(FlowPath(cfg)));
      EXPECT_TRUE(
          fs::exists(fs::path(cfg.output.directory) / "resolved_config.toml"));
      for (int r = 1; r < world.size(); ++r) {
        const fs::path d = base / "mpi" / ("rank" + std::to_string(r));
        EXPECT_EQ(CountFiles(d), 0u) << "files written by rank " << r;
      }
      const driver::RunConfig scfg = Load(base / "serial", extra);
      driver::Run(scfg, Communicator::Serial());
      const Summary m = Summarize(cfg);
      const Summary s = Summarize(scfg);
      ExpectClose("E0", m.e0, s.e0);
      ExpectClose("E_HF", m.e_hf, s.e_hf);
      if (expect_rp2) ExpectClose("Rp2", m.rp2, s.rp2);
      EXPECT_EQ(m.steps, s.steps) << "steps per stage";
      EXPECT_EQ(m.flow_lines, s.flow_lines) << "flow CSV lines";
      std::printf("[mpi-driver] ranks %d, stages %zu, flow lines %zu\n",
                  world.size(), m.steps.size(), m.flow_lines);
    } catch (const std::exception& e) {
      ADD_FAILURE() << "rank 0 comparison failed: " << e.what();
      root_failed = 1;
    }
  }
  world.AllreduceMax(root_failed);  // all ranks leave together
  world.Barrier();
}

TEST(DriverMpi, He4Emax2Magnus) {
  CompareWithSerial("[operators]\nlist = [\"Rp2\"]\n", true);
}

TEST(DriverMpi, He4Emax2PShellTwoStages) {
  CompareWithSerial("[valence]\nspace = \"p-shell\"\n", false);
}

}  // namespace
}  // namespace imsrg::it
