// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// run.task = "transform" on 2 and 4 MPI ranks (design/50 C10): the node-shared
// memory window (`Communicator::AllocateShared`), the Omega stack read once
// per node, the static assignment over the ranks, and the failure manifest
// when items throw. Rank 0 first runs the miniature 27Al flow serially (it
// writes the Omegas and the in-process .snt files); then all ranks run the
// farm, whose files must equal the in-process ones byte for byte. Run as
// `driver_transform_mpi_2ranks` / `_4ranks`. Skips without the 2N file.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/comm/communicator.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/run.h"
#include "imsrg/driver/transform/transform_task.h"
#include "tests/support/mpi_test_main.h"
#include "tests/support/scratch_dir.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace fs = std::filesystem;
using comm::Communicator;
using test_support::ScratchDir;
using test_support::TestWorld;

TEST(SharedMemory, RankZeroWritesEveryRankReads) {
  const Communicator& world = TestWorld();
  const Communicator node = world.SplitShared();
  EXPECT_EQ(node.size(), world.size());  // nested srun: one node
  const std::size_t n = 1000;
  comm::SharedMemory mem = node.AllocateShared(n * sizeof(std::int64_t));
  ASSERT_EQ(mem.size(), n * sizeof(std::int64_t));
  auto* p = reinterpret_cast<std::int64_t*>(mem.data().data());
  if (node.is_root()) {
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<std::int64_t>(3 * i);
  }
  node.Barrier();
  std::int64_t bad = 0;
  for (std::size_t i = 0; i < n; ++i) {
    bad += p[i] != static_cast<std::int64_t>(3 * i) ? 1 : 0;
  }
  EXPECT_EQ(world.AllreduceSum(bad), 0);
  node.Barrier();
}

const char* const kOps[] = {"TCM", "FM0p_150", "FM2p_150", "FSigmap1p_150",
                            "FDelta1p_150"};

std::string Al27Toml(const fs::path& dir, const std::string& run_extra,
                     const std::string& tail) {
  std::string ops = "[operators]\nlist = [";
  for (std::size_t i = 0; i < std::size(kOps); ++i) {
    ops += std::string(i ? ", \"" : "\"") + kOps[i] + "\"";
  }
  ops += "]\n\n";
  return "[run]\ncompat = \"imsrg_fork_79e9330\"\n" + run_extra + "\n" +
         TomlText(dir, "Al27",
                  "[valence]\nspace = \"sd-shell\"\n\n"
                  "[natural_orbitals]\nuse_hf_reference = true\n\n"
                  "[imsrg.space]\nemax = 2\n\n" +
                      ops,
                  false, "0.6\nomega_norm_max = 0.1\neta_criterion = 1e-3",
                  "nat") +
         "stem = \"int\"\n" + tail;
}

driver::RunConfig Load(const std::string& toml) {
  driver::ConfigSources src;
  src.config_text = toml;
  auto loaded = driver::LoadConfig(src);
  if (!loaded.has_value()) {
    std::string msg = "config rejected:";
    for (const auto& issue : loaded.error()) msg += "\n  " + issue.ToString();
    throw std::runtime_error(msg);
  }
  return loaded.value();
}

bool IsTensor(const std::string& name) {
  return name != "TCM" && name.find("0p_") == std::string::npos;
}

TEST(TransformMpi, FarmMatchesSerialInProcessTransform) {
  const Communicator& world = TestWorld();
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  std::unique_ptr<ScratchDir> scratch;
  if (world.is_root()) scratch = std::make_unique<ScratchDir>();
  std::string base_s = scratch ? scratch->path().string() : std::string();
  world.Broadcast(base_s, 0);
  const fs::path base(base_s);
  const fs::path flow_dir = base / "flow";
  const fs::path farm_dir = base / "farm";

  // ---- The flow, serially on rank 0.
  std::int64_t failed = 0;
  if (world.is_root()) {
    try {
      driver::Run(Load(Al27Toml(flow_dir, "", "write_omega = true\n")),
                  Communicator::Serial());
      std::ofstream w(base / "work.txt");
      w << "FPhipp0n_150\nE0\nFM0p_abc\n";
    } catch (const std::exception& e) {
      ADD_FAILURE() << "flow: " << e.what();
      failed = 1;
    }
  }
  if (world.AllreduceMax(failed) != 0) return;

  // ---- The farm on every rank.
  const driver::RunConfig cfg = Load(Al27Toml(
      farm_dir, "task = \"transform\"\n",
      "\n[transform]\nmanifest = \"" + (flow_dir / "int_omegas.toml").string() +
          "\"\nwork_list = \"" + (base / "work.txt").string() + "\"\n"));
  driver::transform::TransformReport report;
  try {
    report = driver::transform::RunTransform(cfg, world);
  } catch (const std::exception& e) {
    ADD_FAILURE() << "rank " << world.rank() << ": " << e.what();
    failed = 1;
  }
  if (world.AllreduceMax(failed) != 0) return;
  EXPECT_EQ(report.num_items, std::size(kOps) + 3);
  EXPECT_EQ(world.AllreduceSum(static_cast<std::int64_t>(report.num_assigned)),
            static_cast<std::int64_t>(report.num_items));
  EXPECT_GE(report.num_assigned, 1U);  // 8 items, at most 4 ranks
  EXPECT_GT(report.omega_window_bytes, 0U);
  std::printf("[transform-mpi] rank %d: %zu items, window %.3f MB\n",
              world.rank(), report.num_assigned,
              static_cast<double>(report.omega_window_bytes) * 1.0e-6);
  // Every rank knows the failures (the status table is reduced).
  ASSERT_EQ(report.failures.size(), 2U);
  EXPECT_EQ(report.failures[0].name, "E0");
  EXPECT_EQ(report.failures[1].name, "FM0p_abc");

  if (world.is_root()) {
    try {
      for (const std::string name : kOps) {
        const std::string suffix = IsTensor(name) ? "_2b.snt" : ".snt";
        EXPECT_EQ(Slurp(farm_dir / ("int" + name + suffix)),
                  Slurp(flow_dir / ("int_" + name + suffix)))
            << name;
      }
      EXPECT_TRUE(fs::exists(farm_dir / "intFPhipp0n_150.snt"));
      std::ifstream in(farm_dir / "failed.txt");
      std::vector<std::string> names;
      for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        names.push_back(line.substr(0, line.find('\t')));
        // The reason of a remote failure comes from its rank's file.
        EXPECT_EQ(line.find("failed on rank"), std::string::npos) << line;
      }
      EXPECT_EQ(names, (std::vector<std::string>{"E0", "FM0p_abc"}));
      for (int r = 0; r < world.size(); ++r) {
        EXPECT_FALSE(
            fs::exists(farm_dir / ("failed.txt.rank" + std::to_string(r))));
      }
    } catch (const std::exception& e) {
      ADD_FAILURE() << "rank 0 comparison: " << e.what();
    }
  }
  world.Barrier();
}

}  // namespace
}  // namespace imsrg::it
