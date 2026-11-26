// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Flow checkpoint/restart and Omega dumps through `driver::Run` (design/50
// C5-C7): the 27Al pipeline in miniature (NAT basis with HF holes at emax 4,
// [imsrg.space] emax 2, two Magnus stages with Omega splits). A run killed
// after a checkpoint (testing aid checkpoint.abort_after_steps) and restarted
// reproduces the uninterrupted run bit for bit with native checkpoints, and
// within float32 precision with ME2JP checkpoints. Skips without the 2N file.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;
namespace fs = std::filesystem;

/// The miniature 27Al configuration; `checkpoint` is appended as given.
std::string Al27Toml(const fs::path& dir, const std::string& extra) {
  return TomlText(dir, "Al27",
                  "[valence]\nspace = \"sd-shell\"\n\n"
                  "[natural_orbitals]\nuse_hf_reference = true\n\n"
                  "[imsrg.space]\nemax = 2\n\n" +
                      extra,
                  false, "0.6\nomega_norm_max = 0.1\neta_criterion = 1e-3",
                  "nat") +
         "write_omega = true\n";
}

RunResult RunWith(const std::string& tag, const std::string& extra,
                  fs::path* out_dir = nullptr) {
  const fs::path dir = FreshDir(tag);
  if (out_dir) *out_dir = dir;
  driver::ConfigSources src;
  src.config_text = Al27Toml(dir, extra);
  return Execute(src, dir);
}

std::string Checkpoint(const fs::path& d, const std::string& format,
                       int abort_after, bool restart) {
  return "[checkpoint]\ndirectory = \"" + d.string() +
         "\"\ninterval_minutes = 0.0\nformat = \"" + format +
         "\"\nabort_after_steps = " + std::to_string(abort_after) +
         "\nrestart = " + (restart ? "true" : "false") + "\n\n";
}

const Json& Stages(const RunResult& r) { return r.results["imsrg"]["stages"]; }

TEST(Checkpoint, NativeRestartIsBitForBit) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  fs::path dir_a;
  const RunResult a = RunWith("ckpt_ref", "", &dir_a);
  ASSERT_FALSE(a.planned) << a.note;
  const int steps1 = static_cast<int>(Stages(a).at(0)["steps"].number);
  const int steps2 = static_cast<int>(Stages(a).at(1)["steps"].number);
  std::printf("[checkpoint] reference: %d + %d steps, %g Omegas, E0 %.12f\n",
              steps1, steps2, a.results["imsrg"]["num_omegas"].number,
              a.results["imsrg"]["e0"].number);
  ASSERT_GE(steps1, 4);
  ASSERT_GE(steps2, 4);
  ASSERT_GE(a.results["imsrg"]["num_omegas"].number, 3.0);
  // Omega manifest and files of write_omega.
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_omegas.toml")));
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_omega_000.opb")));
  EXPECT_TRUE(fs::exists(dir_a / (a.config.output.stem + "_basis.bin")));

  // Kill in stage 1 (step 3) and in stage 2 (2 steps into it); restart.
  for (const int kill : {3, steps1 + 2}) {
    const fs::path ck = FreshDir("ckpt_dir_" + std::to_string(kill));
    EXPECT_THROW(RunWith("ckpt_killed", Checkpoint(ck, "native", kill, false)),
                 std::runtime_error);
    ASSERT_TRUE(fs::exists(ck / "state.txt"));
    fs::path dir_c;
    const RunResult c =
        RunWith("ckpt_restart", Checkpoint(ck, "native", 0, true), &dir_c);
    ASSERT_FALSE(c.planned) << c.note;
    ASSERT_EQ(Stages(c).items.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(Stages(c).at(i)["steps"].number, Stages(a).at(i)["steps"].number);
      EXPECT_EQ(Stages(c).at(i)["s"].number, Stages(a).at(i)["s"].number);
      EXPECT_EQ(Stages(c).at(i)["e0"].number, Stages(a).at(i)["e0"].number);
    }
    EXPECT_EQ(c.results["imsrg"]["e0"].number, a.results["imsrg"]["e0"].number);
    // The valence interaction and every Omega file are identical.
    const std::string stem = a.config.output.stem;
    EXPECT_EQ(Slurp(dir_c / (stem + ".snt")), Slurp(dir_a / (stem + ".snt")));
    const int n = static_cast<int>(a.results["imsrg"]["num_omegas"].number);
    for (int k = 0; k < n; ++k) {
      char name[64];
      std::snprintf(name, sizeof(name), "_omega_%03d.opb", k);
      EXPECT_EQ(Slurp(dir_c / (stem + name)), Slurp(dir_a / (stem + name)))
          << "Omega " << k << " after a kill at step " << kill;
    }
  }
}

TEST(Checkpoint, Me2jpRestartWithinFloat32) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const RunResult a = RunWith("ckpt32_ref", "");
  ASSERT_FALSE(a.planned) << a.note;
  const int steps1 = static_cast<int>(Stages(a).at(0)["steps"].number);
  const fs::path ck = FreshDir("ckpt32_dir");
  EXPECT_THROW(RunWith("ckpt32_killed", Checkpoint(ck, "me2jp", steps1 + 2, false)),
               std::runtime_error);
  ASSERT_TRUE(fs::exists(ck / "state.txt"));
  const RunResult c = RunWith("ckpt32_restart", Checkpoint(ck, "me2jp", 0, true));
  ASSERT_FALSE(c.planned) << c.note;
  const double e0a = a.results["imsrg"]["e0"].number;
  const double e0c = c.results["imsrg"]["e0"].number;
  std::printf("[checkpoint me2jp] E0 %.12f vs %.12f (diff %.3e), steps %g vs %g\n",
              e0c, e0a, e0c - e0a, Stages(c).at(1)["steps"].number,
              Stages(a).at(1)["steps"].number);
  // float32 two-body parts (~6e-8 relative per element): owner O4 accepts a
  // restart within float32 precision, E0 to ~1e-6 MeV (observed 7e-9 MeV).
  EXPECT_NEAR(e0c, e0a, 1.0e-6);
  EXPECT_NEAR(Stages(c).at(1)["steps"].number, Stages(a).at(1)["steps"].number,
              3.0);
}

TEST(Checkpoint, RestartRejectsAnotherConfiguration) {
  if (!fs::exists(DataFile(TwoBodyName()))) GTEST_SKIP() << "missing 2N file";
  const fs::path ck = FreshDir("ckpt_cfg_dir");
  EXPECT_THROW(RunWith("ckpt_cfg_killed", Checkpoint(ck, "native", 2, false)),
               std::runtime_error);
  // Another eta_criterion: refused.
  const fs::path dir = FreshDir("ckpt_cfg_restart");
  driver::ConfigSources src;
  std::string text = Al27Toml(dir, Checkpoint(ck, "native", 0, true));
  const std::string from = "eta_criterion = 1e-3";
  text.replace(text.find(from), from.size(), "eta_criterion = 2e-3");
  src.config_text = text;
  EXPECT_THROW(Execute(src, dir), std::runtime_error);
}

}  // namespace
}  // namespace imsrg::it
