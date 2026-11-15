// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Flow review item 1: the stage-2 step options of a stiff valence generator
// (dsmax 1e-2, imsrg++.cc:1130-1140) reach the solver through the driver, and
// item 9: Omega storage scratch/discard fail with E-PLANNED. Skips without the
// 2N input file.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/run.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

TEST(FlowFixes, StageTwoStiffGeneratorKeepsDsBelowOneHundredth) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir("stage2_ds");
  driver::ConfigSources src;
  src.config_text =
      TomlText(dir, "O16",
               "[valence]\nspace = \"sd-shell\"\n\n[imsrg.generator]\ncore = "
               "\"atan\"\nvalence = \"shell_model_imaginary_time\"\n",
               false, "0.05");
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;
  double max_ds1 = 0.0, max_ds2 = 0.0;
  int rows2 = 0;
  for (std::size_t i = 0; i < r.flow.rows.size(); ++i) {
    const double ds = r.flow.at(i, "ds");
    if (r.flow.at(i, "stage") == 1.0) max_ds1 = std::max(max_ds1, ds);
    if (r.flow.at(i, "stage") == 2.0) {
      max_ds2 = std::max(max_ds2, ds);
      ++rows2;
    }
  }
  std::printf(
      "[stage 2 ds] stage 1 max ds %.4g, stage 2 max ds %.4g (%d rows)\n",
      max_ds1, max_ds2, rows2);
  EXPECT_GT(rows2, 5);
  EXPECT_GT(max_ds1, 1.0e-2);
  EXPECT_LE(max_ds2, 1.0e-2 + 1.0e-15);
}

TEST(FlowFixes, OmegaStorageScratchAndDiscardArePlanned) {
  for (const char* storage : {"scratch", "discard"}) {
    driver::ConfigSources src;
    src.config_text =
        std::string(
            "[basis]\nemax = 2\nhw = 16.0\n[reference]\nnucleus = \"He4\"\n"
            "[interaction.two_body]\nfile = \"x.me2j\"\ne1max = 14\n"
            "e2max = 28\nlmax = 14\n[run]\nscratch_dir = \"/tmp\"\n"
            "[imsrg.omega]\nstorage = \"") +
        storage + "\"\n";
    auto r = driver::LoadConfig(src, driver::LoadOptions{.check_files = false});
    ASSERT_FALSE(r.has_value()) << storage;
    bool planned = false;
    for (const auto& issue : r.error()) {
      planned = planned || (issue.code == "E-PLANNED" &&
                            issue.key_path == "imsrg.omega.storage");
    }
    EXPECT_TRUE(planned) << storage;
  }
}

// Review item 4: the flow methods are accepted by the driver; Omega options
// that need an Omega stack are rejected with a clear message before any work.
TEST(FlowFixes, HunterGathererRequiresMagnus) {
  const auto dir = FreshDir("flow_hg");
  driver::ConfigSources src;
  src.config_text = std::string(
                        "[basis]\nemax = 2\nhw = 16.0\n[reference]\nnucleus = "
                        "\"He4\"\n[interaction.two_body]\nfile = \"x.me2j\"\n"
                        "e1max = 14\ne2max = 28\nlmax = 14\n[imsrg]\nmethod = "
                        "\"flow_rk4\"\n[imsrg.omega]\nhunter_gatherer = "
                        "true\n[output]\ndirectory = \"") +
                    dir.string() + "\"\n";
  auto loaded =
      driver::LoadConfig(src, driver::LoadOptions{.check_files = false});
  ASSERT_TRUE(loaded.has_value());
  EXPECT_THROW(driver::Run(loaded.value()), ConfigError);
  // method = flow_euler is still planned.
  src.config_text = std::string(
                        "[basis]\nemax = 2\nhw = 16.0\n[reference]\nnucleus = "
                        "\"He4\"\n[interaction.two_body]\nfile = \"x.me2j\"\n"
                        "e1max = 14\ne2max = 28\nlmax = 14\n[imsrg]\nmethod = "
                        "\"flow_euler\"\n[output]\ndirectory = \"") +
                    dir.string() + "\"\n";
  auto planned =
      driver::LoadConfig(src, driver::LoadOptions{.check_files = false});
  EXPECT_FALSE(planned.has_value());
}

}  // namespace
}  // namespace imsrg::it
