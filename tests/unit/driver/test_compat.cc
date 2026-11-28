// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of the debug keys run.compat_enable / run.compat_disable
// (design/16 run.compat_*, design/50 section 5).
#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "imsrg/driver/config/compat.h"
#include "imsrg/driver/config/resolve.h"
#include "imsrg/driver/config/toml_reader.h"
#include "imsrg/driver/config/writer.h"
#include "imsrg/driver/pipeline/checkpoint.h"
#include "imsrg/driver/pipeline/operator_frame.h"
#include "imsrg/driver/pipeline/prepared_state.h"
#include "imsrg/driver/transform/transform_task.h"

namespace imsrg::driver {
namespace {

/// Resolves a minimal config with `run_lines` added to its [run] table.
ResolveResult ResolveRun(const std::string& run_lines) {
  const std::string text =
      "[run]\n" + run_lines +
      "\n[basis]\nemax = 4\nhw = 16\n[reference]\nnucleus = \"O16\"\n"
      "[interaction.two_body]\nfile = \"x.me2j\"\ne1max = 14\n";
  RawTree tree;
  const ConfigIssues read = ReadTomlString(text, "test", tree);
  EXPECT_TRUE(read.empty());
  return Resolve(tree, ResolveOptions{false, false});
}

bool HasIssueWith(const ConfigIssues& issues, const std::string& key,
                  const std::string& text) {
  return std::any_of(issues.begin(), issues.end(), [&](const ConfigIssue& i) {
    return i.key_path == key && i.ToString().find(text) != std::string::npos;
  });
}

TEST(Compat, enable_on_profile_none_sets_exactly_that_option) {
  const auto r =
      ResolveRun("compat_enable = [\"tensor_no_fork_diagonal_bug\"]");
  ASSERT_FALSE(HasErrors(r.issues));
  CompatOptions expect;
  expect.tensor_no_fork_diagonal_bug = true;
  EXPECT_TRUE(EffectiveCompatOptions(r.config) == expect);
  EXPECT_TRUE(EffectiveCompatOptions(r.config.run) == expect);
}

TEST(Compat, disable_on_fork_profile_clears_exactly_that_option) {
  const auto r = ResolveRun(
      "compat = \"imsrg_fork_79e9330\"\n"
      "compat_disable = [\"tokyo_writer_fork_79e9330\"]");
  ASSERT_FALSE(HasErrors(r.issues));
  CompatOptions expect = CompatOptionsFor(Compat::kImsrgFork79e9330);
  expect.tokyo_writer_fork_79e9330 = false;
  EXPECT_TRUE(EffectiveCompatOptions(r.config) == expect);
}

TEST(Compat, each_name_flips_exactly_its_own_field) {
  const auto names = CompatOptionNames();
  ASSERT_EQ(names.size(), 7u);
  CompatOptions all;
  for (const auto n : names) {
    CompatOptions o;
    ASSERT_TRUE(SetCompatOption(o, n, true)) << n;
    EXPECT_FALSE(o == CompatOptions{}) << n;
    // Setting any other name changes a different field.
    for (const auto m : names) {
      if (m == n) continue;
      CompatOptions p;
      ASSERT_TRUE(SetCompatOption(p, m, true));
      EXPECT_FALSE(p == o) << n << " " << m;
    }
    ASSERT_TRUE(SetCompatOption(all, n, true));
    ASSERT_TRUE(SetCompatOption(o, n, false));
    EXPECT_TRUE(o == CompatOptions{}) << n;
  }
  EXPECT_TRUE(all == CompatOptionsFor(Compat::kImsrgFork79e9330));
  CompatOptions o;
  EXPECT_FALSE(SetCompatOption(o, "nope", true));
  EXPECT_TRUE(o == CompatOptions{});
}

TEST(Compat, unknown_name_is_a_config_error) {
  const auto r = ResolveRun("compat_enable = [\"fork_step_default\"]");
  EXPECT_TRUE(HasErrors(r.issues));
  EXPECT_TRUE(HasIssueWith(r.issues, "run.compat_enable", "fork_step_default"));
  EXPECT_TRUE(
      HasIssueWith(r.issues, "run.compat_enable", "fork_operator_file_names"));
  const auto d = ResolveRun("compat_disable = [\"bogus\"]");
  EXPECT_TRUE(HasIssueWith(d.issues, "run.compat_disable", "bogus"));
}

TEST(Compat, name_in_both_lists_is_an_error) {
  const auto r = ResolveRun(
      "compat_enable = [\"fork_step_defaults\"]\n"
      "compat_disable = [\"fork_step_defaults\"]");
  EXPECT_TRUE(HasErrors(r.issues));
  EXPECT_TRUE(
      HasIssueWith(r.issues, "run.compat_disable", "fork_step_defaults"));
}

TEST(Compat, enabled_step_defaults_resolve_domega_0p2) {
  EXPECT_EQ(ResolveRun("").config.imsrg.step.domega, 0.1);
  const auto r = ResolveRun("compat_enable = [\"fork_step_defaults\"]");
  ASSERT_FALSE(HasErrors(r.issues));
  EXPECT_EQ(r.config.imsrg.step.domega, 0.2);
  const auto off = ResolveRun(
      "compat = \"imsrg_fork_79e9330\"\n"
      "compat_disable = [\"fork_step_defaults\"]");
  EXPECT_EQ(off.config.imsrg.step.domega, 0.1);
}

TEST(Compat, writer_round_trip_keeps_both_lists) {
  const auto r = ResolveRun(
      "compat_enable = [\"fork_step_defaults\", \"tensor_ninej_threshold_1e8\"]"
      "\ncompat_disable = [\"fork_operator_file_names\"]");
  ASSERT_FALSE(HasErrors(r.issues));
  const std::string text = ToToml(r.config, false);
  RawTree tree;
  ASSERT_TRUE(ReadTomlString(text, "rt", tree).empty());
  const auto back = Resolve(tree, ResolveOptions{false, false});
  ASSERT_FALSE(HasErrors(back.issues));
  EXPECT_EQ(back.config.run.compat_enable, r.config.run.compat_enable);
  EXPECT_EQ(back.config.run.compat_disable, r.config.run.compat_disable);
  EXPECT_EQ(back.config.run.compat_enable.size(), 2u);
  // The default config writes empty lists.
  const std::string def = ToToml(ResolveRun("").config, false);
  EXPECT_NE(def.find("compat_enable = []"), std::string::npos);
  EXPECT_NE(def.find("compat_disable = []"), std::string::npos);
}

TEST(Compat, hashes_ignore_the_debug_keys) {
  RunConfig base = ResolveRun("").config;
  base.reference.nucleus = "Al27";
  base.valence.space = "sd-shell";
  RunConfig c = base;
  c.run.compat_enable = {"fork_step_defaults", "tensor_no_fork_diagonal_bug"};
  c.run.compat_disable = {"fork_operator_file_names"};
  EXPECT_EQ(pipeline::FlowConfigHash(c), pipeline::FlowConfigHash(base));
  EXPECT_EQ(pipeline::PreparationHash(c), pipeline::PreparationHash(base));
  // The run.compat profile still enters the hashes (unchanged behaviour).
  c = base;
  c.run.compat = Compat::kImsrgFork79e9330;
  EXPECT_NE(pipeline::FlowConfigHash(c), pipeline::FlowConfigHash(base));
}

TEST(Compat, transform_output_name_uses_effective_options) {
  RunConfig cfg;
  transform::TransformContext ctx;
  ctx.config = &cfg;
  ctx.out_dir = "out";
  ctx.stem = "int";
  EXPECT_EQ(transform::TransformOutputPath(ctx, "TCM", false),
            std::filesystem::path("out/int_TCM.snt"));
  cfg.run.compat_enable = {"fork_operator_file_names"};
  EXPECT_EQ(transform::TransformOutputPath(ctx, "TCM", false),
            std::filesystem::path("out/intTCM.snt"));
  EXPECT_EQ(transform::TransformOutputPath(ctx, "FM2p_150", true),
            std::filesystem::path("out/intFM2p_150_2b.snt"));
  cfg.run.compat = Compat::kImsrgFork79e9330;
  cfg.run.compat_enable.clear();
  cfg.run.compat_disable = {"fork_operator_file_names"};
  EXPECT_EQ(transform::TransformOutputPath(ctx, "TCM", false),
            std::filesystem::path("out/int_TCM.snt"));
}

}  // namespace
}  // namespace imsrg::driver
