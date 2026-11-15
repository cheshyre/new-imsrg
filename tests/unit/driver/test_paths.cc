// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Relative-path rules of `LoadConfig` (design/16 section 1.8, D37): relative
// paths of a TOML file resolve against the file's directory; legacy and
// command-line (`--set`) values, config text and defaults resolve against the
// working directory; the resolved config records absolute paths.
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/config/writer.h"
#include "tests/support/scratch_dir.h"

namespace imsrg::driver {
namespace {

namespace fs = std::filesystem;

using test_support::ScopedCwd;
using test_support::ScratchDir;

constexpr char kToml[] = R"(
[basis]
emax = 4
hw = 16.0

[reference]
nucleus = "O16"

[interaction.two_body]
file = "data/vnn.me2j.gz"
e1max = 14

[interaction.three_body]
file = "../shared/v3.stream.bin"
type = "no2b"
e1max = 6
e3max = 18

[run]
scratch_dir = "scratch"

[output]
directory = "out"

[operators]
from_file = [
  {name = "Q", file_2b = "ops/q2b.dat", file_3b = "ops/q3b.dat"},
]
)";

constexpr char kMinimalToml[] =
    "[basis]\nemax = 4\nhw = 16.0\n[reference]\nnucleus = \"O16\"\n"
    "[interaction.two_body]\nfile = \"v.me2j\"\ne1max = 14\n";

void WriteFile(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text;
}

RunConfig LoadOk(const ConfigSources& s) {
  auto r = LoadConfig(s, LoadOptions{.check_files = false});
  if (!r) {
    std::string msg;
    for (const auto& i : r.error()) msg += i.ToString() + "\n";
    ADD_FAILURE() << msg;
    return RunConfig{};
  }
  return std::move(r).value();
}

TEST(ConfigPaths, TomlFileResolvesAgainstItsDirectoryNotTheWorkingDirectory) {
  const ScratchDir root;
  const fs::path conf_dir = root.Sub("project/conf");
  const fs::path elsewhere = root.Sub("elsewhere");
  WriteFile(conf_dir / "run.toml", kToml);
  const ScopedCwd cwd(elsewhere);

  ConfigSources s;
  s.config_file = conf_dir / "run.toml";
  const RunConfig c = LoadOk(s);
  EXPECT_EQ(c.interaction.two_body.file.at(0),
            (conf_dir / "data/vnn.me2j.gz").string());
  ASSERT_TRUE(c.interaction.three_body);
  EXPECT_EQ(c.interaction.three_body->file,
            (root.path() / "project/shared/v3.stream.bin").string());
  EXPECT_EQ(c.run.scratch_dir, (conf_dir / "scratch").string());
  EXPECT_EQ(c.output.directory, (conf_dir / "out").string());
  EXPECT_EQ(fs::path(c.output.flow_file).parent_path(), conf_dir / "out");
  ASSERT_EQ(c.operators.from_file.size(), 1u);
  EXPECT_EQ(c.operators.from_file[0].file_2b,
            (conf_dir / "ops/q2b.dat").string());
  EXPECT_EQ(c.operators.from_file[0].file_3b,
            std::optional<std::string>((conf_dir / "ops/q3b.dat").string()));
}

TEST(ConfigPaths, RelativeConfigFilePathAndAbsolutePathsInsideIt) {
  const ScratchDir root;
  const fs::path conf_dir = root.Sub("conf");
  const fs::path other = root.Sub("other");
  const fs::path abs_file = other / "fixed.me2j";
  std::string text = kToml;
  const std::string rel = "data/vnn.me2j.gz";
  text.replace(text.find(rel), rel.size(), abs_file.string());
  WriteFile(conf_dir / "run.toml", text);
  const ScopedCwd cwd(other);

  ConfigSources s;
  s.config_file = fs::path("../conf/run.toml");  // relative to the cwd
  const RunConfig c = LoadOk(s);
  EXPECT_EQ(c.interaction.two_body.file.at(0), abs_file.string());
  EXPECT_EQ(c.output.directory, (root.path() / "conf/out").string());
}

TEST(ConfigPaths, ConfigTextResolvesAgainstTheWorkingDirectory) {
  const ScratchDir root;
  const fs::path cwd_dir = root.Sub("cwd");
  const ScopedCwd cwd(cwd_dir);
  ConfigSources s;
  s.config_text = kToml;
  const RunConfig c = LoadOk(s);
  EXPECT_EQ(c.interaction.two_body.file.at(0),
            (cwd_dir / "data/vnn.me2j.gz").string());
  EXPECT_EQ(c.output.directory, (cwd_dir / "out").string());
}

TEST(ConfigPaths, SetOverridesAndDefaultsUseTheWorkingDirectory) {
  const ScratchDir root;
  const fs::path conf_dir = root.Sub("conf");
  const fs::path cwd_dir = root.Sub("cwd");
  WriteFile(conf_dir / "run.toml", kToml);
  const ScopedCwd cwd(cwd_dir);

  ConfigSources s;
  s.config_file = conf_dir / "run.toml";
  s.set_overrides = {"output.directory=elsewhere",
                     "interaction.two_body.file=cli.me2j"};
  const RunConfig c = LoadOk(s);
  EXPECT_EQ(c.output.directory, (cwd_dir / "elsewhere").string());
  EXPECT_EQ(c.interaction.two_body.file.at(0), (cwd_dir / "cli.me2j").string());
  // Still from the file:
  EXPECT_EQ(c.run.scratch_dir, (conf_dir / "scratch").string());

  // A file without `output.directory`: the default directory is relative to
  // the working directory (it is not a value of the file).
  WriteFile(conf_dir / "min.toml", kMinimalToml);
  ConfigSources m;
  m.config_file = conf_dir / "min.toml";
  const RunConfig d = LoadOk(m);
  EXPECT_EQ(d.output.directory, (cwd_dir / "output").string());
  EXPECT_EQ(d.interaction.two_body.file.at(0), (conf_dir / "v.me2j").string());
  EXPECT_EQ(fs::path(d.output.flow_file).parent_path(), cwd_dir / "output");
}

TEST(ConfigPaths, LegacyValuesResolveAgainstTheWorkingDirectory) {
  const ScratchDir root;
  const fs::path cwd_dir = root.Sub("cwd");
  const ScopedCwd cwd(cwd_dir);
  ConfigSources s;
  s.legacy_args = {"2bme=vnn.me2j",
                   "file2e1max=14",
                   "emax=4",
                   "hw=16",
                   "reference=O16",
                   "valence_space=O16",
                   "intfile=results/run1",
                   "flowfile=results/run1.flow"};
  const RunConfig c = LoadOk(s);
  EXPECT_EQ(c.interaction.two_body.file.at(0), (cwd_dir / "vnn.me2j").string());
  EXPECT_EQ(c.output.directory, (cwd_dir / "results").string());
  EXPECT_EQ(c.output.flow_file, (cwd_dir / "results/run1.flow").string());

  // "intfile=stem" (no directory) means the working directory itself.
  ConfigSources t;
  t.legacy_args = {"2bme=vnn.me2j", "file2e1max=14", "emax=4",
                   "hw=16",         "reference=O16", "intfile=stem"};
  EXPECT_EQ(LoadOk(t).output.directory, cwd_dir.string());
}

TEST(ConfigPaths, ResolvedConfigRecordsAbsolutePaths) {
  const ScratchDir root;
  const fs::path conf_dir = root.Sub("conf");
  const fs::path cwd_dir = root.Sub("cwd");
  WriteFile(conf_dir / "run.toml", kToml);
  const ScopedCwd cwd(cwd_dir);
  ConfigSources s;
  s.config_file = conf_dir / "run.toml";
  const std::string text = ToToml(LoadOk(s), false);
  EXPECT_NE(text.find("\"" + (conf_dir / "data/vnn.me2j.gz").string() + "\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("directory = \"" + (conf_dir / "out").string() + "\""),
            std::string::npos);
  // The written file loads back to the same paths from any directory.
  const fs::path written = root.Sub("written") / "resolved_config.toml";
  WriteFile(written, text);
  const ScopedCwd elsewhere(root.Sub("third"));
  ConfigSources back;
  back.config_file = written;
  const RunConfig again = LoadOk(back);
  EXPECT_EQ(again.interaction.two_body.file.at(0),
            (conf_dir / "data/vnn.me2j.gz").string());
  EXPECT_EQ(again.output.directory, (conf_dir / "out").string());
}

TEST(ConfigPaths, ConfigFileInTheWorkingDirectoryEqualsLegacy) {
  // A config in the working directory and the equivalent legacy command
  // resolve to the same paths.
  const ScratchDir root;
  WriteFile(root.path() / "run.toml", kMinimalToml);
  const ScopedCwd cwd(root.path());
  ConfigSources t;
  t.config_file = fs::path("run.toml");
  ConfigSources l;
  l.legacy_args = {"2bme=v.me2j", "file2e1max=14", "emax=4",
                   "hw=16",       "reference=O16", "valence_space=O16"};
  const RunConfig a = LoadOk(t);
  const RunConfig b = LoadOk(l);
  EXPECT_EQ(a.interaction.two_body.file, b.interaction.two_body.file);
  EXPECT_EQ(a.output.directory, b.output.directory);
  EXPECT_EQ(a.output.flow_file, b.output.flow_file);
}

}  // namespace
}  // namespace imsrg::driver
