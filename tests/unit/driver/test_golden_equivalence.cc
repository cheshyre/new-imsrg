// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Golden equivalence of the design/16 section 2.11 examples: TOML vs legacy
// command line (design/16 section 6.3). The four TOML blocks and the four
// "Equivalent legacy command" lines are extracted from the design document
// itself; examples/config holds verbatim copies.
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/config/writer.h"
#include "tests/support/scratch_dir.h"

#ifndef IMSRG_SOURCE_DIR
#error "IMSRG_SOURCE_DIR must be defined by the build"
#endif

namespace imsrg::driver {
namespace {

std::string Slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) lines.push_back(l);
  return lines;
}

std::vector<std::string> SplitSpaces(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream in(s);
  std::string t;
  while (in >> t) out.push_back(t);
  return out;
}

struct Example {
  char tag;
  const char* file;
  std::string toml;                 ///< from the design doc
  std::vector<std::string> legacy;  ///< from the design doc
};

/// The four examples extracted from design/16 between "### 2.11" and "###
/// 2.12".
std::vector<Example> ExtractExamples() {
  std::vector<Example> ex = {{'a', "a_o16_imsrg2.toml", {}, {}},
                             {'b', "b_o16_imsrg2_3n.toml", {}, {}},
                             {'c', "c_o16_sd_shell.toml", {}, {}},
                             {'d', "d_o16_imsrg3_n7.toml", {}, {}}};
  const auto lines = SplitLines(
      Slurp(std::string(IMSRG_SOURCE_DIR) + "/design/16-config-and-cli.md"));
  bool inside_section = false;
  bool in_block = false;
  std::size_t block = 0;
  std::string current;
  for (const std::string& l : lines) {
    if (l.rfind("### 2.11", 0) == 0) {
      inside_section = true;
      continue;
    }
    if (l.rfind("### 2.12", 0) == 0) inside_section = false;
    if (!inside_section) continue;
    if (in_block) {
      if (l == "```") {
        in_block = false;
        if (block < ex.size()) ex[block].toml = current;
        ++block;
      } else {
        current += l + "\n";
      }
      continue;
    }
    if (l == "```toml") {
      in_block = true;
      current.clear();
      continue;
    }
    for (auto& e : ex) {
      const std::string prefix =
          std::string("Equivalent legacy command for (") + e.tag + "): `imsrg ";
      if (l.rfind(prefix, 0) == 0) {
        const std::size_t end = l.find('`', prefix.size());
        e.legacy = SplitSpaces(l.substr(prefix.size(), end - prefix.size()));
      }
    }
  }
  EXPECT_EQ(block, 4u)
      << "expected exactly 4 toml blocks in design/16 section 2.11";
  return ex;
}

std::string IssuesText(const ConfigIssues& issues) {
  std::string s;
  for (const auto& i : issues) s += "  " + i.ToString() + "\n";
  return s;
}

const std::vector<Example>& Examples() {
  static const auto* const kEx = new std::vector<Example>(ExtractExamples());
  return *kEx;
}

/// Loads `sources` and requires success without warnings.
RunConfig LoadOk(const ConfigSources& sources, const std::string& what) {
  auto r = LoadConfig(sources, LoadOptions{.check_files = false});
  if (!r) {
    ADD_FAILURE() << what << " failed:\n" << IssuesText(r.error());
    return RunConfig{};
  }
  EXPECT_TRUE(r.value().provenance.warnings.empty())
      << what << " produced warnings:\n"
      << IssuesText(r.value().provenance.warnings);
  return std::move(r).value();
}

/// A scratch directory that is the working directory for the test's lifetime.
/// The examples name their input files without a directory: in a TOML file
/// they resolve against the file's directory, in the legacy command against the
/// working directory (D37), so the example file is copied into the working
/// directory to make both sides agree.
struct Workdir {
  test_support::ScratchDir dir;
  test_support::ScopedCwd cwd{dir.path()};
};

RunConfig LoadToml(const Example& e, const Workdir& wd) {
  const std::filesystem::path file = wd.dir.path() / e.file;
  std::ofstream(file) << e.toml;
  ConfigSources s;
  s.config_file = file;
  return LoadOk(s, std::string("TOML of (") + e.tag + ")");
}

RunConfig LoadLegacy(const Example& e) {
  ConfigSources s;
  s.legacy_args = e.legacy;
  return LoadOk(s, std::string("legacy command of (") + e.tag + ")");
}

TEST(GoldenEquivalence, ExtractionFindsFourExamples) {
  ASSERT_EQ(Examples().size(), 4u);
  for (const auto& e : Examples()) {
    EXPECT_FALSE(e.toml.empty()) << e.tag;
    EXPECT_FALSE(e.legacy.empty()) << e.tag;
  }
}

TEST(GoldenEquivalence, ExampleFilesAreVerbatimCopiesOfTheDocBlocks) {
  for (const auto& e : Examples()) {
    const std::string path =
        std::string(IMSRG_SOURCE_DIR) + "/examples/config/" + e.file;
    EXPECT_EQ(Slurp(path), e.toml) << path;
  }
}

TEST(GoldenEquivalence, TomlAndLegacyResolveToTheSameConfig) {
  for (const auto& e : Examples()) {
    SCOPED_TRACE(std::string("example (") + e.tag + ")");
    const Workdir wd;
    const RunConfig t = LoadToml(e, wd);
    RunConfig l = LoadLegacy(e);
    // The one documented difference (D37): an absent `hartree_fock.mixing` is
    // DIIS in TOML and imsrg++'s linear mixing for legacy command lines.
    EXPECT_EQ(t.hartree_fock.mixing, HartreeFockMixing::kDiis);
    EXPECT_EQ(l.hartree_fock.mixing, HartreeFockMixing::kLinear);
    l.hartree_fock.mixing = HartreeFockMixing::kDiis;
    EXPECT_TRUE(t == l) << "RunConfig differs; TOML-side:\n"
                        << ToToml(t, false) << "\nlegacy-side:\n"
                        << ToToml(l, false);
    EXPECT_EQ(ToToml(t, false), ToToml(l, false));
  }
}

TEST(GoldenEquivalence, ExampleA) {
  const Workdir wd;
  const RunConfig c = LoadToml(Examples()[0], wd);
  EXPECT_EQ(c.basis.mass_number, 16);
  EXPECT_EQ(c.imsrg.steps, 1);
  EXPECT_EQ(c.output.stem, "magnus_O16_O16_hw16_e4_A16");
  EXPECT_EQ(
      c.output.flow_file,
      (wd.dir.path() / "output/magnus_O16_O16_hw16_e4_A16.flow").string());
  EXPECT_EQ(c.output.directory, (wd.dir.path() / "output").string());
  EXPECT_EQ(
      c.interaction.two_body.file.at(0),
      (wd.dir.path() / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz")
          .string());
  ASSERT_EQ(c.operators.list.size(), 1u);
  EXPECT_EQ(c.operators.list[0].canonical_name, "Rp2");
  EXPECT_EQ(c.interaction.two_body.e2max, 28);
  EXPECT_EQ(c.interaction.two_body.lmax, 14);
  EXPECT_EQ(c.hamiltonian.beta_cm_hw, 16.0);
  EXPECT_FALSE(c.interaction.three_body.has_value());
}

TEST(GoldenEquivalence, ExampleB) {
  const Workdir wd;
  const RunConfig c = LoadToml(Examples()[1], wd);
  ASSERT_TRUE(c.interaction.three_body.has_value());
  EXPECT_EQ(c.interaction.three_body->type, ThreeBodyType::kNo2b);
  EXPECT_EQ(c.interaction.three_body->e1max, 6);
  EXPECT_EQ(c.interaction.three_body->e2max, std::optional<int>(12));
  EXPECT_EQ(c.interaction.three_body->e3max, 18);
  EXPECT_EQ(c.basis.e3max, 12);
  EXPECT_TRUE(c.operators.list.empty());
}

TEST(GoldenEquivalence, ExampleC) {
  const Workdir wd;
  const RunConfig c = LoadToml(Examples()[2], wd);
  EXPECT_EQ(c.imsrg.steps, 2);
  EXPECT_EQ(c.valence.space, std::optional<std::string>("sd-shell"));
  EXPECT_EQ(c.imsrg.generator.core, Generator::kAtan);
  EXPECT_EQ(c.imsrg.generator.valence, Generator::kShellModelAtan);
  EXPECT_EQ(c.output.stem, "magnus_O16_sd-shell_hw20_e6_A16");
}

TEST(GoldenEquivalence, ExampleD) {
  const Workdir wd;
  const RunConfig c = LoadToml(Examples()[3], wd);
  EXPECT_TRUE(c.imsrg3.in_flow);
  EXPECT_EQ(c.imsrg3.variant, Imsrg3Variant::kN7);
  EXPECT_EQ(c.basis.e3max, 6);
  ASSERT_TRUE(c.interaction.three_body.has_value());
  EXPECT_EQ(c.interaction.three_body->type, ThreeBodyType::kNo2b);
}

TEST(HartreeFockMixingKey, LegacyIsLinearTomlIsDiisAndBothAreRecorded) {
  const Example& e = Examples()[0];
  const Workdir wd;
  const RunConfig toml = LoadToml(e, wd);
  const RunConfig legacy = LoadLegacy(e);
  EXPECT_EQ(toml.hartree_fock.mixing, HartreeFockMixing::kDiis);
  EXPECT_EQ(legacy.hartree_fock.mixing, HartreeFockMixing::kLinear);
  // The key shows up in resolved_config.toml, with the resolved value.
  EXPECT_NE(ToToml(toml, false).find("mixing = \"diis\""), std::string::npos);
  EXPECT_NE(ToToml(legacy, false).find("mixing = \"linear\""),
            std::string::npos);

  // An explicit `--set` wins over the legacy default.
  ConfigSources s;
  s.legacy_args = e.legacy;
  s.set_overrides = {"hartree_fock.mixing=diis"};
  EXPECT_EQ(LoadOk(s, "legacy with --set").hartree_fock.mixing,
            HartreeFockMixing::kDiis);
}

}  // namespace
}  // namespace imsrg::driver
