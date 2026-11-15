// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of the resolved-config writer (design/16 sections 1.3, 1.7, 6.1).
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/schema_keys.h"
#include "imsrg/driver/config/toml_reader.h"
#include "imsrg/driver/config/writer.h"

namespace imsrg::driver {
namespace {

bool HasLine(const std::string& text, std::string_view line) {
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) {
    if (l == line) return true;
  }
  return false;
}

RunConfig SampleConfig() {
  RunConfig cfg;
  cfg.basis.emax = 4;
  cfg.basis.hw = 16.0;
  cfg.imsrg.method = Method::kMagnus;
  cfg.interaction.two_body.file = {"x.me2j"};
  return cfg;
}

OperatorRequest Request(std::string name) {
  OperatorRequest r;
  r.canonical_name = std::move(name);
  return r;
}

TEST(Writer, FirstLineIsHeaderComment) {
  const std::string text = ToToml(SampleConfig(), false);
  const std::string first = text.substr(0, text.find('\n'));
  EXPECT_EQ(first,
            "# imsrg resolved configuration (design/16-config-and-cli.md); "
            "reproduce with: imsrg "
            "--config <this file>");
}

TEST(Writer, ScalarFormats) {
  const std::string text = ToToml(SampleConfig(), false);
  EXPECT_TRUE(HasLine(text, "emax = 4"));
  EXPECT_TRUE(HasLine(text, "hw = 16.0"));
  EXPECT_TRUE(HasLine(text, "domega = 0.10000000000000001"));
  EXPECT_TRUE(HasLine(text, "eta_criterion = 9.9999999999999995e-07"));
  EXPECT_TRUE(HasLine(text, "method = \"magnus\""));
  EXPECT_TRUE(HasLine(text, "use_brueckner_bch = false"));
}

TEST(Writer, TableHeadersAreSeparatedByBlankLine) {
  const std::string text = ToToml(SampleConfig(), false);
  EXPECT_NE(text.find("\n\n[basis]\nemax = 4\n"), std::string::npos);
}

TEST(Writer, SingleFileIsPlainStringTwoFilesAreArray) {
  RunConfig cfg = SampleConfig();
  EXPECT_TRUE(HasLine(ToToml(cfg, false), "file = \"x.me2j\""));
  cfg.interaction.two_body.file = {"sp.dat", "vnn.dat"};
  EXPECT_TRUE(HasLine(ToToml(cfg, false), "file = [\"sp.dat\", \"vnn.dat\"]"));
}

TEST(Writer, EmptyStringArrayIsBrackets) {
  const std::string text = ToToml(SampleConfig(), false);
  EXPECT_TRUE(HasLine(text, "spwf = []"));
}

TEST(Writer, AbsentOptionalsAreOmitted) {
  const std::string text = ToToml(SampleConfig(), false);
  EXPECT_EQ(text.find("lmax3"), std::string::npos);
  EXPECT_EQ(text.find("emax_unocc"), std::string::npos);
  EXPECT_EQ(text.find("nucleus"), std::string::npos);
  EXPECT_EQ(text.find("[interaction.three_body]"), std::string::npos);
  EXPECT_EQ(text.find("[imsrg.space]"), std::string::npos);
  EXPECT_EQ(text.find("[provenance]"), std::string::npos);
}

TEST(Writer, EngagedOptionalsAreWritten) {
  RunConfig cfg = SampleConfig();
  cfg.basis.lmax3 = 3;
  cfg.reference.nucleus = "O16";
  cfg.interaction.three_body = ThreeBodyInputConfig{};
  cfg.interaction.three_body->file = "n.bin";
  cfg.interaction.three_body->e2max = 12;
  cfg.imsrg.space = ImsrgSpaceConfig{};
  cfg.imsrg.space->emax = 2;
  const std::string text = ToToml(cfg, false);
  EXPECT_TRUE(HasLine(text, "lmax3 = 3"));
  EXPECT_TRUE(HasLine(text, "nucleus = \"O16\""));
  EXPECT_TRUE(HasLine(text, "[interaction.three_body]"));
  EXPECT_TRUE(HasLine(text, "file = \"n.bin\""));
  EXPECT_TRUE(HasLine(text, "e2max = 12"));
  EXPECT_TRUE(HasLine(text, "[imsrg.space]"));
  EXPECT_TRUE(HasLine(text, "emax_3body = 0"));
}

TEST(Writer, StringEscaping) {
  RunConfig cfg = SampleConfig();
  cfg.output.directory = "a\"b\\c\nd\te\x01";
  const std::string text = ToToml(cfg, false);
  EXPECT_TRUE(HasLine(text, "directory = \"a\\\"b\\\\c\\nd\\te\\u0001\""));
}

TEST(Writer, OperatorsAreCanonicalNames) {
  RunConfig cfg = SampleConfig();
  cfg.operators.list = {Request("Rp2"), Request("rhop_3")};
  const std::string text = ToToml(cfg, false);
  EXPECT_TRUE(HasLine(text, "list = [\"Rp2\", \"rhop_3\"]"));
  EXPECT_TRUE(HasLine(text, "first_order = []"));
}

TEST(Writer, FromFileInlineTable) {
  RunConfig cfg = SampleConfig();
  OperatorFromFile a;
  a.name = "A";
  a.file_2b = "a.dat";
  OperatorFromFile b;
  b.name = "B";
  b.rank = 0;
  b.particle_rank = 3;
  b.file_2b = "b2.dat";
  b.file_3b = "b3.dat";
  cfg.operators.from_file = {a, b};
  const std::string text = ToToml(cfg, false);
  const std::size_t line_start = text.find("from_file = ");
  ASSERT_NE(line_start, std::string::npos);
  const std::string line =
      text.substr(line_start, text.find('\n', line_start) - line_start);
  EXPECT_NE(
      line.find(
          "{ name = \"A\", rank = 0, dtz = 0, parity = 0, particle_rank = 2, "
          "file_2b = \"a.dat\", format = "),
      std::string::npos)
      << line;
  EXPECT_NE(line.find("file_2b = \"b2.dat\", file_3b = \"b3.dat\""),
            std::string::npos)
      << line;
  EXPECT_EQ(line.find("file_3b"),
            line.rfind("file_3b"));  // only entry B has one
}

TEST(Writer, ProvenanceTable) {
  RunConfig cfg = SampleConfig();
  cfg.provenance.version = "0.1";
  cfg.provenance.git_hash = "abc123";
  cfg.provenance.date = "2026-01-01T00:00:00Z";
  cfg.provenance.command_line = "imsrg --config \"a b\".toml";
  cfg.provenance.legacy_args = {"emax=4", "hw=16"};
  cfg.provenance.applied_rules = {"rule one"};
  cfg.provenance.warnings.push_back(
      MakeWarning("W-TEST", "basis.emax", "a warning",
                  SourceRef{SourceRef::Kind::kNone, "", 0, 0}));
  const std::string with = ToToml(cfg, true);
  const std::string without = ToToml(cfg, false);
  const std::size_t pos = with.rfind("\n[provenance]\n");
  ASSERT_NE(pos, std::string::npos);
  EXPECT_EQ(with.substr(0, pos), without);  // provenance is strictly appended
  const std::string tail = with.substr(pos);
  EXPECT_TRUE(HasLine(tail, "version = \"0.1\""));
  EXPECT_TRUE(HasLine(tail, "git_hash = \"abc123\""));
  EXPECT_TRUE(HasLine(tail, "date = \"2026-01-01T00:00:00Z\""));
  EXPECT_TRUE(
      HasLine(tail, "command_line = \"imsrg --config \\\"a b\\\".toml\""));
  EXPECT_TRUE(HasLine(tail, "legacy_args = [\"emax=4\", \"hw=16\"]"));
  EXPECT_TRUE(HasLine(tail, "applied_rules = [\"rule one\"]"));
  EXPECT_NE(tail.find("warnings = [\""), std::string::npos);
}

TEST(Writer, Deterministic) {
  const RunConfig cfg = SampleConfig();
  EXPECT_EQ(ToToml(cfg, true), ToToml(cfg, true));
  EXPECT_EQ(ToToml(cfg, false), ToToml(cfg, false));
}

TEST(Writer, OutputReadsBackWithoutIssuesAndContainsEveryKey) {
  RunConfig cfg = SampleConfig();
  cfg.reference.nucleus = "O16";
  cfg.operators.list = {Request("Rp2")};
  cfg.interaction.three_body = ThreeBodyInputConfig{};
  cfg.interaction.three_body->file = "n.bin";
  cfg.imsrg.space = ImsrgSpaceConfig{};
  cfg.provenance.version = "1";
  const std::string text = ToToml(cfg, true);

  RawTree tree;
  const ConfigIssues issues = ReadTomlString(text, "roundtrip", tree);
  EXPECT_TRUE(issues.empty()) << (issues.empty() ? "" : issues[0].ToString());

  // Every written `key = ` line (outside [provenance]) is in the tree under its
  // table.
  std::istringstream in(text);
  std::string line;
  std::string table;
  std::size_t checked = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line.front() == '#') continue;
    if (line.front() == '[') {
      table = line.substr(1, line.size() - 2);
      continue;
    }
    if (table == "provenance") continue;
    const std::string key = line.substr(0, line.find(" = "));
    EXPECT_TRUE(tree.contains(table + "." + key)) << table << "." << key;
    ++checked;
  }
  EXPECT_GT(checked, 30u);
}

TEST(Writer, FactorizedRecipeValuesRoundTrip) {
  RunConfig cfg = SampleConfig();
  cfg.reference.nucleus = "O16";
  cfg.imsrg3.factorized = true;
  cfg.imsrg3.factorized_1b_intermediates = false;
  cfg.imsrg3.factorized_2b_intermediates = true;
  cfg.imsrg.omega.hunter_gatherer = true;
  cfg.imsrg.step.omega_norm_max = 0.1;
  const std::string text = ToToml(cfg, false);
  EXPECT_TRUE(HasLine(text, "factorized = true"));
  EXPECT_TRUE(HasLine(text, "factorized_1b_intermediates = false"));
  EXPECT_TRUE(HasLine(text, "factorized_2b_intermediates = true"));
  EXPECT_TRUE(HasLine(text, "hunter_gatherer = true"));
  EXPECT_TRUE(HasLine(text, "omega_norm_max = 0.10000000000000001"));
  RawTree tree;
  EXPECT_TRUE(ReadTomlString(text, "rt", tree).empty());
  for (const char* k :
       {"imsrg3.factorized", "imsrg3.factorized_1b_intermediates",
        "imsrg3.factorized_2b_intermediates", "imsrg.omega.hunter_gatherer",
        "imsrg.step.omega_norm_max"}) {
    EXPECT_TRUE(tree.contains(k)) << k;
  }
}

TEST(Writer, WriteResolvedConfigCreatesDirectories) {
  static std::atomic<int> counter{0};
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("imsrg_test_writer_" +
       std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
       std::to_string(counter++));
  std::filesystem::remove_all(root);
  const std::filesystem::path file = root / "a" / "b" / "resolved_config.toml";
  const RunConfig cfg = SampleConfig();
  WriteResolvedConfig(cfg, file);
  ASSERT_TRUE(std::filesystem::exists(file));
  std::ifstream in(file, std::ios::binary);
  const std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(content, ToToml(cfg));
  std::filesystem::remove_all(root);
}

TEST(Writer, WriteResolvedConfigFailureThrowsNamingPath) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "imsrg_test_writer_fail";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  // The target is an existing directory: it cannot be opened as a file.
  try {
    WriteResolvedConfig(SampleConfig(), root);
    FAIL() << "expected InputError";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find(root.string()), std::string::npos);
  }
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace imsrg::driver
