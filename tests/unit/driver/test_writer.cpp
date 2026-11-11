// Tests of the resolved-config writer (design/16 sections 1.3, 1.7, 6.1).
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/driver/config/schema_keys.hpp"
#include "imsrg/driver/config/toml_reader.hpp"
#include "imsrg/driver/config/writer.hpp"

namespace imsrg::driver {
namespace {

bool has_line(const std::string& text, std::string_view line) {
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) {
    if (l == line) return true;
  }
  return false;
}

RunConfig sample_config() {
  RunConfig cfg;
  cfg.basis.emax = 4;
  cfg.basis.hw = 16.0;
  cfg.imsrg.method = Method::Magnus;
  cfg.interaction.two_body.file = {"x.me2j"};
  return cfg;
}

OperatorRequest request(std::string name) {
  OperatorRequest r;
  r.canonical_name = std::move(name);
  return r;
}

TEST(Writer, FirstLineIsHeaderComment) {
  const std::string text = to_toml(sample_config(), false);
  const std::string first = text.substr(0, text.find('\n'));
  EXPECT_EQ(first,
            "# imsrg resolved configuration (design/16-config-and-cli.md); reproduce with: imsrg "
            "--config <this file>");
}

TEST(Writer, ScalarFormats) {
  const std::string text = to_toml(sample_config(), false);
  EXPECT_TRUE(has_line(text, "emax = 4"));
  EXPECT_TRUE(has_line(text, "hw = 16.0"));
  EXPECT_TRUE(has_line(text, "domega = 0.10000000000000001"));
  EXPECT_TRUE(has_line(text, "eta_criterion = 9.9999999999999995e-07"));
  EXPECT_TRUE(has_line(text, "method = \"magnus\""));
  EXPECT_TRUE(has_line(text, "use_brueckner_bch = false"));
}

TEST(Writer, TableHeadersAreSeparatedByBlankLine) {
  const std::string text = to_toml(sample_config(), false);
  EXPECT_NE(text.find("\n\n[basis]\nemax = 4\n"), std::string::npos);
}

TEST(Writer, SingleFileIsPlainStringTwoFilesAreArray) {
  RunConfig cfg = sample_config();
  EXPECT_TRUE(has_line(to_toml(cfg, false), "file = \"x.me2j\""));
  cfg.interaction.two_body.file = {"sp.dat", "vnn.dat"};
  EXPECT_TRUE(has_line(to_toml(cfg, false), "file = [\"sp.dat\", \"vnn.dat\"]"));
}

TEST(Writer, EmptyStringArrayIsBrackets) {
  const std::string text = to_toml(sample_config(), false);
  EXPECT_TRUE(has_line(text, "spwf = []"));
}

TEST(Writer, AbsentOptionalsAreOmitted) {
  const std::string text = to_toml(sample_config(), false);
  EXPECT_EQ(text.find("lmax3"), std::string::npos);
  EXPECT_EQ(text.find("emax_unocc"), std::string::npos);
  EXPECT_EQ(text.find("nucleus"), std::string::npos);
  EXPECT_EQ(text.find("[interaction.three_body]"), std::string::npos);
  EXPECT_EQ(text.find("[imsrg.space]"), std::string::npos);
  EXPECT_EQ(text.find("[provenance]"), std::string::npos);
}

TEST(Writer, EngagedOptionalsAreWritten) {
  RunConfig cfg = sample_config();
  cfg.basis.lmax3 = 3;
  cfg.reference.nucleus = "O16";
  cfg.interaction.three_body = ThreeBodyInputConfig{};
  cfg.interaction.three_body->file = "n.bin";
  cfg.interaction.three_body->e2max = 12;
  cfg.imsrg.space = ImsrgSpaceConfig{};
  cfg.imsrg.space->emax = 2;
  const std::string text = to_toml(cfg, false);
  EXPECT_TRUE(has_line(text, "lmax3 = 3"));
  EXPECT_TRUE(has_line(text, "nucleus = \"O16\""));
  EXPECT_TRUE(has_line(text, "[interaction.three_body]"));
  EXPECT_TRUE(has_line(text, "file = \"n.bin\""));
  EXPECT_TRUE(has_line(text, "e2max = 12"));
  EXPECT_TRUE(has_line(text, "[imsrg.space]"));
  EXPECT_TRUE(has_line(text, "emax_3body = 0"));
}

TEST(Writer, StringEscaping) {
  RunConfig cfg = sample_config();
  cfg.output.directory = "a\"b\\c\nd\te\x01";
  const std::string text = to_toml(cfg, false);
  EXPECT_TRUE(has_line(text, "directory = \"a\\\"b\\\\c\\nd\\te\\u0001\""));
}

TEST(Writer, OperatorsAreCanonicalNames) {
  RunConfig cfg = sample_config();
  cfg.operators.list = {request("Rp2"), request("rhop_3")};
  const std::string text = to_toml(cfg, false);
  EXPECT_TRUE(has_line(text, "list = [\"Rp2\", \"rhop_3\"]"));
  EXPECT_TRUE(has_line(text, "first_order = []"));
}

TEST(Writer, FromFileInlineTable) {
  RunConfig cfg = sample_config();
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
  const std::string text = to_toml(cfg, false);
  const std::size_t line_start = text.find("from_file = ");
  ASSERT_NE(line_start, std::string::npos);
  const std::string line = text.substr(line_start, text.find('\n', line_start) - line_start);
  EXPECT_NE(line.find("{ name = \"A\", rank = 0, dtz = 0, parity = 0, particle_rank = 2, "
                      "file_2b = \"a.dat\", format = "),
            std::string::npos)
      << line;
  EXPECT_NE(line.find("file_2b = \"b2.dat\", file_3b = \"b3.dat\""), std::string::npos) << line;
  EXPECT_EQ(line.find("file_3b"), line.rfind("file_3b"));  // only entry B has one
}

TEST(Writer, ProvenanceTable) {
  RunConfig cfg = sample_config();
  cfg.provenance.version = "0.1";
  cfg.provenance.git_hash = "abc123";
  cfg.provenance.date = "2026-01-01T00:00:00Z";
  cfg.provenance.command_line = "imsrg --config \"a b\".toml";
  cfg.provenance.legacy_args = {"emax=4", "hw=16"};
  cfg.provenance.applied_rules = {"rule one"};
  cfg.provenance.warnings.push_back(make_warning("W-TEST", "basis.emax", "a warning",
                                                 SourceRef{SourceRef::Kind::None, "", 0, 0}));
  const std::string with = to_toml(cfg, true);
  const std::string without = to_toml(cfg, false);
  const std::size_t pos = with.rfind("\n[provenance]\n");
  ASSERT_NE(pos, std::string::npos);
  EXPECT_EQ(with.substr(0, pos), without);  // provenance is strictly appended
  const std::string tail = with.substr(pos);
  EXPECT_TRUE(has_line(tail, "version = \"0.1\""));
  EXPECT_TRUE(has_line(tail, "git_hash = \"abc123\""));
  EXPECT_TRUE(has_line(tail, "date = \"2026-01-01T00:00:00Z\""));
  EXPECT_TRUE(has_line(tail, "command_line = \"imsrg --config \\\"a b\\\".toml\""));
  EXPECT_TRUE(has_line(tail, "legacy_args = [\"emax=4\", \"hw=16\"]"));
  EXPECT_TRUE(has_line(tail, "applied_rules = [\"rule one\"]"));
  EXPECT_NE(tail.find("warnings = [\""), std::string::npos);
}

TEST(Writer, Deterministic) {
  const RunConfig cfg = sample_config();
  EXPECT_EQ(to_toml(cfg, true), to_toml(cfg, true));
  EXPECT_EQ(to_toml(cfg, false), to_toml(cfg, false));
}

TEST(Writer, OutputReadsBackWithoutIssuesAndContainsEveryKey) {
  RunConfig cfg = sample_config();
  cfg.reference.nucleus = "O16";
  cfg.operators.list = {request("Rp2")};
  cfg.interaction.three_body = ThreeBodyInputConfig{};
  cfg.interaction.three_body->file = "n.bin";
  cfg.imsrg.space = ImsrgSpaceConfig{};
  cfg.provenance.version = "1";
  const std::string text = to_toml(cfg, true);

  RawTree tree;
  const ConfigIssues issues = read_toml_string(text, "roundtrip", tree);
  EXPECT_TRUE(issues.empty()) << (issues.empty() ? "" : issues[0].to_string());

  // Every written `key = ` line (outside [provenance]) is in the tree under its table.
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

TEST(Writer, WriteResolvedConfigCreatesDirectories) {
  static std::atomic<int> counter{0};
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("imsrg_test_writer_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
       "_" + std::to_string(counter++));
  std::filesystem::remove_all(root);
  const std::filesystem::path file = root / "a" / "b" / "resolved_config.toml";
  const RunConfig cfg = sample_config();
  write_resolved_config(cfg, file);
  ASSERT_TRUE(std::filesystem::exists(file));
  std::ifstream in(file, std::ios::binary);
  const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_EQ(content, to_toml(cfg));
  std::filesystem::remove_all(root);
}

TEST(Writer, WriteResolvedConfigFailureThrowsNamingPath) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "imsrg_test_writer_fail";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  // The target is an existing directory: it cannot be opened as a file.
  try {
    write_resolved_config(sample_config(), root);
    FAIL() << "expected InputError";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find(root.string()), std::string::npos);
  }
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace imsrg::driver
