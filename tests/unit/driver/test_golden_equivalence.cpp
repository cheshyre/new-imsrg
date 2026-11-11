// Golden equivalence of the design/16 section 2.11 examples: TOML vs legacy command line
// (design/16 section 6.3). The four TOML blocks and the four "Equivalent legacy command" lines
// are extracted from the design document itself; examples/config holds verbatim copies.
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.hpp"
#include "imsrg/driver/config/writer.hpp"

#ifndef IMSRG_SOURCE_DIR
#error "IMSRG_SOURCE_DIR must be defined by the build"
#endif

namespace imsrg::driver {
namespace {

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string l;
  while (std::getline(in, l)) lines.push_back(l);
  return lines;
}

std::vector<std::string> split_spaces(const std::string& s) {
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

/// The four examples extracted from design/16 between "### 2.11" and "### 2.12".
std::vector<Example> extract_examples() {
  std::vector<Example> ex = {{'a', "a_o16_imsrg2.toml", {}, {}},
                             {'b', "b_o16_imsrg2_3n.toml", {}, {}},
                             {'c', "c_o16_sd_shell.toml", {}, {}},
                             {'d', "d_o16_imsrg3_n7.toml", {}, {}}};
  const auto lines =
      split_lines(slurp(std::string(IMSRG_SOURCE_DIR) + "/design/16-config-and-cli.md"));
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
        e.legacy = split_spaces(l.substr(prefix.size(), end - prefix.size()));
      }
    }
  }
  EXPECT_EQ(block, 4u) << "expected exactly 4 toml blocks in design/16 section 2.11";
  return ex;
}

std::string issues_text(const ConfigIssues& issues) {
  std::string s;
  for (const auto& i : issues) s += "  " + i.to_string() + "\n";
  return s;
}

const std::vector<Example>& examples() {
  static const std::vector<Example> ex = extract_examples();
  return ex;
}

/// Loads `sources` and requires success without warnings.
RunConfig load_ok(const ConfigSources& sources, const std::string& what) {
  auto r = load_config(sources, LoadOptions{.check_files = false});
  if (!r) {
    ADD_FAILURE() << what << " failed:\n" << issues_text(r.error());
    return RunConfig{};
  }
  EXPECT_TRUE(r.value().provenance.warnings.empty()) << what << " produced warnings:\n"
                                                     << issues_text(r.value().provenance.warnings);
  return std::move(r).value();
}

RunConfig load_toml(const Example& e) {
  ConfigSources s;
  s.config_text = e.toml;
  s.config_text_name = e.file;
  return load_ok(s, std::string("TOML of (") + e.tag + ")");
}

RunConfig load_legacy(const Example& e) {
  ConfigSources s;
  s.legacy_args = e.legacy;
  return load_ok(s, std::string("legacy command of (") + e.tag + ")");
}

TEST(GoldenEquivalence, ExtractionFindsFourExamples) {
  ASSERT_EQ(examples().size(), 4u);
  for (const auto& e : examples()) {
    EXPECT_FALSE(e.toml.empty()) << e.tag;
    EXPECT_FALSE(e.legacy.empty()) << e.tag;
  }
}

TEST(GoldenEquivalence, ExampleFilesAreVerbatimCopiesOfTheDocBlocks) {
  for (const auto& e : examples()) {
    const std::string path = std::string(IMSRG_SOURCE_DIR) + "/examples/config/" + e.file;
    EXPECT_EQ(slurp(path), e.toml) << path;
  }
}

TEST(GoldenEquivalence, TomlAndLegacyResolveToTheSameConfig) {
  for (const auto& e : examples()) {
    SCOPED_TRACE(std::string("example (") + e.tag + ")");
    const RunConfig t = load_toml(e);
    const RunConfig l = load_legacy(e);
    EXPECT_TRUE(t == l) << "RunConfig differs; TOML-side:\n"
                        << to_toml(t, false) << "\nlegacy-side:\n"
                        << to_toml(l, false);
    EXPECT_EQ(to_toml(t, false), to_toml(l, false));
  }
}

TEST(GoldenEquivalence, ExampleA) {
  const RunConfig c = load_toml(examples()[0]);
  EXPECT_EQ(c.basis.mass_number, 16);
  EXPECT_EQ(c.imsrg.steps, 1);
  EXPECT_EQ(c.output.stem, "magnus_O16_O16_hw16_e4_A16");
  EXPECT_EQ(c.output.flow_file, "output/magnus_O16_O16_hw16_e4_A16.flow");
  ASSERT_EQ(c.operators.list.size(), 1u);
  EXPECT_EQ(c.operators.list[0].canonical_name, "Rp2");
  EXPECT_EQ(c.interaction.two_body.e2max, 28);
  EXPECT_EQ(c.interaction.two_body.lmax, 14);
  EXPECT_EQ(c.hamiltonian.beta_cm_hw, 16.0);
  EXPECT_FALSE(c.interaction.three_body.has_value());
}

TEST(GoldenEquivalence, ExampleB) {
  const RunConfig c = load_toml(examples()[1]);
  ASSERT_TRUE(c.interaction.three_body.has_value());
  EXPECT_EQ(c.interaction.three_body->type, ThreeBodyType::No2b);
  EXPECT_EQ(c.interaction.three_body->e1max, 6);
  EXPECT_EQ(c.interaction.three_body->e2max, std::optional<int>(12));
  EXPECT_EQ(c.interaction.three_body->e3max, 18);
  EXPECT_EQ(c.basis.e3max, 12);
  EXPECT_TRUE(c.operators.list.empty());
}

TEST(GoldenEquivalence, ExampleC) {
  const RunConfig c = load_toml(examples()[2]);
  EXPECT_EQ(c.imsrg.steps, 2);
  EXPECT_EQ(c.valence.space, std::optional<std::string>("sd-shell"));
  EXPECT_EQ(c.imsrg.generator.core, Generator::Atan);
  EXPECT_EQ(c.imsrg.generator.valence, Generator::ShellModelAtan);
  EXPECT_EQ(c.output.stem, "magnus_O16_sd-shell_hw20_e6_A16");
}

TEST(GoldenEquivalence, ExampleD) {
  const RunConfig c = load_toml(examples()[3]);
  EXPECT_TRUE(c.imsrg3.in_flow);
  EXPECT_EQ(c.imsrg3.variant, Imsrg3Variant::N7);
  EXPECT_EQ(c.basis.e3max, 6);
  ASSERT_TRUE(c.interaction.three_body.has_value());
  EXPECT_EQ(c.interaction.three_body->type, ThreeBodyType::No2b);
}

}  // namespace
}  // namespace imsrg::driver
