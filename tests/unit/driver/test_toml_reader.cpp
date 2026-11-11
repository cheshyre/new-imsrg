// Tests of the TOML reader and of --set (design/16-config-and-cli.md sections 2.11, 2.12, 6.1).
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/toml_reader.hpp"

namespace imsrg::driver {
namespace {

// The four TOML examples of design/16 section 2.11 (a) to (d), verbatim.
constexpr std::string_view kExampleA = R"(
[run]
task = "imsrg"

[basis]
emax = 4
hw = 16.0
kind = "hf"

[reference]
nucleus = "O16"

[interaction.two_body]
file   = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
format = "me2j"
e1max  = 14
e2max  = 28
lmax   = 14

[imsrg]
method = "magnus"
[imsrg.generator]
core = "atan"

[operators]
list = ["Rp2"]

[output]
directory = "output"
)";

constexpr std::string_view kExampleB = R"(
[basis]
emax = 4
hw = 16.0
e3max = 12                      # <= file e3max (18)

[reference]
nucleus = "O16"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
e1max = 14
e2max = 28
lmax  = 14

[interaction.three_body]
file  = "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
type  = "no2b"
e1max = 6
e2max = 12
e3max = 18

[imsrg]
method = "magnus"
)";

constexpr std::string_view kExampleC = R"(
[basis]
emax = 6
hw = 20.0

[reference]
nucleus = "O16"                 # HF reference; core of sd-shell is also O16

[valence]
space = "sd-shell"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"   # hw of file need not equal basis.hw only if the file is hw-independent; user responsibility
e1max = 14
e2max = 28
lmax  = 14

[imsrg]
method = "magnus"
# steps omitted: auto -> 2 because valence orbits exist
[imsrg.generator]
core    = "atan"
valence = "shell_model_atan"

[operators]
list = ["Rp2"]

[output]
valence_file_format = "tokyo"
)";

constexpr std::string_view kExampleD = R"(
[basis]
emax = 2
hw = 16.0
e3max = 6

[reference]
nucleus = "O16"

[interaction.two_body]
file  = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
e1max = 14
e2max = 28
lmax  = 14

[interaction.three_body]
file  = "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"
type  = "no2b"
e1max = 6
e2max = 12
e3max = 18

[imsrg]
method = "magnus"

[imsrg3]
in_flow = true
variant = "n7"
threebody_threshold = 0.0
)";

struct Read {
  RawTree tree;
  ConfigIssues issues;
};

Read read(std::string_view text) {
  Read r;
  r.issues = read_toml_string(text, "<test>", r.tree);
  return r;
}

template <class T>
const T& value_of(const RawTree& tree, std::string_view path) {
  const RawValue* v = tree.find(path);
  EXPECT_NE(v, nullptr) << path;
  static const T fallback{};
  if (v == nullptr) return fallback;
  const T* p = v->get_if<T>();
  EXPECT_NE(p, nullptr) << path << " has type " << v->type_name();
  return p != nullptr ? *p : fallback;
}

// ---- value types -----------------------------------------------------------------------------

TEST(TomlReader, IntegerValue) {
  const Read r = read("[basis]\nemax = 4\n");
  EXPECT_TRUE(r.issues.empty());
  EXPECT_EQ(value_of<std::int64_t>(r.tree, "basis.emax"), 4);
}

TEST(TomlReader, FloatAndIntegerStayDistinct) {
  const Read a = read("[basis]\nhw = 16.0\n");
  EXPECT_TRUE(a.issues.empty());
  EXPECT_EQ(value_of<double>(a.tree, "basis.hw"), 16.0);
  const Read b = read("[basis]\nhw = 16\n");
  EXPECT_TRUE(b.issues.empty());
  EXPECT_EQ(value_of<std::int64_t>(b.tree, "basis.hw"), 16);
}

TEST(TomlReader, BoolStringAndArray) {
  const Read r = read(
      "[imsrg3]\nin_flow = true\n[imsrg]\nmethod = \"magnus\"\n[operators]\n"
      "list = [\"Rp2\", \"E2\"]\n");
  EXPECT_TRUE(r.issues.empty());
  EXPECT_TRUE(value_of<bool>(r.tree, "imsrg3.in_flow"));
  EXPECT_EQ(value_of<std::string>(r.tree, "imsrg.method"), "magnus");
  const RawArray& list = value_of<RawArray>(r.tree, "operators.list");
  ASSERT_EQ(list.size(), 2u);
  EXPECT_EQ(*list[0].get_if<std::string>(), "Rp2");
  EXPECT_EQ(*list[1].get_if<std::string>(), "E2");
}

TEST(TomlReader, InlineTableInsideArray) {
  const Read r = read("[operators]\nlist = [\"Rp2\", { name = \"X\", rank = 2 }]\n");
  EXPECT_TRUE(r.issues.empty());
  const RawArray& list = value_of<RawArray>(r.tree, "operators.list");
  ASSERT_EQ(list.size(), 2u);
  const RawTable* t = list[1].get_if<RawTable>();
  ASSERT_NE(t, nullptr);
  ASSERT_EQ(t->size(), 2u);
  EXPECT_EQ((*t)[0].first, "name");
  EXPECT_EQ(*(*t)[0].second.get_if<std::string>(), "X");
  EXPECT_EQ(*(*t)[1].second.get_if<std::int64_t>(), 2);
}

TEST(TomlReader, ArrayOfTablesFromFile) {
  const Read r = read(
      "[[operators.from_file]]\nname = \"a\"\nrank = 0\nfile_2b = \"f1\"\n"
      "[[operators.from_file]]\nname = \"b\"\nfile_2b = \"f2\"\nfile_3b = \"g2\"\n");
  EXPECT_TRUE(r.issues.empty()) << (r.issues.empty() ? "" : r.issues[0].to_string());
  const RawArray& arr = value_of<RawArray>(r.tree, "operators.from_file");
  ASSERT_EQ(arr.size(), 2u);
  const RawTable* second = arr[1].get_if<RawTable>();
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->size(), 3u);
}

TEST(TomlReader, EmptyArray) {
  const Read r = read("[operators]\nlist = []\n");
  EXPECT_TRUE(r.issues.empty());
  EXPECT_TRUE(value_of<RawArray>(r.tree, "operators.list").empty());
}

TEST(TomlReader, SourcePositionOfValue) {
  const Read r = read("[basis]\nemax = 4\n");
  const RawValue* v = r.tree.find("basis.emax");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->where.kind, SourceRef::Kind::TomlString);
  EXPECT_EQ(v->where.name, "<test>");
  EXPECT_EQ(v->where.line, 2);
  EXPECT_EQ(v->where.column, 8);
}

TEST(TomlReader, DottedKeysAndHeaders) {
  const Read r = read("basis.emax = 4\n[imsrg.generator]\ncore = \"atan\"\n");
  EXPECT_TRUE(r.issues.empty());
  EXPECT_EQ(value_of<std::int64_t>(r.tree, "basis.emax"), 4);
  EXPECT_EQ(value_of<std::string>(r.tree, "imsrg.generator.core"), "atan");
}

// ---- errors ----------------------------------------------------------------------------------

struct UnknownCase {
  const char* name;
  const char* text;
  const char* key_path;
  const char* message;
  const char* suggestion;  // nullptr: none expected
};

class TomlReaderUnknown : public ::testing::TestWithParam<UnknownCase> {};

TEST_P(TomlReaderUnknown, ReportsKeyWithSuggestion) {
  const UnknownCase& c = GetParam();
  const Read r = read(c.text);
  ASSERT_EQ(r.issues.size(), 1u) << r.issues.size();
  const ConfigIssue& issue = r.issues[0];
  EXPECT_EQ(issue.code, "E-KEY-UNKNOWN");
  EXPECT_EQ(issue.severity, Severity::Error);
  EXPECT_EQ(issue.key_path, c.key_path);
  EXPECT_EQ(issue.message, c.message);
  if (c.suggestion != nullptr) {
    ASSERT_TRUE(issue.suggestion.has_value());
    EXPECT_EQ(*issue.suggestion, c.suggestion);
  } else {
    EXPECT_FALSE(issue.suggestion.has_value());
  }
  EXPECT_EQ(issue.where.kind, SourceRef::Kind::TomlString);
  EXPECT_GT(issue.where.line, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, TomlReaderUnknown,
    ::testing::Values(UnknownCase{"KeyInTable", "[basis]\nemaxx = 4\n", "basis.emaxx",
                                  "unknown key 'basis.emaxx'", "basis.emax"},
                      UnknownCase{"TopLevelTable", "[basiss]\nemax = 4\n", "basiss",
                                  "unknown table '[basiss]'", "basis"},
                      UnknownCase{"NestedKey", "[imsrg.generator]\ncor = \"atan\"\n",
                                  "imsrg.generator.cor", "unknown key 'imsrg.generator.cor'",
                                  "imsrg.generator.core"},
                      UnknownCase{"NoSuggestion", "[basis]\nzzzzzzzzzz = 4\n", "basis.zzzzzzzzzz",
                                  "unknown key 'basis.zzzzzzzzzz'", nullptr}),
    [](const ::testing::TestParamInfo<UnknownCase>& info) { return std::string(info.param.name); });

TEST(TomlReader, SeveralUnknownKeysAreReportedTogether) {
  const Read r =
      read("[basis]\nemaxx = 4\nemax = 6\n[basiss]\nx = 1\n[imsrg.generator]\ncor = 1\n");
  ASSERT_EQ(r.issues.size(), 3u);
  EXPECT_EQ(r.issues[0].key_path, "basis.emaxx");
  EXPECT_EQ(r.issues[1].key_path, "basiss");
  EXPECT_EQ(r.issues[2].key_path, "imsrg.generator.cor");
  // The valid key is still stored.
  EXPECT_EQ(value_of<std::int64_t>(r.tree, "basis.emax"), 6);
}

TEST(TomlReader, TableWhereValueExpected) {
  const Read r = read("[basis.emax]\nx = 1\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TYPE");
  EXPECT_EQ(r.issues[0].key_path, "basis.emax");
  EXPECT_FALSE(r.tree.contains("basis.emax"));
}

TEST(TomlReader, InlineTableWhereValueExpected) {
  const Read r = read("[basis]\nemax = { a = 1 }\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TYPE");
  EXPECT_EQ(r.issues[0].where.line, 2);
}

TEST(TomlReader, ValueWhereTableExpected) {
  const Read r = read("basis = 4\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TYPE");
  EXPECT_EQ(r.issues[0].key_path, "basis");
  EXPECT_EQ(r.issues[0].message, "'basis' is a table");
}

TEST(TomlReader, DateIsTypeError) {
  const Read r = read("[basis]\nemax = 2020-01-01\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TYPE");
  EXPECT_EQ(r.issues[0].key_path, "basis.emax");
  EXPECT_EQ(r.issues[0].where.line, 2);
}

TEST(TomlReader, DateInsideArrayIsTypeError) {
  const Read r = read("[operators]\nlist = [12:30:00]\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TYPE");
}

TEST(TomlReader, SyntaxErrorHasLine) {
  const Read r = read("[basis]\nemax = = 4\n");
  ASSERT_EQ(r.issues.size(), 1u);
  EXPECT_EQ(r.issues[0].code, "E-TOML-SYNTAX");
  EXPECT_EQ(r.issues[0].where.kind, SourceRef::Kind::TomlString);
  EXPECT_EQ(r.issues[0].where.line, 2);
  EXPECT_GT(r.issues[0].where.column, 0);
  EXPECT_TRUE(r.tree.empty());
}

TEST(TomlReader, ProvenanceTableIsSkipped) {
  const Read r =
      read("[basis]\nemax = 4\n[provenance]\nversion = \"1\"\nwarnings = []\nwhatever = 3\n");
  EXPECT_TRUE(r.issues.empty());
  EXPECT_EQ(r.tree.entries().size(), 1u);
}

TEST(TomlReader, MissingFileIsEFile) {
  RawTree tree;
  const ConfigIssues issues = read_toml_file("/nonexistent-dir-imsrg/none.toml", tree);
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-FILE");
  EXPECT_EQ(issues[0].message, "cannot read config file '/nonexistent-dir-imsrg/none.toml'");
  EXPECT_TRUE(tree.empty());
}

TEST(TomlReader, FileUsesPathAsGiven) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "imsrg_test_toml_reader_file.toml";
  {
    std::ofstream out(path);
    out << "[basis]\nemax = 4\nemaxx = 1\n";
  }
  RawTree tree;
  const ConfigIssues issues = read_toml_file(path, tree);
  std::filesystem::remove(path);
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].where.kind, SourceRef::Kind::TomlFile);
  EXPECT_EQ(issues[0].where.name, path.string());
  EXPECT_EQ(issues[0].where.line, 3);
  const RawValue* v = tree.find("basis.emax");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->where.kind, SourceRef::Kind::TomlFile);
  EXPECT_EQ(v->where.name, path.string());
}

TEST(TomlReader, ReadingTwiceReplacesEntries) {
  RawTree tree;
  EXPECT_TRUE(read_toml_string("[basis]\nemax = 4\n", "a", tree).empty());
  EXPECT_TRUE(read_toml_string("[basis]\nemax = 6\n", "b", tree).empty());
  EXPECT_EQ(value_of<std::int64_t>(tree, "basis.emax"), 6);
}

// ---- design examples -------------------------------------------------------------------------

class TomlReaderExamples : public ::testing::TestWithParam<std::string_view> {};

TEST_P(TomlReaderExamples, ReadWithoutIssues) {
  const Read r = read(GetParam());
  EXPECT_TRUE(r.issues.empty()) << (r.issues.empty() ? "" : r.issues[0].to_string());
  EXPECT_TRUE(r.tree.contains("basis.emax"));
}

INSTANTIATE_TEST_SUITE_P(Section211, TomlReaderExamples,
                         ::testing::Values(kExampleA, kExampleB, kExampleC, kExampleD));

TEST(TomlReader, ExampleAContent) {
  const Read r = read(kExampleA);
  EXPECT_EQ(value_of<std::int64_t>(r.tree, "interaction.two_body.lmax"), 14);
  EXPECT_EQ(value_of<std::string>(r.tree, "interaction.two_body.format"), "me2j");
  EXPECT_EQ(value_of<std::string>(r.tree, "imsrg.generator.core"), "atan");
  EXPECT_EQ(value_of<RawArray>(r.tree, "operators.list").size(), 1u);
}

TEST(TomlReader, ExampleDHasThreeBodyAndImsrg3) {
  const Read r = read(kExampleD);
  EXPECT_TRUE(r.tree.has_table("interaction.three_body"));
  EXPECT_TRUE(value_of<bool>(r.tree, "imsrg3.in_flow"));
  EXPECT_EQ(value_of<double>(r.tree, "imsrg3.threebody_threshold"), 0.0);
}

// ---- --set -----------------------------------------------------------------------------------

struct SetCase {
  const char* name;
  const char* assignment;
  const char* path;
};

TEST(SetOverride, Integer) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("basis.emax=4", tree).empty());
  EXPECT_EQ(value_of<std::int64_t>(tree, "basis.emax"), 4);
}

TEST(SetOverride, Float) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("basis.hw=16.5", tree).empty());
  EXPECT_EQ(value_of<double>(tree, "basis.hw"), 16.5);
}

TEST(SetOverride, Bool) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("imsrg3.in_flow=true", tree).empty());
  EXPECT_TRUE(value_of<bool>(tree, "imsrg3.in_flow"));
}

TEST(SetOverride, QuotedString) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("imsrg.method=\"magnus\"", tree).empty());
  EXPECT_EQ(value_of<std::string>(tree, "imsrg.method"), "magnus");
}

TEST(SetOverride, BareStringFallsBack) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("imsrg.method=magnus", tree).empty());
  EXPECT_EQ(value_of<std::string>(tree, "imsrg.method"), "magnus");
}

TEST(SetOverride, BareStringWithSpacesIsTrimmed) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("  reference.nucleus = O16 ", tree).empty());
  EXPECT_EQ(value_of<std::string>(tree, "reference.nucleus"), "O16");
}

TEST(SetOverride, Array) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("operators.list=[\"Rp2\",\"E2\"]", tree).empty());
  const RawArray& list = value_of<RawArray>(tree, "operators.list");
  ASSERT_EQ(list.size(), 2u);
  EXPECT_EQ(*list[1].get_if<std::string>(), "E2");
}

TEST(SetOverride, SourceRefIsVerbatimAssignment) {
  RawTree tree;
  apply_set_override("basis.emax=4", tree);
  const RawValue* v = tree.find("basis.emax");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->where.kind, SourceRef::Kind::SetOption);
  EXPECT_EQ(v->where.name, "basis.emax=4");
}

TEST(SetOverride, ValueMayContainEquals) {
  RawTree tree;
  EXPECT_TRUE(apply_set_override("output.directory=a=b", tree).empty());
  EXPECT_EQ(value_of<std::string>(tree, "output.directory"), "a=b");
}

TEST(SetOverride, UnknownKeyWithSuggestion) {
  RawTree tree;
  const ConfigIssues issues = apply_set_override("basis.emaxx=4", tree);
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-KEY-UNKNOWN");
  ASSERT_TRUE(issues[0].suggestion.has_value());
  EXPECT_EQ(*issues[0].suggestion, "basis.emax");
  EXPECT_EQ(issues[0].where.kind, SourceRef::Kind::SetOption);
  EXPECT_TRUE(tree.empty());
}

TEST(SetOverride, MissingEqualsIsSyntaxError) {
  RawTree tree;
  const ConfigIssues issues = apply_set_override("basis.emax", tree);
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-SET-SYNTAX");
  EXPECT_TRUE(tree.empty());
}

TEST(SetOverride, EmptyPathIsSyntaxError) {
  RawTree tree;
  const ConfigIssues issues = apply_set_override("  =4", tree);
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-SET-SYNTAX");
}

TEST(SetOverride, LastWins) {
  RawTree tree;
  apply_set_override("basis.emax=4", tree);
  apply_set_override("basis.emax=8", tree);
  EXPECT_EQ(value_of<std::int64_t>(tree, "basis.emax"), 8);
}

TEST(SetOverride, OverridesFileValue) {
  RawTree tree;
  EXPECT_TRUE(read_toml_string("[basis]\nemax = 4\n", "f", tree).empty());
  apply_set_override("basis.emax=6", tree);
  const RawValue* v = tree.find("basis.emax");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(*v->get_if<std::int64_t>(), 6);
  EXPECT_EQ(v->where.kind, SourceRef::Kind::SetOption);
}

}  // namespace
}  // namespace imsrg::driver
