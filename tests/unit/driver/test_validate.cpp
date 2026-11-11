// Tests of the validation table (design/16 section 6.1) through `resolve` and `validate`.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "imsrg/driver/config/resolve.hpp"

namespace imsrg::driver {
namespace {

namespace fs = std::filesystem;

using Entry = std::pair<std::string, RawValue::Data>;

RawValue::Data I(std::int64_t v) { return v; }
RawValue::Data D(double v) { return v; }
RawValue::Data S(const char* v) { return std::string(v); }
RawValue::Data B(bool v) { return v; }

/// `operators.from_file` with one entry whose dTz is `dtz` (V-62 rows).
RawValue::Data from_file_entry(std::int64_t dtz) {
  auto val = [](RawValue::Data d) {
    return RawValue{std::move(d), SourceRef{SourceRef::Kind::TomlString, "test", 3, 1}};
  };
  RawTable entry{{"name", val(std::string("op"))},
                 {"dtz", val(dtz)},
                 {"file_2b", val(std::string("op.me2j"))}};
  return RawArray{val(std::move(entry))};
}

RawTree minimal() {
  RawTree t;
  auto set = [&](const char* p, RawValue::Data d) {
    t.set(p, RawValue{std::move(d), SourceRef{SourceRef::Kind::TomlString, "test", 1, 1}});
  };
  set("basis.emax", I(4));
  set("basis.hw", I(16));
  set("reference.nucleus", S("O16"));
  set("interaction.two_body.file", S("x.me2j"));
  set("interaction.two_body.e1max", I(14));
  return t;
}

RawTree with(RawTree t, std::initializer_list<Entry> extra) {
  for (const auto& [path, data] : extra) {
    t.set(path, RawValue{data, SourceRef{SourceRef::Kind::TomlString, "test", 2, 1}});
  }
  return t;
}

const ConfigIssue* find_issue(const ConfigIssues& issues, const std::string& code,
                              const std::string& key = "") {
  for (const auto& i : issues) {
    if (i.code == code && (key.empty() || i.key_path == key)) return &i;
  }
  return nullptr;
}

struct Row {
  const char* name;
  std::vector<Entry> extra;
  const char* code;
  const char* key;
  const char* fragment;
  bool legacy_downgrades;  // true: a warning (same code) under legacy parsing
};

std::vector<Row> rows() {
  return {
      {"V02_hw_positive", {{"basis.hw", D(-1.0)}}, "V-02", "basis.hw", "must be > 0", false},
      {"V02_steps", {{"imsrg.steps", I(3)}}, "V-02", "imsrg.steps", "must be 1 or 2", false},
      {"V02_dsmax",
       {{"imsrg.step.dsmax", D(0.0)}},
       "V-02",
       "imsrg.step.dsmax",
       "must be > 0",
       false},
      {"V01_valence_space",
       {{"reference.nucleus", S("O16")}, {"valence.space", S("sd-shel")}},
       "V-01",
       "valence.space",
       "unknown valence space",
       false},
      {"V10_emax_vs_e1max",
       {{"basis.emax", I(6)}, {"interaction.two_body.e1max", I(4)}},
       "V-10",
       "basis.emax",
       "exceeds interaction.two_body.e1max",
       false},
      {"V11_e2max",
       {{"interaction.two_body.e2max", I(6)}},
       "V-11",
       "interaction.two_body.e2max",
       "smaller than 2*basis.emax",
       false},
      {"V12_lmax", {{"basis.lmax", I(20)}}, "V-12", "basis.lmax", "exceeds", false},
      {"V13_three_body_e1max",
       {{"interaction.three_body.file", S("f.me3j")},
        {"interaction.three_body.e1max", I(2)},
        {"interaction.three_body.e3max", I(16)}},
       "V-13",
       "interaction.three_body.e1max",
       "exceeds interaction.three_body.e1max",
       false},
      {"V13_three_body_e3max",
       {{"interaction.three_body.file", S("f.me3j")},
        {"interaction.three_body.e1max", I(14)},
        {"interaction.three_body.e3max", I(6)}},
       "V-13",
       "interaction.three_body.e3max",
       "exceeds interaction.three_body.e3max",
       false},
      {"V20_trap_needs_lab",
       {{"hamiltonian.trap_hw", D(20.0)}, {"hamiltonian.kinetic", S("intrinsic")}},
       "V-20",
       "hamiltonian.trap_hw",
       "requires hamiltonian.kinetic",
       true},
      {"V21_nat_occupations",
       {{"natural_orbitals.use_occupations", B(true)}},
       "V-21",
       "natural_orbitals.use_occupations",
       "requires basis.kind",
       true},
      {"V22_freeze_oscillator",
       {{"basis.kind", S("oscillator")}, {"hartree_fock.freeze_occupations", B(true)}},
       "V-22",
       "hartree_fock.freeze_occupations",
       "no effect",
       false},
      {"V30_space_and_custom",
       {{"reference.nucleus", S("O16")},
        {"valence.space", S("sd-shell")},
        {"valence.custom", S("p0s1,p1d5")}},
       "V-30",
       "valence.custom",
       "both set",
       true},
      {"V31_bad_nucleus",
       {{"reference.nucleus", S("Zz")}},
       "V-31",
       "reference.nucleus",
       "cannot parse",
       false},
      {"V40_steps2_without_valence",
       {{"imsrg.steps", I(2)}},
       "V-40",
       "imsrg.steps",
       "requires a valence space",
       true},
      {"V41_valence_generator_one_step",
       {{"valence.space", S("sd-shell")},
        {"imsrg.steps", I(1)},
        {"imsrg.generator.valence", S("white")}},
       "V-41",
       "imsrg.generator.valence",
       "no effect",
       true},
      {"V42_shell_model_generator_without_valence",
       {{"imsrg.generator.core", S("shell_model_atan")}},
       "V-42",
       "imsrg.generator.core",
       "needs a valence space",
       true},
      {"V43_delta_without_orbit",
       {{"imsrg.denominator.delta", D(0.5)}},
       "V-43",
       "imsrg.denominator.delta_orbit",
       "not set",
       true},
      {"V43_delta_orbit_garbage",
       {{"imsrg.denominator.delta", D(0.5)}, {"imsrg.denominator.delta_orbit", S("zzz")}},
       "V-43",
       "imsrg.denominator.delta_orbit",
       "neither",
       false},
      {"V44_scratch_without_dir",
       {{"imsrg.omega.storage", S("scratch")}},
       "V-44",
       "run.scratch_dir",
       "requires run.scratch_dir",
       false},
      {"V45_triples_needs_magnus",
       {{"imsrg.method", S("flow")}, {"imsrg.perturbative_triples", B(true)}},
       "V-45",
       "imsrg.perturbative_triples",
       "requires imsrg.method = \"magnus\"",
       true},
      {"V46_flow_with_imsrg3",
       {{"imsrg.method", S("flow_rk4")}, {"imsrg3.in_flow", B(true)}},
       "V-46",
       "imsrg.method",
       "cannot be combined with IMSRG(3)",
       false},
      {"V52_at_end_needs_magnus",
       {{"imsrg.method", S("flow")}, {"imsrg3.at_end", B(true)}},
       "V-52",
       "imsrg3.at_end",
       "requires imsrg.method = \"magnus\"",
       false},  // V-46 also fires (every P1 non-magnus method is a flow method)
      {"V62_from_file_dtz",
       {{"operators.from_file", from_file_entry(3)}},
       "V-62",
       "operators.from_file[0].dtz",
       "must be in -2..2",
       false},
      {"V02_non_finite_double",
       {{"imsrg.step.smax", D(std::numeric_limits<double>::infinity())}},
       "V-02",
       "imsrg.step.smax",
       "finite",
       false},
      {"V48_space_emax",
       {{"imsrg.space.emax", I(6)}},
       "V-48",
       "imsrg.space.emax",
       "exceeds basis.emax",
       false},
      {"V50_variant_without_imsrg3",
       {{"imsrg3.variant", S("n7")}},
       "V-50",
       "imsrg3.variant",
       "not active",
       true},
      {"V52_at_end_without_3n",
       {{"imsrg3.at_end", B(true)}},
       "V-52",
       "imsrg3.at_end",
       "no three-body interaction",
       false},
      {"WQ11_at_end_input_3n",
       {{"imsrg3.at_end", B(true)}},
       "W-Q11",
       "imsrg3.at_end",
       "not included",
       false},
      {"V54_de3max_without_imsrg3",
       {{"imsrg3.de3max", D(10.0)}},
       "V-54",
       "imsrg3.de3max",
       "require IMSRG(3)",
       true},
      {"V63_stem_with_separator",
       {{"output.stem", S("a/b")}},
       "V-63",
       "output.stem",
       "path separators",
       false},
      {"PLANNED_method",
       {{"imsrg.method", S("flow_euler")}},
       "E-PLANNED",
       "imsrg.method",
       "not available",
       false},
      {"PLANNED_goose_tank",
       {{"imsrg.goose_tank", B(true)}},
       "E-PLANNED",
       "imsrg.goose_tank",
       "not available",
       false},
      {"PLANNED_no_qqq",
       {{"imsrg3.no_qqq", B(true)}},
       "E-PLANNED",
       "imsrg3.no_qqq",
       "not available",
       false},
      {"PLANNED_fci",
       {{"valence.space", S("FCI-sd")}, {"reference.nucleus", S("O16")}},
       "E-PLANNED",
       "valence.space",
       "not available",
       false},
  };
}

TEST(Validate, table_of_rules) {
  for (const auto& row : rows()) {
    SCOPED_TRACE(row.name);
    RawTree t = minimal();
    // Rows that set no reference.nucleus rely on `extra` to override the minimal tree.
    for (const auto& [path, data] : row.extra) {
      t.set(path, RawValue{data, SourceRef{SourceRef::Kind::TomlString, "test", 2, 1}});
    }
    const auto r = resolve(t, ResolveOptions{false, false});
    const auto* i = find_issue(r.issues, row.code, row.key);
    ASSERT_NE(i, nullptr) << "missing " << row.code << " at " << row.key;
    EXPECT_NE(i->message.find(row.fragment), std::string::npos) << i->message;
    const bool is_warning_rule =
        std::string(row.code) == "W-Q11" || std::string(row.name) == "V22_freeze_oscillator";
    if (!is_warning_rule && std::string(row.name) != "V52_at_end_without_3n") {
      EXPECT_EQ(i->severity, Severity::Error);
    }
  }
}

TEST(Validate, legacy_downgrades_selected_rules_to_warnings) {
  int checked = 0;
  for (const auto& row : rows()) {
    if (!row.legacy_downgrades) continue;
    SCOPED_TRACE(row.name);
    RawTree t = minimal();
    for (const auto& [path, data] : row.extra) {
      t.set(path, RawValue{data, SourceRef{SourceRef::Kind::TomlString, "test", 2, 1}});
    }
    const auto r = resolve(t, ResolveOptions{true, false});
    const auto* i = find_issue(r.issues, row.code, row.key);
    ASSERT_NE(i, nullptr);
    EXPECT_EQ(i->severity, Severity::Warning);
    EXPECT_FALSE(has_errors(r.issues)) << "unexpected errors under legacy";
    ++checked;
  }
  EXPECT_GE(checked, 8);
}

TEST(Validate, warning_rules_stay_warnings_without_legacy) {
  const auto r = resolve(with(minimal(), {{"imsrg3.at_end", B(true)}}), {false, false});
  const auto* i = find_issue(r.issues, "W-Q11");
  ASSERT_NE(i, nullptr);
  EXPECT_EQ(i->severity, Severity::Warning);
  const auto* j = find_issue(r.issues, "V-52", "imsrg3.at_end");
  ASSERT_NE(j, nullptr);
  EXPECT_EQ(j->severity, Severity::Warning);
  EXPECT_FALSE(has_errors(r.issues));
}

TEST(Validate, v32_occupations_file_without_reference_or_valence) {
  RawTree t = minimal();
  t.erase("reference.nucleus");
  t.set("basis.mass_number", RawValue{I(16), {}});
  t.set("reference.occupations_file", RawValue{S("occ.txt"), {}});
  const auto r = resolve(t, {false, false});
  const auto* i = find_issue(r.issues, "V-32", "reference.occupations_file");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("requires reference.nucleus"), std::string::npos);
}

TEST(Validate, v22_only_when_explicitly_set) {
  const auto quiet = resolve(with(minimal(), {{"basis.kind", S("oscillator")}}), {false, false});
  EXPECT_EQ(find_issue(quiet.issues, "V-22"), nullptr);
}

TEST(Validate, independent_errors_accumulate) {
  const auto t = with(minimal(), {{"basis.hw", D(-1.0)},
                                  {"basis.emax", I(6)},
                                  {"interaction.two_body.e1max", I(4)},
                                  {"valence.space", S("sd-shel")},
                                  {"output.stem", S("a/b")}});
  const auto r = resolve(t, {false, false});
  EXPECT_NE(find_issue(r.issues, "V-02", "basis.hw"), nullptr);
  EXPECT_NE(find_issue(r.issues, "V-10", "basis.emax"), nullptr);
  EXPECT_NE(find_issue(r.issues, "V-01", "valence.space"), nullptr);
  EXPECT_NE(find_issue(r.issues, "V-63", "output.stem"), nullptr);
  int errors = 0;
  for (const auto& i : r.issues) errors += i.severity == Severity::Error ? 1 : 0;
  EXPECT_GE(errors, 4);
}

TEST(Validate, unknown_valence_space_suggests_nearest_and_single_reference_hint) {
  const auto r = resolve(with(minimal(), {{"valence.space", S("sd-shel")}}), {false, false});
  const auto* i = find_issue(r.issues, "V-01", "valence.space");
  ASSERT_NE(i, nullptr);
  ASSERT_TRUE(i->suggestion.has_value());
  EXPECT_NE(i->suggestion->find("sd-shell"), std::string::npos);
  EXPECT_NE(i->suggestion->find("reference.nucleus"), std::string::npos);
}

TEST(Validate, valid_tree_has_no_errors) {
  const auto r = resolve(minimal(), {false, false});
  EXPECT_FALSE(has_errors(r.issues));
  EXPECT_TRUE(r.issues.empty());
  const auto again = validate(r.config, minimal(), {false, false});
  EXPECT_TRUE(again.empty());
}

TEST(Validate, valid_single_stage_and_two_stage_trees_have_no_errors) {
  const auto two = resolve(
      with(minimal(), {{"valence.space", S("sd-shell")}, {"imsrg.generator.core", S("atan")}}),
      {false, false});
  EXPECT_FALSE(has_errors(two.issues));
  EXPECT_EQ(two.config.imsrg.steps, 2);
}

// ---- V-14: file checks with real files (check_files = true) ----

class FileChecks : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() / ("imsrg_validate_" + std::to_string(::getpid()));
    fs::create_directories(dir_);
    std::ofstream(dir_ / "x.me2j") << "data\n";
  }
  void TearDown() override {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  RawTree tree(const fs::path& file, const fs::path& out) {
    return with(minimal(), {{"interaction.two_body.file", RawValue::Data(file.generic_string())},
                            {"output.directory", RawValue::Data(out.generic_string())}});
  }
  fs::path dir_;
};

TEST_F(FileChecks, existing_file_and_writable_dir_pass) {
  const auto r = resolve(tree(dir_ / "x.me2j", dir_ / "out"), {false, true});
  EXPECT_EQ(find_issue(r.issues, "V-14"), nullptr);
  EXPECT_FALSE(has_errors(r.issues));
}

TEST_F(FileChecks, missing_file_is_v14) {
  const auto r = resolve(tree(dir_ / "nope.me2j", dir_ / "out"), {false, true});
  const auto* i = find_issue(r.issues, "V-14", "interaction.two_body.file");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("does not exist"), std::string::npos);
}

TEST_F(FileChecks, directory_as_input_file_is_v14) {
  const auto r = resolve(tree(dir_, dir_ / "out"), {false, true});
  EXPECT_NE(find_issue(r.issues, "V-14", "interaction.two_body.file"), nullptr);
}

TEST_F(FileChecks, output_directory_under_a_regular_file_is_v14) {
  const auto r = resolve(tree(dir_ / "x.me2j", dir_ / "x.me2j" / "sub"), {false, true});
  EXPECT_NE(find_issue(r.issues, "V-14", "output.directory"), nullptr);
}

TEST_F(FileChecks, missing_operator_and_occupation_files_are_v14) {
  RawTable e;
  e.emplace_back("name", RawValue{S("Rsq"), {}});
  e.emplace_back("file_2b", RawValue{RawValue::Data((dir_ / "r2.me2j").generic_string()), {}});
  auto t = tree(dir_ / "x.me2j", dir_ / "out");
  t.set("operators.from_file", RawValue{RawArray{RawValue{e, {}}}, {}});
  t.set("reference.occupations_file",
        RawValue{RawValue::Data((dir_ / "occ.txt").generic_string()), {}});
  const auto r = resolve(t, {false, true});
  EXPECT_NE(find_issue(r.issues, "V-14", "operators.from_file[0].file_2b"), nullptr);
  EXPECT_NE(find_issue(r.issues, "V-14", "reference.occupations_file"), nullptr);
}

TEST_F(FileChecks, files_are_not_touched_when_check_files_is_off) {
  const auto r = resolve(tree(dir_ / "nope.me2j", dir_ / "out"), {false, false});
  EXPECT_EQ(find_issue(r.issues, "V-14"), nullptr);
}

TEST_F(FileChecks, legacy_keeps_v14_as_error) {
  const auto r = resolve(tree(dir_ / "nope.me2j", dir_ / "out"), {true, true});
  const auto* i = find_issue(r.issues, "V-14");
  ASSERT_NE(i, nullptr);
  EXPECT_EQ(i->severity, Severity::Error);
}

}  // namespace
}  // namespace imsrg::driver
