// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of `resolve`: typed reading, defaults and derived values (design/16
// sections 1 and 6.1).
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/driver/config/compat.h"
#include "imsrg/driver/config/resolve.h"

namespace imsrg::driver {
namespace {

using Entry = std::pair<std::string, RawValue::Data>;

RawTree MakeTree(std::initializer_list<Entry> entries) {
  RawTree t;
  for (const auto& [path, data] : entries) {
    t.set(path, RawValue{data, SourceRef{SourceRef::Kind::kTomlString, "test",
                                         1, 1}});
  }
  return t;
}

RawValue::Data I(std::int64_t v) { return v; }
RawValue::Data D(double v) { return v; }
RawValue::Data S(const char* v) { return std::string(v); }
RawValue::Data B(bool v) { return v; }

RawTree Minimal() {
  return MakeTree({{"basis.emax", I(4)},
                   {"basis.hw", I(16)},
                   {"reference.nucleus", S("O16")},
                   {"interaction.two_body.file", S("x.me2j")},
                   {"interaction.two_body.e1max", I(14)}});
}

RawTree With(RawTree t, std::initializer_list<Entry> extra) {
  for (const auto& [path, data] : extra) {
    t.set(path, RawValue{data, SourceRef{SourceRef::Kind::kTomlString, "test",
                                         2, 1}});
  }
  return t;
}

ResolveResult ResolveTree(const RawTree& t, bool legacy = false) {
  return Resolve(t, ResolveOptions{legacy, false});
}

const ConfigIssue* FindIssue(const ConfigIssues& issues,
                             const std::string& code,
                             const std::string& key = "") {
  for (const auto& i : issues) {
    if (i.code == code && (key.empty() || i.key_path == key)) return &i;
  }
  return nullptr;
}

TEST(Resolve, minimal_tree_is_valid) {
  const auto r = ResolveTree(Minimal());
  EXPECT_FALSE(HasErrors(r.issues));
  EXPECT_EQ(r.config.basis.emax, 4);
  EXPECT_DOUBLE_EQ(r.config.basis.hw, 16.0);
  EXPECT_EQ(r.config.reference.nucleus, "O16");
  ASSERT_EQ(r.config.interaction.two_body.file.size(), 1u);
  EXPECT_EQ(r.config.interaction.two_body.file[0], "x.me2j");
}

TEST(Resolve, defaults_equal_default_config_except_required_and_derived) {
  const auto r = ResolveTree(Minimal());
  RunConfig expect{};
  expect.basis.emax = 4;
  expect.basis.hw = 16.0;
  expect.basis.e3max = 12;
  expect.basis.mass_number = 16;
  expect.reference.nucleus = "O16";
  expect.hamiltonian.beta_cm_hw = 16.0;
  expect.hamiltonian.kinetic = KineticEnergy::kIntrinsic;
  expect.interaction.two_body.file = {"x.me2j"};
  expect.interaction.two_body.e1max = 14;
  expect.interaction.two_body.e2max = 28;
  expect.interaction.two_body.lmax = 14;
  expect.output.stem = "magnus_O16_O16_hw16_e4_A16";
  expect.output.flow_file = "output/magnus_O16_O16_hw16_e4_A16.flow";
  expect.provenance = r.config.provenance;
  EXPECT_TRUE(r.config == expect);
}

TEST(Resolve, isospin_average_operators_default_true_and_explicit_false) {
  EXPECT_TRUE(ResolveTree(Minimal()).config.output.isospin_average_operators);
  const auto off =
      With(Minimal(), {{"output.isospin_average_operators", B(false)}});
  const auto r = ResolveTree(off);
  EXPECT_FALSE(HasErrors(r.issues));
  EXPECT_FALSE(r.config.output.isospin_average_operators);
}

TEST(Resolve, hf_mixing_defaults_to_diis_in_toml_and_linear_in_legacy) {
  // D37: the TOML default is DIIS; the legacy layer keeps imsrg++'s linear
  // density mixing.
  EXPECT_EQ(ResolveTree(Minimal()).config.hartree_fock.mixing,
            HartreeFockMixing::kDiis);
  EXPECT_EQ(ResolveTree(Minimal(), /*legacy=*/true).config.hartree_fock.mixing,
            HartreeFockMixing::kLinear);
}

// Compat profile imsrg_fork_79e9330 (design/50 section 5): the fork's
// Parameters.hh default domega = 0.2 unless domega is given.
TEST(Resolve, fork_compat_profile_sets_domega_default) {
  EXPECT_EQ(ResolveTree(Minimal()).config.imsrg.step.domega, 0.1);
  const auto fork = With(Minimal(), {{"run.compat", S("imsrg_fork_79e9330")}});
  EXPECT_EQ(ResolveTree(fork).config.imsrg.step.domega, 0.2);
  EXPECT_EQ(ResolveTree(fork).config.run.compat, Compat::kImsrgFork79e9330);
  const auto given = With(fork, {{"imsrg.step.domega", D(0.05)}});
  EXPECT_EQ(ResolveTree(given).config.imsrg.step.domega, 0.05);
}

TEST(Resolve, compat_profile_tokyo_writer_option) {
  EXPECT_FALSE(CompatOptionsFor(Compat::kNone).tensor_one_body_norm_imsrgpp);
  EXPECT_TRUE(
      CompatOptionsFor(Compat::kImsrgFork79e9330).tensor_one_body_norm_imsrgpp);
  EXPECT_FALSE(CompatOptionsFor(Compat::kNone).tokyo_writer_fork_79e9330);
  EXPECT_TRUE(
      CompatOptionsFor(Compat::kImsrgFork79e9330).tokyo_writer_fork_79e9330);
}

TEST(Resolve, hf_mixing_explicit_value_wins_in_both_modes) {
  const auto lin = With(Minimal(), {{"hartree_fock.mixing", S("linear")}});
  const auto diis = With(Minimal(), {{"hartree_fock.mixing", S("diis")}});
  EXPECT_EQ(ResolveTree(lin).config.hartree_fock.mixing,
            HartreeFockMixing::kLinear);
  EXPECT_EQ(ResolveTree(diis, true).config.hartree_fock.mixing,
            HartreeFockMixing::kDiis);
}

TEST(Resolve, hf_mixing_unknown_spelling_is_v01) {
  const auto r =
      ResolveTree(With(Minimal(), {{"hartree_fock.mixing", S("dis")}}));
  EXPECT_NE(FindIssue(r.issues, "V-01", "hartree_fock.mixing"), nullptr);
}

TEST(Resolve, factorized_sets_the_3f2_recipe_defaults) {
  const auto plain = ResolveTree(Minimal());
  EXPECT_FALSE(plain.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(plain.config.imsrg.step.omega_norm_max, 0.25);
  EXPECT_TRUE(plain.config.imsrg3.factorized_1b_intermediates);
  EXPECT_FALSE(plain.config.imsrg3.factorized_2b_intermediates);

  const auto r = ResolveTree(With(Minimal(), {{"imsrg3.factorized", B(true)}}));
  EXPECT_FALSE(HasErrors(r.issues));
  EXPECT_TRUE(r.config.imsrg3.factorized);
  EXPECT_TRUE(r.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(r.config.imsrg.step.omega_norm_max, 0.1);
  EXPECT_TRUE(r.config.imsrg3.factorized_1b_intermediates);
  EXPECT_FALSE(r.config.imsrg3.factorized_2b_intermediates);
}

TEST(Resolve, explicit_keys_win_over_the_3f2_recipe) {
  const auto r =
      ResolveTree(With(Minimal(), {{"imsrg3.factorized", B(true)},
                                   {"imsrg.omega.hunter_gatherer", B(false)},
                                   {"imsrg.step.omega_norm_max", D(0.25)}}));
  EXPECT_FALSE(r.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(r.config.imsrg.step.omega_norm_max, 0.25);
  const auto r2 =
      ResolveTree(With(Minimal(), {{"imsrg3.factorized", B(true)},
                                   {"imsrg.step.omega_norm_max", D(0.3)}}));
  EXPECT_TRUE(r2.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(r2.config.imsrg.step.omega_norm_max, 0.3);
  const auto r3 = ResolveTree(
      With(Minimal(), {{"imsrg3.factorized", B(true)},
                       {"imsrg3.factorized_1b_intermediates", B(false)},
                       {"imsrg3.factorized_2b_intermediates", B(true)}}));
  EXPECT_FALSE(r3.config.imsrg3.factorized_1b_intermediates);
  EXPECT_TRUE(r3.config.imsrg3.factorized_2b_intermediates);
}

TEST(Resolve, mass_number_from_reference_and_valence_core) {
  EXPECT_EQ(ResolveTree(Minimal()).config.basis.mass_number, 16);
  auto t = Minimal();
  t.erase("reference.nucleus");
  t.set("valence.space", RawValue{S("sd-shell"), {}});
  const auto r = ResolveTree(t);
  EXPECT_EQ(r.config.basis.mass_number, 16);
  EXPECT_EQ(FindIssue(r.issues, "V-23"), nullptr);
}

TEST(Resolve, mass_number_explicit_wins) {
  const auto r = ResolveTree(With(Minimal(), {{"basis.mass_number", I(18)}}));
  EXPECT_EQ(r.config.basis.mass_number, 18);
}

TEST(Resolve, mass_number_undeterminable_is_v23) {
  auto t = Minimal();
  t.erase("reference.nucleus");
  const auto r = ResolveTree(t);
  const auto* i = FindIssue(r.issues, "V-23");
  ASSERT_NE(i, nullptr);
  EXPECT_EQ(i->severity, Severity::kError);
  EXPECT_NE(i->message.find("cannot determine the mass number"),
            std::string::npos);
}

TEST(Resolve, stems) {
  EXPECT_EQ(ResolveTree(Minimal()).config.output.stem,
            "magnus_O16_O16_hw16_e4_A16");
  auto t = Minimal();
  t.erase("reference.nucleus");
  t.set("valence.space", RawValue{S("sd-shell"), {}});
  t.set("basis.hw", RawValue{I(20), {}});
  t.set("basis.emax", RawValue{I(6), {}});
  EXPECT_EQ(ResolveTree(t).config.output.stem,
            "magnus_O16_sd-shell_hw20_e6_A16");
}

TEST(Resolve, stem_uses_task_when_not_imsrg_and_fractional_hw) {
  const auto r = ResolveTree(
      With(Minimal(), {{"run.task", S("hf")}, {"basis.hw", D(13.5)}}));
  EXPECT_EQ(r.config.output.stem, "hf_O16_O16_hw13.5_e4_A16");
}

TEST(Resolve, flow_file_follows_directory_and_stem) {
  EXPECT_EQ(ResolveTree(Minimal()).config.output.flow_file,
            "output/magnus_O16_O16_hw16_e4_A16.flow");
  const auto r = ResolveTree(With(
      Minimal(), {{"output.directory", S("out/d")}, {"output.stem", S("s")}}));
  EXPECT_EQ(r.config.output.flow_file, "out/d/s.flow");
}

TEST(Resolve, steps_follow_valence_space) {
  EXPECT_EQ(ResolveTree(Minimal()).config.imsrg.steps, 1);
  auto t = Minimal();
  t.set("valence.space", RawValue{S("sd-shell"), {}});
  EXPECT_EQ(ResolveTree(t).config.imsrg.steps, 2);
  t.set("imsrg.steps", RawValue{I(1), {}});
  EXPECT_EQ(ResolveTree(t).config.imsrg.steps, 1);
}

TEST(Resolve, valence_generator_reset_for_single_stage) {
  const auto r =
      ResolveTree(With(Minimal(), {{"imsrg.generator.valence", S("white")}}));
  EXPECT_EQ(r.config.imsrg.generator.valence, GeneratorConfig{}.valence);
  auto t = With(Minimal(), {{"imsrg.generator.valence", S("white")},
                            {"imsrg.generator.core", S("shell-model-atan")},
                            {"valence.space", S("sd-shell")}});
  EXPECT_EQ(ResolveTree(t).config.imsrg.generator.valence, Generator::kWhite);
}

TEST(Resolve, imsrg_space_defaults) {
  const auto r = ResolveTree(With(Minimal(), {{"imsrg.space.emax", I(3)}}));
  ASSERT_TRUE(r.config.imsrg.space.has_value());
  EXPECT_EQ(r.config.imsrg.space->emax, 3);
  EXPECT_EQ(r.config.imsrg.space->e2max, 6);
  EXPECT_EQ(r.config.imsrg.space->e3max, 9);
  EXPECT_EQ(r.config.imsrg.space->emax_3body, 3);
  const auto r2 = ResolveTree(With(Minimal(), {{"imsrg.space.e2max", I(4)}}));
  ASSERT_TRUE(r2.config.imsrg.space.has_value());
  EXPECT_EQ(r2.config.imsrg.space->emax, 4);
  EXPECT_EQ(r2.config.imsrg.space->e2max, 4);
  EXPECT_FALSE(ResolveTree(Minimal()).config.imsrg.space.has_value());
}

TEST(Resolve, e3max_clamped_to_three_emax) {
  EXPECT_EQ(ResolveTree(Minimal()).config.basis.e3max, 12);
  const auto r = ResolveTree(
      With(Minimal(), {{"basis.emax", I(2)}, {"basis.e3max", I(10)}}));
  EXPECT_EQ(r.config.basis.e3max, 6);
  const auto r2 = ResolveTree(With(Minimal(), {{"basis.e3max", I(8)}}));
  EXPECT_EQ(r2.config.basis.e3max, 8);
}

TEST(Resolve, kinetic_default_depends_on_format_and_trap) {
  EXPECT_EQ(ResolveTree(Minimal()).config.hamiltonian.kinetic,
            KineticEnergy::kIntrinsic);
  EXPECT_EQ(ResolveTree(With(Minimal(), {{"hamiltonian.trap_hw", I(20)}}))
                .config.hamiltonian.kinetic,
            KineticEnergy::kLab);
  EXPECT_EQ(ResolveTree(With(Minimal(),
                             {{"interaction.two_body.format", S("nushellx")}}))
                .config.hamiltonian.kinetic,
            KineticEnergy::kNone);
  EXPECT_EQ(ResolveTree(With(Minimal(), {{"hamiltonian.kinetic", S("lab")}}))
                .config.hamiltonian.kinetic,
            KineticEnergy::kLab);
}

TEST(Resolve, two_body_cuts_and_beta_cm_default) {
  const auto r = ResolveTree(Minimal());
  EXPECT_EQ(r.config.interaction.two_body.e2max, 28);
  EXPECT_EQ(r.config.interaction.two_body.lmax, 14);
  EXPECT_DOUBLE_EQ(r.config.hamiltonian.beta_cm_hw, 16.0);
  const auto r2 =
      ResolveTree(With(Minimal(), {{"interaction.two_body.e2max", I(10)},
                                   {"interaction.two_body.lmax", I(3)},
                                   {"hamiltonian.beta_cm_hw", D(7.5)}}));
  EXPECT_EQ(r2.config.interaction.two_body.e2max, 10);
  EXPECT_EQ(r2.config.interaction.two_body.lmax, 3);
  EXPECT_DOUBLE_EQ(r2.config.hamiltonian.beta_cm_hw, 7.5);
}

TEST(Resolve, two_body_file_string_or_array) {
  const auto r = ResolveTree(With(
      Minimal(), {{"interaction.two_body.file",
                   RawArray{RawValue{S("a"), {}}, RawValue{S("b"), {}}}}}));
  ASSERT_EQ(r.config.interaction.two_body.file.size(), 2u);
  EXPECT_EQ(r.config.interaction.two_body.file[1], "b");
  const auto r3 =
      ResolveTree(With(Minimal(), {{"interaction.two_body.file", RawArray{}}}));
  const auto* i = FindIssue(r3.issues, "V-02", "interaction.two_body.file");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("1 or 2"), std::string::npos);
}

TEST(Resolve, double_accepts_integer_and_bool_rejects_int) {
  const auto r = ResolveTree(With(
      Minimal(), {{"basis.hw", I(20)}, {"imsrg.use_brueckner_bch", B(true)}}));
  EXPECT_DOUBLE_EQ(r.config.basis.hw, 20.0);
  EXPECT_TRUE(r.config.imsrg.use_brueckner_bch);
  const auto r2 =
      ResolveTree(With(Minimal(), {{"imsrg.use_brueckner_bch", I(1)}}));
  EXPECT_NE(FindIssue(r2.issues, "E-TYPE", "imsrg.use_brueckner_bch"), nullptr);
}

TEST(Resolve, float_for_int_is_type_error) {
  const auto r = ResolveTree(With(Minimal(), {{"basis.emax", D(4.5)}}));
  const auto* i = FindIssue(r.issues, "E-TYPE", "basis.emax");
  ASSERT_NE(i, nullptr);
  EXPECT_EQ(i->message, "expected an integer, got float");
  EXPECT_EQ(i->where.line, 2);
}

TEST(Resolve, int_overflow_is_v02) {
  const auto r =
      ResolveTree(With(Minimal(), {{"basis.emax", I(std::int64_t{1} << 40)}}));
  EXPECT_NE(FindIssue(r.issues, "V-02", "basis.emax"), nullptr);
}

TEST(Resolve, string_for_number_is_type_error) {
  const auto r = ResolveTree(With(Minimal(), {{"basis.hw", S("sixteen")}}));
  const auto* i = FindIssue(r.issues, "E-TYPE", "basis.hw");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("got string"), std::string::npos);
}

TEST(Resolve, unknown_enum_gives_v01_with_suggestion) {
  const auto r =
      ResolveTree(With(Minimal(), {{"imsrg.method", S("magnus_eular")}}));
  const auto* i = FindIssue(r.issues, "V-01", "imsrg.method");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("unknown"), std::string::npos);
  EXPECT_NE(i->message.find("magnus_modified_euler"),
            std::string::npos);  // allowed list
  ASSERT_TRUE(i->suggestion.has_value());
  EXPECT_EQ(i->suggestion->rfind("magnus", 0), 0u);
}

TEST(Resolve, optional_values_are_engaged_when_present) {
  const auto r = ResolveTree(With(
      Minimal(), {{"basis.lmax", I(3)}, {"hamiltonian.trap_hw", D(12.5)}}));
  EXPECT_EQ(r.config.basis.lmax, 3);
  ASSERT_TRUE(r.config.hamiltonian.trap_hw.has_value());
  EXPECT_DOUBLE_EQ(*r.config.hamiltonian.trap_hw, 12.5);
  EXPECT_FALSE(r.config.basis.lmax3.has_value());
}

TEST(Resolve, required_keys_missing_are_v03) {
  const auto r = ResolveTree(RawTree{});
  for (const char* key : {"basis.emax", "basis.hw", "interaction.two_body.file",
                          "interaction.two_body.e1max"}) {
    const auto* i = FindIssue(r.issues, "V-03", key);
    ASSERT_NE(i, nullptr) << key;
    EXPECT_EQ(i->message, std::string("required key '") + key + "' is missing");
    EXPECT_EQ(i->severity, Severity::kError);
  }
  // Absent three_body table: no three-body requirements.
  EXPECT_EQ(FindIssue(r.issues, "V-03", "interaction.three_body.file"),
            nullptr);
}

TEST(Resolve, three_body_required_keys_only_when_table_present) {
  const auto r = ResolveTree(
      With(Minimal(), {{"interaction.three_body.type", S("full")}}));
  ASSERT_TRUE(r.config.interaction.three_body.has_value());
  for (const char* key :
       {"interaction.three_body.file", "interaction.three_body.e1max",
        "interaction.three_body.e3max"}) {
    EXPECT_NE(FindIssue(r.issues, "V-03", key), nullptr) << key;
  }
  const auto ok =
      ResolveTree(With(Minimal(), {{"interaction.three_body.file", S("f.me3j")},
                                   {"interaction.three_body.e1max", I(14)},
                                   {"interaction.three_body.e3max", I(16)}}));
  ASSERT_TRUE(ok.config.interaction.three_body.has_value());
  EXPECT_FALSE(HasErrors(ok.issues));
  EXPECT_EQ(ok.config.interaction.three_body->e3max, 16);
  EXPECT_FALSE(
      ResolveTree(Minimal()).config.interaction.three_body.has_value());
}

TEST(Resolve, from_file_entries_are_read) {
  RawTable e;
  e.emplace_back("name", RawValue{S("Rsq"), {}});
  e.emplace_back("file_2b", RawValue{S("r2.me2j"), {}});
  e.emplace_back("particle_rank", RawValue{I(2), {}});
  const auto r = ResolveTree(
      With(Minimal(), {{"operators.from_file", RawArray{RawValue{e, {}}}}}));
  ASSERT_EQ(r.config.operators.from_file.size(), 1u);
  EXPECT_EQ(r.config.operators.from_file[0].name, "Rsq");
  EXPECT_EQ(r.config.operators.from_file[0].file_2b, "r2.me2j");
  EXPECT_FALSE(r.config.operators.from_file[0].file_3b.has_value());
}

TEST(Resolve, from_file_unknown_key_and_missing_required) {
  RawTable e;
  e.emplace_back("nme", RawValue{S("Rsq"), {}});
  const auto r = ResolveTree(
      With(Minimal(), {{"operators.from_file", RawArray{RawValue{e, {}}}}}));
  const auto* u = FindIssue(r.issues, "E-KEY-UNKNOWN");
  ASSERT_NE(u, nullptr);
  ASSERT_TRUE(u->suggestion.has_value());
  EXPECT_NE(u->suggestion->find("name"), std::string::npos);
  EXPECT_NE(FindIssue(r.issues, "V-03", "operators.from_file[0].name"),
            nullptr);
  EXPECT_NE(FindIssue(r.issues, "V-03", "operators.from_file[0].file_2b"),
            nullptr);
}

TEST(Resolve, valid_tree_yields_no_issues_at_all) {
  const auto r = ResolveTree(Minimal());
  EXPECT_TRUE(r.issues.empty());
}

}  // namespace
}  // namespace imsrg::driver
