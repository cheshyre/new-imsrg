// Tests of the legacy key=value translation (design/16 section 6.3, items 1-5), at the raw-tree
// level.
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.hpp"
#include "imsrg/driver/config/schema_keys.hpp"
#include "imsrg/driver/legacy/legacy_parameters.hpp"

namespace imsrg::driver {
namespace {

/// Translates `args`, prepending `2bme=x.me2j` unless `with_2bme` is false.
LegacyTranslation tr(std::vector<std::string> args, bool with_2bme = true) {
  if (with_2bme) args.insert(args.begin(), "2bme=x.me2j");
  return translate_legacy(args);
}

const ConfigIssue* issue(const LegacyTranslation& t, std::string_view code) {
  for (const ConfigIssue& i : t.issues)
    if (i.code == code) return &i;
  return nullptr;
}

bool has(const LegacyTranslation& t, std::string_view code) { return issue(t, code) != nullptr; }

std::optional<std::string> str(const LegacyTranslation& t, std::string_view p) {
  const RawValue* v = t.tree.find(p);
  if (v == nullptr) return std::nullopt;
  const auto* s = std::get_if<std::string>(&v->data);
  return s ? std::optional<std::string>(*s) : std::nullopt;
}
std::optional<std::int64_t> num_i(const LegacyTranslation& t, std::string_view p) {
  const RawValue* v = t.tree.find(p);
  if (v == nullptr) return std::nullopt;
  const auto* s = std::get_if<std::int64_t>(&v->data);
  return s ? std::optional<std::int64_t>(*s) : std::nullopt;
}
std::optional<double> num_d(const LegacyTranslation& t, std::string_view p) {
  const RawValue* v = t.tree.find(p);
  if (v == nullptr) return std::nullopt;
  const auto* s = std::get_if<double>(&v->data);
  return s ? std::optional<double>(*s) : std::nullopt;
}
std::optional<bool> flag(const LegacyTranslation& t, std::string_view p) {
  const RawValue* v = t.tree.find(p);
  if (v == nullptr) return std::nullopt;
  const auto* s = std::get_if<bool>(&v->data);
  return s ? std::optional<bool>(*s) : std::nullopt;
}
std::vector<std::string> arr(const LegacyTranslation& t, std::string_view p) {
  std::vector<std::string> out;
  const RawValue* v = t.tree.find(p);
  if (v == nullptr) return out;
  if (const auto* a = std::get_if<RawArray>(&v->data))
    for (const RawValue& e : *a)
      if (const auto* s = std::get_if<std::string>(&e.data)) out.push_back(*s);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Item 1: key coverage.

TEST(LegacyKeys, ListMatchesGoldenFile) {
  std::ifstream in(std::string(IMSRG_DRIVER_TEST_DATA) + "/legacy_keys.txt");
  ASSERT_TRUE(in.good());
  std::vector<std::string> golden;
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) golden.push_back(line);
  const auto keys = legacy_keys();
  ASSERT_EQ(keys.size(), 85u);
  ASSERT_EQ(golden.size(), keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) EXPECT_EQ(std::string(keys[i].key), golden[i]);
  std::set<std::string_view> unique;
  for (const auto& k : keys) unique.insert(k.key);
  EXPECT_EQ(unique.size(), 85u);
}

TEST(LegacyKeys, NewKeysAreSchemaPaths) {
  for (const LegacyKeyInfo& k : legacy_keys()) {
    if (k.status == LegacyStatus::NotSupported) {
      EXPECT_TRUE(k.new_key.empty()) << k.key;
      continue;
    }
    ASSERT_FALSE(k.new_key.empty()) << k.key;
    if (k.new_key == "-") continue;  // accepted and dropped (physical_system)
    std::stringstream ss{std::string(k.new_key)};
    for (std::string part; std::getline(ss, part, ',');) {
      const auto b = part.find_first_not_of(' ');
      part = part.substr(b);
      bool ok = find_schema_key(part) != nullptr;
      for (const auto& tab : schema_tables())
        if (tab == part) ok = true;
      if (!ok)  // prefix of a schema key (e.g. a table such as operators.from_file)
        for (const auto& tab : schema_tables())
          if (std::string_view(part).substr(0, tab.size()) == tab) ok = true;
      EXPECT_TRUE(ok) << k.key << " -> " << part;
    }
  }
}

TEST(LegacyKeys, FindAndText) {
  EXPECT_NE(find_legacy_key("emax"), nullptr);
  EXPECT_EQ(find_legacy_key("emaxx"), nullptr);
  const std::string text = legacy_keys_text();
  std::size_t lines = 0;
  for (char c : text) lines += c == '\n';
  EXPECT_EQ(lines, 86u);
  EXPECT_NE(text.find("2bme"), std::string::npos);
}

TEST(LegacyKeys, NotSupportedRows) {
  for (std::string_view k : {"atomicZ", "OperatorsRPA", "OperatorsTDA"}) {
    const auto* r = find_legacy_key(k);
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->status, LegacyStatus::NotSupported);
    EXPECT_FALSE(r->note.empty());
  }
}

// ---------------------------------------------------------------------------------------------
// Item 2: per-key rows.

TEST(LegacyTranslate, SuppliedDefaults) {
  const auto t = tr({});
  EXPECT_FALSE(has_errors(t.issues));
  EXPECT_EQ(num_i(t, "basis.emax"), 6);
  EXPECT_EQ(num_d(t, "basis.hw"), 20.0);
  EXPECT_EQ(num_i(t, "interaction.two_body.e1max"), 12);
  EXPECT_EQ(str(t, "interaction.two_body.file"), "x.me2j");
  EXPECT_EQ(t.tree.find("basis.emax")->where.kind, SourceRef::Kind::LegacyDefault);
  EXPECT_FALSE(t.tree.contains("interaction.three_body.e1max"));
  const auto t3 = tr({"3bme=x.me3j"});
  EXPECT_EQ(num_i(t3, "interaction.three_body.e1max"), 12);
  EXPECT_EQ(num_i(t3, "interaction.three_body.e3max"), 12);
}

// An invalid value of a defaulted key is reported once; the old default is still supplied so that
// resolution adds no consequential "required key missing" (found in review).
TEST(LegacyTranslate, InvalidValueOfDefaultedKeyStillGetsDefault) {
  const auto t = tr({"emax=4.5", "hw=abc"});
  EXPECT_TRUE(has(t, "E-LEGACY-VALUE"));
  EXPECT_EQ(num_i(t, "basis.emax"), 6);
  EXPECT_EQ(num_d(t, "basis.hw"), 20.0);
  ConfigSources s;
  s.legacy_args = {"2bme=x.me2j", "emax=4.5", "reference=O16"};
  const auto loaded = load_config(s, LoadOptions{.check_files = false});
  ASSERT_FALSE(loaded.has_value());
  for (const ConfigIssue& i : loaded.error()) EXPECT_NE(i.code, "V-03") << i.to_string();
}

// imsrg++ read numbers with istringstream: a leading '+' is accepted; inf/nan are not values.
TEST(LegacyTranslate, NumberSyntax) {
  EXPECT_EQ(num_i(tr({"emax=+4"}), "basis.emax"), 4);
  EXPECT_EQ(num_d(tr({"hw=+16.5"}), "basis.hw"), 16.5);
  EXPECT_TRUE(has(tr({"emax=+-4"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"smax=inf"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"smax=nan"}), "E-LEGACY-VALUE"));
}

TEST(LegacyTranslate, ArgSourceRef) {
  const auto t = tr({"emax=8"});
  const RawValue* v = t.tree.find("basis.emax");
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->where.kind, SourceRef::Kind::LegacyArg);
  EXPECT_EQ(v->where.name, "emax=8");
  EXPECT_EQ(num_i(t, "basis.emax"), 8);
}

struct ValueRow {
  std::vector<std::string> args;
  const char* path;
  std::string expected;  // string value
};

class EnumRows : public ::testing::TestWithParam<ValueRow> {};
TEST_P(EnumRows, Maps) {
  const auto& r = GetParam();
  const auto t = tr(r.args);
  EXPECT_FALSE(has_errors(t.issues)) << r.args[0];
  EXPECT_EQ(str(t, r.path), r.expected) << r.args[0];
}
INSTANTIATE_TEST_SUITE_P(
    Spellings, EnumRows,
    ::testing::Values(
        ValueRow{{"core_generator=white"}, "imsrg.generator.core", "white"},
        ValueRow{{"core_generator=atan"}, "imsrg.generator.core", "atan"},
        ValueRow{{"core_generator=imaginary-time"}, "imsrg.generator.core", "imaginary_time"},
        ValueRow{{"core_generator=imaginary_time"}, "imsrg.generator.core", "imaginary_time"},
        ValueRow{{"core_generator=wegner"}, "imsrg.generator.core", "wegner"},
        ValueRow{{"core_generator=shell-model"}, "imsrg.generator.core", "shell_model"},
        ValueRow{{"core_generator=shell-model-atan"}, "imsrg.generator.core", "shell_model_atan"},
        ValueRow{
            {"core_generator=shell-model-wegner"}, "imsrg.generator.core", "shell_model_wegner"},
        ValueRow{{"core_generator=shell-model-atan-npnh"},
                 "imsrg.generator.core",
                 "shell_model_atan_npnh"},
        ValueRow{{"core_generator=shell-model-imaginary-time"},
                 "imsrg.generator.core",
                 "shell_model_imaginary_time"},
        ValueRow{{"core_generator=hartree-fock"}, "imsrg.generator.core", "hartree_fock"},
        ValueRow{{"core_generator=1PA"}, "imsrg.generator.core", "one_pa"},
        ValueRow{
            {"valence_generator=shell-model-atan"}, "imsrg.generator.valence", "shell_model_atan"},
        ValueRow{{"basis=HF"}, "basis.kind", "hf"},
        ValueRow{{"basis=oscillator"}, "basis.kind", "oscillator"},
        ValueRow{{"basis=NAT"}, "basis.kind", "nat"}, ValueRow{{"basis=nat"}, "basis.kind", "nat"},
        ValueRow{{"denominator_partitioning=Epstein_Nesbet"},
                 "imsrg.denominator.partitioning",
                 "epstein_nesbet"},
        ValueRow{{"denominator_partitioning=Moller_Plesset"},
                 "imsrg.denominator.partitioning",
                 "moller_plesset"},
        ValueRow{{"denominator_partitioning=MP_isospin"},
                 "imsrg.denominator.partitioning",
                 "mp_isospin"},
        ValueRow{{"3bme=a.me3j", "3bme_type=no2b"}, "interaction.three_body.type", "no2b"},
        ValueRow{{"3bme=a.me3j", "3bme_type=mono"}, "interaction.three_body.type", "mono"},
        ValueRow{{"3bme=a.me3j", "no2b_precision=half"},
                 "interaction.three_body.no2b_precision",
                 "half"},
        ValueRow{{"fmt2=me2j"}, "interaction.two_body.format", "me2j"},
        ValueRow{{"fmt2=navratil"}, "interaction.two_body.format", "navratil"},
        ValueRow{{"fmt2=oslo"}, "interaction.two_body.format", "oslo"},
        ValueRow{{"fmt2=oakridge"}, "interaction.two_body.format", "oakridge"},
        ValueRow{{"fmt2=oakridgebin"}, "interaction.two_body.format", "oakridge_bin"},
        ValueRow{{"fmt2=takayuki"}, "interaction.two_body.format", "takayuki"},
        ValueRow{{"fmt2=nushellx"}, "interaction.two_body.format", "nushellx"},
        ValueRow{{"3bme=a.me3j", "fmt3=me3j"}, "interaction.three_body.format", "me3j"},
        ValueRow{{"3bme=a.me3j", "fmt3=navratil"}, "interaction.three_body.format", "navratil"},
        ValueRow{{"valence_file_format=nushellx"}, "output.valence_file_format", "nushellx"},
        ValueRow{{"valence_file_format=antoine"}, "output.valence_file_format", "antoine"},
        ValueRow{{"basis=NAT", "NAT_order=energy"}, "natural_orbitals.order", "energy"},
        ValueRow{{"method=magnus_euler"}, "imsrg.method", "magnus"},
        ValueRow{{"method=flow_adaptive"}, "imsrg.method", "flow"},
        ValueRow{{"method=flow_RK4"}, "imsrg.method", "flow_rk4"},
        ValueRow{{"method=flow_rk4"}, "imsrg.method", "flow_rk4"},
        ValueRow{{"method=magnus_backoff"}, "imsrg.method", "magnus_backoff"}));

TEST(LegacyTranslate, NushellxSetsKineticNone) {
  const auto t = tr({"fmt2=nushellx"});
  EXPECT_EQ(str(t, "hamiltonian.kinetic"), "none");
}

TEST(LegacyTranslate, QtransferAtan) {
  const auto t = tr({"core_generator=qtransfer-atan_2"});
  EXPECT_FALSE(has_errors(t.issues));
  EXPECT_EQ(str(t, "imsrg.generator.core"), "qtransfer_atan");
  EXPECT_EQ(num_i(t, "imsrg.generator.qtransfer_n"), 2);
}

TEST(LegacyTranslate, Sentinels) {
  // lmax: 99999 or >= emax is absent.
  EXPECT_FALSE(tr({"lmax=99999"}).tree.contains("basis.lmax"));
  EXPECT_FALSE(tr({"emax=6", "lmax=6"}).tree.contains("basis.lmax"));
  EXPECT_EQ(num_i(tr({"emax=6", "lmax=3"}), "basis.lmax"), 3);
  EXPECT_FALSE(tr({"lmax3=-1"}).tree.contains("basis.lmax3"));
  EXPECT_FALSE(tr({"lmax3=0"}).tree.contains("basis.lmax3"));
  EXPECT_EQ(num_i(tr({"lmax3=4"}), "basis.lmax3"), 4);
  EXPECT_FALSE(tr({"emax_unocc=-1"}).tree.contains("basis.emax_unocc"));
  EXPECT_EQ(num_i(tr({"emax_unocc=8"}), "basis.emax_unocc"), 8);
  EXPECT_FALSE(tr({"A=-1"}).tree.contains("basis.mass_number"));
  EXPECT_EQ(num_i(tr({"A=16"}), "basis.mass_number"), 16);
  EXPECT_FALSE(tr({"hwBetaCM=-1"}).tree.contains("hamiltonian.beta_cm_hw"));
  EXPECT_EQ(num_d(tr({"hwBetaCM=24"}), "hamiltonian.beta_cm_hw"), 24.0);
  EXPECT_FALSE(tr({"hw_trap=-1"}).tree.contains("hamiltonian.trap_hw"));
  {
    const auto t = tr({"hw_trap=20"});
    EXPECT_EQ(num_d(t, "hamiltonian.trap_hw"), 20.0);
    EXPECT_EQ(str(t, "hamiltonian.kinetic"), "lab");
  }
  EXPECT_FALSE(tr({"OccNat3Cut=-1"}).tree.contains("imsrg3.occ_nat3_cut"));
  EXPECT_FALSE(tr({"OccNat3Cut=0"}).tree.contains("imsrg3.occ_nat3_cut"));
  EXPECT_FALSE(tr({"file2e2max=-1"}).tree.contains("interaction.two_body.e2max"));
  EXPECT_EQ(num_i(tr({"file2e2max=24"}), "interaction.two_body.e2max"), 24);
  EXPECT_FALSE(tr({"file2lmax=-1"}).tree.contains("interaction.two_body.lmax"));
  {
    const auto t = tr({"file2lmax=0"});
    EXPECT_FALSE(t.tree.contains("interaction.two_body.lmax"));
    EXPECT_TRUE(has(t, "W-Q4"));
  }
  EXPECT_FALSE(tr({"3bme=a.me3j", "file3e2max=-1"}).tree.contains("interaction.three_body.e2max"));
  EXPECT_EQ(num_i(tr({"3bme=a.me3j", "file3e2max=14"}), "interaction.three_body.e2max"), 14);
  EXPECT_FALSE(tr({"nsteps=-1"}).tree.contains("imsrg.steps"));
  EXPECT_EQ(num_i(tr({"nsteps=1", "valence_space=sd-shell"}), "imsrg.steps"), 1);
  EXPECT_EQ(num_i(tr({"nsteps=2", "valence_space=sd-shell"}), "imsrg.steps"), 2);
  EXPECT_TRUE(has(tr({"nsteps=3"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"nsteps=0"}), "E-LEGACY-VALUE"));
  for (const char* k : {"emax_imsrg", "e2max_imsrg", "e3max_imsrg", "emax_3body_imsrg"}) {
    const auto none = tr({std::string(k) + "=-1"});
    EXPECT_EQ(none.tree.entries().size(), tr({}).tree.entries().size()) << k;
    const auto set = tr({std::string(k) + "=0"});
    EXPECT_EQ(set.tree.entries().size(), tr({}).tree.entries().size() + 1) << k;
  }
}

TEST(LegacyTranslate, NumbersAndBools) {
  const auto t = tr({"smax=3.5", "dsmax=0.25", "BetaCM=1.5", "write_omega=true",
                     "freeze_occupations=true", "hunter_gatherer=true", "e3max=14"});
  EXPECT_FALSE(has_errors(t.issues));
  EXPECT_EQ(num_d(t, "imsrg.step.smax"), 3.5);
  EXPECT_EQ(num_d(t, "imsrg.step.dsmax"), 0.25);
  EXPECT_EQ(num_d(t, "hamiltonian.beta_cm"), 1.5);
  EXPECT_EQ(flag(t, "output.write_omega"), true);
  EXPECT_EQ(flag(t, "hartree_fock.freeze_occupations"), true);
  EXPECT_EQ(flag(t, "imsrg.omega.hunter_gatherer"), true);
  EXPECT_EQ(num_i(t, "basis.e3max"), 14);
}

TEST(LegacyTranslate, BadValues) {
  EXPECT_TRUE(has(tr({"emax=4.5"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"emax=abc"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"hw=20x"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"IMSRG3=True"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"IMSRG3=1"}), "E-LEGACY-VALUE"));
  EXPECT_TRUE(has(tr({"emax="}), "E-LEGACY-SYNTAX"));
  EXPECT_TRUE(has(tr({"foo"}), "E-LEGACY-SYNTAX"));
  EXPECT_TRUE(has(tr({"basis=bogus"}), "V-01"));
  EXPECT_TRUE(has(tr({"core_generator=nonsense"}), "V-01"));
  {
    const auto t = tr({"method=magnus_eular"});
    const ConfigIssue* i = issue(t, "V-01");
    ASSERT_NE(i, nullptr);
    ASSERT_TRUE(i->suggestion.has_value());
    EXPECT_EQ(*i->suggestion, "magnus_euler");
  }
}

TEST(LegacyTranslate, MissingTwoBodyIsV03) {
  EXPECT_TRUE(has(tr({}, false), "V-03"));
  EXPECT_TRUE(has(tr({"2bme=none"}, false), "V-03"));
  EXPECT_FALSE(has(tr({"2bme=Minnesota", "fmt2=schematic"}, false), "V-03"));
  const auto t = tr({"fmt2=oakridge", "2bme=sp,vnn"}, false);
  EXPECT_EQ(arr(t, "interaction.two_body.file").size(), 2u);
}

TEST(LegacyTranslate, ThreeBodyKeysWithout3bmeWarn) {
  const auto t = tr({"3bme_type=mono", "no2b_precision=half", "store_3bme_pn=true"});
  EXPECT_TRUE(has(t, "W-L1"));
  EXPECT_FALSE(t.tree.contains("interaction.three_body.type"));
  EXPECT_FALSE(t.tree.contains("interaction.three_body.store_pn"));
  EXPECT_FALSE(has_errors(t.issues));
}

TEST(LegacyTranslate, DuplicateKeys) {
  const auto t = tr({"emax=4", "emax=8"});
  EXPECT_TRUE(has(t, "W-DUPLICATE"));
  EXPECT_EQ(num_i(t, "basis.emax"), 8);
  const auto v = tr({"Operators=a", "Operators=b"});
  EXPECT_FALSE(has(v, "W-DUPLICATE"));
  EXPECT_EQ(arr(v, "operators.list"), (std::vector<std::string>{"a", "b"}));
}

// ---------------------------------------------------------------------------------------------
// Item 3: shorthands.

TEST(LegacyShorthands, NSmagnus) {
  const auto t = tr({"method=NSmagnus"});
  EXPECT_EQ(str(t, "imsrg.method"), "magnus");
  EXPECT_EQ(num_d(t, "imsrg.step.omega_norm_max"), 50000.0);
  EXPECT_FALSE(t.applied_rules.empty());
}

TEST(LegacyShorthands, Brueckner) {
  const auto t = tr({"method=brueckner"});
  EXPECT_EQ(str(t, "imsrg.method"), "magnus");
  EXPECT_EQ(flag(t, "imsrg.use_brueckner_bch"), true);
  EXPECT_EQ(num_d(t, "imsrg.step.omega_norm_max"), 500.0);
}

TEST(LegacyShorthands, Brueckner1step) {
  const auto t = tr(
      {"method=brueckner1step", "valence_space=sd-shell", "valence_generator=shell-model-wegner"});
  EXPECT_EQ(str(t, "imsrg.method"), "magnus");
  EXPECT_EQ(flag(t, "imsrg.use_brueckner_bch"), true);
  EXPECT_EQ(num_i(t, "imsrg.steps"), 1);
  EXPECT_EQ(str(t, "imsrg.generator.core"), "shell_model_wegner");
  EXPECT_EQ(num_d(t, "imsrg.step.omega_norm_max"), 500.0);
}

TEST(LegacyShorthands, Brueckner2IsPlanned) {
  EXPECT_TRUE(has(tr({"method=brueckner2"}), "E-PLANNED"));
}

TEST(LegacyShorthands, RunTasks) {
  EXPECT_EQ(str(tr({"method=HF"}), "run.task"), "hf");
  EXPECT_EQ(str(tr({"method=MP3"}), "run.task"), "mp3");
  EXPECT_EQ(str(tr({"method=FCI"}), "run.task"), "fci_export");
  EXPECT_FALSE(tr({"method=HF"}).tree.contains("imsrg.method"));
}

TEST(LegacyShorthands, Scratch) {
  {
    const auto t = tr({"scratch=/dev/null"});
    EXPECT_EQ(str(t, "imsrg.omega.storage"), "discard");
    EXPECT_FALSE(t.tree.contains("run.scratch_dir"));
  }
  {
    const auto t = tr({"scratch=/dev/null", "Operators=E2"});
    EXPECT_EQ(str(t, "imsrg.omega.storage"), "discard");  // validation judges Operators later
    EXPECT_EQ(arr(t, "operators.list"), (std::vector<std::string>{"E2"}));
  }
  {
    const auto t = tr({"scratch=/tmp/scr"});
    EXPECT_EQ(str(t, "imsrg.omega.storage"), "scratch");
    EXPECT_EQ(str(t, "run.scratch_dir"), "/tmp/scr");
  }
}

TEST(LegacyShorthands, ValenceSpace) {
  {
    const auto t = tr({"valence_space=sd-shell"});
    EXPECT_EQ(str(t, "valence.space"), "sd-shell");
    EXPECT_FALSE(t.tree.contains("reference.nucleus"));
  }
  {
    const auto t = tr({"valence_space=O16"});
    EXPECT_EQ(str(t, "reference.nucleus"), "O16");
    EXPECT_FALSE(t.tree.contains("valence.space"));
  }
  for (const char* v : {"0hw-shell", "jj-shell", "sd-shell-FCI", "p-shell,sd-shell"}) {
    const auto t = tr({std::string("valence_space=") + v});
    EXPECT_FALSE(has_errors(t.issues)) << v;
    EXPECT_EQ(str(t, "valence.space"), v);
  }
  EXPECT_TRUE(has(tr({"valence_space=bogus"}), "V-01"));
  {
    const auto t = tr({"reference=O16", "valence_space=Ne20"});
    EXPECT_TRUE(has(t, "E-LEGACY-NS"));
  }
}

TEST(LegacyShorthands, CustomValenceSpace) {
  const auto t = tr({"custom_valence_space=p0,p1"});
  EXPECT_EQ(str(t, "valence.custom"), "p0,p1");
  EXPECT_FALSE(has(t, "V-30"));
  const auto both = tr({"custom_valence_space=p0,p1", "valence_space=sd-shell"});
  EXPECT_TRUE(has(both, "V-30"));
  EXPECT_EQ(str(both, "valence.custom"), "p0,p1");
}

TEST(LegacyShorthands, Intfile) {
  {
    const auto t = tr({"intfile=out/run1"});
    EXPECT_EQ(str(t, "output.directory"), "out");
    EXPECT_EQ(str(t, "output.stem"), "run1");
  }
  {
    const auto t = tr({"intfile=run1"});
    EXPECT_EQ(str(t, "output.stem"), "run1");
    EXPECT_EQ(str(t, "output.directory"), ".");
  }
  EXPECT_FALSE(tr({"intfile=default"}).tree.contains("output.stem"));
}

TEST(LegacyShorthands, OperatorsFromFileTag) {
  const auto t = tr({"OperatorsFromFile=Name^2_0_0_0^f2.op"});
  EXPECT_FALSE(has_errors(t.issues));
  const RawValue* v = t.tree.find("operators.from_file");
  ASSERT_NE(v, nullptr);
  const auto* a = std::get_if<RawArray>(&v->data);
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(a->size(), 1u);
  EXPECT_TRUE(std::holds_alternative<RawTable>((*a)[0].data));
}

TEST(LegacyShorthands, OperatorsPassThrough) {
  const auto t = tr({"Operators=rhop_all,E2"});
  EXPECT_EQ(arr(t, "operators.list"), (std::vector<std::string>{"rhop_all", "E2"}));
  const auto p = tr({"OperatorsPT1=E2", "SPWF=0s1/2"});
  EXPECT_EQ(arr(p, "operators.first_order"), (std::vector<std::string>{"E2"}));
  EXPECT_EQ(arr(p, "output.spwf"), (std::vector<std::string>{"0s1/2"}));
}

// ---------------------------------------------------------------------------------------------
// Item 4: warning rules.

TEST(LegacyWarnings, Nsteps1ValenceGenerator) {
  const auto t = tr({"nsteps=1", "valence_space=sd-shell", "valence_generator=shell-model-wegner"});
  EXPECT_TRUE(has(t, "W-NSTEPS1"));
  EXPECT_FALSE(t.tree.contains("imsrg.generator.valence"));
  EXPECT_FALSE(has_errors(t.issues));
}

TEST(LegacyWarnings, Q9) {
  const auto t = tr({"nsteps=2"});
  EXPECT_TRUE(has(t, "W-Q9"));
  EXPECT_FALSE(t.tree.contains("imsrg.steps"));
}

TEST(LegacyWarnings, Q3) {
  const auto t = tr({"denominator_delta=0.5"});
  EXPECT_TRUE(has(t, "W-Q3"));
  EXPECT_FALSE(t.tree.contains("imsrg.denominator.delta"));
  const auto ok = tr({"denominator_delta=0.5", "denominator_delta_orbit=p0p1"});
  EXPECT_FALSE(has(ok, "W-Q3"));
  EXPECT_EQ(num_d(ok, "imsrg.denominator.delta"), 0.5);
  EXPECT_EQ(str(ok, "imsrg.denominator.delta_orbit"), "p0p1");
}

TEST(LegacyWarnings, Q10) {
  const auto t = tr({"method=flow", "perturbative_triples=true"});
  EXPECT_TRUE(has(t, "W-Q10"));
  EXPECT_FALSE(t.tree.contains("imsrg.perturbative_triples"));
  const auto ok = tr({"method=magnus", "perturbative_triples=true"});
  EXPECT_FALSE(has(ok, "W-Q10"));
  EXPECT_EQ(flag(ok, "imsrg.perturbative_triples"), true);
}

TEST(LegacyWarnings, L2) {
  const auto t = tr({"imsrg3_n7=true", "only_2b_eta=true"});
  EXPECT_TRUE(has(t, "W-L2"));
  EXPECT_FALSE(t.tree.contains("imsrg3.variant"));
  EXPECT_FALSE(t.tree.contains("imsrg3.only_2b_eta"));
  const auto ok = tr({"IMSRG3=true", "imsrg3_n7=true"});
  EXPECT_FALSE(has(ok, "W-L2"));
  EXPECT_EQ(str(ok, "imsrg3.variant"), "n7");
  EXPECT_EQ(flag(ok, "imsrg3.in_flow"), true);
}

TEST(LegacyWarnings, Q15) {
  const auto t = tr({"method=NSmagnus", "omega_norm_max=0.5"});
  EXPECT_TRUE(has(t, "W-Q15"));
  EXPECT_EQ(num_d(t, "imsrg.step.omega_norm_max"), 50000.0);
  EXPECT_FALSE(has(tr({"method=NSmagnus", "omega_norm_max=50000"}), "W-Q15"));
}

TEST(LegacyWarnings, Q2) {
  EXPECT_TRUE(has(tr({"ds_0=0.1"}), "W-Q2"));
  EXPECT_FALSE(has(tr({"ds_0=0.1", "method=flow"}), "W-Q2"));
}

TEST(LegacyWarnings, V21) {
  const auto t = tr({"use_NAT_occupations=true"});
  EXPECT_TRUE(has(t, "V-21"));
  EXPECT_FALSE(t.tree.contains("natural_orbitals.use_occupations"));
  const auto ok = tr({"basis=NAT", "use_NAT_occupations=true"});
  EXPECT_FALSE(has(ok, "V-21"));
  EXPECT_EQ(flag(ok, "natural_orbitals.use_occupations"), true);
}

TEST(LegacyWarnings, V52) {
  const auto t = tr({"method=flow", "imsrg3_at_end=true"});
  EXPECT_TRUE(has(t, "V-52"));
  EXPECT_FALSE(t.tree.contains("imsrg3.at_end"));
}

// ---------------------------------------------------------------------------------------------
// Item 5: rejections.

TEST(LegacyRejections, NotSupported) {
  for (const char* a : {"physical_system=atomic", "atomicZ=8", "OperatorsRPA=x", "OperatorsTDA=x",
                        "3bme=a.h5", "fmt3=hdf5"}) {
    std::vector<std::string> args{a};
    if (std::string(a) == "fmt3=hdf5") args.push_back("3bme=a.h5");
    const auto t = tr(args);
    const ConfigIssue* i = issue(t, "E-LEGACY-NS");
    if (std::string(a) == "3bme=a.h5") {
      EXPECT_FALSE(has_errors(t.issues));
      continue;
    }
    ASSERT_NE(i, nullptr) << a;
    EXPECT_FALSE(i->message.empty());
  }
  const auto h = tr({"3bme=a.h5", "fmt3=hdf5"});
  const ConfigIssue* i = issue(h, "E-LEGACY-NS");
  ASSERT_NE(i, nullptr);
  EXPECT_NE(i->message.find("HDF5 3N files are not supported"), std::string::npos);
}

TEST(LegacyRejections, NuclearAccepted) {
  const auto t = tr({"physical_system=nuclear"});
  EXPECT_FALSE(has_errors(t.issues));
  EXPECT_TRUE(has(tr({"physical_system=bogus"}), "E-LEGACY-VALUE"));
}

TEST(LegacyRejections, UnknownKey) {
  const auto t = tr({"emaxx=4"});
  const ConfigIssue* i = issue(t, "E-KEY-UNKNOWN");
  ASSERT_NE(i, nullptr);
  ASSERT_TRUE(i->suggestion.has_value());
  EXPECT_EQ(*i->suggestion, "emax");
}

}  // namespace
}  // namespace imsrg::driver
