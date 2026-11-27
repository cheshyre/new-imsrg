// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Round-trip tests of the resolved-config writer (design/16 sections 1.3, 6.3):
// for every loadable config cfg1, LoadConfig(ToToml(cfg1)) succeeds, equals
// cfg1, and writes the same text.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/config/schema_keys.h"
#include "imsrg/driver/config/writer.h"

#ifndef IMSRG_SOURCE_DIR
#error "IMSRG_SOURCE_DIR must be defined by the build"
#endif

namespace imsrg::driver {
namespace {

std::string IssuesText(const ConfigIssues& issues) {
  std::string s;
  for (const auto& i : issues) s += "  " + i.ToString() + "\n";
  return s;
}

std::string Slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Expected<RunConfig, ConfigIssues> LoadText(const std::string& text,
                                           const std::string& name) {
  ConfigSources s;
  s.config_text = text;
  s.config_text_name = name;
  return LoadConfig(s, LoadOptions{.check_files = false});
}

/// Checks the round-trip property of `text`; returns the loaded config's
/// written TOML (or "" on failure of the property, after reporting it).
void ExpectRoundTrip(const std::string& text, const std::string& name) {
  auto r1 = LoadText(text, name);
  ASSERT_TRUE(r1.has_value()) << name << " does not load:\n"
                              << IssuesText(r1.error());
  const RunConfig& cfg1 = r1.value();
  const std::string written = ToToml(cfg1, true);
  auto r2 = LoadText(written, name + " (written)");
  ASSERT_TRUE(r2.has_value()) << name << ": written TOML does not load:\n"
                              << IssuesText(r2.error()) << "written text:\n"
                              << written << "\noriginal text:\n"
                              << text;
  const RunConfig& cfg2 = r2.value();
  EXPECT_TRUE(cfg2 == cfg1) << name << ": reloaded config differs\n"
                            << "original text:\n"
                            << text << "\nwritten:\n"
                            << written << "\nrewritten:\n"
                            << ToToml(cfg2, false);
  EXPECT_EQ(ToToml(cfg2, false), ToToml(cfg1, false))
      << name << "\noriginal text:\n"
      << text;
}

// ---- the four design examples
// ------------------------------------------------------------

TEST(RoundTrip, DesignExamples) {
  for (const char* f : {"a_o16_imsrg2.toml", "b_o16_imsrg2_3n.toml",
                        "c_o16_sd_shell.toml", "d_o16_imsrg3_n7.toml"}) {
    SCOPED_TRACE(f);
    ExpectRoundTrip(
        Slurp(std::string(IMSRG_SOURCE_DIR) + "/examples/config/" + f), f);
  }
}

// ---- every key set to a valid non-default value
// --------------------------------------------

const char* const kAllKeys = R"toml(
[run]
task = "imsrg"
scratch_dir = "/tmp/imsrg_scratch"
log_level = "debug"

[basis]
emax = 4
hw = 18.5
e3max = 8
lmax = 4
lmax3 = 3
emax_unocc = 3
mass_number = 16
kind = "nat"

[reference]
nucleus = "O16"
occupations_file = "occupations.dat"

[valence]
space = "sd-shell"

[hamiltonian]
kinetic = "lab"
trap_hw = 20.0
beta_cm = 0.5
beta_cm_hw = 22.0
nucleon_mass_correction = true

[interaction.two_body]
file = ["a.me2j.gz"]
format = "me2j"
e1max = 6
e2max = 10
lmax = 5

[interaction.three_body]
file = "three.stream.bin"
type = "no2b"
format = "me3j"
no2b_precision = "half"
e1max = 4
e2max = 8
e3max = 8
store_pn = true

[hartree_fock]
freeze_occupations = true

[natural_orbitals]
use_occupations = true
order = "energy"
sort = "signed_occupation"

[imsrg]
method = "magnus"
steps = 2
use_brueckner_bch = true
perturbative_triples = true

[imsrg.generator]
core = "white"
valence = "shell_model"
qtransfer_n = 3

[imsrg.denominator]
partitioning = "moller_plesset"
delta = 0.5
delta_orbit = "all"

[imsrg.step]
smax = 100.0
dsmax = 0.25
ds_0 = 0.125
domega = 0.05
omega_norm_max = 0.3
eta_criterion = 1e-5
ode_tolerance = 1e-7

[imsrg.omega]
storage = "memory"  # scratch/discard are E-PLANNED
hunter_gatherer = true

[imsrg.space]
emax = 3
e2max = 6
e3max = 7
emax_3body = 3

[imsrg3]
in_flow = true
at_end = true
variant = "n7"
factorized = false  # V-57: true excludes in_flow / at_end (see test_writer)
factorized_1b_intermediates = true
factorized_2b_intermediates = false
only_2b_eta = true
only_2b_omega = true
discard_residual_input_3n = true
threebody_threshold = 1e-6
de3max = 20.0
occ_nat3_cut = 0.01

[operators]
list = ["rhop_0.2", "OBD_p0s1_p0s1", "HCM_16", { name = "rhon", r = 1.5 }]
from_file = [{ name = "Aop", rank = 0, dtz = 0, parity = 0, particle_rank = 2, file_2b = "aop.dat", file_3b = "aop3.dat", format = "miyagi" }]

[output]
directory = "out"
stem = "my_stem"
flow_file = "out/my_stem.flow"
valence_file_format = "tokyo"
write_omega = true
spwf = ["valence", "p0s1"]
)toml";

/// Paths of the keys written in `toml` ("table.key"), by a line scan of the
/// writer's output.
std::set<std::string> WrittenKeyPaths(const std::string& toml) {
  std::set<std::string> paths;
  std::istringstream in(toml);
  std::string line, table;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line[0] == '[') {
      const auto end = line.find(']');
      table = line.substr(1, end - 1);
      continue;
    }
    const auto eq = line.find(" =");
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    paths.insert(table.empty() ? key : table + "." + key);
  }
  return paths;
}

TEST(RoundTrip, EveryKeySetToANonDefaultValue) {
  auto r = LoadText(kAllKeys, "all_keys");
  ASSERT_TRUE(r.has_value()) << IssuesText(r.error());
  const RunConfig& cfg = r.value();
  EXPECT_TRUE(cfg.provenance.warnings.empty())
      << IssuesText(cfg.provenance.warnings);
  ExpectRoundTrip(kAllKeys, "all_keys");

  const std::string written = ToToml(cfg, false);
  const std::set<std::string> present = WrittenKeyPaths(written);
  for (const KeyInfo& k : SchemaKeys()) {
    const std::string path(k.path);
    // Deliberately left out: planned keys (default only) and valence.custom
    // (exclusive with valence.space).
    if (k.phase == Phase::kPlanned || path == "valence.custom") continue;
    EXPECT_TRUE(present.count(path) == 1)
        << "key not in the written output: " << path;
  }
  // Spot-check some values of the config itself.
  EXPECT_EQ(cfg.operators.list.size(), 4u);
  ASSERT_EQ(cfg.operators.from_file.size(), 1u);
  EXPECT_EQ(cfg.operators.from_file[0].rank, 0);
  EXPECT_TRUE(cfg.imsrg3.in_flow);
  EXPECT_EQ(cfg.imsrg.denominator.delta_orbit,
            std::optional<std::string>("all"));
  EXPECT_EQ(cfg.output.spwf.size(), 2u);
  EXPECT_TRUE(cfg.imsrg.space.has_value());
  EXPECT_TRUE(cfg.interaction.three_body.has_value());
}

// A one-stage valence-space run: generator.valence is inert, so the writer
// leaves it out; an explicit value would be V-41 on re-reading (found in
// review).
TEST(RoundTrip, OneStageValenceSpaceRunReloads) {
  ExpectRoundTrip(
      "[basis]\nemax = 4\nhw = 20.0\n[reference]\nnucleus = "
      "\"O16\"\n[valence]\n"
      "space = \"sd-shell\"\n[interaction.two_body]\nfile = \"x.me2j\"\ne1max "
      "= 14\n"
      "[imsrg]\nsteps = 1\n[imsrg.generator]\ncore = \"shell_model_atan\"\n",
      "one_stage_toml");

  ConfigSources s;
  s.legacy_args = {"2bme=x.me2j",   "file2e1max=14",          "emax=4",
                   "reference=O16", "valence_space=sd-shell", "nsteps=1"};
  auto legacy = LoadConfig(s, LoadOptions{.check_files = false});
  ASSERT_TRUE(legacy.has_value()) << IssuesText(legacy.error());
  EXPECT_EQ(ToToml(legacy.value(), false).find("\nvalence = "),
            std::string::npos);
  ExpectRoundTrip(ToToml(legacy.value(), false), "one_stage_legacy");
}

TEST(RoundTrip, SpellingsOfOneOperatorWriteTheSameCanonicalName) {
  const std::string base =
      "[basis]\nemax = 2\nhw = 16.0\n[reference]\nnucleus = \"O16\"\n"
      "[interaction.two_body]\nfile = \"f.me2j\"\ne1max = 4\n";
  auto a = LoadText(base + "[operators]\nlist = [\"rhop_0.2\"]\n", "a");
  auto b = LoadText(
      base + "[operators]\nlist = [{ name = \"rhop\", r = 0.2 }]\n", "b");
  ASSERT_TRUE(a.has_value()) << IssuesText(a.error());
  ASSERT_TRUE(b.has_value()) << IssuesText(b.error());
  EXPECT_TRUE(a.value() == b.value());
  EXPECT_EQ(ToToml(a.value(), false), ToToml(b.value(), false));
}

// ---- randomised
// ----------------------------------------------------------------------------

class Gen {
 public:
  explicit Gen(std::uint64_t seed) : rng_(seed) {}

  double U() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
  int In(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng_);
  }
  bool Coin(double p = 0.5) { return U() < p; }
  template <class T>
  const T& Pick(const std::vector<T>& v) {
    return v[static_cast<std::size_t>(In(0, static_cast<int>(v.size()) - 1))];
  }
  /// Full-precision double text.
  static std::string Num(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", x);
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos)
      s += ".0";  // keep it a TOML float
    return s;
  }
  double AnyHw() { return 0.1 + U() * 7.3 * 5; }
  double Decade() { return std::pow(10.0, -8.0 + 10.0 * U()); }

  std::string Candidate() {
    std::ostringstream t;
    const int emax = In(0, 6);
    const int e1max = emax + In(0, 4);
    const bool valence = Coin(0.4);
    std::string nucleus = "O16";
    std::string space;
    if (valence) {
      const int s = In(0, 2);
      space = s == 0 ? "sd-shell" : s == 1 ? "p-shell" : "fp-shell";
      nucleus = s == 0 ? "O16" : s == 1 ? "He4" : "Ca40";
    } else {
      nucleus = Pick<std::string>({"O16", "Ca40", "He4", "C12", "Ni56"});
    }
    const bool three = Coin(0.4);
    const bool shell_valence = valence;

    t << "[run]\n";
    if (Coin())
      t << "log_level = \"" << Pick<std::string>({"quiet", "normal", "debug"})
        << "\"\n";
    const bool scratch = Coin(0.3);
    if (scratch) t << "scratch_dir = \"/tmp/imsrg_scratch\"\n";

    t << "[basis]\nemax = " << emax << "\nhw = " << Num(AnyHw()) << "\n";
    if (Coin()) t << "e3max = " << In(emax, 3 * emax + 3) << "\n";
    if (Coin()) t << "lmax = " << In(0, emax) << "\n";
    if (Coin(0.3)) t << "emax_unocc = " << In(0, emax) << "\n";
    if (Coin(0.3))
      t << "kind = \"" << Pick<std::string>({"hf", "oscillator"}) << "\"\n";

    t << "[reference]\nnucleus = \"" << nucleus << "\"\n";
    if (valence) t << "[valence]\nspace = \"" << space << "\"\n";

    t << "[hamiltonian]\n";
    if (Coin())
      t << "kinetic = \"" << Pick<std::string>({"intrinsic", "lab", "none"})
        << "\"\n";
    if (Coin()) t << "beta_cm = " << Num(0.1 + U() * 2.0) << "\n";
    if (Coin()) t << "beta_cm_hw = " << Num(AnyHw()) << "\n";
    if (Coin())
      t << "nucleon_mass_correction = " << (Coin() ? "true" : "false") << "\n";

    t << "[interaction.two_body]\nfile = \"vnn.me2j.gz\"\ne1max = " << e1max
      << "\n";
    if (Coin()) t << "e2max = " << In(emax, 2 * e1max) << "\n";
    if (Coin()) t << "lmax = " << In(0, e1max) << "\n";

    if (three) {
      const int e3 = In(emax, 3 * emax + 6);
      t << "[interaction.three_body]\nfile = \"v3n.bin\"\ne1max = "
        << (emax + In(0, 2)) << "\ne3max = " << e3 << "\n";
      if (Coin())
        t << "type = \"" << Pick<std::string>({"full", "no2b"}) << "\"\n";
      if (Coin()) t << "e2max = " << In(emax, 2 * emax + 2) << "\n";
      if (Coin()) t << "store_pn = " << (Coin() ? "true" : "false") << "\n";
    }

    t << "[hartree_fock]\n";
    if (Coin())
      t << "freeze_occupations = " << (Coin() ? "true" : "false") << "\n";
    t << "[natural_orbitals]\n";
    if (Coin())
      t << "order = \"" << Pick<std::string>({"occupation", "energy"})
        << "\"\n";
    if (Coin())
      t << "sort = \"" << Pick<std::string>({"abs_occupation",
                                              "signed_occupation"})
        << "\"\n";

    t << "[imsrg]\n";
    if (Coin())
      t << "method = \"" << Pick<std::string>({"magnus", "flow", "flow_rk4"})
        << "\"\n";
    if (Coin())
      t << "use_brueckner_bch = " << (Coin() ? "true" : "false") << "\n";
    if (Coin())
      t << "perturbative_triples = " << (Coin() ? "true" : "false") << "\n";
    t << "[imsrg.generator]\n";
    if (Coin()) {
      t << "core = \""
        << Pick<std::string>({"white", "atan", "imaginary_time", "wegner"})
        << "\"\n";
    }
    if (shell_valence && Coin()) {
      t << "valence = \""
        << Pick<std::string>({"shell_model", "shell_model_atan",
                              "shell_model_imaginary_time"})
        << "\"\n";
    }
    t << "[imsrg.denominator]\n";
    if (Coin())
      t << "partitioning = \""
        << Pick<std::string>({"epstein_nesbet", "moller_plesset"}) << "\"\n";
    if (Coin()) t << "delta = " << Num(Decade()) << "\ndelta_orbit = \"all\"\n";
    t << "[imsrg.step]\n";
    if (Coin()) t << "smax = " << Num(1.0 + U() * 300.0) << "\n";
    if (Coin()) t << "dsmax = " << Num(0.01 + U()) << "\n";
    if (Coin()) t << "domega = " << Num(Decade() / 100.0) << "\n";
    if (Coin()) t << "eta_criterion = " << Num(Decade() / 100.0) << "\n";
    if (Coin()) t << "ode_tolerance = " << Num(Decade() / 100.0) << "\n";
    t << "[imsrg.omega]\n";
    if (Coin()) {
      t << "storage = \"memory\"\n";  // scratch/discard: E-PLANNED
    }
    if (Coin())
      t << "hunter_gatherer = " << (Coin() ? "true" : "false") << "\n";
    if (Coin(0.3)) {
      t << "[imsrg.space]\nemax = " << In(0, emax) << "\n";
      if (Coin()) t << "e2max = " << In(0, 2 * emax) << "\n";
    }

    if (three && Coin(0.5)) {
      t << "[imsrg3]\n" << (Coin() ? "in_flow" : "at_end") << " = true\n";
      if (Coin())
        t << "variant = \"" << Pick<std::string>({"full", "n7"}) << "\"\n";
      if (Coin()) t << "factorized = " << (Coin() ? "true" : "false") << "\n";
      if (Coin()) t << "threebody_threshold = " << Num(Decade()) << "\n";
      if (Coin()) t << "de3max = " << Num(1.0 + U() * 50.0) << "\n";
      if (Coin()) t << "occ_nat3_cut = " << Num(U() * 0.1) << "\n";
    }

    if (Coin(0.7)) {
      const std::vector<std::string> catalog = {
          "Trel",   "TCM",      "Rp2",           "Rn2",     "Rm2",
          "R2CM",   "E0",       "Iso2",          "Tz2",     "LdotS",
          "HCM_16", "rhop_0.2", "OBD_p0s1_p0s1", "rhon_1.5"};
      t << "[operators]\nlist = [";
      const int n = In(1, 4);
      for (int i = 0; i < n; ++i)
        t << (i ? ", " : "") << '"' << Pick(catalog) << '"';
      t << "]\n";
    }
    t << "[output]\n";
    if (Coin()) t << "directory = \"out" << In(0, 9) << "\"\n";
    if (Coin()) t << "write_omega = " << (Coin() ? "true" : "false") << "\n";
    return t.str();
  }

 private:
  std::mt19937_64 rng_;
};

TEST(RoundTrip, RandomConfigs) {
  constexpr std::uint64_t kSeed = 20260512ULL;
  constexpr int kCandidates = 400;
  Gen gen(kSeed);
  int accepted = 0, rejected = 0;
  for (int i = 0; i < kCandidates; ++i) {
    const std::string text = gen.Candidate();
    auto r = LoadText(text, "random#" + std::to_string(i));
    if (!r.has_value()) {
      ++rejected;
      continue;
    }
    ++accepted;
    SCOPED_TRACE("seed " + std::to_string(kSeed) + ", candidate " +
                 std::to_string(i));
    ExpectRoundTrip(text, "random#" + std::to_string(i));
    if (::testing::Test::HasFailure()) {
      ADD_FAILURE() << "seed " << kSeed << " candidate " << i
                    << " failing text:\n"
                    << text;
      return;
    }
  }
  EXPECT_GE(accepted, 100) << "seed " << kSeed << ": accepted " << accepted
                           << ", rejected " << rejected << " of "
                           << kCandidates;
}

}  // namespace
}  // namespace imsrg::driver
