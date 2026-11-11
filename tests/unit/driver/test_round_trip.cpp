// Round-trip tests of the resolved-config writer (design/16 sections 1.3, 6.3): for every
// loadable config cfg1, load_config(to_toml(cfg1)) succeeds, equals cfg1, and writes the same text.
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

#include "imsrg/driver/config/load_config.hpp"
#include "imsrg/driver/config/schema_keys.hpp"
#include "imsrg/driver/config/writer.hpp"

#ifndef IMSRG_SOURCE_DIR
#error "IMSRG_SOURCE_DIR must be defined by the build"
#endif

namespace imsrg::driver {
namespace {

std::string issues_text(const ConfigIssues& issues) {
  std::string s;
  for (const auto& i : issues) s += "  " + i.to_string() + "\n";
  return s;
}

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Expected<RunConfig, ConfigIssues> load_text(const std::string& text, const std::string& name) {
  ConfigSources s;
  s.config_text = text;
  s.config_text_name = name;
  return load_config(s, LoadOptions{.check_files = false});
}

/// Checks the round-trip property of `text`; returns the loaded config's written TOML (or "" on
/// failure of the property, after reporting it).
void expect_round_trip(const std::string& text, const std::string& name) {
  auto r1 = load_text(text, name);
  ASSERT_TRUE(r1.has_value()) << name << " does not load:\n" << issues_text(r1.error());
  const RunConfig& cfg1 = r1.value();
  const std::string written = to_toml(cfg1, true);
  auto r2 = load_text(written, name + " (written)");
  ASSERT_TRUE(r2.has_value()) << name << ": written TOML does not load:\n"
                              << issues_text(r2.error()) << "written text:\n"
                              << written << "\noriginal text:\n"
                              << text;
  const RunConfig& cfg2 = r2.value();
  EXPECT_TRUE(cfg2 == cfg1) << name << ": reloaded config differs\n"
                            << "original text:\n"
                            << text << "\nwritten:\n"
                            << written << "\nrewritten:\n"
                            << to_toml(cfg2, false);
  EXPECT_EQ(to_toml(cfg2, false), to_toml(cfg1, false)) << name << "\noriginal text:\n" << text;
}

// ---- the four design examples ------------------------------------------------------------

TEST(RoundTrip, DesignExamples) {
  for (const char* f : {"a_o16_imsrg2.toml", "b_o16_imsrg2_3n.toml", "c_o16_sd_shell.toml",
                        "d_o16_imsrg3_n7.toml"}) {
    SCOPED_TRACE(f);
    expect_round_trip(slurp(std::string(IMSRG_SOURCE_DIR) + "/examples/config/" + f), f);
  }
}

// ---- every key set to a valid non-default value --------------------------------------------

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
storage = "scratch"
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
factorized = true
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

/// Paths of the keys written in `toml` ("table.key"), by a line scan of the writer's output.
std::set<std::string> written_key_paths(const std::string& toml) {
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
  auto r = load_text(kAllKeys, "all_keys");
  ASSERT_TRUE(r.has_value()) << issues_text(r.error());
  const RunConfig& cfg = r.value();
  EXPECT_TRUE(cfg.provenance.warnings.empty()) << issues_text(cfg.provenance.warnings);
  expect_round_trip(kAllKeys, "all_keys");

  const std::string written = to_toml(cfg, false);
  const std::set<std::string> present = written_key_paths(written);
  for (const KeyInfo& k : schema_keys()) {
    const std::string path(k.path);
    // Deliberately left out: planned keys (default only) and valence.custom (exclusive with
    // valence.space).
    if (k.phase == Phase::Planned || path == "valence.custom") continue;
    EXPECT_TRUE(present.count(path) == 1) << "key not in the written output: " << path;
  }
  // Spot-check some values of the config itself.
  EXPECT_EQ(cfg.operators.list.size(), 4u);
  ASSERT_EQ(cfg.operators.from_file.size(), 1u);
  EXPECT_EQ(cfg.operators.from_file[0].rank, 0);
  EXPECT_TRUE(cfg.imsrg3.in_flow);
  EXPECT_EQ(cfg.imsrg.denominator.delta_orbit, std::optional<std::string>("all"));
  EXPECT_EQ(cfg.output.spwf.size(), 2u);
  EXPECT_TRUE(cfg.imsrg.space.has_value());
  EXPECT_TRUE(cfg.interaction.three_body.has_value());
}

TEST(RoundTrip, SpellingsOfOneOperatorWriteTheSameCanonicalName) {
  const std::string base =
      "[basis]\nemax = 2\nhw = 16.0\n[reference]\nnucleus = \"O16\"\n"
      "[interaction.two_body]\nfile = \"f.me2j\"\ne1max = 4\n";
  auto a = load_text(base + "[operators]\nlist = [\"rhop_0.2\"]\n", "a");
  auto b = load_text(base + "[operators]\nlist = [{ name = \"rhop\", r = 0.2 }]\n", "b");
  ASSERT_TRUE(a.has_value()) << issues_text(a.error());
  ASSERT_TRUE(b.has_value()) << issues_text(b.error());
  EXPECT_TRUE(a.value() == b.value());
  EXPECT_EQ(to_toml(a.value(), false), to_toml(b.value(), false));
}

// ---- randomised ----------------------------------------------------------------------------

class Gen {
 public:
  explicit Gen(std::uint64_t seed) : rng_(seed) {}

  double u() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
  int in(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng_); }
  bool coin(double p = 0.5) { return u() < p; }
  template <class T>
  const T& pick(const std::vector<T>& v) {
    return v[static_cast<std::size_t>(in(0, static_cast<int>(v.size()) - 1))];
  }
  /// Full-precision double text.
  static std::string num(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", x);
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";  // keep it a TOML float
    return s;
  }
  double any_hw() { return 0.1 + u() * 7.3 * 5; }
  double decade() { return std::pow(10.0, -8.0 + 10.0 * u()); }

  std::string candidate() {
    std::ostringstream t;
    const int emax = in(0, 6);
    const int e1max = emax + in(0, 4);
    const bool valence = coin(0.4);
    std::string nucleus = "O16";
    std::string space;
    if (valence) {
      const int s = in(0, 2);
      space = s == 0 ? "sd-shell" : s == 1 ? "p-shell" : "fp-shell";
      nucleus = s == 0 ? "O16" : s == 1 ? "He4" : "Ca40";
    } else {
      nucleus = pick<std::string>({"O16", "Ca40", "He4", "C12", "Ni56"});
    }
    const bool three = coin(0.4);
    const bool shell_valence = valence;

    t << "[run]\n";
    if (coin()) t << "log_level = \"" << pick<std::string>({"quiet", "normal", "debug"}) << "\"\n";
    const bool scratch = coin(0.3);
    if (scratch) t << "scratch_dir = \"/tmp/imsrg_scratch\"\n";

    t << "[basis]\nemax = " << emax << "\nhw = " << num(any_hw()) << "\n";
    if (coin()) t << "e3max = " << in(emax, 3 * emax + 3) << "\n";
    if (coin()) t << "lmax = " << in(0, emax) << "\n";
    if (coin(0.3)) t << "emax_unocc = " << in(0, emax) << "\n";
    if (coin(0.3)) t << "kind = \"" << pick<std::string>({"hf", "oscillator"}) << "\"\n";

    t << "[reference]\nnucleus = \"" << nucleus << "\"\n";
    if (valence) t << "[valence]\nspace = \"" << space << "\"\n";

    t << "[hamiltonian]\n";
    if (coin()) t << "kinetic = \"" << pick<std::string>({"intrinsic", "lab", "none"}) << "\"\n";
    if (coin()) t << "beta_cm = " << num(0.1 + u() * 2.0) << "\n";
    if (coin()) t << "beta_cm_hw = " << num(any_hw()) << "\n";
    if (coin()) t << "nucleon_mass_correction = " << (coin() ? "true" : "false") << "\n";

    t << "[interaction.two_body]\nfile = \"vnn.me2j.gz\"\ne1max = " << e1max << "\n";
    if (coin()) t << "e2max = " << in(emax, 2 * e1max) << "\n";
    if (coin()) t << "lmax = " << in(0, e1max) << "\n";

    if (three) {
      const int e3 = in(emax, 3 * emax + 6);
      t << "[interaction.three_body]\nfile = \"v3n.bin\"\ne1max = " << (emax + in(0, 2))
        << "\ne3max = " << e3 << "\n";
      if (coin()) t << "type = \"" << pick<std::string>({"full", "no2b"}) << "\"\n";
      if (coin()) t << "e2max = " << in(emax, 2 * emax + 2) << "\n";
      if (coin()) t << "store_pn = " << (coin() ? "true" : "false") << "\n";
    }

    t << "[hartree_fock]\n";
    if (coin()) t << "freeze_occupations = " << (coin() ? "true" : "false") << "\n";
    t << "[natural_orbitals]\n";
    if (coin()) t << "order = \"" << pick<std::string>({"occupation", "energy"}) << "\"\n";

    t << "[imsrg]\n";
    if (coin()) t << "method = \"" << pick<std::string>({"magnus", "flow", "flow_rk4"}) << "\"\n";
    if (coin()) t << "use_brueckner_bch = " << (coin() ? "true" : "false") << "\n";
    if (coin()) t << "perturbative_triples = " << (coin() ? "true" : "false") << "\n";
    t << "[imsrg.generator]\n";
    if (coin()) {
      t << "core = \"" << pick<std::string>({"white", "atan", "imaginary_time", "wegner"})
        << "\"\n";
    }
    if (shell_valence && coin()) {
      t << "valence = \""
        << pick<std::string>({"shell_model", "shell_model_atan", "shell_model_imaginary_time"})
        << "\"\n";
    }
    t << "[imsrg.denominator]\n";
    if (coin())
      t << "partitioning = \"" << pick<std::string>({"epstein_nesbet", "moller_plesset"}) << "\"\n";
    if (coin()) t << "delta = " << num(decade()) << "\ndelta_orbit = \"all\"\n";
    t << "[imsrg.step]\n";
    if (coin()) t << "smax = " << num(1.0 + u() * 300.0) << "\n";
    if (coin()) t << "dsmax = " << num(0.01 + u()) << "\n";
    if (coin()) t << "domega = " << num(decade() / 100.0) << "\n";
    if (coin()) t << "eta_criterion = " << num(decade() / 100.0) << "\n";
    if (coin()) t << "ode_tolerance = " << num(decade() / 100.0) << "\n";
    t << "[imsrg.omega]\n";
    if (coin()) {
      t << "storage = \""
        << (scratch ? pick<std::string>({"memory", "scratch", "discard"})
                    : pick<std::string>({"memory", "discard"}))
        << "\"\n";
    }
    if (coin()) t << "hunter_gatherer = " << (coin() ? "true" : "false") << "\n";
    if (coin(0.3)) {
      t << "[imsrg.space]\nemax = " << in(0, emax) << "\n";
      if (coin()) t << "e2max = " << in(0, 2 * emax) << "\n";
    }

    if (three && coin(0.5)) {
      t << "[imsrg3]\n" << (coin() ? "in_flow" : "at_end") << " = true\n";
      if (coin()) t << "variant = \"" << pick<std::string>({"full", "n7"}) << "\"\n";
      if (coin()) t << "factorized = " << (coin() ? "true" : "false") << "\n";
      if (coin()) t << "threebody_threshold = " << num(decade()) << "\n";
      if (coin()) t << "de3max = " << num(1.0 + u() * 50.0) << "\n";
      if (coin()) t << "occ_nat3_cut = " << num(u() * 0.1) << "\n";
    }

    if (coin(0.7)) {
      const std::vector<std::string> catalog = {
          "Trel", "TCM", "Rp2",   "Rn2",    "Rm2",      "R2CM",          "E0",
          "Iso2", "Tz2", "LdotS", "HCM_16", "rhop_0.2", "OBD_p0s1_p0s1", "rhon_1.5"};
      t << "[operators]\nlist = [";
      const int n = in(1, 4);
      for (int i = 0; i < n; ++i) t << (i ? ", " : "") << '"' << pick(catalog) << '"';
      t << "]\n";
    }
    t << "[output]\n";
    if (coin()) t << "directory = \"out" << in(0, 9) << "\"\n";
    if (coin()) t << "write_omega = " << (coin() ? "true" : "false") << "\n";
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
    const std::string text = gen.candidate();
    auto r = load_text(text, "random#" + std::to_string(i));
    if (!r.has_value()) {
      ++rejected;
      continue;
    }
    ++accepted;
    SCOPED_TRACE("seed " + std::to_string(kSeed) + ", candidate " + std::to_string(i));
    expect_round_trip(text, "random#" + std::to_string(i));
    if (::testing::Test::HasFailure()) {
      ADD_FAILURE() << "seed " << kSeed << " candidate " << i << " failing text:\n" << text;
      return;
    }
  }
  EXPECT_GE(accepted, 100) << "seed " << kSeed << ": accepted " << accepted << ", rejected "
                           << rejected << " of " << kCandidates;
}

}  // namespace
}  // namespace imsrg::driver
