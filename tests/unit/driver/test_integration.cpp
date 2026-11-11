// End-to-end integration tests of the `imsrg` driver pipeline (task P3 of M8): IT-2, IT-3, IT-4,
// IT-6, IT-7 and IT-12, compared against the imsrg++ goldens under tests/golden/e2e.
//
// Output contract assumed of `driver::run` (all files in `config.output.directory`):
//   results.json  {"e_reference", "imsrg":{"e0","num_omegas","stages":[{"stage","generator",
//                  "steps","s","e0","norm_eta","stop"}]}, "operators":{name:{"reference",
//                  "imsrg"}}, "valence_interaction": path | null}
//   <stem>.flow   CSV, header stage,istep,s,ds,E0,norm_H,norm_eta_1b,norm_eta_2b,norm_eta,
//                 norm_omega_1b,norm_omega_2b,norm_omega,num_omegas,wall_s; istep 0 included
//   <stem>.snt    valence-space Hamiltonian (tokyo); operators in <stem>_<name>.snt
// While `run()` still throws E-PLANNED the tests skip with a message.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "integration/mini_json.hpp"
#include "integration/run_support.hpp"
#include "integration/snt_compare.hpp"
#include <gtest/gtest.h>

namespace {

using namespace imsrg::it;
namespace driver = imsrg::driver;

/// Skips the current test when `path` does not exist.
#define IT_REQUIRE_FILE(path)                                                 \
  do {                                                                        \
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << (path); \
  } while (0)

#define IT_REQUIRE_RUN(r)                                                   \
  do {                                                                      \
    if ((r).planned) GTEST_SKIP() << "driver::run not ready: " << (r).note; \
  } while (0)

/// Total number of steps over all stages of a run, from results.json.
long total_steps(const Json& results) {
  long n = 0;
  for (const auto& st : results["imsrg"]["stages"].items)
    n += static_cast<long>(st["steps"].number);
  return n;
}

/// Shared IT-2 / IT-6 / IT-12 pipeline run: O16, Magnus, atan, 2N only, with Rp2 requested.
/// Run once; the result is cached (nullopt if inputs are missing).
const RunResult* shared_it2_run() {
  static const RunResult* cached = [&]() -> const RunResult* {
    if (!std::filesystem::exists(data_file(two_body_name()))) return nullptr;
    const auto dir = fresh_dir("it2_toml");
    driver::ConfigSources src;
    src.config_text = toml_text(dir, "O16", "[operators]\nlist = [\"Rp2\"]\n");
    static RunResult result = execute(src, dir);
    return &result;
  }();
  return cached;
}

// ---------------------------------------------------------------------------------------------
// IT-2: O16 IMSRG(2), Magnus, atan, 2N only.
// ---------------------------------------------------------------------------------------------

TEST(DriverIntegration, IT2_O16_Magnus_Atan) {
  const RunResult* r = shared_it2_run();
  if (r == nullptr) GTEST_SKIP() << "missing 2N input file";
  IT_REQUIRE_RUN(*r);

  const double golden_e0 = -164.648772308;
  const double e0 = r->results["imsrg"]["e0"].number;
  std::printf("[IT-2] E0 = %.9f  golden %.9f  dE0 = %.3e\n", e0, golden_e0, e0 - golden_e0);
  std::printf("[IT-2] steps = %ld  golden 256  (stages %zu)\n", total_steps(r->results),
              r->results["imsrg"]["stages"].size());
  EXPECT_NEAR(e0, golden_e0, 1e-3);

  // First ten trajectory steps against the golden trajectory.
  const auto traj_path = golden_file("IT-2/trajectory.csv");
  IT_REQUIRE_FILE(traj_path);
  const FlowTable traj = read_flow(traj_path);
  ASSERT_FALSE(r->flow.rows.empty()) << "no flow CSV written";
  double dev_s = 0, dev_e = 0, dev_eta = 0, dev_om = 0;
  const std::size_t n = std::min<std::size_t>(10, std::min(traj.rows.size(), r->flow.rows.size()));
  ASSERT_EQ(n, 10u);
  for (std::size_t i = 0; i < n; ++i) {
    // The golden's step column indexes steps of the single stage; flow rows are in step order.
    dev_s = std::max(dev_s, std::abs(r->flow.at(i, "s") - traj.at(i, "s")));
    dev_e = std::max(dev_e, std::abs(r->flow.at(i, "E0") - traj.at(i, "E0")));
    dev_eta = std::max(dev_eta, std::abs(r->flow.at(i, "norm_eta") - traj.at(i, "norm_eta")));
    dev_om = std::max(dev_om, std::abs(r->flow.at(i, "norm_omega") - traj.at(i, "norm_omega")));
  }
  std::printf(
      "[IT-2] first-10 trajectory max dev: s %.3e  E0 %.3e  norm_eta %.3e  norm_omega %.3e\n",
      dev_s, dev_e, dev_eta, dev_om);
  EXPECT_LE(dev_s, 1e-6);
  EXPECT_LE(dev_e, 1e-6);
  EXPECT_LE(dev_eta, 1e-6);
  EXPECT_LE(dev_om, 1e-6);
}

// ---------------------------------------------------------------------------------------------
// IT-6: IT-2 plus the Rp2 operator (shares the IT-2 run).
// ---------------------------------------------------------------------------------------------

TEST(DriverIntegration, IT6_Rp2_Operator) {
  const RunResult* r = shared_it2_run();
  if (r == nullptr) GTEST_SKIP() << "missing 2N input file";
  IT_REQUIRE_RUN(*r);
  const auto golden_path = golden_file("IT-6/intRp2.snt");
  IT_REQUIRE_FILE(golden_path);

  const SntData golden = read_snt(golden_path);
  ASSERT_TRUE(r->results["operators"].has("Rp2"));
  const double ref = r->results["operators"]["Rp2"]["reference"].number;
  const double imsrg = r->results["operators"]["Rp2"]["imsrg"].number;
  std::printf("[IT-6] Rp2 imsrg = %.8f  golden %.8f  dev %.3e\n", imsrg, golden.zero_body,
              imsrg - golden.zero_body);
  std::printf("[IT-6] Rp2 HF    = %.8f  golden 4.294907  dev %.3e\n", ref, ref - 4.294907);
  EXPECT_NEAR(imsrg, golden.zero_body, 1e-4);
  EXPECT_NEAR(ref, 4.294907, 1e-5);
}

// ---------------------------------------------------------------------------------------------
// IT-12: legacy key=value tokens give the same RunConfig (except output) and the same run as TOML.
// ---------------------------------------------------------------------------------------------

TEST(DriverIntegration, IT12_Legacy_Matches_Toml) {
  const RunResult* toml = shared_it2_run();
  if (toml == nullptr) GTEST_SKIP() << "missing 2N input file";

  const auto dir = fresh_dir("it2_legacy");
  driver::ConfigSources src;
  src.legacy_args = {"2bme=" + data_file(two_body_name()).string(),
                     "file2e1max=14",
                     "file2e2max=28",
                     "file2lmax=14",
                     "hw=16",
                     "emax=4",
                     "reference=O16",
                     "valence_space=O16",
                     "method=magnus",
                     "Operators=Rp2",
                     "intfile=" + (dir / "legacy").string()};
  const RunResult legacy = execute(src, dir);

  // Configs agree in every section except output (provenance is excluded from == already).
  const auto& a = toml->config;
  const auto& b = legacy.config;
  EXPECT_TRUE(a.run == b.run) << "run";
  EXPECT_TRUE(a.basis == b.basis) << "basis";
  EXPECT_TRUE(a.reference == b.reference) << "reference";
  EXPECT_TRUE(a.valence == b.valence) << "valence";
  EXPECT_TRUE(a.hamiltonian == b.hamiltonian) << "hamiltonian";
  EXPECT_TRUE(a.interaction == b.interaction) << "interaction";
  EXPECT_TRUE(a.hartree_fock == b.hartree_fock) << "hartree_fock";
  EXPECT_TRUE(a.natural_orbitals == b.natural_orbitals) << "natural_orbitals";
  EXPECT_TRUE(a.imsrg == b.imsrg) << "imsrg";
  EXPECT_TRUE(a.imsrg3 == b.imsrg3) << "imsrg3";
  EXPECT_TRUE(a.operators == b.operators) << "operators";
  EXPECT_NE(a.output.directory, b.output.directory);

  IT_REQUIRE_RUN(*toml);
  IT_REQUIRE_RUN(legacy);
  const double e0a = toml->results["imsrg"]["e0"].number;
  const double e0b = legacy.results["imsrg"]["e0"].number;
  std::printf("[IT-12] E0 toml %.17g legacy %.17g  diff %.3e\n", e0a, e0b, e0a - e0b);
  std::printf("[IT-12] steps toml %ld legacy %ld\n", total_steps(toml->results),
              total_steps(legacy.results));
  EXPECT_EQ(e0a, e0b);
  EXPECT_EQ(total_steps(toml->results), total_steps(legacy.results));

  // Flow CSVs identical in every column except wall_s.
  ASSERT_EQ(toml->flow.columns, legacy.flow.columns);
  ASSERT_EQ(toml->flow.rows.size(), legacy.flow.rows.size());
  double maxdev = 0;
  for (std::size_t c = 0; c < toml->flow.columns.size(); ++c) {
    if (toml->flow.columns[c] == "wall_s") continue;
    for (std::size_t i = 0; i < toml->flow.rows.size(); ++i)
      maxdev = std::max(maxdev, std::abs(toml->flow.rows[i][c] - legacy.flow.rows[i][c]));
  }
  std::printf("[IT-12] flow CSV max deviation (excluding wall_s) = %.3e over %zu rows\n", maxdev,
              toml->flow.rows.size());
  EXPECT_EQ(maxdev, 0.0);
}

// ---------------------------------------------------------------------------------------------
// IT-4: 2N + NO2B three-body, smax = 10.
// ---------------------------------------------------------------------------------------------

TEST(DriverIntegration, IT4_ThreeBody_NO2B_Smax10) {
  IT_REQUIRE_FILE(data_file(two_body_name()));
  IT_REQUIRE_FILE(data_file(three_body_name()));
  const auto dir = fresh_dir("it4");
  driver::ConfigSources src;
  src.config_text = toml_text(dir, "O16", "", /*three_body=*/true, "10.0");
  const RunResult r = execute(src, dir);
  IT_REQUIRE_RUN(r);

  const double e_hf = r.results["e_reference"].number;
  const double e0 = r.results["imsrg"]["e0"].number;
  std::printf("[IT-4] E_HF = %.7f  golden -313.1684814  dev %.3e\n", e_hf, e_hf + 313.1684814);
  std::printf("[IT-4] E0(s=10) = %.9f  golden -343.761780688  dev %.3e\n", e0, e0 + 343.761780688);
  std::printf("[IT-4] steps = %ld  golden 29\n", total_steps(r.results));
  EXPECT_NEAR(e_hf, -313.1684814, 1e-4);
  EXPECT_NEAR(e0, -343.761780688, 1e-3);
}

// ---------------------------------------------------------------------------------------------
// IT-3 / IT-7: valence-space decoupling, sd-shell, compared with imsrg++ int.snt.
// ---------------------------------------------------------------------------------------------

void run_valence_case(const char* tag, const char* nucleus, const char* golden_rel,
                      long golden_steps) {
  IT_REQUIRE_FILE(data_file(two_body_name()));
  const auto golden_path = golden_file(golden_rel);
  IT_REQUIRE_FILE(golden_path);
  const auto dir = fresh_dir(tag);
  driver::ConfigSources src;
  src.config_text = toml_text(dir, nucleus, "[valence]\nspace = \"sd-shell\"\n");
  const RunResult r = execute(src, dir);
  IT_REQUIRE_RUN(r);

  const auto& stages = r.results["imsrg"]["stages"];
  std::string per_stage;
  for (const auto& st : stages.items)
    per_stage += " " + std::to_string(static_cast<long>(st["steps"].number));
  std::printf("[%s] stages %zu, steps per stage:%s  total %ld  golden %ld\n", tag, stages.size(),
              per_stage.c_str(), total_steps(r.results), golden_steps);

  ASSERT_TRUE(r.results.has("valence_interaction"));
  ASSERT_EQ(r.results["valence_interaction"].kind, Json::Kind::String);
  const std::filesystem::path snt = r.results["valence_interaction"].text;
  ASSERT_TRUE(std::filesystem::exists(snt)) << snt;

  const SntData got = read_snt(snt);
  const SntData want = read_snt(golden_path);
  const SntComparison c = compare_snt(got, want, 1e-3);
  std::printf("[%s] zero body %.6f vs %.6f (dev %.3e)\n", tag, got.zero_body, want.zero_body,
              c.zero_dev);
  std::printf("[%s] one-body elements %zu vs %zu, max dev %.3e\n", tag, c.n1b_a, c.n1b_b,
              c.max_dev_1b);
  std::printf(
      "[%s] two-body elements %zu vs %zu (lines %zu vs %zu, duplicates %zu/%zu), "
      "max dev %.3e, missing>1e-3 %zu\n",
      tag, c.n2b_a, c.n2b_b, got.two_body_lines, want.two_body_lines, got.duplicate_lines,
      want.duplicate_lines, c.max_dev_2b, c.missing);
  std::printf("[%s] orbits equal: %s; worst 2B: %s\n", tag, c.orbits_equal ? "yes" : "NO",
              c.worst.c_str());

  EXPECT_LE(c.zero_dev, 1e-3);
  EXPECT_LE(c.max_dev_1b, 1e-3);
  EXPECT_LE(c.max_dev_2b, 1e-3);
}

TEST(DriverIntegration, SntReaderSelfComparison) {
  const auto path = golden_file("IT-3/int.snt");
  IT_REQUIRE_FILE(path);
  const SntData d = read_snt(path);
  const SntComparison c = compare_snt(d, d, 1e-3);
  std::printf("[snt] IT-3 golden: zero %.6f, %zu 1B, %zu 2B (%zu lines, %zu duplicates)\n",
              d.zero_body, d.one_body.size(), d.two_body.size(), d.two_body_lines,
              d.duplicate_lines);
  EXPECT_EQ(c.max_dev(), 0.0);
  EXPECT_NEAR(d.zero_body, -164.648772, 1e-6);
  EXPECT_FALSE(d.two_body.empty());
}

TEST(DriverIntegration, IT3_O16_SdShell_Valence) {
  run_valence_case("IT-3", "O16", "IT-3/int.snt", 421);
}

TEST(DriverIntegration, IT7_Ne20_SdShell_Valence) {
  run_valence_case("IT-7", "Ne20", "IT-7/int.snt", 431);
}

}  // namespace
