// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// End-to-end integration tests of the `imsrg` driver pipeline (task P3 of M8):
// IT-2, IT-3, IT-4, IT-6, IT-7 and IT-12, compared against the imsrg++ goldens
// under tests/golden/e2e.
//
// Output contract assumed of `driver::Run` (all files in
// `config.output.directory`):
//   results.json  {"e_reference",
//                  "imsrg":{"e0","num_omegas",
//                           "stages":[{"stage","generator","steps","s","e0",
//                                      "norm_eta","stop"}]},
//                  "operators":{name:{"reference", "imsrg"}},
//                  "valence_interaction": path | null}
//   <stem>.flow   CSV, header stage,istep,s,ds,E0,norm_H,norm_eta_1b,
//                 norm_eta_2b,norm_eta,norm_omega_1b,norm_omega_2b,
//                 norm_omega,num_omegas,wall_s; istep 0 included
//   <stem>.snt    valence-space Hamiltonian (tokyo); operators in
//                 <stem>_<name>.snt
// While `Run()` still throws E-PLANNED the tests skip with a message.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"
#include "tests/unit/driver/integration/snt_compare.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// Skips the current test when `path` does not exist.
#define IMSRG_IT_REQUIRE_FILE(path)                                           \
  do {                                                                        \
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << (path); \
  } while (0)

#define IMSRG_IT_REQUIRE_RUN(r)                                             \
  do {                                                                      \
    if ((r).planned) GTEST_SKIP() << "driver::run not ready: " << (r).note; \
  } while (0)

/// Total number of steps over all stages of a run, from results.json.
std::int64_t TotalSteps(const Json& results) {
  std::int64_t n = 0;
  for (const auto& st : results["imsrg"]["stages"].items)
    n += static_cast<std::int64_t>(st["steps"].number);
  return n;
}

/// Shared IT-2 / IT-6 / IT-12 pipeline run: O16, Magnus, atan, 2N only, with
/// Rp2 requested. Run once; the result is cached (nullopt if inputs are
/// missing).
const RunResult* SharedIt2Run() {
  static const RunResult* cached = [&]() -> const RunResult* {
    if (!std::filesystem::exists(DataFile(TwoBodyName()))) return nullptr;
    const auto dir = FreshDir("it2_toml");
    driver::ConfigSources src;
    src.config_text = TomlText(dir, "O16", "[operators]\nlist = [\"Rp2\"]\n");
    static const RunResult* const kResult = new RunResult(Execute(src, dir));
    return kResult;
  }();
  return cached;
}

// -----------------------------------------------------------------------------
// IT-2: O16 IMSRG(2), Magnus, atan, 2N only.
// -----------------------------------------------------------------------------

TEST(DriverIntegration, IT2_O16_Magnus_Atan) {
  const RunResult* r = SharedIt2Run();
  if (r == nullptr) GTEST_SKIP() << "missing 2N input file";
  IMSRG_IT_REQUIRE_RUN(*r);

  const double golden_e0 = -164.648772308;
  const double e0 = r->results["imsrg"]["e0"].number;
  std::printf("[IT-2] E0 = %.9f  golden %.9f  dE0 = %.3e\n", e0, golden_e0,
              e0 - golden_e0);
  std::printf("[IT-2] steps = %" PRId64 "  golden 256  (stages %zu)\n",
              TotalSteps(r->results), r->results["imsrg"]["stages"].size());
  EXPECT_NEAR(e0, golden_e0, 1.0e-3);

  // First ten trajectory steps against the golden trajectory.
  const auto traj_path = GoldenFile("IT-2/trajectory.csv");
  IMSRG_IT_REQUIRE_FILE(traj_path);
  const FlowTable traj = ReadFlow(traj_path);
  ASSERT_FALSE(r->flow.rows.empty()) << "no flow CSV written";
  double dev_s = 0, dev_e = 0, dev_eta = 0, dev_om = 0;
  const std::size_t n = std::min<std::size_t>(
      10, std::min(traj.rows.size(), r->flow.rows.size()));
  ASSERT_EQ(n, 10u);
  for (std::size_t i = 0; i < n; ++i) {
    // The golden's step column indexes steps of the single stage; flow rows are
    // in step order.
    dev_s = std::max(dev_s, std::abs(r->flow.at(i, "s") - traj.at(i, "s")));
    dev_e = std::max(dev_e, std::abs(r->flow.at(i, "E0") - traj.at(i, "E0")));
    dev_eta = std::max(
        dev_eta, std::abs(r->flow.at(i, "norm_eta") - traj.at(i, "norm_eta")));
    dev_om = std::max(dev_om, std::abs(r->flow.at(i, "norm_omega") -
                                       traj.at(i, "norm_omega")));
  }
  std::printf(
      "[IT-2] first-10 trajectory max dev: s %.3e  E0 %.3e  norm_eta %.3e  "
      "norm_omega %.3e\n",
      dev_s, dev_e, dev_eta, dev_om);
  EXPECT_LE(dev_s, 1.0e-6);
  EXPECT_LE(dev_e, 1.0e-6);
  EXPECT_LE(dev_eta, 1.0e-6);
  EXPECT_LE(dev_om, 1.0e-6);
}

// -----------------------------------------------------------------------------
// IT-6: IT-2 plus the Rp2 operator (shares the IT-2 run).
// -----------------------------------------------------------------------------

TEST(DriverIntegration, IT6_Rp2_Operator) {
  const RunResult* r = SharedIt2Run();
  if (r == nullptr) GTEST_SKIP() << "missing 2N input file";
  IMSRG_IT_REQUIRE_RUN(*r);
  const auto golden_path = GoldenFile("IT-6/intRp2.snt");
  IMSRG_IT_REQUIRE_FILE(golden_path);

  const SntData golden = ReadSnt(golden_path);
  ASSERT_TRUE(r->results["operators"].has("Rp2"));
  const double ref = r->results["operators"]["Rp2"]["reference"].number;
  const double imsrg = r->results["operators"]["Rp2"]["imsrg"].number;
  std::printf("[IT-6] Rp2 imsrg = %.8f  golden %.8f  dev %.3e\n", imsrg,
              golden.zero_body, imsrg - golden.zero_body);
  std::printf("[IT-6] Rp2 HF    = %.8f  golden 4.294907  dev %.3e\n", ref,
              ref - 4.294907);
  // Tightened from 1e-4 with D35: the golden now comes from the patched
  // imsrg++ (Rp2 without Rso), so the old 9.1e-5 offset is gone (dev ~2e-9).
  EXPECT_NEAR(imsrg, golden.zero_body, 1.0e-6);
  EXPECT_NEAR(ref, 4.294907, 1.0e-5);
}

// -----------------------------------------------------------------------------
// IT-12: legacy key=value tokens give the same RunConfig (except output) and
// the same run as TOML.
// -----------------------------------------------------------------------------

TEST(DriverIntegration, IT12_Legacy_Matches_Toml) {
  const RunResult* toml = SharedIt2Run();
  if (toml == nullptr) GTEST_SKIP() << "missing 2N input file";

  const auto dir = FreshDir("it2_legacy");
  driver::ConfigSources src;
  src.legacy_args = {"2bme=" + DataFile(TwoBodyName()).string(),
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
  const RunResult legacy = Execute(src, dir);

  // Configs agree in every section except output (provenance is excluded from
  // == already).
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

  IMSRG_IT_REQUIRE_RUN(*toml);
  IMSRG_IT_REQUIRE_RUN(legacy);
  const double e0a = toml->results["imsrg"]["e0"].number;
  const double e0b = legacy.results["imsrg"]["e0"].number;
  std::printf("[IT-12] E0 toml %.17g legacy %.17g  diff %.3e\n", e0a, e0b,
              e0a - e0b);
  std::printf("[IT-12] steps toml %" PRId64 " legacy %" PRId64 "\n",
              TotalSteps(toml->results), TotalSteps(legacy.results));
  EXPECT_EQ(e0a, e0b);
  EXPECT_EQ(TotalSteps(toml->results), TotalSteps(legacy.results));

  // Flow CSVs identical in every column except wall_s.
  ASSERT_EQ(toml->flow.columns, legacy.flow.columns);
  ASSERT_EQ(toml->flow.rows.size(), legacy.flow.rows.size());
  double maxdev = 0;
  for (std::size_t c = 0; c < toml->flow.columns.size(); ++c) {
    if (toml->flow.columns[c] == "wall_s") continue;
    for (std::size_t i = 0; i < toml->flow.rows.size(); ++i)
      maxdev = std::max(
          maxdev, std::abs(toml->flow.rows[i][c] - legacy.flow.rows[i][c]));
  }
  std::printf(
      "[IT-12] flow CSV max deviation (excluding wall_s) = %.3e over %zu "
      "rows\n",
      maxdev, toml->flow.rows.size());
  EXPECT_EQ(maxdev, 0.0);
}

// -----------------------------------------------------------------------------
// IT-4: 2N + NO2B three-body, smax = 10.
// -----------------------------------------------------------------------------

TEST(DriverIntegration, IT4_ThreeBody_NO2B_Smax10) {
  IMSRG_IT_REQUIRE_FILE(DataFile(TwoBodyName()));
  IMSRG_IT_REQUIRE_FILE(DataFile(ThreeBodyName()));
  const auto dir = FreshDir("it4");
  driver::ConfigSources src;
  src.config_text = TomlText(dir, "O16", "", /*three_body=*/true, "10.0");
  const RunResult r = Execute(src, dir);
  IMSRG_IT_REQUIRE_RUN(r);

  const double e_hf = r.results["e_reference"].number;
  const double e0 = r.results["imsrg"]["e0"].number;
  std::printf("[IT-4] E_HF = %.7f  golden -313.1684814  dev %.3e\n", e_hf,
              e_hf + 313.1684814);
  std::printf("[IT-4] E0(s=10) = %.9f  golden -343.761780688  dev %.3e\n", e0,
              e0 + 343.761780688);
  std::printf("[IT-4] steps = %" PRId64 "  golden 29\n", TotalSteps(r.results));
  EXPECT_NEAR(e_hf, -313.1684814, 1.0e-4);
  EXPECT_NEAR(e0, -343.761780688, 1.0e-3);
}

// -----------------------------------------------------------------------------
// IT-3 / IT-7: valence-space decoupling, sd-shell, compared with imsrg++
// int.snt.
// -----------------------------------------------------------------------------

void RunValenceCase(const char* tag, const char* nucleus,
                    const char* golden_rel, std::int64_t golden_steps) {
  IMSRG_IT_REQUIRE_FILE(DataFile(TwoBodyName()));
  const auto golden_path = GoldenFile(golden_rel);
  IMSRG_IT_REQUIRE_FILE(golden_path);
  const auto dir = FreshDir(tag);
  driver::ConfigSources src;
  src.config_text = TomlText(dir, nucleus, "[valence]\nspace = \"sd-shell\"\n");
  const RunResult r = Execute(src, dir);
  IMSRG_IT_REQUIRE_RUN(r);

  const auto& stages = r.results["imsrg"]["stages"];
  std::string per_stage;
  for (const auto& st : stages.items)
    per_stage +=
        " " + std::to_string(static_cast<std::int64_t>(st["steps"].number));
  std::printf("[%s] stages %zu, steps per stage:%s  total %" PRId64
              "  golden %" PRId64 "\n",
              tag, stages.size(), per_stage.c_str(), TotalSteps(r.results),
              golden_steps);

  ASSERT_TRUE(r.results.has("valence_interaction"));
  ASSERT_EQ(r.results["valence_interaction"].kind, Json::Kind::kString);
  const std::filesystem::path snt = r.results["valence_interaction"].text;
  ASSERT_TRUE(std::filesystem::exists(snt)) << snt;

  const SntData got = ReadSnt(snt);
  const SntData want = ReadSnt(golden_path);
  const SntComparison c = CompareSnt(got, want, 1.0e-3);
  std::printf("[%s] zero body %.6f vs %.6f (dev %.3e)\n", tag, got.zero_body,
              want.zero_body, c.zero_dev);
  std::printf("[%s] one-body elements %zu vs %zu, max dev %.3e\n", tag, c.n1b_a,
              c.n1b_b, c.max_dev_1b);
  std::printf(
      "[%s] two-body elements %zu vs %zu (lines %zu vs %zu, duplicates "
      "%zu/%zu), "
      "max dev %.3e, missing>1e-3 %zu\n",
      tag, c.n2b_a, c.n2b_b, got.two_body_lines, want.two_body_lines,
      got.duplicate_lines, want.duplicate_lines, c.max_dev_2b, c.missing);
  std::printf("[%s] orbits equal: %s; worst 2B: %s\n", tag,
              c.orbits_equal ? "yes" : "NO", c.worst.c_str());

  EXPECT_LE(c.zero_dev, 1.0e-3);
  EXPECT_LE(c.max_dev_1b, 1.0e-3);
  EXPECT_LE(c.max_dev_2b, 1.0e-3);
}

TEST(DriverIntegration, SntReaderSelfComparison) {
  const auto path = GoldenFile("IT-3/int.snt");
  IMSRG_IT_REQUIRE_FILE(path);
  const SntData d = ReadSnt(path);
  const SntComparison c = CompareSnt(d, d, 1.0e-3);
  std::printf(
      "[snt] IT-3 golden: zero %.6f, %zu 1B, %zu 2B (%zu lines, %zu "
      "duplicates)\n",
      d.zero_body, d.one_body.size(), d.two_body.size(), d.two_body_lines,
      d.duplicate_lines);
  EXPECT_EQ(c.max_dev(), 0.0);
  EXPECT_NEAR(d.zero_body, -164.648772, 1.0e-6);
  EXPECT_FALSE(d.two_body.empty());
}

TEST(DriverIntegration, IT3_O16_SdShell_Valence) {
  RunValenceCase("IT-3", "O16", "IT-3/int.snt", 421);
}

TEST(DriverIntegration, IT7_Ne20_SdShell_Valence) {
  RunValenceCase("IT-7", "Ne20", "IT-7/int.snt", 431);
}

}  // namespace
}  // namespace imsrg::it
