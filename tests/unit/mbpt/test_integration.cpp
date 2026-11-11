// IT-1 (EMP2, emax 4, 2N) and IT-5 (NAT-basis E_HF, emax 4, 2N + NO2B 3N) from the real inputs.
#include "mbpt_test_support.hpp"

#include "imsrg/basis/model_space.hpp"
#include "imsrg/io/me2j.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/opslib/center_of_mass.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::mbtest;

op::Operator build_hamiltonian(const fs::path& me2j) {
  const auto ms = basis::ModelSpace::from_names({.emax = 4}, 16.0, "O16", "O16");
  op::Operator v = io::read_me2j(me2j, {.file = {14, 28, 14}}, ms);
  return opslib::t_rel(ms, 16) + v;
}

fs::path me2j() {
  const fs::path p =
      fs::path(IMSRG_TEST_DATA_DIR) / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  return fs::exists(p) ? p : fs::path{};
}

TEST(MbptIntegration, IT1_Emp2) {
  const fs::path f = me2j();
  if (f.empty()) GTEST_SKIP() << "me2j file not found in " << IMSRG_TEST_DATA_DIR;
  op::Operator H = build_hamiltonian(f);
  const hf::HFResult r = hf::solve(H, nullptr);
  op::Operator hno = hf::normal_ordered_hamiltonian(H, nullptr, r);
  const double mp2 = mbpt::mp2_energy(hno);
  const Json res = read_json(fs::path(IMSRG_GOLDEN_DIR) / "e2e/IT-1/result.json");
  const double ref = res.at("mbpt").at("EMP2").number;
  std::cout << "[IT-1] dEMP2 " << mp2 - ref << "\n";
  EXPECT_NEAR(mp2, ref, 1e-4);
}

TEST(MbptIntegration, IT5_NatBasis) {
  const fs::path f = me2j();
  if (f.empty()) GTEST_SKIP() << "me2j file not found in " << IMSRG_TEST_DATA_DIR;
  const fs::path nf = no2b_file();
  if (nf.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = load_v3(nf, {4, 8, 12, 4});
  op::Operator H = build_hamiltonian(f);
  NatResult res = build_nat(H, &v3);
  const Json j = read_json(fs::path(IMSRG_GOLDEN_DIR) / "e2e/IT-5/result.json");
  const double ehf_hf = j.at("hf").at("EHF").number;
  const double ehf_nat = j.at("Hbare_0b").number;
  const double emp2_nat = j.at("mbpt").at("EMP2").number;
  const double mp2 = mbpt::mp2_energy(res.h_nat);
  std::cout << "[IT-5] dE_HF(HF) " << res.e_hf_hf - ehf_hf << " dE_HF(NAT) "
            << res.e_hf_nat - ehf_nat << " dE0(NAT H) " << res.h_nat.zero_body() - ehf_nat
            << " dE_MP2(NAT) " << mp2 - emp2_nat << "\n";
  EXPECT_NEAR(res.e_hf_hf, ehf_hf, 1e-4);
  EXPECT_NEAR(res.e_hf_nat, ehf_nat, 1e-4);
  EXPECT_NEAR(res.h_nat.zero_body(), ehf_nat, 1e-4);
  EXPECT_NEAR(mp2, emp2_nat, 1e-3);
}

}  // namespace
