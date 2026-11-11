// MBPT(2) density matrix and natural orbitals at emax 2 (O16, 2N): structural invariants.
#include <algorithm>

#include "mbpt_test_support.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::mbtest;

TEST(MbptNaturalOrbitals, Emax2Invariants) {
  op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  NatResult res = build_nat(H, nullptr);
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  const auto N = ms.num_orbits();
  const auto& C = res.nat.c_hf2nat;

  double trace = 0.0;
  for (std::int32_t a = 0; a < N; ++a)
    trace += (basis.orbit(OrbitIndex(a)).j2.twice + 1) * res.rho_hf(a, a);
  EXPECT_NEAR(trace, ms.reference().A(), 1e-6);

  double max_orth = 0.0, max_offblock = 0.0;
  for (std::int32_t a = 0; a < N; ++a) {
    const OrbitIndex oa(a);
    EXPECT_GE(C(a, a), 0.0);
    const auto ch = basis.same_ljtz(oa);
    for (std::int32_t b = 0; b < N; ++b) {
      double s = 0.0;
      for (std::int32_t k = 0; k < N; ++k) s += C(k, a) * C(k, b);
      max_orth = std::max(max_orth, std::abs(s - (a == b ? 1.0 : 0.0)));
      if (std::find(ch.begin(), ch.end(), OrbitIndex(b)) == ch.end())
        max_offblock = std::max(max_offblock, std::abs(C(a, b)));
    }
    for (std::size_t k = 1; k < ch.size(); ++k)
      EXPECT_GE(res.nat.occupations[static_cast<std::size_t>(ch[k - 1].value())],
                res.nat.occupations[static_cast<std::size_t>(ch[k].value())]);
  }
  EXPECT_LT(max_orth, 1e-12);
  EXPECT_EQ(max_offblock, 0.0);

  std::cout << "[nat emax2] E_HF(HF) " << res.e_hf_hf << " E_HF(NAT) " << res.e_hf_nat
            << " E0(NAT H) " << res.h_nat.zero_body() << " MP2(NAT) "
            << mbpt::mp2_energy(res.h_nat) << "\n";
  EXPECT_GE(res.e_hf_nat, res.e_hf_hf - 1e-8);  // HF is the variational minimum
  EXPECT_NEAR(res.h_nat.zero_body(), res.e_hf_nat, 1e-9);
}

}  // namespace
