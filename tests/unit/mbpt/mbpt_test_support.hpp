#pragma once
/// Test-side assembly of the NAT-basis Hamiltonian. mbpt does not depend on hf, so the test
/// drives the hf internals (design/13, M9-Q1): the HF result is modified to describe the NAT
/// orbitals (C = C_ho2hf C_hf2nat, rho = reference density in the oscillator basis,
/// F = T + V2[rho] + 1/2 V3[rho], E_HF for rho) and passed to hf::normal_ordered_hamiltonian.
#include <memory>

#include "../hf/hf_test_support.hpp"

#include "imsrg/hf/monopole.hpp"
#include "imsrg/mbpt/energies.hpp"
#include "imsrg/mbpt/natural_orbitals.hpp"

namespace imsrg::mbtest {

using namespace imsrg::hftest;

struct NatResult {
  op::Operator h_nat;             ///< normal-ordered H in the NAT basis
  mbpt::NaturalOrbitals nat;      ///< NAT orbitals in the HF basis
  linalg::Matrix<double> rho_hf;  ///< MBPT(2) density in the HF basis
  double e_hf_hf = 0.0;           ///< E_HF in the HF basis
  double e_hf_nat = 0.0;          ///< E_HF in the NAT basis
};

/// Runs HF, the MBPT(2) density, natural orbitals and builds the NAT-basis Hamiltonian.
inline NatResult build_nat(op::Operator H, const op::No2bThreeBody* v3) {
  H.sync_to_host();
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  const std::int32_t norb = basis.num_orbits();
  hf::HFResult r = hf::solve(H, v3);
  op::Operator hno = hf::normal_ordered_hamiltonian(H, v3, r);
  linalg::Matrix<double> rho_hf = mbpt::mbpt2_density_matrix(hno);
  mbpt::NaturalOrbitals nat = mbpt::natural_orbitals(rho_hf, ms);
  const linalg::Matrix<double> c_ho2nat = mbpt::matmul(r.C, nat.c_hf2nat);
  const linalg::Matrix<double> rho_ho = mbpt::reference_density_ho(c_ho2nat, ms);

  hf::detail::Monopole2B mono2(H);
  linalg::Matrix<double> v2ij, v3ij;
  mono2.apply(rho_ho, v2ij);
  std::unique_ptr<hf::detail::Monopole3B> mono3;
  if (v3 != nullptr) {
    mono3 = std::make_unique<hf::detail::Monopole3B>(H, *v3, 12);
    mono3->apply(rho_ho, v3ij);
  }
  hf::HFResult m = r;
  m.C = c_ho2nat;
  m.rho = rho_ho;
  m.F = linalg::Matrix<double>(norb, norb);
  double e1 = 0, e2 = 0, e3 = 0;
  for (std::int32_t a = 0; a < norb; ++a) {
    const OrbitIndex oa(a);
    const double w = basis.orbit(oa).j2.twice + 1.0;
    for (const OrbitIndex b : basis.same_ljtz(oa)) {
      const double ke = H.one_body().get(oa, b);
      double f = ke + v2ij(a, b.value());
      const double rr = w * rho_ho(a, b.value());
      e1 += rr * ke;
      e2 += 0.5 * rr * v2ij(a, b.value());
      if (v3 != nullptr) {
        f += 0.5 * v3ij(a, b.value());
        e3 += rr * v3ij(a, b.value()) / 6.0;
      }
      m.F(a, b.value()) = f;
    }
  }
  m.e1 = e1;
  m.e2 = e2;
  m.e3 = e3;
  m.e_hf = e1 + e2 + e3;
  for (std::int32_t a = 0; a < norb; ++a)
    m.occupations[static_cast<std::size_t>(a)] = ms.reference().occupation(OrbitIndex(a));
  op::Operator hnat = hf::normal_ordered_hamiltonian(H, v3, m);
  return NatResult{std::move(hnat), std::move(nat), std::move(rho_hf), r.e_hf, m.e_hf};
}

}  // namespace imsrg::mbtest
