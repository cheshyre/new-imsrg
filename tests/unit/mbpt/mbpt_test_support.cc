// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/mbpt/mbpt_test_support.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/hf/hartree_fock.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/mbpt/natural_orbitals.h"

namespace imsrg::mbtest {

NatResult BuildNat(op::Operator H, const op::No2bThreeBody* v3) {
  H.SyncToHost();
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  hf::HFResult r = hf::Solve(H, v3);
  op::Operator hno = hf::NormalOrderedHamiltonian(H, v3, r);
  linalg::Matrix<double> rho_hf = mbpt::Mbpt2DensityMatrix(hno);
  mbpt::NaturalOrbitals nat = mbpt::ComputeNaturalOrbitals(rho_hf, ms);
  const linalg::Matrix<double> c_ho2nat = mbpt::Matmul(r.C, nat.c_hf2nat);
  std::vector<double> occ(static_cast<std::size_t>(basis.num_orbits()));
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a)
    occ[static_cast<std::size_t>(a)] = ms.reference().occupation(OrbitIndex(a));
  const hf::HFResult m = hf::SolutionInBasis(H, v3, c_ho2nat, occ);
  op::Operator hnat = hf::NormalOrderedHamiltonian(H, v3, m);
  return NatResult{std::move(hnat), std::move(nat), std::move(rho_hf), r.e_hf,
                   m.e_hf};
}

}  // namespace imsrg::mbtest
