// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_MBPT_MBPT_TEST_SUPPORT_H_
#define TESTS_UNIT_MBPT_MBPT_TEST_SUPPORT_H_
/// Test-side assembly of the NAT-basis Hamiltonian from the public APIs: HF,
/// the MBPT(2) density and natural orbitals (mbpt), `hf::SolutionInBasis` for
/// C = C_ho2hf C_hf2nat and `hf::NormalOrderedHamiltonian` (mbpt does not
/// depend on hf; design/13, M9-Q1).
#include <memory>
#include <utility>
#include <vector>

#include "imsrg/hf/hartree_fock.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/mbpt/natural_orbitals.h"
#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::mbtest {

struct NatResult {
  op::Operator h_nat;             ///< normal-ordered H in the NAT basis
  mbpt::NaturalOrbitals nat;      ///< NAT orbitals in the HF basis
  linalg::Matrix<double> rho_hf;  ///< MBPT(2) density in the HF basis
  double e_hf_hf = 0.0;           ///< E_HF in the HF basis
  double e_hf_nat = 0.0;          ///< E_HF in the NAT basis
};

/// Runs HF, the MBPT(2) density, natural orbitals (ordered by `sorting`;
/// holes reset to the HF orbitals if `use_hf_holes`, fork
/// `use_HF_reference_in_NAT`) and builds the NAT-basis Hamiltonian.
NatResult BuildNat(
    op::Operator H, const op::No2bThreeBody* v3, bool use_hf_holes = false,
    mbpt::NatSorting sorting = mbpt::NatSorting::kSignedDescending);

}  // namespace imsrg::mbtest

#endif  // TESTS_UNIT_MBPT_MBPT_TEST_SUPPORT_H_
