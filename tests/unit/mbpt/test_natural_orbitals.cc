// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// MBPT(2) density matrix and natural orbitals at emax 2 (O16, 2N): structural
// invariants.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "tests/unit/mbpt/mbpt_test_support.h"

namespace imsrg::mbtest {

namespace fs = std::filesystem;
using hftest::LoadHamiltonian;
namespace {

TEST(MbptNaturalOrbitals, Emax2Invariants) {
  op::Operator H =
      LoadHamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  NatResult res = BuildNat(H, nullptr);
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  const auto N = ms.num_orbits();
  const auto& C = res.nat.c_hf2nat;

  double trace = 0.0;
  for (std::int32_t a = 0; a < N; ++a)
    trace += (basis.orbit(OrbitIndex(a)).j2.twice + 1) * res.rho_hf(a, a);
  EXPECT_NEAR(trace, ms.reference().A(), 1.0e-6);

  double max_orth = 0.0, max_offblock = 0.0;
  for (std::int32_t a = 0; a < N; ++a) {
    const OrbitIndex oa(a);
    EXPECT_GE(C(a, a), 0.0);
    const auto ch = basis.SameLjtz(oa);
    for (std::int32_t b = 0; b < N; ++b) {
      double s = 0.0;
      for (std::int32_t k = 0; k < N; ++k) s += C(k, a) * C(k, b);
      max_orth = std::max(max_orth, std::abs(s - (a == b ? 1.0 : 0.0)));
      if (std::find(ch.begin(), ch.end(), OrbitIndex(b)) == ch.end())
        max_offblock = std::max(max_offblock, std::abs(C(a, b)));
    }
    for (std::size_t k = 1; k < ch.size(); ++k)
      EXPECT_GE(
          res.nat.occupations[static_cast<std::size_t>(ch[k - 1].value())],
          res.nat.occupations[static_cast<std::size_t>(ch[k].value())]);
  }
  EXPECT_LT(max_orth, 1.0e-12);
  EXPECT_EQ(max_offblock, 0.0);

  std::cout << "[nat emax2] E_HF(HF) " << res.e_hf_hf << " E_HF(NAT) "
            << res.e_hf_nat << " E0(NAT H) " << res.h_nat.zero_body()
            << " MP2(NAT) " << mbpt::Mp2Energy(res.h_nat) << "\n";
  EXPECT_GE(res.e_hf_nat,
            res.e_hf_hf - 1.0e-8);  // HF is the variational minimum
  EXPECT_NEAR(res.h_nat.zero_body(), res.e_hf_nat, 1.0e-9);
}

// Rows/columns of the (l,j,tz) channel of orbit i, found by comparing quantum
// numbers (independent of SingleParticleBasis::SameLjtz).
bool SameChannel(const basis::SingleParticleBasis& b, std::int64_t i,
                 std::int64_t k) {
  const auto& oi = b.orbit(OrbitIndex(static_cast<std::int32_t>(i)));
  const auto& ok = b.orbit(OrbitIndex(static_cast<std::int32_t>(k)));
  return oi.l == ok.l && oi.j2.twice == ok.j2.twice &&
         oi.tz2.twice == ok.tz2.twice;
}

// Direct re-implementation of the fork's HFMBPT::UseHFForHoleStates
// (imsrg++ 79e9330, HFMBPT.cc:892-1136) on a dense matrix, with the fork's
// loops (all orbits, channel membership tested per element), followed by
// ReorderHFMBPTCoefficients' sign rule.
linalg::Matrix<double> ForkUseHfForHoleStates(linalg::Matrix<double> C,
                                              const basis::ModelSpace& ms) {
  const auto& b = ms.basis();
  const std::int64_t N = ms.num_orbits();
  auto occ = [&](std::int64_t i) {
    return ms.reference().occupation(OrbitIndex(static_cast<std::int32_t>(i)));
  };
  for (std::int64_t i = 0; i < N; ++i) {
    if (occ(i) > 1e-8) {
      for (std::int64_t j = 0; j < N; ++j) {
        if (j == i) {
          C(j, i) = 1.0;
        } else {
          C(j, i) = 0.0;
          C(i, j) = 0.0;
        }
      }
    }
  }
  for (std::int64_t i = 0; i < N; ++i) {
    for (std::int64_t j = 0; j < N; ++j) {
      if (!SameChannel(b, i, j) || j >= i) continue;
      double ij_overlap = 0.0;
      double jj_overlap = 0.0;
      for (std::int64_t k = 0; k < N; ++k) {
        if (!SameChannel(b, i, k)) continue;
        ij_overlap += C(k, i) * C(k, j);
        jj_overlap += C(k, j) * C(k, j);
      }
      for (std::int64_t k = 0; k < N; ++k) {
        if (!SameChannel(b, i, k)) continue;
        C(k, i) -= ij_overlap / jj_overlap * C(k, j);
      }
    }
  }
  for (std::int64_t i = 0; i < N; ++i) {
    double ii_overlap = 0.0;
    for (std::int64_t j = 0; j < N; ++j)
      if (SameChannel(b, i, j)) ii_overlap += C(j, i) * C(j, i);
    for (std::int64_t j = 0; j < N; ++j)
      if (SameChannel(b, i, j)) C(j, i) *= 1 / std::sqrt(ii_overlap);
  }
  for (std::int64_t i = 0; i < N; ++i) {
    if (C(i, i) < 0) {
      for (std::int64_t k = 0; k < N; ++k) C(k, i) *= -1;
    }
  }
  return C;
}

// A random orthogonal matrix, block diagonal in (l,j,tz) (Gram-Schmidt of
// random columns), as a stand-in for C_hf2nat.
linalg::Matrix<double> RandomChannelOrthogonal(const basis::ModelSpace& ms,
                                               unsigned seed) {
  const auto& b = ms.basis();
  const std::int64_t N = ms.num_orbits();
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  linalg::Matrix<double> C(N, N);
  for (std::int64_t i = 0; i < N; ++i)
    for (std::int64_t k = 0; k < N; ++k)
      if (SameChannel(b, i, k)) C(k, i) = u(rng) + (k == i ? 2.0 : 0.0);
  for (std::int64_t i = 0; i < N; ++i) {
    for (std::int64_t j = 0; j < i; ++j) {
      if (!SameChannel(b, i, j)) continue;
      double d = 0.0;
      for (std::int64_t k = 0; k < N; ++k) d += C(k, i) * C(k, j);
      for (std::int64_t k = 0; k < N; ++k) C(k, i) -= d * C(k, j);
    }
    double n = 0.0;
    for (std::int64_t k = 0; k < N; ++k) n += C(k, i) * C(k, i);
    for (std::int64_t k = 0; k < N; ++k) C(k, i) /= std::sqrt(n);
  }
  return C;
}

// Al27 ensemble reference (fractional proton 0d5/2) at emax 4: the fork's
// algorithm on a random C agrees with UseHfForHoleStates; holes are unit
// vectors, virtuals carry no hole component, C is orthogonal and C(i,i) >= 0.
TEST(MbptNaturalOrbitals, UseHfForHoleStatesMatchesForkAlgorithm) {
  const auto ms = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 4, .lmax = -1}, 16.0, "Al27", "sd-shell");
  const std::int64_t N = ms->num_orbits();
  bool fractional = false;
  for (std::int64_t i = 0; i < N; ++i) {
    const double n =
        ms->reference().occupation(OrbitIndex(static_cast<std::int32_t>(i)));
    fractional = fractional || (n > 1.0e-8 && n < 1.0 - 1.0e-8);
  }
  ASSERT_TRUE(fractional);
  for (unsigned seed : {1U, 2U, 3U}) {
    mbpt::NaturalOrbitals nat{
        RandomChannelOrthogonal(*ms, seed),
        std::vector<double>(static_cast<std::size_t>(N), 0.0)};
    const linalg::Matrix<double> expect =
        ForkUseHfForHoleStates(nat.c_hf2nat, *ms);
    mbpt::UseHfForHoleStates(nat, *ms);
    const auto& C = nat.c_hf2nat;
    double max_diff = 0.0, max_orth = 0.0;
    for (std::int64_t i = 0; i < N; ++i) {
      const double ni =
          ms->reference().occupation(OrbitIndex(static_cast<std::int32_t>(i)));
      EXPECT_GE(C(i, i), 0.0);
      for (std::int64_t j = 0; j < N; ++j) {
        max_diff = std::max(max_diff, std::abs(C(i, j) - expect(i, j)));
        if (ni > 1.0e-8) {
          EXPECT_EQ(C(j, i), j == i ? 1.0 : 0.0);  // hole column = e_i
          if (j != i) {
            EXPECT_EQ(C(i, j), 0.0);  // hole row of others
          }
        }
        double s = 0.0;
        for (std::int64_t k = 0; k < N; ++k) s += C(k, i) * C(k, j);
        max_orth = std::max(max_orth, std::abs(s - (i == j ? 1.0 : 0.0)));
      }
    }
    EXPECT_LT(max_diff, 1.0e-14) << "seed " << seed;
    EXPECT_LT(max_orth, 1.0e-13) << "seed " << seed;
  }
}

// O16 at emax 2 (2N golden H): with the holes reset to the HF orbitals the
// mean-field state of the NAT basis is the HF state, E_ref(NAT) = E_HF (fork
// log: E_Nat = EHF).
TEST(MbptNaturalOrbitals, HfHolesReproduceHfEnergy) {
  op::Operator H =
      LoadHamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  NatResult plain = BuildNat(H, nullptr);
  NatResult res = BuildNat(H, nullptr, /*use_hf_holes=*/true);
  EXPECT_NEAR(res.e_hf_nat, res.e_hf_hf, 1.0e-10);
  EXPECT_GT(plain.e_hf_nat, plain.e_hf_hf + 1.0e-6);  // the reset matters
  EXPECT_NEAR(res.h_nat.zero_body(), res.e_hf_hf, 1.0e-10);
}

// |occupation| sorting (fork re-sorting, HFMBPT.cc:131-182) on a diagonal
// density at emax 4 (O16): in the proton s1/2 channel (0s, 1s, 2s) the
// occupations 1, 0.01, -0.03 are relabelled 1, -0.03, 0.01; the signed order
// keeps 1, 0.01, -0.03. Neutron 1s gets +0.02 to keep Tr = A.
TEST(MbptNaturalOrbitals, AbsoluteOccupationSorting) {
  const auto ms = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 4, .lmax = -1}, 16.0, "O16", "O16");
  const auto& b = ms->basis();
  const std::int64_t N = ms->num_orbits();
  linalg::Matrix<double> rho(N, N);
  std::vector<std::int64_t> ps, ns;  // proton / neutron s1/2 orbits by n
  for (std::int64_t i = 0; i < N; ++i) {
    const auto& o = b.orbit(OrbitIndex(static_cast<std::int32_t>(i)));
    rho(i, i) =
        ms->reference().occupation(OrbitIndex(static_cast<std::int32_t>(i)));
    if (o.l == 0) (o.tz2.twice < 0 ? ps : ns).push_back(i);
  }
  ASSERT_EQ(ps.size(), 3U);
  ASSERT_EQ(ns.size(), 3U);
  rho(ps[1], ps[1]) = 0.01;
  rho(ps[2], ps[2]) = -0.03;
  rho(ns[1], ns[1]) = 0.02;
  const auto signed_nat = mbpt::ComputeNaturalOrbitals(rho, *ms);
  const auto abs_nat =
      mbpt::ComputeNaturalOrbitals(rho, *ms, mbpt::NatSorting::kAbsDescending);
  auto occ = [](const mbpt::NaturalOrbitals& n, std::int64_t i) {
    return n.occupations[static_cast<std::size_t>(i)];
  };
  EXPECT_NEAR(occ(signed_nat, ps[1]), 0.01, 1.0e-15);
  EXPECT_NEAR(occ(signed_nat, ps[2]), -0.03, 1.0e-15);
  EXPECT_NEAR(occ(abs_nat, ps[0]), 1.0, 1.0e-15);
  EXPECT_NEAR(occ(abs_nat, ps[1]), -0.03, 1.0e-15);
  EXPECT_NEAR(occ(abs_nat, ps[2]), 0.01, 1.0e-15);
  // The slot of 1s now holds the 2s vector and vice versa.
  EXPECT_NEAR(std::abs(abs_nat.c_hf2nat(ps[2], ps[1])), 1.0, 1.0e-15);
  EXPECT_NEAR(std::abs(abs_nat.c_hf2nat(ps[1], ps[2])), 1.0, 1.0e-15);
  // Channels without negative occupations are unchanged.
  EXPECT_NEAR(occ(abs_nat, ns[1]), 0.02, 1.0e-15);
  for (std::int64_t i = 0; i < N; ++i) {
    EXPECT_GE(abs_nat.c_hf2nat(i, i), 0.0);
    if (b.orbit(OrbitIndex(static_cast<std::int32_t>(i))).l == 0 &&
        b.orbit(OrbitIndex(static_cast<std::int32_t>(i))).tz2.twice < 0)
      continue;
    for (std::int64_t k = 0; k < N; ++k)
      EXPECT_EQ(abs_nat.c_hf2nat(k, i), signed_nat.c_hf2nat(k, i));
  }
}

}  // namespace
}  // namespace imsrg::mbtest
