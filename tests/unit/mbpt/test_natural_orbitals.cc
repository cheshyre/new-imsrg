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
#include "imsrg/operator/random_operator.h"
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
  const auto signed_nat = mbpt::ComputeNaturalOrbitals(
      rho, *ms, mbpt::NatSorting::kSignedDescending);
  // The default is the |occupation| order (D72, AL-NAT).
  const auto abs_nat = mbpt::ComputeNaturalOrbitals(rho, *ms);
  const auto abs_explicit =
      mbpt::ComputeNaturalOrbitals(rho, *ms, mbpt::NatSorting::kAbsDescending);
  EXPECT_EQ(abs_nat.occupations, abs_explicit.occupations);
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

// Al27 ensemble reference at emax 4, random Hermitian H with ordered
// single-particle energies: the (proton 1d5/2, proton 0d5/2) element of the
// MBPT(2) density equals imsrg++'s DensityMatrixPP + DensityMatrixPH
// (HFMBPT.cc), transcribed here loop by loop. The PP term couples the
// particle 1d5/2 to the partially filled 0d5/2 (n = 5/6): imsrg++ tests
// 1 - n_b, not the particle list (lead D, 2026-11-26: missing this term
// moved EMP2 of the 27Al production run by 0.026 MeV).
TEST(MbptNaturalOrbitals, PartiallyFilledOrbitEntersPpBlock) {
  const auto ms = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 4, .lmax = -1}, 16.0, "Al27", "sd-shell");
  const auto& basis = ms->basis();
  const auto& ref = ms->reference();
  const std::int32_t N = ms->num_orbits();
  op::Operator H = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian),
      {.seed = 7, .one_body_scale = 0.0, .two_body_scale = 0.5});
  H.SyncToHost();
  for (std::int32_t a = 0; a < N; ++a) {
    const OrbitIndex oa(a);
    const auto& o = basis.orbit(oa);
    H.one_body().set(oa, oa, 3.0 * o.e() - 9.0 + 0.1 * a);
  }
  H.SyncToDevice();
  std::int32_t a = -1, b = -1;  // proton 1d5/2, proton 0d5/2
  for (std::int32_t k = 0; k < N; ++k) {
    const auto& o = basis.orbit(OrbitIndex(k));
    if (o.l == 2 && o.j2.twice == 5 && o.tz2.twice < 0) (o.n == 0 ? b : a) = k;
  }
  ASSERT_GE(a, 0);
  ASSERT_GE(b, 0);
  ASSERT_NEAR(ref.occupation(OrbitIndex(b)), 5.0 / 6.0, 1.0e-12);

  op::Operator h = H;
  const linalg::Matrix<double> rho = mbpt::Mbpt2DensityMatrix(h);

  const auto n = [&](std::int32_t k) { return ref.occupation(OrbitIndex(k)); };
  const auto e = [&](std::int32_t k) {
    return H.one_body().get(OrbitIndex(k), OrbitIndex(k));
  };
  const auto j2 = [&](std::int32_t k) {
    return basis.orbit(OrbitIndex(k)).j2.twice;
  };
  const auto V = [&](std::int32_t p, std::int32_t q, std::int32_t r,
                     std::int32_t s, int J) {
    const auto &P = basis.orbit(OrbitIndex(p)), &Q = basis.orbit(OrbitIndex(q)),
               &R = basis.orbit(OrbitIndex(r)), &S = basis.orbit(OrbitIndex(s));
    if ((P.l + Q.l + R.l + S.l) % 2 != 0) return 0.0;
    if (P.tz2.twice + Q.tz2.twice != R.tz2.twice + S.tz2.twice) return 0.0;
    return H.two_body().get_unnormalized(OrbitIndex(p), OrbitIndex(q),
                                         OrbitIndex(r), OrbitIndex(s),
                                         TwoJ{2 * J});
  };
  const double cut = 1.0e-6;
  // DensityMatrixPP, element (a, b), b < a.
  double pp = 0.0;
  for (std::int32_t c = 0; c < N; ++c) {
    if (n(c) >= cut || 1 - n(c) < cut) continue;  // particles
    for (std::int32_t i = 0; i < N; ++i) {
      if (n(i) < cut) continue;  // holes
      for (std::int32_t j = 0; j < N; ++j) {
        if (n(j) < cut) continue;
        const double eac = e(a) + e(c) - e(i) - e(j);
        const double ebc = e(b) + e(c) - e(i) - e(j);
        if (std::abs(eac * ebc) < 1.0e-8) continue;
        const int jmin =
            std::max({std::abs(j2(a) - j2(c)), std::abs(j2(i) - j2(j)),
                      std::abs(j2(b) - j2(c))}) /
            2;
        const int jmax =
            std::min({j2(a) + j2(c), j2(i) + j2(j), j2(b) + j2(c)}) / 2;
        double t = 0.0;
        for (int J = jmin; J <= jmax; ++J)
          t += (2 * J + 1) * V(a, c, i, j, J) * V(i, j, b, c, J);
        t *= (1 - n(a)) * (1 - n(b)) * (1 - n(c)) * (1 - n(c)) * n(i) * n(i) *
             n(j) * n(j);
        const double eps = 0.5 * std::sqrt(std::abs(eac * ebc));
        pp += 0.5 * (std::sqrt(t + eps * eps) - eps) / std::sqrt(t + eps * eps);
      }
    }
  }
  pp *= 0.5 / (j2(a) + 1);
  // DensityMatrixPH, element (a, i = b).
  const std::int32_t i = b;
  double r1 = 0.0, r2 = 0.0;
  const double eai = e(a) - e(i);
  for (std::int32_t p = 0; p < N; ++p) {
    if (n(p) >= cut) continue;  // particles
    for (std::int32_t q = 0; q < N; ++q) {
      if (n(q) >= cut) continue;
      for (std::int32_t j = 0; j < N; ++j) {
        if (n(j) < cut) continue;
        const double d = e(p) + e(q) - e(i) - e(j);
        if (eai * d < 1.0e-8) continue;
        const int jmin =
            std::max({std::abs(j2(a) - j2(j)), std::abs(j2(p) - j2(q)),
                      std::abs(j2(i) - j2(j))}) /
            2;
        const int jmax =
            std::min({j2(a) + j2(j), j2(p) + j2(q), j2(i) + j2(j)}) / 2;
        double t = 0.0;
        for (int J = jmin; J <= jmax; ++J)
          t += (2 * J + 1) * V(a, j, p, q, J) * V(p, q, i, j, J);
        r1 +=
            t * (1 - n(a)) * (1 - n(p)) * (1 - n(q)) * n(i) * n(j) / (eai * d);
      }
    }
    for (std::int32_t j = 0; j < N; ++j) {
      if (n(j) < cut) continue;
      for (std::int32_t k = 0; k < N; ++k) {
        if (n(k) < cut) continue;
        const double d = e(a) + e(p) - e(k) - e(j);
        if (eai * d < 1.0e-8) continue;
        const int jmin =
            std::max({std::abs(j2(k) - j2(j)), std::abs(j2(i) - j2(p)),
                      std::abs(j2(a) - j2(p))}) /
            2;
        const int jmax =
            std::min({j2(k) + j2(j), j2(i) + j2(p), j2(a) + j2(p)}) / 2;
        double t = 0.0;
        for (int J = jmin; J <= jmax; ++J)
          t += (2 * J + 1) * V(k, j, i, p, J) * V(a, p, k, j, J);
        r2 += t * (1 - n(a)) * n(i) * n(j) * n(k) * (1 - n(p)) / (eai * d);
      }
    }
  }
  const double ph = (r1 - r2) * 0.5 / (j2(a) + 1);
  ASSERT_GT(std::abs(pp), 1.0e-8);  // the term under test is not trivial
  EXPECT_NEAR(rho(a, b), pp + ph, 1.0e-12 * (std::abs(pp) + std::abs(ph)));
  EXPECT_EQ(rho(a, b), rho(b, a));
}

}  // namespace
}  // namespace imsrg::mbtest
