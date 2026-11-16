// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 terms on a three-body space with an OccNat3 cut (imsrg3.
// occ_nat3_cut, D34/M10-Q62): a ket is kept iff prod_i n_i (1 - n_i) >= cut
// with natural occupations n_i that differ from the reference occupations.
// Every N7 term and the fused `Commutator` with TermMask::Imsrg3N7 against the
// m-scheme brute force on the kept space (He4 and O16 at emax 1, He4 at emax
// 2 with E3max 4). The criterion is max|dZ| <= 1e-12 max(1, max|Z|), part by
// part (`RelativeDeviation`, never element-wise relative).

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/three_body_test_support.h"
#include "tests/unit/commutator/n7_test_support.h"

namespace imsrg::commutator {
namespace {

using testing::BruteForceN7Term;
using testing::EvaluateN7Term;
using testing::N7Inputs;
using testing::RandomN7Inputs;
using testing::RelativeDeviation;

constexpr double kTol = 1.0e-12;

constexpr Term kN7Terms[] = {
    Term::kComm330ss, Term::kComm331ss, Term::kComm231ss, Term::kComm132ss,
    Term::kComm232ss, Term::kComm133ss, Term::kComm223ss};

/// Synthetic fractional natural occupations, one per orbit: a deterministic
/// pseudo-random value in [0.05, 0.45], so n (1 - n) lies in [0.0475, 0.2475]
/// and the products of three orbits spread over about two decades.
std::vector<double> SyntheticNaturalOccupations(int num_orbits) {
  std::vector<double> occ(static_cast<std::size_t>(num_orbits));
  for (int i = 0; i < num_orbits; ++i) {
    occ[static_cast<std::size_t>(i)] = 0.05 + 0.4 * ((7 * i + 3) % 11) / 10.0;
  }
  return occ;
}

/// A model space of `reference` at `emax` with the three-body space of
/// E3max `e3max` and the OccNat3 cut `cut` on the synthetic occupations.
std::shared_ptr<const basis::ModelSpace> CutSpace(int emax, const char* ref,
                                                  int e3max, double cut) {
  const auto ms = testing::MakeModelSpace(emax, ref);
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  cuts.occnat3_cut = cut;
  cuts.natural_occupations = SyntheticNaturalOccupations(ms->num_orbits());
  return ms->WithThreeBody(cuts);
}

/// Rows of the three-body space without any cut.
std::int64_t UncutRows(int emax, const char* ref, int e3max) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  return testing::MakeThreeBodySpace(emax, ref, cuts)
      ->three_body_space()
      ->NumRows();
}

/// The zero-, one- and two-body parts of `O` as a rank-2 operator.
op::Operator TwoBodyPart(const op::Operator& O) {
  O.SyncToHost();
  op::OperatorSpec spec = O.spec();
  spec.particle_rank = 2;
  op::Operator out(O.model_space_ptr(), spec);
  out.zero_body() = O.zero_body();
  out.one_body() = O.one_body();
  out.two_body() = O.two_body();
  out.SyncToDevice();
  return out;
}

/// All seven terms and the fused commutator against the brute force.
void CheckAgainstBruteForce(const char* label, N7Inputs& c) {
  op::Operator fused_ref;
  bool first = true;
  for (const Term t : kN7Terms) {
    const op::Operator z = EvaluateN7Term(t, c.x, c.y, *c.ctx);
    const op::Operator ref = BruteForceN7Term(t, c.x, c.y, c.ms3);
    const double dev = RelativeDeviation(z, ref);
    std::printf("[occnat] %s %s: deviation from the brute force %.3g\n", label,
                std::string(TermName(t)).c_str(), dev);
    EXPECT_LE(dev, kTol) << label << " " << TermName(t);
    // Sum of the brute-force terms = the brute force of the fused N7 set.
    if (first) {
      fused_ref = ref;
      first = false;
    } else {
      fused_ref += ref;
    }
  }
  Workspace ws(c.ms3);
  CommutatorOptions o;
  o.terms = TermMask::Imsrg3N7();
  op::Operator z;
  Commutator(c.x, c.y, z, o, ws);
  // The preset also holds the IMSRG(2) terms, which see only the zero-, one-
  // and two-body parts of X and Y: add them from rank-2 copies.
  Workspace ws2(c.ms3);
  op::Operator z2;
  Commutator(TwoBodyPart(c.x), TwoBodyPart(c.y), z2, CommutatorOptions{}, ws2);
  fused_ref += z2;
  const double dev = RelativeDeviation(z, fused_ref);
  std::printf("[occnat] %s fused N7: deviation from the brute force %.3g\n",
              label, dev);
  EXPECT_LE(dev, kTol) << label << " fused";
  EXPECT_GT(op::ThreeBodyNorm(fused_ref) + op::TwoBodyNorm(fused_ref), 0.0);
}

/// Emax 1: the cut drops some but not all kets (asserted); X non-Hermitian, Y
/// Hermitian random three-body operators.
TEST(N7OccNat, Emax1MatchesBruteForce) {
  for (const char* ref : {"He4", "O16"}) {
    const std::int64_t all = UncutRows(1, ref, 12);
    const auto ms3 = CutSpace(1, ref, 12, 2.0e-3);
    const std::int64_t kept = ms3->three_body_space()->NumRows();
    std::printf("[occnat] %s emax 1: %" PRId64 " of %" PRId64 " rows kept\n",
                ref, kept, all);
    ASSERT_GT(kept, 0);
    ASSERT_LT(kept, all);
    N7Inputs c = RandomN7Inputs(ms3, Hermiticity::kNonHermitian,
                                Hermiticity::kHermitian, 71);
    CheckAgainstBruteForce(ref, c);
  }
}

/// Anti-Hermitian X with Hermitian Y (the flow's [eta, H] / [Omega, H]).
TEST(N7OccNat, Emax1AntiHermitianMatchesBruteForce) {
  const auto ms3 = CutSpace(1, "He4", 12, 2.0e-3);
  N7Inputs c = RandomN7Inputs(ms3, Hermiticity::kAntiHermitian,
                              Hermiticity::kHermitian, 73);
  CheckAgainstBruteForce("He4 anti-Hermitian", c);
}

/// He4 at emax 2 (E3max 4): one-body channels with several orbits.
TEST(N7OccNat, Emax2MatchesBruteForce) {
  const std::int64_t all = UncutRows(2, "He4", 4);
  const auto ms3 = CutSpace(2, "He4", 4, 1.0e-3);
  const std::int64_t kept = ms3->three_body_space()->NumRows();
  std::printf("[occnat] He4 emax 2 E3max 4: %" PRId64 " of %" PRId64
              " rows kept\n",
              kept, all);
  ASSERT_GT(kept, 0);
  ASSERT_LT(kept, all);
  N7Inputs c = RandomN7Inputs(ms3, Hermiticity::kNonHermitian,
                              Hermiticity::kHermitian, 75);
  CheckAgainstBruteForce("He4 emax2", c);
}

}  // namespace
}  // namespace imsrg::commutator
