// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 terms, part 4: the full commutator with the N7 term set
// (`Commutator` with TermMask::Imsrg3N7, dispatch in commutator.cc) against
// imsrg++'s full `Commutator::Commutator` with IMSRG3 and imsrg3_n7
// (tests/golden/n7/<case>/n7.dump), Z reuse and the threebody_threshold skip;
// all seven terms against the m-scheme brute force for same-hermiticity and
// two non-Hermitian operands (anti-Hermitian and non-Hermitian Z); and an
// emax-2 fractional case, where every one-body channel holds several orbits
// (at emax 1 comm132ss vanishes and the 231/331 outputs are diagonal).
// Adapted from the probes of the Opus kernel review.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/reference.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/workspace.h"
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
using testing::GoldenN7Inputs;
using testing::N7Inputs;
using testing::RandomN7Inputs;
using testing::RelativeDeviation;

constexpr double kTol = 1.0e-10;  // design/30 section 2

constexpr Term kN7Terms[] = {
    Term::kComm330ss, Term::kComm331ss, Term::kComm231ss, Term::kComm132ss,
    Term::kComm232ss, Term::kComm133ss, Term::kComm223ss};

op::Operator FullN7(const op::Operator& X, const op::Operator& Y, Workspace& ws,
                    double threshold) {
  CommutatorOptions o;
  o.terms = TermMask::Imsrg3N7();
  o.threebody_threshold = threshold;
  op::Operator z;
  Commutator(X, Y, z, o, ws);
  return z;
}

TEST(N7Full, MatchesImsrgDevelFullCommutator) {
  for (const char* ref : {"He4", "O16"}) {
    const N7Inputs in = GoldenN7Inputs(2, ref, 3);
    Workspace ws(in.ms3);
    const op::Operator z = FullN7(in.x, in.y, ws, 0.0);
    ASSERT_EQ(z.spec().particle_rank, 3);
    const op::Operator golden =
        testing::ReadN7Golden(std::string(ref) + "_emax2", "n7.dump", in.ms3);
    const double dev = RelativeDeviation(z, golden);
    std::printf("full N7 %s, E3max 3: relative deviation from imsrg++ %.3g\n",
                ref, dev);
    EXPECT_LE(dev, kTol) << ref;
    // Reusing Z (the flow-loop case) gives the same result.
    op::Operator z2 = z;
    CommutatorOptions o;
    o.terms = TermMask::Imsrg3N7();
    Commutator(in.x, in.y, z2, o, ws);
    EXPECT_LE(RelativeDeviation(z2, golden), kTol) << ref << " (reused Z)";
    // Above the threshold the three-body terms are skipped (imsrg++
    // Commutator.cc:396); Z keeps rank 3 with the IMSRG(2) terms only.
    const op::Operator z3 = FullN7(in.x, in.y, ws, 1.0e9);
    EXPECT_EQ(z3.spec().particle_rank, 3);
    EXPECT_EQ(op::ThreeBodyNorm(z3), 0.0);
    EXPECT_GT(op::TwoBodyNorm(z3), 0.0);
  }
}

TEST(N7Full, MatchesImsrgDevelNormsAtFullE3max) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    const N7Inputs in = GoldenN7Inputs(2, ref, 6);
    Workspace ws(in.ms3);
    const op::Operator z = FullN7(in.x, in.y, ws, 0.0);
    const auto rel = [](double a, double b) {
      return std::abs(a - b) / std::max(1.0, std::abs(b));
    };
    EXPECT_LE(
        rel(z.zero_body(), testing::N7GoldenValue(dir, 6, "n7", "zero_body")),
        kTol);
    EXPECT_LE(rel(op::OneBodyNorm(z),
                  testing::N7GoldenValue(dir, 6, "n7", "one_body_norm")),
              kTol);
    EXPECT_LE(rel(op::TwoBodyNorm(z),
                  testing::N7GoldenValue(dir, 6, "n7", "two_body_norm")),
              kTol);
    EXPECT_LE(rel(op::ThreeBodyNorm(z),
                  testing::N7GoldenValue(dir, 6, "n7", "three_body_norm")),
              kTol)
        << dir;
  }
}

std::vector<N7Inputs> SameHermiticityCases() {
  std::vector<N7Inputs> cases;
  const basis::ThreeBodyCuts full;
  cases.push_back(RandomN7Inputs(testing::MakeThreeBodySpace(1, "He4", full),
                                 Hermiticity::kAntiHermitian,
                                 Hermiticity::kAntiHermitian, 51));
  cases.push_back(RandomN7Inputs(
      testing::AddThreeBodySpace(testing::FractionalModelSpace(1, "He4"), full),
      Hermiticity::kHermitian, Hermiticity::kHermitian, 53));
  cases.push_back(RandomN7Inputs(testing::MakeThreeBodySpace(1, "C12", full),
                                 Hermiticity::kNonHermitian,
                                 Hermiticity::kNonHermitian, 55));
  return cases;
}

TEST(N7Full, SameHermiticityAndNonHermitianMatchBruteForce) {
  for (N7Inputs& c : SameHermiticityCases()) {
    for (const Term t : kN7Terms) {
      const op::Operator z = EvaluateN7Term(t, c.x, c.y, *c.ctx);
      const op::Operator ref = BruteForceN7Term(t, c.x, c.y, c.ms3);
      EXPECT_LE(RelativeDeviation(z, ref), kTol)
          << TermName(t) << " on " << c.ms3->reference().name();
    }
  }
}

// emax 2 with fractional occupations in the 0s, 1s and 0p3/2 orbits and
// E3max 4: the one-body channels hold several orbits, so comm132ss and the
// off-diagonal one-body outputs of comm231ss/comm331ss are tested.
std::shared_ptr<const basis::ModelSpace> Emax2Fractional() {
  const auto ms = testing::MakeModelSpace(2, "He4");
  std::vector<double> occ(ms->reference().occupations().begin(),
                          ms->reference().occupations().end());
  for (int o = 0; o < ms->num_orbits(); ++o) {
    const auto& orb = ms->basis().orbit(OrbitIndex{o});
    const bool proton = orb.tz2.twice < 0;
    const auto k = static_cast<std::size_t>(o);
    if (orb.n == 0 && orb.l == 0) occ[k] = proton ? 0.6 : 0.7;
    if (orb.n == 1 && orb.l == 0) occ[k] = proton ? 0.1 : 0.05;
    if (orb.n == 0 && orb.l == 1 && orb.j2.twice == 3) {
      occ[k] = proton ? 0.25 : 0.15;
    }
  }
  basis::ThreeBodyCuts cuts;
  cuts.e3max = 4;
  return ms
      ->WithReference(basis::Reference::FromOccupations(
          ms->basis(), std::move(occ), "frac2"))
      ->WithThreeBody(cuts);
}

TEST(N7Full, Emax2OneBodyCouplingsMatchBruteForce) {
  N7Inputs c = RandomN7Inputs(Emax2Fractional(), Hermiticity::kNonHermitian,
                              Hermiticity::kHermitian, 61);
  for (const Term t : {Term::kComm132ss, Term::kComm231ss, Term::kComm133ss,
                       Term::kComm331ss}) {
    const op::Operator z = EvaluateN7Term(t, c.x, c.y, *c.ctx);
    const op::Operator ref = BruteForceN7Term(t, c.x, c.y, c.ms3);
    const double dev = RelativeDeviation(z, ref);
    std::printf("emax 2 fractional %s: relative deviation %.3g\n",
                std::string(TermName(t)).c_str(), dev);
    EXPECT_GT(
        op::OneBodyNorm(ref) + op::TwoBodyNorm(ref) + op::ThreeBodyNorm(ref),
        0.0)
        << TermName(t) << ": the reference must not vanish";
    EXPECT_LE(dev, kTol) << TermName(t);
  }
}

}  // namespace
}  // namespace imsrg::commutator
