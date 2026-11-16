// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 terms comm231ss and comm331ss (imsrg3.h) against the m-scheme
// brute force (emax 1, closed-shell, open-shell and fractional references,
// general hermiticity), against imsrg-devel (emax 2) and for antisymmetry
// under X <-> Y.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
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
using testing::N7GoldenValue;
using testing::N7Inputs;
using testing::RandomN7Inputs;
using testing::RelativeDeviation;

constexpr double kTol = 1.0e-10;

double Rel(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::abs(b));
}

// The emax-1 brute-force cases (as in test_n7.cc).
std::vector<N7Inputs> BruteForceCases() {
  std::vector<N7Inputs> cases;
  const basis::ThreeBodyCuts full;
  cases.push_back(RandomN7Inputs(testing::MakeThreeBodySpace(1, "He4", full),
                                 Hermiticity::kAntiHermitian,
                                 Hermiticity::kHermitian, 21));
  cases.push_back(RandomN7Inputs(testing::MakeThreeBodySpace(1, "C12", full),
                                 Hermiticity::kAntiHermitian,
                                 Hermiticity::kHermitian, 23));
  cases.push_back(RandomN7Inputs(
      testing::AddThreeBodySpace(testing::FractionalModelSpace(1, "He4"), full),
      Hermiticity::kAntiHermitian, Hermiticity::kHermitian, 25));
  cases.push_back(RandomN7Inputs(
      testing::AddThreeBodySpace(testing::FractionalModelSpace(1, "He4"), full),
      Hermiticity::kNonHermitian, Hermiticity::kHermitian, 27));
  return cases;
}

void ExpectMatchesBruteForce(Term t) {
  for (N7Inputs& c : BruteForceCases()) {
    const op::Operator z = EvaluateN7Term(t, c.x, c.y, *c.ctx);
    const op::Operator ref = BruteForceN7Term(t, c.x, c.y, c.ms3);
    const double dev = RelativeDeviation(z, ref);
    EXPECT_LE(dev, kTol) << TermName(t) << " on " << c.ms3->reference().name();
    std::printf("%s %s: relative deviation from the brute force %.3g\n",
                std::string(TermName(t)).c_str(),
                std::string(c.ms3->reference().name()).c_str(), dev);
  }
}

void ExpectMatchesImsrgDevel(Term t, const char* name) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    const N7Inputs in3 = GoldenN7Inputs(2, ref, 3);
    const op::Operator z3 = EvaluateN7Term(t, in3.x, in3.y, *in3.ctx);
    const op::Operator golden =
        testing::ReadN7Golden(dir, std::string(name) + ".dump", in3.ms3);
    const double dev = RelativeDeviation(z3, golden);
    EXPECT_LE(dev, kTol) << name << " " << dir << " e3max 3";
    std::printf("%s %s e3max 3: deviation from imsrg++ %.3g\n", name,
                dir.c_str(), dev);
    const N7Inputs in6 = GoldenN7Inputs(2, ref, 6);
    const auto t0 = std::chrono::steady_clock::now();
    const op::Operator z6 = EvaluateN7Term(t, in6.x, in6.y, *in6.ctx);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double expect = N7GoldenValue(dir, 6, name, "one_body_norm");
    const double got = op::OneBodyNorm(z6);
    EXPECT_LE(Rel(got, expect), kTol)
        << name << " " << dir << " e3max 6: " << got << " vs " << expect;
    std::printf("%s %s e3max 6: one-body norm deviation %.3g, %.1f ms\n", name,
                dir.c_str(), Rel(got, expect), ms);
  }
}

TEST(N7Comm231, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm231ss);
}

TEST(N7Comm231, MatchesImsrgDevel) {
  ExpectMatchesImsrgDevel(Term::kComm231ss, "comm231ss");
}

TEST(N7Comm331, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm331ss);
}

TEST(N7Comm331, MatchesImsrgDevel) {
  ExpectMatchesImsrgDevel(Term::kComm331ss, "comm331ss");
}

// [Y, X] = -[X, Y] (the one-body outputs; Z's hermiticity is the same).
TEST(N7Comm231331, AntisymmetricInXY) {
  const N7Inputs in = GoldenN7Inputs(2, "He4", 3);
  for (const Term t : {Term::kComm231ss, Term::kComm331ss}) {
    const op::Operator xy = EvaluateN7Term(t, in.x, in.y, *in.ctx);
    const op::Operator yx = EvaluateN7Term(t, in.y, in.x, *in.ctx);
    const std::vector<double> a = xy.one_body().buffer().ToVector();
    const std::vector<double> b = yx.one_body().buffer().ToVector();
    ASSERT_EQ(a.size(), b.size());
    double scale = 1.0;
    double worst = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      scale = std::max(scale, std::abs(a[k]));
      worst = std::max(worst, std::abs(a[k] + b[k]));
    }
    EXPECT_GT(op::OneBodyNorm(xy), 0.0) << TermName(t);
    EXPECT_LE(worst / scale, kTol) << TermName(t);
  }
}

}  // namespace
}  // namespace imsrg::commutator
