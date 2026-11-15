// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 term comm232ss (imsrg3.h) against the m-scheme brute force
// (emax 1, closed-shell, open-shell and fractional references, general
// hermiticity), against imsrg-devel (emax 2) and for antisymmetry under
// X <-> Y.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
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

constexpr double kTol = 1.0e-10;  // design/30 section 2

double Rel(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::abs(b));
}

double Milliseconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// The emax-1 brute-force cases of test_n7.cc (closed shell, open shell, C12
// fills 0p3/2, fractional occupations; X anti-Hermitian or non-Hermitian, Y
// Hermitian) plus both operands non-Hermitian.
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
  cases.push_back(RandomN7Inputs(testing::MakeThreeBodySpace(1, "C12", full),
                                 Hermiticity::kNonHermitian,
                                 Hermiticity::kNonHermitian, 29));
  return cases;
}

TEST(N7Comm232, MatchesBruteForce) {
  for (N7Inputs& c : BruteForceCases()) {
    const op::Operator z = EvaluateN7Term(Term::kComm232ss, c.x, c.y, *c.ctx);
    const op::Operator ref =
        BruteForceN7Term(Term::kComm232ss, c.x, c.y, c.ms3);
    const double dev = RelativeDeviation(z, ref);
    EXPECT_GT(op::TwoBodyNorm(ref), 1.0);
    EXPECT_LE(dev, kTol) << c.ms3->reference().name();
    std::printf("comm232ss %s: relative deviation from the brute force %.3g\n",
                std::string(c.ms3->reference().name()).c_str(), dev);
  }
}

// Element-wise against the imsrg++ dumps (E3max 3) and the two-body norms of
// values.json (E3max 3 and 6, where the hhh-ppp couplings exist).
TEST(N7Comm232, MatchesImsrgDevel) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    for (const int e3max : {3, 6}) {
      const N7Inputs in = GoldenN7Inputs(2, ref, e3max);
      const auto t0 = std::chrono::steady_clock::now();
      const op::Operator z =
          EvaluateN7Term(Term::kComm232ss, in.x, in.y, *in.ctx);
      const double ms = Milliseconds(t0);
      if (e3max == 3) {
        const op::Operator golden =
            testing::ReadN7Golden(dir, "comm232ss.dump", in.ms3);
        const double dev = RelativeDeviation(z, golden);
        EXPECT_LE(dev, kTol) << dir;
        std::printf("comm232ss %s e3max 3: deviation from imsrg++ %.3g\n",
                    dir.c_str(), dev);
      }
      const double expect =
          N7GoldenValue(dir, e3max, "comm232ss", "two_body_norm");
      const double got = op::TwoBodyNorm(z);
      EXPECT_LE(Rel(got, expect), kTol)
          << dir << " e3max " << e3max << ": " << got << " vs " << expect;
      std::printf(
          "comm232ss %s e3max %d: two-body norm deviation %.3g, %.1f ms\n",
          dir.c_str(), e3max, Rel(got, expect), ms);
    }
  }
}

// [Y, X] = -[X, Y] (the two-body outputs; Z's hermiticity is the same).
TEST(N7Comm232, AntisymmetricInXY) {
  const N7Inputs in = GoldenN7Inputs(2, "O16", 6);
  const op::Operator xy = EvaluateN7Term(Term::kComm232ss, in.x, in.y, *in.ctx);
  const op::Operator yx = EvaluateN7Term(Term::kComm232ss, in.y, in.x, *in.ctx);
  const std::vector<double> a = xy.two_body().matrix().buffer().ToVector();
  const std::vector<double> b = yx.two_body().matrix().buffer().ToVector();
  ASSERT_EQ(a.size(), b.size());
  double scale = 1.0;
  double worst = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    scale = std::max(scale, std::abs(a[k]));
    worst = std::max(worst, std::abs(a[k] + b[k]));
  }
  EXPECT_GT(op::TwoBodyNorm(xy), 1.0);
  EXPECT_LE(worst / scale, 1.0e-13);
}

// The term adds into Z and leaves the other parts alone.
TEST(N7Comm232, AddsIntoZ) {
  const N7Inputs in = GoldenN7Inputs(2, "He4", 3);
  op::Operator once = EvaluateN7Term(Term::kComm232ss, in.x, in.y, *in.ctx);
  terms::Comm232ss(in.x, in.y, once, *in.ctx);
  const op::Operator single =
      EvaluateN7Term(Term::kComm232ss, in.x, in.y, *in.ctx);
  EXPECT_NEAR(op::TwoBodyNorm(once), 2.0 * op::TwoBodyNorm(single),
              1.0e-12 * op::TwoBodyNorm(single));
  EXPECT_EQ(op::OneBodyNorm(once), 0.0);
  EXPECT_EQ(op::ThreeBodyNorm(once), 0.0);
  EXPECT_EQ(once.zero_body(), 0.0);
}

TEST(N7Comm232, RequiresATwoBodyOutput) {
  const N7Inputs in = GoldenN7Inputs(2, "He4", 3);
  op::Operator z(in.ms3, op::OperatorSpec::Scalar(1, Hermiticity::kHermitian));
  EXPECT_THROW(terms::Comm232ss(in.x, in.y, z, *in.ctx), InvariantError);
}

}  // namespace
}  // namespace imsrg::commutator
