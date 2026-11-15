// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 terms comm133ss and comm132ss (imsrg3.h) against the m-scheme
// brute force (emax 1, closed-shell, open-shell and fractional references,
// general hermiticity) and against imsrg-devel (emax 2).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
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
using testing::ReadN7Golden;
using testing::RelativeDeviation;

constexpr double kTermTol = 1.0e-10;
constexpr double kGoldenTol = 1.0e-10;

double Rel(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::abs(b));
}

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
    EXPECT_LE(dev, kTermTol)
        << TermName(t) << " on " << c.ms3->reference().name();
    std::printf("%s %s: relative deviation from the brute force %.3g\n",
                std::string(TermName(t)).c_str(),
                std::string(c.ms3->reference().name()).c_str(), dev);
  }
}

void ExpectMatchesImsrgDevel(Term t, const char* file, const char* name,
                             bool three_body) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    {
      const N7Inputs in = GoldenN7Inputs(2, ref, 3);
      const auto t0 = std::chrono::steady_clock::now();
      const op::Operator z = EvaluateN7Term(t, in.x, in.y, *in.ctx);
      const double sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      const op::Operator golden = ReadN7Golden(dir, file, in.ms3);
      const double dev = RelativeDeviation(z, golden);
      EXPECT_LE(dev, kGoldenTol) << dir << " e3max 3";
      std::printf("%s %s e3max 3: deviation from imsrg++ %.3g (%.3f s)\n", name,
                  ref, dev, sec);
    }
    {
      const N7Inputs in = GoldenN7Inputs(2, ref, 6);
      const auto t0 = std::chrono::steady_clock::now();
      const op::Operator z = EvaluateN7Term(t, in.x, in.y, *in.ctx);
      const double sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      const double norm =
          three_body ? op::ThreeBodyNorm(z) : op::TwoBodyNorm(z);
      const double golden = N7GoldenValue(
          dir, 6, name, three_body ? "three_body_norm" : "two_body_norm");
      EXPECT_LE(Rel(norm, golden), kGoldenTol)
          << dir << " e3max 6: " << norm << " vs " << golden;
      std::printf("%s %s e3max 6: norm %.10g vs %.10g (%.3f s)\n", name, ref,
                  norm, golden, sec);
    }
  }
}

TEST(N7Comm133, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm133ss);
}

TEST(N7Comm132, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm132ss);
}

TEST(N7Comm133, MatchesImsrgDevel) {
  ExpectMatchesImsrgDevel(Term::kComm133ss, "comm133ss.dump", "comm133ss",
                          true);
}

TEST(N7Comm132, MatchesImsrgDevel) {
  ExpectMatchesImsrgDevel(Term::kComm132ss, "comm132ss.dump", "comm132ss",
                          false);
}

TEST(N7Comm133, RequiresAThreeBodyOutput) {
  const N7Inputs in = GoldenN7Inputs(2, "He4", 3);
  op::Operator z(in.ms3, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  EXPECT_THROW(terms::Comm133ss(in.x, in.y, z, *in.ctx), InvariantError);
}

}  // namespace
}  // namespace imsrg::commutator
