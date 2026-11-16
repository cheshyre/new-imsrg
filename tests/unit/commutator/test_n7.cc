// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 terms (imsrg3.h), part 1: the golden inputs (drift check against
// the imsrg++ norms of tests/golden/n7), comm330ss and comm223ss against the
// m-scheme brute force (emax 1, closed-shell, open-shell and fractional
// references, general hermiticity) and against imsrg-devel (emax 2).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
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

constexpr double kTermTol = 1.0e-10;    // design/30 section 2
constexpr double kGoldenTol = 1.0e-10;  // goldens are %.17g, parallel sums

double Rel(double a, double b) {
  return std::abs(a - b) / std::max(1.0, std::abs(b));
}

// The emax-1 brute-force cases: closed shell, open shell (C12 fills 0p3/2),
// fractional occupations; X anti-Hermitian, Y Hermitian, and a non-Hermitian
// X.
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

// The regenerated golden inputs have the imsrg++ norms recorded by the
// golden script (a drift of RandomThreeBodyOperator shows up here first).
TEST(N7Inputs, NormsMatchTheGoldenInputs) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    for (const int e3max : {3, 6}) {
      const N7Inputs in = GoldenN7Inputs(2, ref, e3max);
      for (const auto& [name, op] :
           {std::pair<const char*, const op::Operator*>{"X", &in.x},
            std::pair<const char*, const op::Operator*>{"Y", &in.y}}) {
        EXPECT_LE(Rel(op::OneBodyNorm(*op),
                      N7GoldenValue(dir, e3max, name, "one_body_norm")),
                  kGoldenTol);
        EXPECT_LE(Rel(op::TwoBodyNorm(*op),
                      N7GoldenValue(dir, e3max, name, "two_body_norm")),
                  kGoldenTol);
        EXPECT_LE(Rel(op::ThreeBodyNorm(*op),
                      N7GoldenValue(dir, e3max, name, "three_body_norm")),
                  kGoldenTol)
            << dir << " e3max " << e3max << " " << name;
      }
    }
  }
}

TEST(N7Comm330, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm330ss);
}

TEST(N7Comm330, MatchesImsrgDevel) {
  for (const char* ref : {"He4", "O16"}) {
    const std::string dir = std::string(ref) + "_emax2";
    for (const int e3max : {3, 6}) {
      const N7Inputs in = GoldenN7Inputs(2, ref, e3max);
      const op::Operator z =
          EvaluateN7Term(Term::kComm330ss, in.x, in.y, *in.ctx);
      const double golden = N7GoldenValue(dir, e3max, "comm330ss", "zero_body");
      EXPECT_LE(Rel(z.zero_body(), golden), kGoldenTol)
          << dir << " e3max " << e3max << ": " << z.zero_body() << " vs "
          << golden;
    }
  }
}

TEST(N7Comm223, MatchesBruteForce) {
  ExpectMatchesBruteForce(Term::kComm223ss);
}

TEST(N7Comm223, AddsComm223) {
  const N7Inputs in = GoldenN7Inputs(2, "He4", 3);
  const op::Operator z = EvaluateN7Term(Term::kComm223ss, in.x, in.y, *in.ctx);
  const op::Operator ref =
      testing::ReadN7Golden("He4_emax2", "comm223ss.dump", in.ms3);
  EXPECT_LE(RelativeDeviation(z, ref), kGoldenTol);
}

}  // namespace
}  // namespace imsrg::commutator
