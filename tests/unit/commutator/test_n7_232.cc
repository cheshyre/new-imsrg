// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// IMSRG(3)-N7 term comm232ss (imsrg3.h) against the m-scheme brute force
// (emax 1, closed-shell, open-shell and fractional references, general
// hermiticity), against imsrg-devel (emax 2), for antisymmetry under
// X <-> Y, and the pair-third path of terms::Comm232ss against the
// element-wise reference (detail/comm232.h).

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
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
#include "imsrg/operator/random_operator.h"
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

// Z = comm232ss(X, Y) through the element-wise reference.
op::Operator EvaluateReference(const op::Operator& X, const op::Operator& Y,
                               const ThreeBodyContext& ctx) {
  op::Operator z = testing::ZeroN7Output(X, Y, ctx.model_space_ptr());
  detail::Comm232ssReference(X, Y, z, ctx);
  return z;
}

// Largest |a - b| over the two-body elements, relative to max(1, max |b|).
double TwoBodyDeviation(const op::Operator& a, const op::Operator& b) {
  const std::vector<double> va = a.two_body().matrix().buffer().ToVector();
  const std::vector<double> vb = b.two_body().matrix().buffer().ToVector();
  EXPECT_EQ(va.size(), vb.size());
  double scale = 1.0;
  double worst = 0.0;
  for (std::size_t k = 0; k < va.size() && k < vb.size(); ++k) {
    scale = std::max(scale, std::abs(vb[k]));
    worst = std::max(worst, std::abs(va[k] - vb[k]));
  }
  return worst / scale;
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

// Three-body cuts define the kept space (D34): E3max 2 (He4 at emax 1 drops
// the ppp kets) and dE3max 1 (C12, e_F = 1: at most one 0s orbit per ket)
// drop kets, and both paths match the brute force on the remaining space.
TEST(N7Comm232, MatchesBruteForceWithCuts) {
  basis::ThreeBodyCuts e3;
  e3.e3max = 2;
  basis::ThreeBodyCuts de3;
  de3.de3max = 1.0;
  const basis::ThreeBodyCuts full;
  for (const auto& [ref, cuts] :
       {std::pair<const char*, basis::ThreeBodyCuts>{"He4", e3},
        std::pair<const char*, basis::ThreeBodyCuts>{"C12", de3}}) {
    N7Inputs c =
        RandomN7Inputs(testing::MakeThreeBodySpace(1, ref, cuts),
                       Hermiticity::kNonHermitian, Hermiticity::kHermitian, 31);
    const std::int64_t kept = c.ms3->three_body_space()->NumRows();
    EXPECT_GT(kept, 0);
    EXPECT_LT(kept, testing::MakeThreeBodySpace(1, ref, full)
                        ->three_body_space()
                        ->NumRows());
    const op::Operator bf = BruteForceN7Term(Term::kComm232ss, c.x, c.y, c.ms3);
    const op::Operator fast =
        EvaluateN7Term(Term::kComm232ss, c.x, c.y, *c.ctx);
    const op::Operator slow = EvaluateReference(c.x, c.y, *c.ctx);
    EXPECT_GT(op::TwoBodyNorm(bf), 1.0);
    EXPECT_LE(RelativeDeviation(fast, bf), kTol) << ref;
    EXPECT_LE(RelativeDeviation(slow, bf), kTol) << ref;
    std::printf("comm232ss %s e3max %d de3max %g (%" PRId64
                " kept rows): deviation from the brute force %.3g\n",
                ref, cuts.e3max, cuts.de3max, kept,
                RelativeDeviation(fast, bf));
  }
}

// An operand without a three-body part (particle rank 2) contributes only
// through its two-body part: [X2, Y3] alone, [X3, Y2] alone, nothing for two
// rank-2 operands.
TEST(N7Comm232, MissingThreeBodyParts) {
  const auto ms3 =
      testing::MakeThreeBodySpace(1, "C12", basis::ThreeBodyCuts{});
  const ThreeBodyContext ctx(ms3);
  const op::Operator x3 =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 41);
  const op::Operator y3 =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 42);
  op::RandomOperatorOptions opt;
  opt.seed = 43;
  const op::Operator x2 = op::RandomOperator(
      ms3, op::OperatorSpec::Scalar(2, Hermiticity::kNonHermitian), opt);
  opt.seed = 44;
  const op::Operator y2 = op::RandomOperator(
      ms3, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian), opt);
  for (const auto& [x, y] :
       {std::pair<const op::Operator*, const op::Operator*>{&x2, &y3},
        std::pair<const op::Operator*, const op::Operator*>{&x3, &y2}}) {
    const op::Operator bf = BruteForceN7Term(Term::kComm232ss, *x, *y, ms3);
    const op::Operator fast = EvaluateN7Term(Term::kComm232ss, *x, *y, ctx);
    const op::Operator slow = EvaluateReference(*x, *y, ctx);
    EXPECT_GT(op::TwoBodyNorm(bf), 1.0);
    EXPECT_LE(RelativeDeviation(fast, bf), kTol);
    EXPECT_LE(RelativeDeviation(slow, bf), kTol);
  }
  const op::Operator none = EvaluateN7Term(Term::kComm232ss, x2, y2, ctx);
  EXPECT_EQ(op::TwoBodyNorm(none), 0.0);
}

// Element-wise against the imsrg++ dumps (E3max 3) and the two-body norms of
// values.json (E3max 3, and 6 where the hhh-ppp couplings exist).
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

// The pair-third path (terms::Comm232ss) against the element-wise reference
// (golden inputs, both E3max; the brute-force cases with non-Hermitian
// operands).
TEST(N7Comm232, PairThirdMatchesReference) {
  for (const char* ref : {"He4", "O16"}) {
    for (const int e3max : {3, 6}) {
      const N7Inputs in = GoldenN7Inputs(2, ref, e3max);
      auto t0 = std::chrono::steady_clock::now();
      const op::Operator fast =
          EvaluateN7Term(Term::kComm232ss, in.x, in.y, *in.ctx);
      const double ms_fast = Milliseconds(t0);
      t0 = std::chrono::steady_clock::now();
      const op::Operator slow = EvaluateReference(in.x, in.y, *in.ctx);
      const double ms_slow = Milliseconds(t0);
      const double dev = TwoBodyDeviation(fast, slow);
      EXPECT_LE(dev, 1.0e-12) << ref << " e3max " << e3max;
      std::printf(
          "comm232ss %s emax 2 e3max %d: pair third vs element-wise "
          "%.3g (%.1f ms vs %.1f ms)\n",
          ref, e3max, dev, ms_fast, ms_slow);
    }
  }
  for (N7Inputs& c : BruteForceCases()) {
    const op::Operator fast =
        EvaluateN7Term(Term::kComm232ss, c.x, c.y, *c.ctx);
    const op::Operator slow = EvaluateReference(c.x, c.y, *c.ctx);
    EXPECT_LE(TwoBodyDeviation(fast, slow), 1.0e-12)
        << c.ms3->reference().name();
  }
}

// Channel batching: a budget of one byte (one channel per batch) gives the
// result of the default budget (one batch at emax 2).
TEST(N7Comm232, BatchesAgree) {
  const N7Inputs in = GoldenN7Inputs(2, "O16", 6);
  op::Operator one = testing::ZeroN7Output(in.x, in.y, in.ms3);
  op::Operator many = testing::ZeroN7Output(in.x, in.y, in.ms3);
  detail::Comm232ssPairThird(in.x, in.y, one, *in.ctx,
                             detail::kComm232MaxBlockBytes);
  detail::Comm232ssPairThird(in.x, in.y, many, *in.ctx, 1);
  EXPECT_GT(op::TwoBodyNorm(one), 1.0);
  EXPECT_LE(TwoBodyDeviation(many, one), 1.0e-13);
  // Non-Hermitian operands (two block sets per operand).
  for (N7Inputs& c : BruteForceCases()) {
    op::Operator a = testing::ZeroN7Output(c.x, c.y, c.ms3);
    op::Operator b = testing::ZeroN7Output(c.x, c.y, c.ms3);
    detail::Comm232ssPairThird(c.x, c.y, a, *c.ctx,
                               detail::kComm232MaxBlockBytes);
    detail::Comm232ssPairThird(c.x, c.y, b, *c.ctx, 1);
    EXPECT_LE(TwoBodyDeviation(b, a), 1.0e-13) << c.ms3->reference().name();
  }
}

// Timing of the pair-third path (printed; repeated calls on fresh outputs).
TEST(N7Comm232, Timing) {
  for (const char* ref : {"He4", "O16"}) {
    const N7Inputs in = GoldenN7Inputs(2, ref, 6);
    std::vector<double> ms;
    for (int rep = 0; rep < 7; ++rep) {
      op::Operator z = testing::ZeroN7Output(in.x, in.y, in.ms3);
      const auto t0 = std::chrono::steady_clock::now();
      terms::Comm232ss(in.x, in.y, z, *in.ctx);
      ms.push_back(Milliseconds(t0));
    }
    std::sort(ms.begin(), ms.end());
    std::printf("comm232ss %s emax 2 e3max 6: min %.2f ms, median %.2f ms\n",
                ref, ms.front(), ms[ms.size() / 2]);
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
