// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The row-tiled comm331ss ([X3, Y3] -> Z1, kernels/comm331.cc) and comm231ss
// ([X2, Y3] + [X3, Y2] -> Z1, kernels/comm231.cc):
// * the batch paths (detail::Comm331Rows, detail::Comm231Rows) against the
//   element-wise references (detail::Comm331ssReference,
//   detail::Comm231ssReference) in the form max|dZ| <= 1e-12 max(1, max|Z|) of
//   design/14 section 6.5 (deviations printed): emax 1 and 2, He4 and O16 at
//   E3max 3 emax, a fractional-occupation space and a dE3max-cut space,
//   Hermitian, anti-Hermitian and non-Hermitian operands, several batch
//   budgets (down to one triple per batch);
// * partition invariance with masked operands
//   (tests/support/n7_partition_support.h);
// * operands without a three-body part, accumulation into Z, and bitwise
//   reproducibility from run to run (no atomics, fixed order).

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/n7_partition_support.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::commutator::detail {
namespace {

/// The design/30 section 2 tolerance of a rewritten kernel against its
/// reference.
constexpr double kTol = 1.0e-12;
/// A budget that splits the emax-2 spaces into several batches.
constexpr std::int64_t kMidBudget = std::int64_t{2} << 20;

/// A term under test: its batch loop and element-wise reference.
struct Term {
  const char* name;
  void (*rows)(const op::Operator&, const op::Operator&, op::Operator&,
               const ThreeBodyContext&, const basis::RowSet&, std::int64_t);
  void (*reference)(const op::Operator&, const op::Operator&, op::Operator&,
                    const ThreeBodyContext&);
  void (*pub)(const op::Operator&, const op::Operator&, op::Operator&,
              const ThreeBodyContext&);
};

const Term kTerm331 = {"331", Comm331Rows, Comm331ssReference,
                       terms::Comm331ss};
const Term kTerm231 = {"231", Comm231Rows, Comm231ssReference,
                       terms::Comm231ss};

/// A zeroed rank-3 output with the hermiticity of [X, Y].
op::Operator ZeroOutput(const op::Operator& x, const op::Operator& y,
                        const ThreeBodyContext& ctx) {
  op::OperatorSpec spec = CommutatorSpec(x.spec(), y.spec());
  spec.particle_rank = 3;
  return op::Operator(ctx.model_space_ptr(), spec);
}

/// max|z - ref| / max(1, max|ref|) over all parts.
double Deviation(const op::Operator& z, const op::Operator& ref) {
  const testing::OperatorDiff d = testing::CompareOperators(z, ref);
  return d.max_abs_diff / std::max(1.0, d.max_abs_ref);
}

/// The whole-space batch path of `t` with `budget` as a new Z.
op::Operator RunWhole(const Term& t, const ThreeBodyContext& ctx,
                      const op::Operator& x, const op::Operator& y,
                      std::int64_t budget) {
  op::Operator z = ZeroOutput(x, y, ctx);
  t.rows(x, y, z, ctx,
         basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
         budget);
  return z;
}

/// The element-wise reference of `t` as a new Z.
op::Operator RunReference(const Term& t, const ThreeBodyContext& ctx,
                          const op::Operator& x, const op::Operator& y) {
  op::Operator z = ZeroOutput(x, y, ctx);
  t.reference(x, y, z, ctx);
  return z;
}

/// The batch path with each of `budgets` against the reference for X, Y;
/// prints the deviations and expects them below kTol and the term nonzero.
void CheckOperands(const Term& t, const ThreeBodyContext& ctx,
                   const op::Operator& x, const op::Operator& y,
                   const std::vector<std::int64_t>& budgets,
                   const std::string& tag) {
  const op::Operator ref = RunReference(t, ctx, x, y);
  const double norm = op::OneBodyNorm(ref);
  EXPECT_GT(norm, 1.0e-3) << t.name << " " << tag;
  for (const std::int64_t budget : budgets) {
    const double dev = Deviation(RunWhole(t, ctx, x, y, budget), ref);
    std::printf("[%s batch] %s budget %" PRId64
                ": max|dZ| / max(1, max|Z|) = %.3e (|Z|_1 = %.4g)\n",
                t.name, tag.c_str(), budget, dev, norm);
    EXPECT_LE(dev, kTol) << t.name << " " << tag << " budget " << budget;
  }
}

/// A model space of the comparison cases with its label.
struct SpaceCase {
  std::string name;
  std::shared_ptr<const basis::ModelSpace> ms;
};

/// He4 and O16 at E3max 3 emax, fractional He4 (E3max 3 emax), and O16 with
/// a dE3max 2 cut (E3max 3 emax).
std::vector<SpaceCase> Spaces(int emax) {
  const basis::ThreeBodyCuts full{.e3max = 3 * emax};
  basis::ThreeBodyCuts cut = full;
  cut.de3max = 2.0;
  std::vector<SpaceCase> out;
  out.push_back({"He4 emax " + std::to_string(emax),
                 testing::MakeThreeBodySpace(emax, "He4", full)});
  out.push_back({"O16 emax " + std::to_string(emax),
                 testing::MakeThreeBodySpace(emax, "O16", full)});
  out.push_back({"fractional He4 emax " + std::to_string(emax),
                 testing::AddThreeBodySpace(
                     testing::FractionalModelSpace(emax, "He4"), full)});
  out.push_back({"O16 dE3max 2 emax " + std::to_string(emax),
                 testing::MakeThreeBodySpace(emax, "O16", cut)});
  return out;
}

/// One letter per hermiticity, for the labels.
const char* Letter(Hermiticity h) {
  switch (h) {
    case Hermiticity::kHermitian:
      return "H";
    case Hermiticity::kAntiHermitian:
      return "A";
    case Hermiticity::kNonHermitian:
      return "N";
  }
  return "?";
}

/// The operand pairs (X, Y) of the comparisons: (A, H), (H, H), (H, A),
/// (A, A), (N, H), (H, N), (N, N).
std::vector<std::pair<Hermiticity, Hermiticity>> Pairs() {
  constexpr Hermiticity kA = Hermiticity::kAntiHermitian;
  constexpr Hermiticity kH = Hermiticity::kHermitian;
  constexpr Hermiticity kN = Hermiticity::kNonHermitian;
  return {{kA, kH}, {kH, kH}, {kH, kA}, {kA, kA}, {kN, kH}, {kH, kN}, {kN, kN}};
}

/// The batch path against the reference on every space of `Spaces(emax)`, for
/// every pair of `Pairs` and each budget of `budgets`.
void CheckAgainstReference(const Term& t, int emax,
                           const std::vector<std::int64_t>& budgets) {
  std::uint64_t seed = 300;
  for (const SpaceCase& c : Spaces(emax)) {
    ASSERT_GT(c.ms->three_body_space()->NumRows(), 0) << c.name;
    const ThreeBodyContext ctx(c.ms);
    for (const auto& [hx, hy] : Pairs()) {
      const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
      const op::Operator y =
          testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
      seed += 2;
      const std::string tag =
          c.name + " (" + Letter(hx) + ", " + Letter(hy) + ")";
      // [A, A] and [H, H] of a rank-3 pair have a one-body part, but the
      // cases whose Z1 vanishes by hermiticity (zero contribution) are
      // compared too, only without the nonzero check.
      const op::Operator ref = RunReference(t, ctx, x, y);
      const double norm = op::OneBodyNorm(ref);
      for (const std::int64_t budget : budgets) {
        const double dev = Deviation(RunWhole(t, ctx, x, y, budget), ref);
        std::printf("[%s batch] %s budget %" PRId64
                    ": max|dZ| / max(1, max|Z|) = %.3e (|Z|_1 = %.4g)\n",
                    t.name, tag.c_str(), budget, dev, norm);
        EXPECT_LE(dev, kTol) << t.name << " " << tag << " budget " << budget;
      }
    }
  }
}

TEST(N7Comm331Batch, MatchesReferenceEmax1) {
  // Budget 1: every triple is a batch of its own.
  CheckAgainstReference(kTerm331, 1, {1, kMidBudget, kN7BatchBytes});
}

TEST(N7Comm331Batch, MatchesReferenceEmax2) {
  CheckAgainstReference(kTerm331, 2, {kMidBudget, kN7BatchBytes});
}

TEST(N7Comm231Batch, MatchesReferenceEmax1) {
  CheckAgainstReference(kTerm231, 1, {1, kMidBudget, kN7BatchBytes});
}

TEST(N7Comm231Batch, MatchesReferenceEmax2) {
  CheckAgainstReference(kTerm231, 2, {kMidBudget, kN7BatchBytes});
}

// The terms must be nonzero in the compared cases, or the comparison tests
// nothing: the O16 (A, H) and (H, H) pairs at emax 2.
TEST(N7Comm331Batch, TermIsNonzero) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 11);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 12);
  CheckOperands(kTerm331, ctx, x, y, {kN7BatchBytes}, "O16 emax 2 (A, H)");
  CheckOperands(kTerm231, ctx, x, y, {kN7BatchBytes}, "O16 emax 2 (A, H)");
}

// The dE3max cut must drop kets, or the cut case tests nothing new.
TEST(N7Comm331Batch, CutSpaceDropsKets) {
  for (const int emax : {1, 2}) {
    const std::vector<SpaceCase> s = Spaces(emax);
    EXPECT_LT(s[3].ms->three_body_space()->NumRows(),
              s[1].ms->three_body_space()->NumRows())
        << "emax " << emax;
  }
}

// The mid budget must split the emax-2 space into several batches.
TEST(N7Comm331Batch, MidBudgetSplitsTheSpace) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const basis::RowSet whole = basis::RowSet::Whole(*ms->three_body_space());
  EXPECT_GT(PlanBatches(ctx.impl(), whole, kMidBudget, 2).size(),
            std::size_t{2});
  EXPECT_EQ(PlanBatches(ctx.impl(), whole, kN7BatchBytes, 2).size(),
            std::size_t{1});
}

// An operand without a three-body part: 331 adds nothing, 231 gets the term
// of the operand pair that has a two-body part and a three-body part.
TEST(N7Comm331Batch, MissingThreeBodyParts) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x3 =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 41);
  const op::Operator y3 =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 42);
  op::RandomOperatorOptions opt;
  opt.seed = 43;
  const op::Operator x2 = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian), opt);
  opt.seed = 44;
  const op::Operator y2 = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian), opt);
  // 331 needs both three-body parts.
  for (const auto& [a, b] :
       {std::pair<const op::Operator*, const op::Operator*>(&x2, &y3),
        {&x3, &y2},
        {&x2, &y2}}) {
    op::Operator z = ZeroOutput(*a, *b, ctx);
    Comm331Rows(*a, *b, z, ctx, basis::RowSet::Whole(*ms->three_body_space()),
                kMidBudget);
    EXPECT_EQ(op::Norm(z), 0.0);
  }
  // 231: [X2, Y3] alone and [X3, Y2] alone, two rank-2 operands nothing.
  CheckOperands(kTerm231, ctx, x2, y3, {kMidBudget}, "O16 (A2, H3)");
  CheckOperands(kTerm231, ctx, x3, y2, {kMidBudget}, "O16 (A3, H2)");
  op::Operator z = ZeroOutput(x2, y2, ctx);
  Comm231Rows(x2, y2, z, ctx, basis::RowSet::Whole(*ms->three_body_space()),
              kN7BatchBytes);
  EXPECT_EQ(op::Norm(z), 0.0);
}

// The terms add into Z: a second call doubles the contribution.
void CheckAddsIntoZ(const Term& t) {
  const auto ms = testing::MakeThreeBodySpace(2, "He4", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 51);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 52);
  op::Operator once = ZeroOutput(x, y, ctx);
  t.pub(x, y, once, ctx);
  op::Operator twice = ZeroOutput(x, y, ctx);
  t.pub(x, y, twice, ctx);
  t.pub(x, y, twice, ctx);
  EXPECT_GT(op::OneBodyNorm(once), 1.0e-3) << t.name;
  op::Operator doubled = once;
  doubled *= 2.0;
  EXPECT_LE(Deviation(twice, doubled), kTol) << t.name;
}

TEST(N7Comm331Batch, AddsIntoZ) { CheckAddsIntoZ(kTerm331); }

TEST(N7Comm231Batch, AddsIntoZ) { CheckAddsIntoZ(kTerm231); }

// Two runs with the same thread count agree bitwise (no floating-point
// atomics, every Z1 element accumulated by one work item in a fixed order),
// and the public term is the whole-space batch path with the default budget.
void CheckDeterministic(const Term& t) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const Hermiticity kinds[] = {Hermiticity::kAntiHermitian,
                               Hermiticity::kNonHermitian};
  std::uint64_t seed = 60;
  for (const Hermiticity hx : kinds) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
    const op::Operator y = testing::RandomThreeBodyOperator(
        ctx, Hermiticity::kHermitian, seed + 1);
    seed += 2;
    const std::string tag = std::string("(") + Letter(hx) + ", H)";
    const op::Operator a = RunWhole(t, ctx, x, y, kMidBudget);
    const op::Operator b = RunWhole(t, ctx, x, y, kMidBudget);
    EXPECT_TRUE(testing::CompareOperators(a, b).bitwise_equal)
        << t.name << " " << tag;
    op::Operator pub = ZeroOutput(x, y, ctx);
    t.pub(x, y, pub, ctx);
    const op::Operator whole = RunWhole(t, ctx, x, y, kN7BatchBytes);
    EXPECT_TRUE(testing::CompareOperators(pub, whole).bitwise_equal)
        << t.name << " " << tag;
    EXPECT_GT(op::OneBodyNorm(pub), 1.0e-3) << t.name << " " << tag;
  }
}

TEST(N7Comm331Batch, Deterministic) { CheckDeterministic(kTerm331); }

TEST(N7Comm231Batch, Deterministic) { CheckDeterministic(kTerm231); }

/// Partition invariance and masked operands of `t` (mid budget) for the
/// pairs (A, H), (H, H) and (N, H) on `ref` at emax 2, E3max 6.
void CheckPartitions(const Term& t, const std::string& ref) {
  const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const std::pair<Hermiticity, Hermiticity> pairs[] = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kHermitian},
      {Hermiticity::kNonHermitian, Hermiticity::kHermitian}};
  std::uint64_t seed = 70;
  for (const auto& [hx, hy] : pairs) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
    const op::Operator y = testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
    seed += 2;
    const auto zero = [&]() { return ZeroOutput(x, y, ctx); };
    const testing::N7RowsFn run = [&](const op::Operator& a,
                                      const op::Operator& b, op::Operator& z,
                                      const basis::RowSet& rows) {
      t.rows(a, b, z, ctx, rows, kMidBudget);
    };
    testing::ExpectPartitionInvariance(*ms, x, y, zero, run, kTol, false,
                                       std::string("comm") + t.name + " " +
                                           ref + " (" + Letter(hx) + ", " +
                                           Letter(hy) + ")");
  }
}

TEST(N7Comm331Batch, PartitionInvarianceHe4) {
  CheckPartitions(kTerm331, "He4");
}

TEST(N7Comm331Batch, PartitionInvarianceO16) {
  CheckPartitions(kTerm331, "O16");
}

TEST(N7Comm231Batch, PartitionInvarianceHe4) {
  CheckPartitions(kTerm231, "He4");
}

TEST(N7Comm231Batch, PartitionInvarianceO16) {
  CheckPartitions(kTerm231, "O16");
}

}  // namespace
}  // namespace imsrg::commutator::detail
