// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The row-tiled comm132ss ([X1, Y3] + [X3, Y1] -> Z2, kernels/comm132.cc) and
// comm330ss ([X3, Y3] -> Z0, kernels/comm330.cc):
// * the batch paths (detail::Comm132Rows, detail::Comm330Rows) against the
//   element-wise references (detail::Comm132ssReference,
//   detail::Comm330ssReference) in the form max|dZ| <= 1e-12 max(1, max|Z|)
//   of design/14 section 6.5 (deviations printed): emax 1 and 2, He4 and
//   O16 at E3max 3 emax, a fractional-occupation space and a dE3max-cut
//   space, Hermitian, anti-Hermitian and non-Hermitian operand pairs,
//   operands without a three-body part, several batch budgets (down to one
//   triple per batch), and accumulation into Z;
// * partition invariance with masked operands
//   (tests/support/n7_partition_support.h);
// * determinism: two runs give bitwise equal results.

#include <algorithm>
#include <cinttypes>
#include <cmath>
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
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/n7_terms.h"
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

/// The `30` section 2 tolerance of a rewritten kernel against its reference.
constexpr double kTol = 1.0e-12;
/// A budget that splits the emax-2 spaces into several batches.
constexpr std::int64_t kMidBudget = std::int64_t{2} << 20;

/// A zeroed output with the hermiticity of [X, Y] (rank 2: Z0 and Z2).
op::Operator ZeroOutput(const op::Operator& x, const op::Operator& y,
                        const ThreeBodyContext& ctx) {
  op::OperatorSpec spec = CommutatorSpec(x.spec(), y.spec());
  spec.particle_rank = 2;
  return op::Operator(ctx.model_space_ptr(), spec);
}

/// max|z - ref| / max(1, max|ref|) over all parts; `ref_max` receives
/// max|ref|.
double Deviation(const op::Operator& z, const op::Operator& ref,
                 double* ref_max = nullptr) {
  const testing::OperatorDiff d = testing::CompareOperators(z, ref);
  if (ref_max != nullptr) *ref_max = d.max_abs_ref;
  return d.max_abs_diff / std::max(1.0, d.max_abs_ref);
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

/// The 3x3 operand pairs (X, Y) of every hermiticity (a non-Hermitian operand
/// makes [X, Y] non-Hermitian). [A, A] terms with an anti-Hermitian one-body
/// part vanish at emax 1 (one orbit per one-body channel).
std::vector<std::pair<Hermiticity, Hermiticity>> Pairs() {
  const Hermiticity kinds[] = {Hermiticity::kAntiHermitian,
                               Hermiticity::kHermitian,
                               Hermiticity::kNonHermitian};
  std::vector<std::pair<Hermiticity, Hermiticity>> out;
  for (const Hermiticity hx : kinds) {
    for (const Hermiticity hy : kinds) out.emplace_back(hx, hy);
  }
  return out;
}

/// The Rows function of a term and the reference it is checked against.
struct Term {
  const char* name;
  void (*rows)(const op::Operator&, const op::Operator&, op::Operator&,
               const ThreeBodyContext&, const basis::RowSet&, std::int64_t);
  void (*reference)(const op::Operator&, const op::Operator&, op::Operator&,
                    const ThreeBodyContext&);
};

const Term kTerm132 = {"132", &Comm132Rows, &Comm132ssReference};
const Term kTerm330 = {"330", &Comm330Rows, &Comm330ssReference};

/// The whole-space batch path of `term` with `budget` against its reference
/// for X, Y; prints and returns the deviation; `ref_max` receives max|ref|.
double BatchVsReference(const Term& term, const ThreeBodyContext& ctx,
                        const op::Operator& x, const op::Operator& y,
                        std::int64_t budget, const std::string& tag,
                        double* ref_max) {
  op::Operator ref = ZeroOutput(x, y, ctx);
  term.reference(x, y, ref, ctx);
  op::Operator z = ZeroOutput(x, y, ctx);
  term.rows(x, y, z, ctx,
            basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
            budget);
  const double dev = Deviation(z, ref, ref_max);
  std::printf("[%s batch] %s budget %" PRId64
              ": max|dZ| / max(1, max|Z|) = %.3e (max|Z| = %.4g)\n",
              term.name, tag.c_str(), budget, dev, *ref_max);
  return dev;
}

/// The batch path of `term` against its reference on every space of
/// `Spaces(emax)`, for every pair of `Pairs` and each budget of `budgets`;
/// with `need_nonzero` some case must give a nonzero term (132 vanishes at
/// emax 1: one orbit per one-body channel, so no b != a).
void CheckAgainstReference(const Term& term, int emax,
                           const std::vector<std::int64_t>& budgets,
                           bool need_nonzero) {
  std::uint64_t seed = 900 + static_cast<std::uint64_t>(emax);
  int nonzero = 0;
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
      for (const std::int64_t budget : budgets) {
        double ref_max = 0.0;
        EXPECT_LE(BatchVsReference(term, ctx, x, y, budget, tag, &ref_max),
                  kTol)
            << tag;
        if (ref_max > 1.0e-8) ++nonzero;
      }
    }
  }
  if (need_nonzero) EXPECT_GT(nonzero, 0) << term.name << " is zero in every case";
}

TEST(N7Comm132Batch, MatchesReferenceEmax1) {
  // Budget 1: every triple is a batch of its own.
  CheckAgainstReference(kTerm132, 1, {1, kMidBudget, kN7BatchBytes}, false);
}

TEST(N7Comm132Batch, MatchesReferenceEmax2) {
  CheckAgainstReference(kTerm132, 2, {kMidBudget, kN7BatchBytes}, true);
}

TEST(N7Comm330Batch, MatchesReferenceEmax1) {
  CheckAgainstReference(kTerm330, 1, {1, kMidBudget, kN7BatchBytes}, true);
}

TEST(N7Comm330Batch, MatchesReferenceEmax2) {
  CheckAgainstReference(kTerm330, 2, {1, kMidBudget, kN7BatchBytes}, true);
}

// The dE3max cut must drop kets, or the cut case tests nothing new.
TEST(N7Comm132Batch, CutSpaceDropsKets) {
  for (const int emax : {1, 2}) {
    const std::vector<SpaceCase> s = Spaces(emax);
    EXPECT_LT(s[3].ms->three_body_space()->NumRows(),
              s[1].ms->three_body_space()->NumRows())
        << "emax " << emax;
  }
}

// An operand without a three-body part contributes only through its
// one-body part: [X1, Y3] alone and [X3, Y1] alone (132), nothing (330).
TEST(N7Comm132Batch, MissingThreeBodyParts) {
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
  double m = 0.0;
  EXPECT_LE(BatchVsReference(kTerm132, ctx, x2, y3, kMidBudget, "O16 (A2, H3)",
                             &m),
            kTol);
  EXPECT_GT(m, 1.0e-6);
  EXPECT_LE(BatchVsReference(kTerm132, ctx, x3, y2, kMidBudget, "O16 (A3, H2)",
                             &m),
            kTol);
  EXPECT_GT(m, 1.0e-6);
  // Two rank-2 operands: nothing is added by either term.
  op::Operator z = ZeroOutput(x2, y2, ctx);
  const basis::RowSet whole = basis::RowSet::Whole(*ms->three_body_space());
  Comm132Rows(x2, y2, z, ctx, whole, kN7BatchBytes);
  Comm330Rows(x2, y2, z, ctx, whole, kN7BatchBytes);
  EXPECT_EQ(Deviation(z, ZeroOutput(x2, y2, ctx)), 0.0);
  // 330 needs both three-body parts.
  op::Operator z330 = ZeroOutput(x2, y3, ctx);
  Comm330Rows(x2, y3, z330, ctx, whole, kN7BatchBytes);
  EXPECT_EQ(z330.zero_body(), 0.0);
}

// The 330 term of anti-Hermitian Z vanishes (Z0 stays exactly zero).
TEST(N7Comm330Batch, AntiHermitianZ0IsZero) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 31);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 32);
  op::Operator z = ZeroOutput(x, y, ctx);
  ASSERT_EQ(z.hermiticity(), Hermiticity::kAntiHermitian);
  terms::Comm330ss(x, y, z, ctx);
  EXPECT_EQ(z.zero_body(), 0.0);
}

// The terms add into Z: a second call doubles the contribution.
TEST(N7Comm132Batch, AddsIntoZ) {
  const auto ms = testing::MakeThreeBodySpace(2, "He4", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 51);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 52);
  op::Operator once = ZeroOutput(x, y, ctx);
  terms::Comm132ss(x, y, once, ctx);
  terms::Comm330ss(x, y, once, ctx);
  op::Operator twice = ZeroOutput(x, y, ctx);
  terms::Comm132ss(x, y, twice, ctx);
  terms::Comm330ss(x, y, twice, ctx);
  terms::Comm132ss(x, y, twice, ctx);
  terms::Comm330ss(x, y, twice, ctx);
  op::Operator doubled = once;
  doubled *= 2.0;
  double m = 0.0;
  EXPECT_LE(Deviation(twice, doubled, &m), kTol);
  EXPECT_GT(m, 1.0e-6);
}

// The public terms are the batch paths with the default budget, and two
// runs give bitwise equal results (no atomics, fixed summation order).
TEST(N7Comm132Batch, PublicTermsAreDeterministic) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const basis::RowSet whole = basis::RowSet::Whole(*ms->three_body_space());
  const std::pair<Hermiticity, Hermiticity> pairs[] = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kNonHermitian, Hermiticity::kHermitian}};
  std::uint64_t seed = 60;
  for (const auto& [hx, hy] : pairs) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
    const op::Operator y = testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
    seed += 2;
    op::Operator z1 = ZeroOutput(x, y, ctx);
    terms::Comm132ss(x, y, z1, ctx);
    terms::Comm330ss(x, y, z1, ctx);
    op::Operator z2 = ZeroOutput(x, y, ctx);
    terms::Comm132ss(x, y, z2, ctx);
    terms::Comm330ss(x, y, z2, ctx);
    op::Operator z3 = ZeroOutput(x, y, ctx);
    Comm132Rows(x, y, z3, ctx, whole, kN7BatchBytes);
    Comm330Rows(x, y, z3, ctx, whole, kN7BatchBytes);
    const std::string tag =
        std::string("(") + Letter(hx) + ", " + Letter(hy) + ")";
    EXPECT_TRUE(testing::CompareOperators(z1, z2).bitwise_equal) << tag;
    EXPECT_TRUE(testing::CompareOperators(z1, z3).bitwise_equal) << tag;
  }
}

/// Partition invariance and masked operands of the Rows function of `term`
/// (mid budget) for two pairs on `ref` at emax 2, E3max 6.
void CheckPartitions(const Term& term, const std::string& ref) {
  const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const std::pair<Hermiticity, Hermiticity> pairs[] = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
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
      term.rows(a, b, z, ctx, rows, kMidBudget);
    };
    testing::ExpectPartitionInvariance(
        *ms, x, y, zero, run, kTol, false,
        std::string("comm") + term.name + " " + ref + " (" + Letter(hx) +
            ", " + Letter(hy) + ")");
  }
}

TEST(N7Comm132Batch, PartitionInvarianceHe4) {
  CheckPartitions(kTerm132, "He4");
}

TEST(N7Comm132Batch, PartitionInvarianceO16) {
  CheckPartitions(kTerm132, "O16");
}

TEST(N7Comm330Batch, PartitionInvarianceHe4) {
  CheckPartitions(kTerm330, "He4");
}

TEST(N7Comm330Batch, PartitionInvarianceO16) {
  CheckPartitions(kTerm330, "O16");
}

}  // namespace
}  // namespace imsrg::commutator::detail
