// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// comm232ss on the row-tiled batch interface (detail::Comm232Batch, run by
// detail::Comm232Rows; design/14 section 6): the batched path against the
// element-wise reference within max|dZ| <= 1e-12 max(1, max|Z|) (the
// design/30 section 2 form), partition invariance with masked operands
// (every part of the standard partitions reads three-body entries of its own
// lines only), and the same result for any batch budget. Emax 2, E3max 6:
// He4 and O16 with the operand hermiticities (A,H), (H,H), (N,H), (H,N), a
// fractional reference and a dE3max cut. The deviations are printed.

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/n7_partition_support.h"
#include "tests/support/three_body_test_support.h"
#include "tests/unit/commutator/n7_test_support.h"

namespace imsrg::commutator::detail {
namespace {

using testing::N7Inputs;

/// The tolerance of every non-bitwise comparison (design/14 section 6.5).
constexpr double kTol = 1.0e-12;

/// Operands of one test case with a label for the messages.
struct Case {
  std::string tag;
  N7Inputs in;
};

/// The letter of a hermiticity in the labels.
char Letter(Hermiticity h) {
  switch (h) {
    case Hermiticity::kHermitian:
      return 'H';
    case Hermiticity::kAntiHermitian:
      return 'A';
    default:
      return 'N';
  }
}

/// X of hermiticity `hx` and Y of `hy` on emax 2, E3max 6 of `ref`.
Case HermiticityCase(const char* ref, Hermiticity hx, Hermiticity hy,
                     std::uint64_t seed) {
  Case c;
  c.tag = std::string(ref) + " (" + Letter(hx) + "," + Letter(hy) + ")";
  c.in = testing::RandomN7Inputs(
      testing::MakeThreeBodySpace(2, ref, {.e3max = 6}), hx, hy, seed);
  return c;
}

/// The four operand hermiticities on `ref`.
std::vector<Case> HermiticityCases(const char* ref) {
  constexpr Hermiticity kA = Hermiticity::kAntiHermitian;
  constexpr Hermiticity kH = Hermiticity::kHermitian;
  constexpr Hermiticity kN = Hermiticity::kNonHermitian;
  std::vector<Case> cases;
  cases.push_back(HermiticityCase(ref, kA, kH, 51));
  cases.push_back(HermiticityCase(ref, kH, kH, 53));
  cases.push_back(HermiticityCase(ref, kN, kH, 55));
  cases.push_back(HermiticityCase(ref, kH, kN, 57));
  return cases;
}

/// A fractional reference (O16 with fractional last holes and first
/// particles, so all four pair classes and 0 < n_c < 1 occur; X
/// non-Hermitian, Y Hermitian) and a dE3max cut (O16, dE3max 2, X
/// anti-Hermitian, Y Hermitian), both at emax 2, E3max 6.
std::vector<Case> SpecialCases() {
  std::vector<Case> cases;
  Case frac;
  frac.tag = "O16 fractional (N,H)";
  frac.in = testing::RandomN7Inputs(
      testing::AddThreeBodySpace(testing::FractionalModelSpace(2, "O16"),
                                 {.e3max = 6}),
      Hermiticity::kNonHermitian, Hermiticity::kHermitian, 59);
  cases.push_back(std::move(frac));
  Case cut;
  cut.tag = "O16 dE3max 2 (A,H)";
  cut.in = testing::RandomN7Inputs(
      testing::MakeThreeBodySpace(2, "O16", {.e3max = 6, .de3max = 2.0}),
      Hermiticity::kAntiHermitian, Hermiticity::kHermitian, 61);
  const std::int64_t kept = cut.in.ms3->three_body_space()->NumRows();
  const std::int64_t full = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6})
                                ->three_body_space()
                                ->NumRows();
  EXPECT_GT(kept, 0);
  EXPECT_LT(kept, full);
  cases.push_back(std::move(cut));
  return cases;
}

/// A zeroed output for [X, Y] of `in`.
op::Operator Zero(const N7Inputs& in) {
  return testing::ZeroN7Output(in.x, in.y, in.ms3);
}

/// max|a - ref| / max(1, max|ref|) over all parts of the operators.
double Deviation(const op::Operator& a, const op::Operator& ref) {
  const testing::OperatorDiff d = testing::CompareOperators(a, ref);
  return d.max_abs_diff / std::max(1.0, d.max_abs_ref);
}

/// Operand blocks of the H/W intermediates of a three-body part (as
/// `ForEachN7Batch` counts them): 0 if missing, 2 if dense, else 1.
std::int32_t OperandBlocks(const op::Operator& x) {
  if (!x.three_body().has_value()) return 0;
  return x.three_body()->packed() ? 1 : 2;
}

/// A batch budget that splits the whole space of `in` into several batches
/// of several triples: a fifth of the bytes of the intermediates of the
/// whole space (checked with `PlanBatches`).
std::int64_t MidBudget(const N7Inputs& in, const std::string& tag) {
  const ThreeBodyContext::Impl& im = in.ctx->impl();
  const basis::ThreeBodySpace& space = *in.ms3->three_body_space();
  const basis::RowSet whole = basis::RowSet::Whole(space);
  const std::int32_t blocks =
      std::max(1, OperandBlocks(in.x) + OperandBlocks(in.y));
  std::int64_t total = 0;
  for (const LineBatch& b : PlanBatches(
           im, whole, std::numeric_limits<std::int64_t>::max(), blocks)) {
    total += 8 * blocks * (b.h_doubles + b.w_doubles);
  }
  const std::int64_t mid = total / 5;
  const std::size_t batches = PlanBatches(im, whole, mid, blocks).size();
  EXPECT_GT(batches, std::size_t{2}) << tag;
  EXPECT_LT(batches, static_cast<std::size_t>(space.num_triples())) << tag;
  return mid;
}

/// Z = comm232ss(X, Y) on the lines of `rows` with the batch budget `budget`.
op::Operator RunRows(const N7Inputs& in, const basis::RowSet& rows,
                     std::int64_t budget) {
  op::Operator z = Zero(in);
  Comm232Rows(in.x, in.y, z, *in.ctx, rows, budget);
  return z;
}

/// (a) The batched path of `terms::Comm232ss` against the element-wise
/// reference.
void ExpectMatchesReference(const Case& c) {
  op::Operator fast = Zero(c.in);
  terms::Comm232ss(c.in.x, c.in.y, fast, *c.in.ctx);
  op::Operator slow = Zero(c.in);
  Comm232ssReference(c.in.x, c.in.y, slow, *c.in.ctx);
  const double dev = Deviation(fast, slow);
  EXPECT_GT(op::TwoBodyNorm(slow), 1.0) << c.tag;
  EXPECT_LE(dev, kTol) << c.tag;
  std::printf("[comm232 batch] %s: batched vs element-wise %.3e\n",
              c.tag.c_str(), dev);
}

/// (b) Partition invariance and masked operands with a mid budget.
void ExpectPartitionInvariant(const Case& c) {
  const N7Inputs& in = c.in;
  const std::int64_t mid = MidBudget(in, c.tag);
  const auto zero = [&in]() { return Zero(in); };
  const testing::N7RowsFn run =
      [&in, mid](const op::Operator& a, const op::Operator& b, op::Operator& z,
                 const basis::RowSet& rows) {
        Comm232Rows(a, b, z, *in.ctx, rows, mid);
      };
  testing::ExpectPartitionInvariance(*in.ms3, in.x, in.y, zero, run, kTol,
                                     false, "comm232 " + c.tag);
}

/// (c) Budgets of 1 byte (one triple per batch), mid and whole agree.
void ExpectBudgetsAgree(const Case& c) {
  const basis::RowSet whole =
      basis::RowSet::Whole(*c.in.ms3->three_body_space());
  const op::Operator ref = RunRows(c.in, whole, kN7BatchBytes);
  EXPECT_GT(op::TwoBodyNorm(ref), 1.0) << c.tag;
  const std::int64_t mid = MidBudget(c.in, c.tag);
  for (const std::int64_t budget : {std::int64_t{1}, mid}) {
    const double dev = Deviation(RunRows(c.in, whole, budget), ref);
    EXPECT_LE(dev, kTol) << c.tag << " budget " << budget;
    std::printf("[comm232 batch] %s: budget %" PRId64 " vs whole %.3e\n",
                c.tag.c_str(), budget, dev);
  }
}

TEST(N7Comm232Batch, MatchesReferenceHe4) {
  for (const Case& c : HermiticityCases("He4")) ExpectMatchesReference(c);
}

TEST(N7Comm232Batch, MatchesReferenceO16) {
  for (const Case& c : HermiticityCases("O16")) ExpectMatchesReference(c);
}

TEST(N7Comm232Batch, MatchesReferenceFractionalAndCut) {
  for (const Case& c : SpecialCases()) ExpectMatchesReference(c);
}

TEST(N7Comm232Batch, PartitionInvariantHe4) {
  for (const Case& c : HermiticityCases("He4")) ExpectPartitionInvariant(c);
}

TEST(N7Comm232Batch, PartitionInvariantO16) {
  for (const Case& c : HermiticityCases("O16")) ExpectPartitionInvariant(c);
}

TEST(N7Comm232Batch, PartitionInvariantFractionalAndCut) {
  for (const Case& c : SpecialCases()) ExpectPartitionInvariant(c);
}

TEST(N7Comm232Batch, BudgetsAgreeHe4) {
  for (const Case& c : HermiticityCases("He4")) ExpectBudgetsAgree(c);
}

TEST(N7Comm232Batch, BudgetsAgreeO16) {
  for (const Case& c : HermiticityCases("O16")) ExpectBudgetsAgree(c);
}

TEST(N7Comm232Batch, BudgetsAgreeFractionalAndCut) {
  for (const Case& c : SpecialCases()) ExpectBudgetsAgree(c);
}

// No lines, no columns: nothing is added.
TEST(N7Comm232Batch, EmptyRowSetAddsNothing) {
  const std::vector<Case> cases = HermiticityCases("He4");
  const op::Operator z = RunRows(cases.front().in, basis::RowSet(), 1);
  EXPECT_EQ(op::TwoBodyNorm(z), 0.0);
}

}  // namespace
}  // namespace imsrg::commutator::detail
