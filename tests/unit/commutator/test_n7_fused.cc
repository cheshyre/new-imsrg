// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The fused N7 batch driver (detail::RunN7, n7_batches.cc) and its dispatch
// from Commutator():
// * all seven terms in one batch loop against the seven single-term
//   `terms::` calls in imsrg++ order (133, 330, 223, 232, 331, 231, 132) into
//   a zeroed Z, in the form max|dZ| <= 1e-12 max(1, max|Z|) over Z0..Z3 of
//   design/30 section 2 (never element-wise relative; deviations printed):
//   He4 and O16 at emax 1 and 2 (E3max 3 emax), a fractional-occupation space
//   and a dE3max-cut space, operand pairs (H, A), (A, H), (A, A) and (N, H)
//   (dense Z), budgets of one triple per batch and the default;
// * a single selected term against its `terms::` call and its single-term
//   batch loop (bitwise: the batch plans coincide);
// * operands without a three-body part (the early returns of the terms);
// * partition invariance with masked operands
//   (tests/support/n7_partition_support.h) of all seven terms, and of the
//   other six terms into a dense Z;
// * the checks: an empty selection, N8/N9 terms, a dense Z3 with 133 on a
//   part of the space, a missing output part;
// * Commutator() with TermMask::Imsrg3N7() (or an N7 subset) = its IMSRG(2)
//   part (three-body gate closed) + RunN7.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/n7_partition_support.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::commutator::detail {
namespace {

/// The design/30 section 2 tolerance of a reordered sum against the original.
constexpr double kTol = 1.0e-12;
/// A budget below the size of a single triple: every triple is a batch of its
/// own (the smallest batches, one line or a few per batch).
constexpr std::int64_t kOneTriple = 1;
/// A budget that splits the emax-2 spaces into several batches.
constexpr std::int64_t kMidBudget = std::int64_t{2} << 20;
/// The seven N7 terms.
constexpr TermMask kAllN7 = TermMask::Imsrg3N7().ThreeBodyTerms();

/// A single-term function of the public API.
using TermFn = void (*)(const op::Operator&, const op::Operator&, op::Operator&,
                        const ThreeBodyContext&);
/// A single-term batch loop (`Comm<term>Rows`).
using RowsFn = void (*)(const op::Operator&, const op::Operator&, op::Operator&,
                        const ThreeBodyContext&, const basis::RowSet&,
                        std::int64_t);

/// AddComm223Rows in the form of the other single-term batch loops (223
/// plans no batches, the budget is unused).
void Comm223RowsForTest(const op::Operator& x, const op::Operator& y,
                        op::Operator& z, const ThreeBodyContext& ctx,
                        const basis::RowSet& rows, std::int64_t /*budget*/) {
  AddComm223Rows(x, y, *z.three_body(), ctx, rows);
}

/// An N7 term with its label, public function and single-term batch loop.
struct N7Term {
  Term term;
  const char* label;
  TermFn pub;
  RowsFn rows;
};

/// The seven N7 terms in imsrg++'s call order (Commutator.cc:403-430).
const N7Term kCallOrder[] = {
    {Term::kComm133ss, "133", terms::Comm133ss, Comm133Rows},
    {Term::kComm330ss, "330", terms::Comm330ss, Comm330Rows},
    {Term::kComm223ss, "223", terms::Comm223ss, Comm223RowsForTest},
    {Term::kComm232ss, "232", terms::Comm232ss, Comm232Rows},
    {Term::kComm331ss, "331", terms::Comm331ss, Comm331Rows},
    {Term::kComm231ss, "231", terms::Comm231ss, Comm231Rows},
    {Term::kComm132ss, "132", terms::Comm132ss, Comm132Rows}};
static_assert(sizeof(kCallOrder) / sizeof(kCallOrder[0]) == kNumN7Terms);

/// A zeroed rank-3 output with the hermiticity of [X, Y].
op::Operator ZeroOutput(const op::Operator& x, const op::Operator& y,
                        const ThreeBodyContext& ctx) {
  op::OperatorSpec spec = CommutatorSpec(x.spec(), y.spec());
  spec.particle_rank = 3;
  return op::Operator(ctx.model_space_ptr(), spec);
}

/// The whole kept space of the context.
basis::RowSet Whole(const ThreeBodyContext& ctx) {
  return basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space());
}

/// The seven `terms::` calls in imsrg++ order into a zeroed Z.
op::Operator PerTerm(const op::Operator& x, const op::Operator& y,
                     const ThreeBodyContext& ctx) {
  op::Operator z = ZeroOutput(x, y, ctx);
  for (const N7Term& t : kCallOrder) t.pub(x, y, z, ctx);
  return z;
}

/// RunN7 of `terms` on the whole space with `budget` into a zeroed Z.
op::Operator Fused(const op::Operator& x, const op::Operator& y,
                   const ThreeBodyContext& ctx, TermMask terms,
                   std::int64_t budget) {
  op::Operator z = ZeroOutput(x, y, ctx);
  RunN7(x, y, z, ctx, terms, Whole(ctx), budget);
  return z;
}

/// max|z - ref| / max(1, max|ref|) over Z0..Z3; prints it with `tag` and
/// whether z and ref are bitwise equal.
double Deviation(const op::Operator& z, const op::Operator& ref,
                 const std::string& tag) {
  const testing::OperatorDiff d = testing::CompareOperators(z, ref);
  const double dev = d.max_abs_diff / std::max(1.0, d.max_abs_ref);
  std::printf("[fused] %s: max|dZ| / max(1, max|Z|) = %.3e (%s, max|Z| %.4g)\n",
              tag.c_str(), dev, d.bitwise_equal ? "bitwise" : "not bitwise",
              d.max_abs_ref);
  return dev;
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
  const std::string e = " emax " + std::to_string(emax);
  std::vector<SpaceCase> out;
  out.push_back({"He4" + e, testing::MakeThreeBodySpace(emax, "He4", full)});
  out.push_back({"O16" + e, testing::MakeThreeBodySpace(emax, "O16", full)});
  out.push_back({"fractional He4" + e,
                 testing::AddThreeBodySpace(
                     testing::FractionalModelSpace(emax, "He4"), full)});
  out.push_back(
      {"O16 dE3max 2" + e, testing::MakeThreeBodySpace(emax, "O16", cut)});
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

/// "(X, Y)" for the labels.
std::string PairLabel(Hermiticity hx, Hermiticity hy) {
  return std::string("(") + Letter(hx) + ", " + Letter(hy) + ")";
}

/// The operand pairs (X, Y) of the comparisons: Hermitian Z from (H, A) and
/// (A, H), anti-Hermitian Z from (A, A), and a dense (general) Z from (N, H).
const std::pair<Hermiticity, Hermiticity> kPairs[] = {
    {Hermiticity::kHermitian, Hermiticity::kAntiHermitian},
    {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
    {Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian},
    {Hermiticity::kNonHermitian, Hermiticity::kHermitian}};

/// RunN7 with all seven terms against the per-term sequence on every space of
/// `Spaces(emax)`, every pair of `kPairs`, budgets of one triple per batch and
/// the default.
void CheckFusedAgainstPerTerm(int emax) {
  std::uint64_t seed = 700 + 100 * static_cast<std::uint64_t>(emax);
  // Cases with a nonzero Z0, Z1, Z2. They vanish where the space has no
  // particle orbit (O16 at emax 1, also with the cut), Z0 also without a
  // channel holding an all-hole and an all-particle triple (He4 at emax 1),
  // and by symmetry (an anti-Hermitian Z0, an anti-Hermitian Z1 at emax 1),
  // so the parts are counted over the cases instead of asserted per case
  // (at emax 1: Z0 in 3 cases, Z1 and Z2 in 6 and 8).
  int nonzero[3] = {0, 0, 0};
  for (const SpaceCase& c : Spaces(emax)) {
    ASSERT_GT(c.ms->three_body_space()->NumRows(), 0) << c.name;
    const ThreeBodyContext ctx(c.ms);
    for (const auto& [hx, hy] : kPairs) {
      const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
      const op::Operator y =
          testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
      seed += 2;
      const std::string tag = c.name + " " + PairLabel(hx, hy);
      const op::Operator ref = PerTerm(x, y, ctx);
      std::printf("[fused] %s: Z0 %.4g, |Z1| %.4g, |Z2| %.4g, |Z3| %.4g\n",
                  tag.c_str(), ref.zero_body(), op::OneBodyNorm(ref),
                  op::TwoBodyNorm(ref), op::ThreeBodyNorm(ref));
      EXPECT_GT(op::ThreeBodyNorm(ref), 1.0e-3) << tag;  // 223 at least
      nonzero[0] += std::abs(ref.zero_body()) > 1.0e-6 ? 1 : 0;
      nonzero[1] += op::OneBodyNorm(ref) > 1.0e-3 ? 1 : 0;
      nonzero[2] += op::TwoBodyNorm(ref) > 1.0e-3 ? 1 : 0;
      for (const std::int64_t budget : {kOneTriple, kN7BatchBytes}) {
        const op::Operator z = Fused(x, y, ctx, kAllN7, budget);
        EXPECT_LE(Deviation(z, ref,
                            "all seven vs per-term " + tag + " budget " +
                                std::to_string(budget)),
                  kTol)
            << tag << " budget " << budget;
      }
    }
  }
  for (int part = 0; part < 3; ++part) {
    EXPECT_GE(nonzero[part], 3) << "emax " << emax << ": too few cases with "
                                << "a nonzero Z" << part;
  }
}

TEST(N7Fused, MatchesPerTermSequenceEmax1) { CheckFusedAgainstPerTerm(1); }

TEST(N7Fused, MatchesPerTermSequenceEmax2) { CheckFusedAgainstPerTerm(2); }

// A single selected term reproduces its `terms::` call bitwise (both run the
// same batch plan: the default budget, the whole space and operands with
// every part), including the dense-Z path of 133 (Comm133ssReference), and
// its single-term batch loop on a part of the space with a budget that
// splits it into several batches (same plan again).
TEST(N7Fused, SingleTermMatchesItsTerm) {
  std::uint64_t seed = 900;
  int nonzero[kNumN7Terms] = {};  // cases in which the term is nonzero
  for (const int emax : {1, 2}) {
    for (const char* ref : {"He4", "O16"}) {
      const auto ms = testing::MakeThreeBodySpace(emax, ref, {.e3max = 9});
      const ThreeBodyContext ctx(ms);
      const std::vector<basis::RowSet> halves =
          testing::ContiguousParts(*ms->three_body_space(), 2);
      for (const auto& [hx, hy] : kPairs) {
        const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
        const op::Operator y =
            testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
        seed += 2;
        const bool dense = hx == Hermiticity::kNonHermitian;
        for (int i = 0; i < kNumN7Terms; ++i) {
          const N7Term& t = kCallOrder[i];
          const std::string tag = std::string(ref) + " emax " +
                                  std::to_string(emax) + " " +
                                  PairLabel(hx, hy) + " " + t.label;
          op::Operator expect = ZeroOutput(x, y, ctx);
          t.pub(x, y, expect, ctx);
          const double size =
              testing::CompareOperators(expect, ZeroOutput(x, y, ctx))
                  .max_abs_diff;
          nonzero[i] += size > 1.0e-6 ? 1 : 0;
          const op::Operator z =
              Fused(x, y, ctx, TermMask::Only(t.term), kN7BatchBytes);
          EXPECT_TRUE(testing::CompareOperators(z, expect).bitwise_equal)
              << tag << " vs terms::";
          // Dense Z: 133 runs only on the whole space (checked below).
          if (dense && t.term == Term::kComm133ss) continue;
          op::Operator expect_rows = ZeroOutput(x, y, ctx);
          t.rows(x, y, expect_rows, ctx, halves[1], kMidBudget);
          op::Operator z_rows = ZeroOutput(x, y, ctx);
          RunN7(x, y, z_rows, ctx, TermMask::Only(t.term), halves[1],
                kMidBudget);
          EXPECT_TRUE(
              testing::CompareOperators(z_rows, expect_rows).bitwise_equal)
              << tag << " vs its batch loop on the second half";
        }
      }
    }
  }
  // Every term is exercised (16 cases each; 330 vanishes for an
  // anti-Hermitian Z, at emax 1 132 vanishes, 133 for (A, A), and every term
  // but 133 and 223 for O16).
  for (int i = 0; i < kNumN7Terms; ++i) {
    std::printf("[fused] single term %s: nonzero in %d of 16 cases\n",
                kCallOrder[i].label, nonzero[i]);
    EXPECT_GE(nonzero[i], 6) << kCallOrder[i].label;
  }
}

// An operand without a three-body part: every term does what its single-term
// function does (the early returns), with one batch loop for the rest.
TEST(N7Fused, MissingThreeBodyParts) {
  for (const char* ref : {"He4", "O16"}) {
    const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
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
    struct Case {
      const op::Operator* x;
      const op::Operator* y;
      const char* label;
    };
    const Case cases[] = {
        {&x2, &y3, "(A2, H3)"}, {&x3, &y2, "(A3, H2)"}, {&x2, &y2, "(A2, H2)"}};
    for (const Case& c : cases) {
      const op::Operator& x = *c.x;
      const op::Operator& y = *c.y;
      const std::string tag = std::string(ref) + " " + c.label;
      const op::Operator want = PerTerm(x, y, ctx);
      EXPECT_GT(op::ThreeBodyNorm(want), 1.0e-3) << tag;  // 223 at least
      for (const std::int64_t budget : {kMidBudget, kN7BatchBytes}) {
        EXPECT_LE(Deviation(Fused(x, y, ctx, kAllN7, budget), want,
                            "missing 3B " + tag + " budget " +
                                std::to_string(budget)),
                  kTol)
            << tag;
      }
    }
  }
}

/// Partition invariance and masked operands of RunN7 with `terms` (mid
/// budget) for the pairs `pairs` on `ref` at emax 2, E3max 6.
void CheckPartitions(
    const std::string& ref, TermMask terms,
    const std::vector<std::pair<Hermiticity, Hermiticity>>& pairs,
    const std::string& what, bool ownership = false) {
  const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  std::uint64_t seed = 70;
  for (const auto& [hx, hy] : pairs) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
    const op::Operator y = testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
    seed += 2;
    const auto zero = [&]() { return ZeroOutput(x, y, ctx); };
    const testing::N7RowsFn run = [&](const op::Operator& a,
                                      const op::Operator& b, op::Operator& z,
                                      const basis::RowSet& rows) {
      RunN7(a, b, z, ctx, terms, rows, kMidBudget);
    };
    testing::ExpectPartitionInvariance(
        *ms, x, y, zero, run, kTol, false,
        "RunN7 " + what + " " + ref + " " + PairLabel(hx, hy),
        {.ownership = ownership});
  }
}

TEST(N7Fused, PartitionInvarianceHe4) {
  CheckPartitions("He4", kAllN7,
                  {{Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
                   {Hermiticity::kHermitian, Hermiticity::kHermitian}},
                  "all seven");
}

TEST(N7Fused, PartitionInvarianceO16) {
  CheckPartitions("O16", kAllN7,
                  {{Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
                   {Hermiticity::kHermitian, Hermiticity::kHermitian}},
                  "all seven");
}

// A dense Z (non-Hermitian operand): every term but 133, which needs the
// whole space for a dense Z, is partition invariant (223 on dense rows).
TEST(N7Fused, PartitionInvarianceDenseZ) {
  TermMask without133 = kAllN7;
  without133.reset(Term::kComm133ss);
  CheckPartitions("O16", without133,
                  {{Hermiticity::kNonHermitian, Hermiticity::kHermitian}},
                  "all but 133", true);
}

// Output ownership (design/14 section 6.5): without 133 (whose transposed
// ket side lands in other rows), Z3 holds only 223, and a part writes Z3
// only in its own rows, in every partition.
TEST(N7Fused, OutputOwnershipWithout133) {
  TermMask without133 = kAllN7;
  without133.reset(Term::kComm133ss);
  for (const char* ref : {"He4", "O16"}) {
    CheckPartitions(ref, without133,
                    {{Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
                     {Hermiticity::kHermitian, Hermiticity::kHermitian}},
                    "all but 133", true);
  }
}

// The checks: an empty selection does nothing, N8/N9 terms throw, 133 into a
// dense Z3 needs the whole space, the output parts the terms write must
// exist.
TEST(N7Fused, Checks) {
  const auto ms = testing::MakeThreeBodySpace(1, "He4", {.e3max = 3});
  const ThreeBodyContext ctx(ms);
  const op::Operator a =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 81);
  const op::Operator h =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 82);
  const op::Operator n =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kNonHermitian, 83);
  const basis::RowSet whole = Whole(ctx);
  const std::vector<basis::RowSet> halves =
      testing::ContiguousParts(*ms->three_body_space(), 2);

  // Empty N7 selections (the IMSRG(2) terms are ignored).
  for (const TermMask m : {TermMask::None(), TermMask::Imsrg2()}) {
    op::Operator z = ZeroOutput(a, h, ctx);
    RunN7(a, h, z, ctx, m, whole, kN7BatchBytes);
    EXPECT_TRUE(
        testing::CompareOperators(z, ZeroOutput(a, h, ctx)).bitwise_equal);
  }
  // The IMSRG(2) terms of a mask are ignored: Imsrg3N7 = the seven terms.
  EXPECT_TRUE(testing::CompareOperators(
                  Fused(a, h, ctx, TermMask::Imsrg3N7(), kN7BatchBytes),
                  Fused(a, h, ctx, kAllN7, kN7BatchBytes))
                  .bitwise_equal);
  // N8/N9 terms are not implemented.
  {
    op::Operator z = ZeroOutput(a, h, ctx);
    EXPECT_THROW(
        RunN7(a, h, z, ctx, TermMask::Imsrg3Full(), whole, kN7BatchBytes),
        NotImplementedError);
    EXPECT_THROW(RunN7(a, h, z, ctx, TermMask::Only(Term::kComm233PpHhss),
                       whole, kN7BatchBytes),
                 NotImplementedError);
  }
  // 133 into a dense Z3: the whole space only (InvariantError before any
  // work); the other terms run on a part of the space.
  {
    op::Operator z = ZeroOutput(n, h, ctx);
    EXPECT_THROW(RunN7(n, h, z, ctx, kAllN7, halves[0], kN7BatchBytes),
                 InvariantError);
    EXPECT_TRUE(
        testing::CompareOperators(z, ZeroOutput(n, h, ctx)).bitwise_equal);
    TermMask without133 = kAllN7;
    without133.reset(Term::kComm133ss);
    EXPECT_NO_THROW(RunN7(n, h, z, ctx, without133, halves[0], kN7BatchBytes));
    EXPECT_GT(op::ThreeBodyNorm(z), 0.0);
  }
  // A Z without a three-body part fails the checks of 133 and 223, as their
  // single-term functions do.
  {
    op::Operator z2 =
        op::Operator(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
    for (const Term t : {Term::kComm133ss, Term::kComm223ss}) {
      EXPECT_THROW(
          RunN7(a, h, z2, ctx, TermMask::Only(t), whole, kN7BatchBytes),
          InvariantError)
          << static_cast<int>(t);
    }
    // The other terms do not need Z3.
    TermMask no_z3 = kAllN7;
    no_z3.reset(Term::kComm133ss).reset(Term::kComm223ss);
    EXPECT_NO_THROW(RunN7(a, h, z2, ctx, no_z3, whole, kN7BatchBytes));
  }
  // Z with the wrong hermiticity (PrepareN7).
  {
    op::Operator z = ZeroOutput(a, a, ctx);  // anti-Hermitian, [A, H] is H
    EXPECT_THROW(RunN7(a, h, z, ctx, kAllN7, whole, kN7BatchBytes),
                 InvariantError);
  }
}

/// Commutator() with `terms` against its IMSRG(2) part (the same call with
/// the three-body gate closed, which leaves a rank-3 Z with a zero
/// three-body part) plus RunN7 of `terms` on the whole space: bitwise.
void CheckDispatch(const op::Operator& x, const op::Operator& y, Workspace& ws,
                   TermMask terms, const std::string& tag) {
  CommutatorOptions o;
  o.terms = terms;
  op::Operator full;
  Commutator(x, y, full, o, ws);
  CommutatorOptions gated = o;
  gated.threebody_threshold = 1.0e9;
  op::Operator expect;
  Commutator(x, y, expect, gated, ws);
  ASSERT_EQ(op::ThreeBodyNorm(expect), 0.0) << tag;
  const ThreeBodyContext& ctx = *ws.three_body_context();
  RunN7(x, y, expect, ctx, terms, Whole(ctx), kN7BatchBytes);
  EXPECT_GT(op::ThreeBodyNorm(full), 1.0e-3) << tag;
  EXPECT_TRUE(testing::CompareOperators(full, expect).bitwise_equal) << tag;
}

TEST(N7Fused, CommutatorDispatch) {
  std::uint64_t seed = 950;
  for (const char* ref : {"He4", "O16"}) {
    const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
    Workspace ws(ms);
    ASSERT_NE(ws.three_body_context(), nullptr);
    const ThreeBodyContext& ctx = *ws.three_body_context();
    for (const auto& [hx, hy] : kPairs) {
      const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
      const op::Operator y =
          testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
      seed += 2;
      const std::string tag = std::string(ref) + " " + PairLabel(hx, hy);
      CheckDispatch(x, y, ws, TermMask::Imsrg3N7(), tag + " Imsrg3N7");
      // A subset of the N7 terms (one of each kind of output).
      const TermMask subset =
          TermMask::Imsrg2() | TermMask::Only(Term::kComm223ss) |
          TermMask::Only(Term::kComm132ss) | TermMask::Only(Term::kComm331ss);
      CheckDispatch(x, y, ws, subset, tag + " subset");
    }
  }
}

}  // namespace
}  // namespace imsrg::commutator::detail
