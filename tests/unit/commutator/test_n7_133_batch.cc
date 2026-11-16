// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The row-tiled comm133ss ([X1, Y3] + [X3, Y1] -> Z3, kernels/comm133.cc):
// * the pair-third batch path (detail::Comm133Rows) against the element-wise
//   reference (detail::Comm133ssReference) in the form max|dZ| <= 1e-12
//   max(1, max|Z|) of design/14 section 6.5 (deviations printed): emax 1
//   and 2, He4 and O16 at E3max 3 emax, a fractional-occupation space and a
//   dE3max-cut space, operand pairs (A, H), (H, H), (H, A) (and (A, A) from
//   emax 2), operands without a three-body part, several batch budgets, and
//   accumulation into Z;
// * the same for three-body parts off the physical subspace of the linearly
//   dependent canonical kets (random and cubically distorted elements, as the
//   element-wise atan generator of the IMSRG(3) flow produces them);
// * the dispatch of the public term: packed Z through the batch path, dense
//   Z (a non-Hermitian operand) through the reference;
// * partition invariance with masked operands
//   (tests/support/n7_partition_support.h).

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
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
/// A budget that splits the emax-2 spaces into several batches (the whole
/// space needs about 22 MB of H and W for two operand blocks).
constexpr std::int64_t kMidBudget = std::int64_t{2} << 20;

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

/// The whole-space batch path with `budget` against the reference for X, Y;
/// prints and returns the deviation. Expects a nonzero term.
double BatchVsReference(const ThreeBodyContext& ctx, const op::Operator& x,
                        const op::Operator& y, std::int64_t budget,
                        const std::string& tag) {
  op::Operator ref = ZeroOutput(x, y, ctx);
  Comm133ssReference(x, y, ref, ctx);
  op::Operator z = ZeroOutput(x, y, ctx);
  Comm133Rows(x, y, z, ctx,
              basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
              budget);
  const double norm = op::ThreeBodyNorm(ref);
  EXPECT_GT(norm, 1.0e-3) << tag;
  const double dev = Deviation(z, ref);
  std::printf("[133 batch] %s budget %" PRId64
              ": max|dZ| / max(1, max|Z|) = %.3e (|Z|_3 = %.4g)\n",
              tag.c_str(), budget, dev, norm);
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

/// The operand pairs (X, Y) of definite hermiticity: (A, H), (H, H), (H, A),
/// and with `with_aa` (A, A). [A, A] has a 133 term only if an anti-Hermitian
/// one-body part acts in the space: not at emax 1 (one orbit per one-body
/// channel, so it vanishes), and not in the dE3max-cut O16 space at emax 2
/// (it couples only 0s and 1s there, and no kept triple has a 0s orbit).
std::vector<std::pair<Hermiticity, Hermiticity>> Pairs(bool with_aa) {
  std::vector<std::pair<Hermiticity, Hermiticity>> out = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}};
  if (with_aa) {
    out.emplace_back(Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian);
  }
  return out;
}

/// The batch path against the reference on every space of `Spaces(emax)`,
/// for every pair of `Pairs` and each budget of `budgets`.
void CheckAgainstReference(int emax, const std::vector<std::int64_t>& budgets) {
  std::uint64_t seed = 300;
  for (const SpaceCase& c : Spaces(emax)) {
    ASSERT_GT(c.ms->three_body_space()->NumRows(), 0) << c.name;
    const ThreeBodyContext ctx(c.ms);
    const bool cut = c.name.find("dE3max") != std::string::npos;
    for (const auto& [hx, hy] : Pairs(emax >= 2 && !cut)) {
      const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
      const op::Operator y =
          testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
      seed += 2;
      const std::string tag =
          c.name + " (" + Letter(hx) + ", " + Letter(hy) + ")";
      for (const std::int64_t budget : budgets) {
        EXPECT_LE(BatchVsReference(ctx, x, y, budget, tag), kTol) << tag;
      }
    }
  }
}

TEST(N7Comm133Batch, MatchesReferenceEmax1) {
  // Budget 1: every triple is a batch of its own.
  CheckAgainstReference(1, {1, kMidBudget, kN7BatchBytes});
}

TEST(N7Comm133Batch, MatchesReferenceEmax2) {
  CheckAgainstReference(2, {kMidBudget, kN7BatchBytes});
}

/// A copy of `x` (which has a three-body part) whose three-body part is off
/// the physical subspace of the linearly dependent canonical kets of triples
/// with equal orbits: `random` replaces every stored element (i >= j) by a
/// uniform value in [-1, 1] (generator seeded with `seed`; the
/// anti-Hermitian diagonal stays 0), otherwise v -> v + 0.3 v^3 (odd, so the
/// hermiticity holds; a stand-in for the element-wise atan of the IMSRG(3)
/// generator). Synced to the device.
op::Operator Unphysical(const op::Operator& x, bool random,
                        std::uint64_t seed) {
  op::Operator u = x;
  op::ThreeBody& t = *u.three_body();
  t.SyncToHost();
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const bool anti = t.hermiticity() == Hermiticity::kAntiHermitian;
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    const Channel3Index ch{c};
    for (std::int32_t i = 0; i < t.dim(ch); ++i) {
      for (std::int32_t j = 0; j <= i; ++j) {
        if (anti && i == j) continue;
        const basis::KeptRow3Index ri{i};
        const basis::KeptRow3Index rj{j};
        const double v = t.get(ch, ri, rj);
        t.set(ch, ri, rj, random ? dist(gen) : v + 0.3 * v * v * v);
      }
    }
  }
  t.SyncToDevice();
  return u;
}

// The batch path reads exactly the (ordered) three-body elements the
// reference reads, so it matches it also for three-body parts off the
// physical subspace, which the IMSRG(3) flow produces through the
// element-wise atan generator (a formulation through the comm223 recoupling
// map agreed only on physical operands and made the IT-9 O16 emax-3 flow take
// an extra step; kernels/comm133.cc).
TEST(N7Comm133Batch, MatchesReferenceOnUnphysicalOperands) {
  std::uint64_t seed = 500;
  for (const int emax : {1, 2}) {
    for (const char* ref : {"He4", "O16"}) {
      const auto ms = testing::MakeThreeBodySpace(emax, ref, {.e3max = 9});
      const ThreeBodyContext ctx(ms);
      for (const auto& [hx, hy] : Pairs(emax >= 2)) {
        const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
        const op::Operator y =
            testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
        seed += 2;
        for (const bool random : {true, false}) {
          const std::string tag = std::string(ref) + " emax " +
                                  std::to_string(emax) + " (" + Letter(hx) +
                                  ", " + Letter(hy) + ")" +
                                  (random ? " random" : " cubic");
          const op::Operator xu = Unphysical(x, random, seed);
          const op::Operator yu = Unphysical(y, random, seed + 7);
          EXPECT_LE(BatchVsReference(ctx, xu, y, kMidBudget, tag + " X3"),
                    kTol);
          EXPECT_LE(BatchVsReference(ctx, x, yu, kMidBudget, tag + " Y3"),
                    kTol);
          EXPECT_LE(BatchVsReference(ctx, xu, yu, kMidBudget, tag + " both"),
                    kTol);
        }
      }
    }
  }
}

// The dE3max cut must drop kets, or the cut case tests nothing new.
TEST(N7Comm133Batch, CutSpaceDropsKets) {
  for (const int emax : {1, 2}) {
    const std::vector<SpaceCase> s = Spaces(emax);
    EXPECT_LT(s[3].ms->three_body_space()->NumRows(),
              s[1].ms->three_body_space()->NumRows())
        << "emax " << emax;
  }
}

// An operand without a three-body part contributes only through its
// one-body part: [X1, Y3] alone and [X3, Y1] alone.
TEST(N7Comm133Batch, MissingThreeBodyParts) {
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
  EXPECT_LE(BatchVsReference(ctx, x2, y3, kMidBudget, "O16 (A2, H3)"), kTol);
  EXPECT_LE(BatchVsReference(ctx, x3, y2, kMidBudget, "O16 (A3, H2)"), kTol);
  // Two rank-2 operands: nothing is added.
  op::Operator z = ZeroOutput(x2, y2, ctx);
  Comm133Rows(x2, y2, z, ctx, basis::RowSet::Whole(*ms->three_body_space()),
              kN7BatchBytes);
  EXPECT_EQ(op::ThreeBodyNorm(z), 0.0);
}

// The term adds into Z: a second call doubles the contribution (up to the
// order of the bra and ket additions).
TEST(N7Comm133Batch, AddsIntoZ) {
  const auto ms = testing::MakeThreeBodySpace(2, "He4", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 51);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 52);
  op::Operator once = ZeroOutput(x, y, ctx);
  terms::Comm133ss(x, y, once, ctx);
  op::Operator twice = ZeroOutput(x, y, ctx);
  terms::Comm133ss(x, y, twice, ctx);
  terms::Comm133ss(x, y, twice, ctx);
  op::Operator doubled = once;
  doubled *= 2.0;
  EXPECT_LE(Deviation(twice, doubled), kTol);
}

// The public term: packed Z = the batch path with the default budget
// (bitwise), dense Z (a non-Hermitian operand) = the reference (bitwise),
// which the brute-force and imsrg-devel tests of test_n7_133_132.cc cover.
TEST(N7Comm133Batch, PublicTermDispatch) {
  const auto ms = testing::MakeThreeBodySpace(2, "O16", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const basis::RowSet whole = basis::RowSet::Whole(*ms->three_body_space());
  const Hermiticity kinds[] = {Hermiticity::kAntiHermitian,
                               Hermiticity::kHermitian,
                               Hermiticity::kNonHermitian};
  std::uint64_t seed = 60;
  for (const Hermiticity hx : kinds) {
    for (const Hermiticity hy : kinds) {
      const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
      const op::Operator y =
          testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
      seed += 2;
      const std::string tag =
          std::string("(") + Letter(hx) + ", " + Letter(hy) + ")";
      op::Operator z = ZeroOutput(x, y, ctx);
      terms::Comm133ss(x, y, z, ctx);
      op::Operator expect = ZeroOutput(x, y, ctx);
      const bool dense =
          hx == Hermiticity::kNonHermitian || hy == Hermiticity::kNonHermitian;
      if (dense) {
        Comm133ssReference(x, y, expect, ctx);
        op::Operator zr = ZeroOutput(x, y, ctx);
        EXPECT_THROW(Comm133Rows(x, y, zr, ctx, whole, kN7BatchBytes),
                     NotImplementedError)
            << tag;
      } else {
        Comm133Rows(x, y, expect, ctx, whole, kN7BatchBytes);
      }
      EXPECT_TRUE(testing::CompareOperators(z, expect).bitwise_equal) << tag;
      EXPECT_GT(op::ThreeBodyNorm(z), 1.0e-3) << tag;
    }
  }
}

/// Partition invariance and masked operands of Comm133Rows (mid budget) for
/// the (A, H) and (H, H) pairs on `ref` at emax 2, E3max 6.
void CheckPartitions(const std::string& ref) {
  const auto ms = testing::MakeThreeBodySpace(2, ref, {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const std::pair<Hermiticity, Hermiticity> pairs[] = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kHermitian}};
  std::uint64_t seed = 70;
  for (const auto& [hx, hy] : pairs) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, seed);
    const op::Operator y = testing::RandomThreeBodyOperator(ctx, hy, seed + 1);
    seed += 2;
    const auto zero = [&]() { return ZeroOutput(x, y, ctx); };
    const testing::N7RowsFn run = [&](const op::Operator& a,
                                      const op::Operator& b, op::Operator& z,
                                      const basis::RowSet& rows) {
      Comm133Rows(a, b, z, ctx, rows, kMidBudget);
    };
    testing::ExpectPartitionInvariance(
        *ms, x, y, zero, run, kTol, false,
        "comm133 " + ref + " (" + Letter(hx) + ", " + Letter(hy) + ")");
  }
}

TEST(N7Comm133Batch, PartitionInvarianceHe4) { CheckPartitions("He4"); }

TEST(N7Comm133Batch, PartitionInvarianceO16) { CheckPartitions("O16"); }

}  // namespace
}  // namespace imsrg::commutator::detail
