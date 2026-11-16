// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The batch loop of the row-tiled N7 terms (detail::ForEachN7Batch) and the
// partition/masking test support (tests/support/n7_partition_support.h),
// exercised with the [2,2]->3 rows entry point, whose partition invariance
// is bitwise.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "tests/support/n7_partition_support.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::commutator::detail {
namespace {

/// A zeroed rank-3 output with the hermiticity of [X, Y].
op::Operator ZeroOutput(const op::Operator& x, const op::Operator& y,
                        const ThreeBodyContext& ctx) {
  op::OperatorSpec spec = CommutatorSpec(x.spec(), y.spec());
  spec.particle_rank = 3;
  return op::Operator(ctx.model_space_ptr(), spec);
}

/// Visits every batch of the whole space with W requested and checks the
/// inputs: lines cover the space once, the intermediates of present parts
/// are set, and the transposed blocks follow the storage mode.
void CheckLoop(const ThreeBodyContext& ctx, const op::Operator& x,
               const op::Operator& y, std::int64_t budget) {
  op::Operator z = ZeroOutput(x, y, ctx);
  const N7Views v = PrepareN7("loop", x, y, z, ctx);
  const basis::ThreeBodySpace& s = *ctx.model_space_ptr()->three_body_space();
  std::int64_t lines = 0;
  std::int32_t batches = 0;
  ForEachN7Batch(v, ctx, basis::RowSet::Whole(s), budget, N7Needs{false, true},
                 [&](const N7BatchInputs& in) {
                   lines += in.batch.num_lines;
                   ++batches;
                   EXPECT_NE(in.hx, nullptr);
                   EXPECT_NE(in.wy, nullptr);
                   const bool dense_x = v.x3.packed == 0;
                   EXPECT_EQ(in.hxt == in.hx, !dense_x);
                   EXPECT_EQ(in.wxt == in.wx, !dense_x);
                   EXPECT_EQ(in.sxt, dense_x ? 1.0 : v.x3.sign);
                 });
  EXPECT_EQ(lines, s.NumRows());
  EXPECT_GE(batches, 1);
}

TEST(N7BatchLoop, VisitsEveryLineOnceWithTheIntermediates) {
  const auto ms = testing::MakeThreeBodySpace(2, "He4", {.e3max = 6});
  const ThreeBodyContext ctx(ms);
  const Hermiticity kinds[] = {Hermiticity::kHermitian,
                               Hermiticity::kAntiHermitian,
                               Hermiticity::kNonHermitian};
  for (const Hermiticity hx : kinds) {
    const op::Operator x = testing::RandomThreeBodyOperator(ctx, hx, 3);
    const op::Operator y =
        testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 4);
    for (const std::int64_t budget : {std::int64_t{1}, kN7BatchBytes}) {
      CheckLoop(ctx, x, y, budget);
    }
  }
}

/// The partition support on the [2,2]->3 rows entry point (bitwise).
void CheckComm223Partitions(int emax, int e3max, const std::string& ref) {
  const auto ms = testing::MakeThreeBodySpace(emax, ref, {.e3max = e3max});
  const ThreeBodyContext ctx(ms);
  const op::Operator x =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 5);
  const op::Operator y =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 6);
  const auto zero = [&]() { return ZeroOutput(x, y, ctx); };
  const testing::N7RowsFn run = [&](const op::Operator& a,
                                    const op::Operator& b, op::Operator& z,
                                    const basis::RowSet& rows) {
    AddComm223Rows(a, b, *z.three_body(), ctx, rows);
  };
  testing::ExpectPartitionInvariance(*ms, x, y, zero, run, 0.0, true,
                                     "comm223 " + ref);
}

TEST(N7BatchLoop, PartitionSupportOnComm223He4) {
  CheckComm223Partitions(2, 6, "He4");
}

TEST(N7BatchLoop, PartitionSupportOnComm223O16) {
  CheckComm223Partitions(2, 6, "O16");
}

}  // namespace
}  // namespace imsrg::commutator::detail
