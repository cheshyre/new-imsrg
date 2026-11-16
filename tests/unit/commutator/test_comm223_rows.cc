// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// `detail::AddComm223Rows`: the [2,2]->3 term restricted to the stored rows of
// a `basis::RowSet`. Partition invariance (the parts of a partition, added
// into one zeroed Z, reproduce `AddComm223` bit for bit) and locality (a part
// alone writes only its rows), for packed Hermitian, packed anti-Hermitian
// and dense Z.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/n7_cost.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/row_mask.h"
#include "tests/support/three_body_test_support.h"
#include "tests/unit/commutator/n7_test_support.h"

namespace imsrg::commutator {
namespace {

using basis::RowRange;
using basis::RowSet;
using basis::Triple3Index;
using testing::N7Inputs;

/// A partition into `parts` shares of contiguous global triple blocks (a
/// block is cut at channel boundaries into one range per channel).
std::vector<RowSet> ContiguousParts(const basis::ThreeBodySpace& s,
                                    std::int32_t parts) {
  const std::int32_t nt = s.num_triples();
  std::vector<RowSet> out;
  for (std::int32_t p = 0; p < parts; ++p) {
    const std::int32_t lo =
        static_cast<std::int32_t>(static_cast<std::int64_t>(p) * nt / parts);
    const std::int32_t hi = static_cast<std::int32_t>(
        static_cast<std::int64_t>(p + 1) * nt / parts);
    std::vector<RowRange> r;
    std::int32_t t = lo;
    while (t < hi) {
      const basis::OrbitTriple& first = s.triple(Triple3Index{t});
      const Channel3Index ch = first.channel;
      const std::int32_t end = std::min(hi, s.channel(ch).triple_end.value());
      const basis::OrbitTriple& last = s.triple(Triple3Index{end - 1});
      r.push_back(RowRange{ch, Triple3Index{t}, Triple3Index{end},
                           first.row_begin, last.row_end});
      t = end;
    }
    out.push_back(RowSet::FromRanges(s, std::move(r)));
  }
  return out;
}

/// The parts of `RowPartition::Balanced` with the N7 line weights.
std::vector<RowSet> BalancedParts(const basis::ModelSpace& ms,
                                  std::int32_t parts) {
  const basis::RowPartition p = basis::RowPartition::Balanced(
      *ms.three_body_space(), N7LineWeights(ms), parts);
  std::vector<RowSet> out;
  for (std::int32_t i = 0; i < p.num_parts(); ++i) out.push_back(p.part(i));
  return out;
}

/// True iff the host buffers of `a` and `b` are equal bit for bit.
bool BitwiseEqual(const op::ThreeBody& a, const op::ThreeBody& b) {
  a.SyncToHost();
  b.SyncToHost();
  const auto sa = a.host_span();
  const auto sb = b.host_span();
  return sa.size() == sb.size() &&
         std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(double)) == 0;
}

/// Z = AddComm223 reference; then each of `parts` alone writes only its rows
/// and all parts added into one zeroed Z give the reference bit for bit.
void CheckPartition(const N7Inputs& in, const std::vector<RowSet>& parts,
                    const std::string& tag) {
  SCOPED_TRACE(tag);
  const Hermiticity h = in.x.hermiticity() == Hermiticity::kNonHermitian ||
                                in.y.hermiticity() == Hermiticity::kNonHermitian
                            ? Hermiticity::kNonHermitian
                        : in.x.hermiticity() == in.y.hermiticity()
                            ? Hermiticity::kAntiHermitian
                            : Hermiticity::kHermitian;
  op::ThreeBody ref(in.ms3, h);
  detail::AddComm223(in.x, in.y, ref, *in.ctx);
  op::ThreeBody sum(in.ms3, h);
  for (const RowSet& rows : parts) {
    op::ThreeBody alone(in.ms3, h);
    detail::AddComm223Rows(in.x, in.y, alone, *in.ctx, rows);
    EXPECT_TRUE(testing::OnlyRowsNonzero(alone, rows));
    detail::AddComm223Rows(in.x, in.y, sum, *in.ctx, rows);
  }
  EXPECT_TRUE(BitwiseEqual(ref, sum));
}

/// All operand hermiticity combinations at one model space.
void CheckSpace(int emax, int e3max, const std::string& ref) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  const auto ms = testing::MakeThreeBodySpace(emax, ref, cuts);
  const std::pair<Hermiticity, Hermiticity> combos[] = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kHermitian},
      {Hermiticity::kNonHermitian, Hermiticity::kHermitian}};
  std::uint64_t seed = 5;
  for (const auto& [hx, hy] : combos) {
    const N7Inputs in = testing::RandomN7Inputs(ms, hx, hy, seed++);
    const basis::ThreeBodySpace& s = *ms->three_body_space();
    const std::string tag = ref + " emax " + std::to_string(emax) + " herm " +
                            std::to_string(static_cast<int>(hx)) + "," +
                            std::to_string(static_cast<int>(hy));
    for (const std::int32_t p : {2, 3, 7}) {
      CheckPartition(in, ContiguousParts(s, p),
                     tag + " contiguous " + std::to_string(p));
    }
    for (const std::int32_t p : {4, 16}) {
      CheckPartition(in, BalancedParts(*ms, p),
                     tag + " balanced " + std::to_string(p));
    }
    CheckPartition(in, {RowSet::Whole(s)}, tag + " whole");
  }
}

TEST(Comm223RowsTest, PartitionInvarianceHe4Emax2) { CheckSpace(2, 6, "He4"); }

TEST(Comm223RowsTest, PartitionInvarianceO16Emax2) { CheckSpace(2, 6, "O16"); }

TEST(Comm223RowsTest, PartitionInvarianceO16Emax3) { CheckSpace(3, 9, "O16"); }

}  // namespace
}  // namespace imsrg::commutator
