// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// `N7CostModel` and `N7LineWeights` (n7_cost.h): the counted pair-third
// states, active states and R entries per triple against the real
// `op::PairThirdSpace` and `Comm232Tables` at emax 2 and 3, the exact counts
// of emax 3 / E3max 9, the line weights and the batch scratch.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/n7_cost.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/pair_third.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::commutator {
namespace {

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

std::shared_ptr<const basis::ModelSpace> Space(int emax, int e3max,
                                               const std::string& ref) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  return testing::MakeThreeBodySpace(emax, ref, cuts);
}

/// Occupation class 3 rule of `PairClass` applied to a real state.
bool ActiveState(const std::span<const double>& occ, std::int32_t p,
                 std::int32_t q) {
  const double np = occ[U(p)];
  const double nq = occ[U(q)];
  return (np != 0.0 && nq != 0.0) || (np != 1.0 && nq != 1.0);
}

/// Checks the counted states, active states and R entries of every triple
/// against the real pair-third space and the comm232ss tables.
void CompareWithRealSpace(int emax, int e3max, const std::string& ref) {
  const auto ms = Space(emax, e3max, ref);
  const ThreeBodyContext ctx(ms);
  const op::PairThirdSpace& pt = ctx.impl().pair_third;
  const detail::Comm232Tables& t232 = ctx.impl().comm232;
  const basis::ThreeBodySpace& space = *ms->three_body_space();
  const op::PairThirdView v = pt.host_view();
  const auto occ = ms->reference().occupations();
  const N7CostModel model(*ms);

  ASSERT_EQ(model.num_triples(), space.num_triples());
  EXPECT_EQ(model.num_states(), pt.num_states());
  std::int64_t active_total = 0;
  std::int64_t nnz_model = 0;
  std::int64_t nnz_real = 0;
  for (std::int32_t t = 0; t < space.num_triples(); ++t) {
    const basis::Triple3Index ti{t};
    const basis::OrbitTriple& ot = space.triple(ti);
    const N7TripleCost& c = model.triples()[U(t)];
    const std::int32_t s0 = pt.state_begin(ti);
    const std::int32_t s1 = pt.state_end(ti);
    EXPECT_EQ(c.states, s1 - s0) << "triple " << t;
    EXPECT_EQ(c.rows, ot.row_end.value() - ot.row_begin.value());
    const std::int32_t base = pt.state_offset(ot.channel);
    std::int32_t active = 0;
    for (std::int32_t s = s0; s < s1; ++s) {
      if (ActiveState(occ, v.st_p[base + s], v.st_q[base + s])) ++active;
    }
    EXPECT_EQ(c.active, active) << "triple " << t;
    active_total += active;
    const std::int64_t row0 = space.FlatRow(ot.channel, ot.row_begin);
    const std::int64_t nnz = v.r.ptr[row0 + c.rows] - v.r.ptr[row0];
    // Upper bound: R drops the coefficients that vanish (|v| < 1e-13).
    EXPECT_GE(c.nnz_r, nnz)
        << "triple " << t << " (" << ot.a.value() << "," << ot.b.value() << ","
        << ot.c.value() << ") 2J " << space.channel(ot.channel).two_j.twice;
    nnz_model += c.nnz_r;
    nnz_real += nnz;
  }
  EXPECT_EQ(model.NumActive(), active_total);
  EXPECT_LE(static_cast<double>(nnz_model - nnz_real),
            0.02 * static_cast<double>(nnz_real));
  for (std::int32_t ch = 0; ch < space.num_channels(); ++ch) {
    const N7ChannelCost& cc = model.channels()[U(ch)];
    EXPECT_EQ(cc.states, pt.num_states(Channel3Index{ch}));
    if (cc.states > 0) {
      EXPECT_EQ(cc.active, t232.ch_active_host[U(ch)]) << "channel " << ch;
    }
  }
}

TEST(N7CostTest, MatchesPairThirdSpaceEmax2) {
  CompareWithRealSpace(2, 6, "O16");
  CompareWithRealSpace(2, 6, "He4");
}

TEST(N7CostTest, MatchesPairThirdSpaceEmax3) {
  CompareWithRealSpace(3, 9, "O16");
}

TEST(N7CostTest, ExactCountsEmax3E3max9) {
  const auto ms = Space(3, 9, "O16");
  const N7CostModel model(*ms);
  EXPECT_EQ(model.num_rows(), 13223);
  EXPECT_EQ(model.PackedDoubles(), 2261446);
  EXPECT_EQ(model.num_states(), 36564);
  EXPECT_EQ(model.MaxDim(), 554);
}

TEST(N7CostTest, TermFormulasAtEmax2) {
  const auto ms = Space(2, 6, "O16");
  const N7CostModel model(*ms);
  const auto tot = model.TermTotals();
  // 330: 2 mem_t summed over the triples is twice the packed size.
  EXPECT_DOUBLE_EQ(tot[static_cast<std::size_t>(N7Term::k330)],
                   2.0 * static_cast<double>(model.PackedDoubles()));
  double sum = 0.0;
  for (const double x : tot) sum += x;
  EXPECT_DOUBLE_EQ(sum, model.TotalMultiplyAdds());
  const auto parts = model.Comm223Parts();
  EXPECT_NEAR(parts[0] + parts[1], tot[static_cast<std::size_t>(N7Term::k223)],
              1.0e-9 * (parts[0] + parts[1]));
  // Per channel: the W build is 2 S (sum_t k_t m_t).
  for (const N7TripleCost& c : model.triples()) {
    EXPECT_GE(c.Total(), 0.0);
  }
}

TEST(N7CostTest, LineWeights) {
  const auto ms = Space(3, 9, "O16");
  const basis::LineWeights w = N7LineWeights(*ms, 0.0);
  const basis::ThreeBodySpace& space = *ms->three_body_space();
  ASSERT_EQ(w.compute.size(), U(space.num_triples()));
  ASSERT_EQ(w.memory.size(), U(space.num_triples()));
  double mem = 0.0;
  double compute = 0.0;
  for (std::size_t t = 0; t < w.memory.size(); ++t) {
    EXPECT_GE(w.compute[t], 0.0);
    mem += w.memory[t];
    compute += w.compute[t];
  }
  EXPECT_DOUBLE_EQ(mem, 2261446.0);
  const N7CostModel model(*ms);
  EXPECT_NEAR(compute, model.TotalMultiplyAdds(), 1.0e-9 * compute);
}

TEST(N7CostTest, LineWeightsRowOverhead) {
  const auto ms = Space(3, 9, "O16");
  const basis::LineWeights w0 = N7LineWeights(*ms, 0.0);
  const basis::LineWeights w1 = N7LineWeights(*ms, 1000.0);
  const basis::LineWeights wd = N7LineWeights(*ms);
  const N7CostModel model(*ms);
  ASSERT_EQ(w1.compute.size(), w0.compute.size());
  for (std::size_t t = 0; t < w0.compute.size(); ++t) {
    // The per-row term: overhead x rows of the triple; memory is unchanged.
    EXPECT_NEAR(w1.compute[t] - w0.compute[t], 1000.0 * model.triples()[t].rows,
                1.0e-9 * w1.compute[t]);
    EXPECT_DOUBLE_EQ(w1.memory[t], w0.memory[t]);
    EXPECT_NEAR(wd.compute[t] - w0.compute[t],
                kRowOverheadMadd * model.triples()[t].rows,
                1.0e-9 * wd.compute[t] + 1.0e-12);
  }
  EXPECT_THROW(N7LineWeights(*ms, -1.0), imsrg::InvariantError);
}

TEST(N7CostTest, BatchScratch) {
  const auto ms = Space(2, 6, "O16");
  const N7CostModel model(*ms);
  // One batch per channel: S n + S S + n (n + 1) / 2, largest over channels.
  std::int64_t expect = 0;
  for (const N7ChannelCost& c : model.channels()) {
    const std::int64_t n = c.dim;
    expect =
        std::max(expect, c.states * n + c.states * c.states + n * (n + 1) / 2);
  }
  const N7BatchScratch all = model.BatchScratch(model.MaxDim());
  EXPECT_EQ(all.max_doubles, expect);
  // Smaller batches never need more scratch.
  const N7BatchScratch small = model.BatchScratch(8);
  EXPECT_LE(small.max_doubles, all.max_doubles);
  EXPECT_GE(small.max_lines, 1);
  EXPECT_THROW(model.BatchScratch(0), imsrg::InvariantError);
}

}  // namespace
}  // namespace imsrg::commutator
