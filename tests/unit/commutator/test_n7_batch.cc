// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Batch planner and shared pair-third intermediates H and W of the row-tiled
// IMSRG(3)-N7 terms (detail/n7_batch.h, design/14 section 6.4-6.5):
// * the planner against its contract (exact cover, triple alignment, one tile
//   per channel and batch, budget, offsets, determinism);
// * H and W of every tile against dense host matrices (E B and E B E^T with
//   the rows permuted to the sorted order), for Hermitian, anti-Hermitian
//   and non-Hermitian (dense, with the transposed variant) operands, and
//   bitwise equal for any budget;
// * masked operands (review fix M1): NaN on every entry off the lines of a
//   part must not change H and W of the part's batches.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/n7_cost.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/pair_third.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/row_mask.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::commutator::detail {
namespace {

constexpr std::int64_t kGiB = std::int64_t{1} << 30;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::size_t Sz(std::int64_t i) { return static_cast<std::size_t>(i); }

// ---------------------------------------------------------------- planner

/// Total bytes of the tiles of `batches` (sum over the batches).
std::int64_t TotalBytes(const std::vector<LineBatch>& batches,
                        std::int64_t blocks) {
  std::int64_t total = 0;
  for (const LineBatch& b : batches) {
    total += 8 * blocks * (b.h_doubles + b.w_doubles);
  }
  return total;
}

/// True iff the tiles of `a` and `b` are equal field by field.
bool SameBatches(const std::vector<LineBatch>& a,
                 const std::vector<LineBatch>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].num_lines != b[i].num_lines || a[i].h_doubles != b[i].h_doubles ||
        a[i].w_doubles != b[i].w_doubles ||
        a[i].tiles.size() != b[i].tiles.size()) {
      return false;
    }
    for (std::size_t t = 0; t < a[i].tiles.size(); ++t) {
      if (std::memcmp(&a[i].tiles[t], &b[i].tiles[t], sizeof(LineTile)) != 0) {
        // Padding bytes may differ: compare the fields.
        const LineTile& x = a[i].tiles[t];
        const LineTile& y = b[i].tiles[t];
        if (x.channel != y.channel || x.triple_begin != y.triple_begin ||
            x.triple_end != y.triple_end || x.row_begin != y.row_begin ||
            x.row_end != y.row_end || x.state_begin != y.state_begin ||
            x.state_end != y.state_end || x.h_offset != y.h_offset ||
            x.w_offset != y.w_offset) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Checks every contract of `PlanBatches` for `rows` at one budget.
void CheckPlan(const ThreeBodyContext::Impl& im,
               const basis::ThreeBodySpace& space, const basis::RowSet& rows,
               std::int64_t budget, std::int32_t blocks) {
  SCOPED_TRACE("budget " + std::to_string(budget) + " blocks " +
               std::to_string(blocks));
  const std::vector<LineBatch> batches = PlanBatches(im, rows, budget, blocks);
  EXPECT_TRUE(SameBatches(batches, PlanBatches(im, rows, budget, blocks)));
  const op::PairThirdView pt = im.pair_third.host_view();
  std::vector<std::vector<int>> cover;
  for (const basis::KeptChannel3& c : space.channels()) {
    cover.emplace_back(Sz(c.dim), 0);
  }
  for (const LineBatch& b : batches) {
    std::vector<int> channel_seen(Sz(space.num_channels()), 0);
    std::int64_t h = 0;
    std::int64_t w = 0;
    std::int64_t lines = 0;
    for (const LineTile& t : b.tiles) {
      ASSERT_EQ(channel_seen[Sz(t.channel)]++, 0) << "two tiles of a channel";
      const std::int64_t s = pt.ch_num_states[t.channel];
      const std::int64_t k = t.row_end - t.row_begin;
      const std::int64_t m = t.state_end - t.state_begin;
      ASSERT_GE(t.triple_end, t.triple_begin + 1);
      const basis::OrbitTriple& first = space.triples()[Sz(t.triple_begin)];
      const basis::OrbitTriple& last = space.triples()[Sz(t.triple_end - 1)];
      EXPECT_EQ(first.row_begin.value(), t.row_begin);
      EXPECT_EQ(last.row_end.value(), t.row_end);
      EXPECT_EQ(first.channel.value(), t.channel);
      EXPECT_EQ(last.channel.value(), t.channel);
      EXPECT_EQ(pt.tr_state_begin[t.triple_begin], t.state_begin);
      EXPECT_EQ(pt.tr_state_end[t.triple_end - 1], t.state_end);
      // The tile lies inside one range of the set.
      bool inside = false;
      for (const basis::RowRange& r : rows.ranges()) {
        inside = inside || (r.channel.value() == t.channel &&
                            r.triple_begin.value() <= t.triple_begin &&
                            t.triple_end <= r.triple_end.value());
      }
      EXPECT_TRUE(inside);
      EXPECT_EQ(t.h_offset, h);
      EXPECT_EQ(t.w_offset, w);
      h += s * k;
      w += s * m;
      lines += k;
      for (std::int32_t r = t.row_begin; r < t.row_end; ++r) {
        ++cover[Sz(t.channel)][Sz(r)];
      }
      if (t.triple_end - t.triple_begin > 1) {
        EXPECT_LE(8 * blocks * (s * k + s * m), budget);
      }
    }
    EXPECT_EQ(b.h_doubles, h);
    EXPECT_EQ(b.w_doubles, w);
    EXPECT_EQ(b.num_lines, lines);
    if (b.tiles.size() > 1 ||
        (!b.tiles.empty() &&
         b.tiles[0].triple_end - b.tiles[0].triple_begin > 1)) {
      EXPECT_LE(8 * blocks * (b.h_doubles + b.w_doubles), budget);
    }
  }
  for (std::size_t c = 0; c < cover.size(); ++c) {
    for (std::size_t k = 0; k < cover[c].size(); ++k) {
      const bool in = rows.Contains(Channel3Index{static_cast<int>(c)},
                                    basis::KeptRow3Index{static_cast<int>(k)});
      ASSERT_EQ(cover[c][k], in ? 1 : 0) << "ch " << c << " row " << k;
    }
  }
}

/// All budgets of the acceptance test for one row set.
void CheckPlanAllBudgets(const ThreeBodyContext::Impl& im,
                         const basis::ThreeBodySpace& space,
                         const basis::RowSet& rows) {
  for (const std::int32_t blocks : {1, 2}) {
    const std::int64_t total =
        TotalBytes(PlanBatches(im, rows, kGiB, blocks), blocks);
    for (const std::int64_t budget :
         {std::int64_t{1}, std::max<std::int64_t>(1, total / 7), kGiB}) {
      CheckPlan(im, space, rows, budget, blocks);
    }
  }
}

void CheckPlanner(const std::string& ref) {
  SCOPED_TRACE(ref);
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  const auto ms3 = testing::MakeThreeBodySpace(2, ref, e3);
  const ThreeBodyContext ctx(ms3);
  const ThreeBodyContext::Impl& im = ctx.impl();
  const basis::ThreeBodySpace& space = *ms3->three_body_space();
  CheckPlanAllBudgets(im, space, basis::RowSet::Whole(space));
  const basis::RowPartition part =
      basis::RowPartition::Balanced(space, N7LineWeights(*ms3), 4);
  for (std::int32_t p = 0; p < part.num_parts(); ++p) {
    SCOPED_TRACE("part " + std::to_string(p));
    CheckPlanAllBudgets(im, space, part.part(p));
  }
  // The whole-set plan at a tiny budget has one single-triple tile per batch.
  for (const LineBatch& b :
       PlanBatches(im, basis::RowSet::Whole(space), 1, 1)) {
    ASSERT_EQ(b.tiles.size(), 1u);
    EXPECT_EQ(b.tiles[0].triple_end, b.tiles[0].triple_begin + 1);
  }
}

TEST(N7Batch, PlannerHe4) { CheckPlanner("He4"); }
TEST(N7Batch, PlannerO16) { CheckPlanner("O16"); }

// ------------------------------------------------------- H and W vs dense

/// Dense references of one channel: H = E B (S x n, sorted rows) and W =
/// E B E^T (S sorted rows x S pair-third states), column-major.
struct DenseRef {
  std::int32_t n = 0;
  std::int32_t s = 0;
  std::vector<double> h;
  std::vector<double> w;
};

/// Dense matrix of the rows `rows` (flat states of `pt`) of E, S x n
/// column-major.
std::vector<double> DenseE(const op::PairThirdView& pt,
                           const std::vector<std::int32_t>& rows,
                           std::int32_t n) {
  const std::int32_t s = static_cast<std::int32_t>(rows.size());
  std::vector<double> e(Sz(std::int64_t{s} * n), 0.0);
  for (std::int32_t i = 0; i < s; ++i) {
    const std::int32_t f = rows[Sz(i)];
    for (std::int64_t q = pt.e.ptr[f]; q < pt.e.ptr[f + 1]; ++q) {
      e[Sz(i + std::int64_t{pt.e.col[q]} * s)] += pt.e.val[q];
    }
  }
  return e;
}

/// C = A (m x k) B (k x n), all column-major, plain triple loop.
std::vector<double> MatMul(const std::vector<double>& a,
                           const std::vector<double>& b, std::int32_t m,
                           std::int32_t k, std::int32_t n) {
  std::vector<double> c(Sz(std::int64_t{m} * n), 0.0);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t l = 0; l < k; ++l) {
      const double blj = b[Sz(l + std::int64_t{j} * k)];
      if (blj == 0.0) continue;
      for (std::int32_t i = 0; i < m; ++i) {
        c[Sz(i + std::int64_t{j} * m)] += a[Sz(i + std::int64_t{l} * m)] * blj;
      }
    }
  }
  return c;
}

/// Transpose of an m x n column-major matrix.
std::vector<double> Transpose(const std::vector<double>& a, std::int32_t m,
                              std::int32_t n) {
  std::vector<double> t(a.size());
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < m; ++i) {
      t[Sz(j + std::int64_t{i} * n)] = a[Sz(i + std::int64_t{j} * m)];
    }
  }
  return t;
}

/// The dense references of every channel for the operand part `b` (host
/// mirror synced here), `transposed` selects B^T.
std::vector<DenseRef> MakeReference(const ThreeBodyContext::Impl& im,
                                    op::ThreeBody& b, bool transposed) {
  b.SyncToHost();
  const op::PairThirdView pt = im.pair_third.host_view();
  const std::vector<std::int32_t> orig = im.comm232.orig.ToVector();
  std::vector<DenseRef> out(Sz(pt.num_channels));
  for (std::int32_t ch = 0; ch < pt.num_channels; ++ch) {
    DenseRef& r = out[Sz(ch)];
    r.s = pt.ch_num_states[ch];
    r.n = b.dim(Channel3Index{ch});
    if (r.s == 0 || r.n == 0) continue;
    std::vector<double> bd(Sz(std::int64_t{r.n} * r.n));
    for (std::int32_t j = 0; j < r.n; ++j) {
      for (std::int32_t i = 0; i < r.n; ++i) {
        const double v = transposed
                             ? b.get(Channel3Index{ch}, basis::KeptRow3Index{j},
                                     basis::KeptRow3Index{i})
                             : b.get(Channel3Index{ch}, basis::KeptRow3Index{i},
                                     basis::KeptRow3Index{j});
        bd[Sz(i + std::int64_t{j} * r.n)] = v;
      }
    }
    std::vector<std::int32_t> sorted_rows(Sz(r.s));
    std::vector<std::int32_t> flat_rows(Sz(r.s));
    const std::int32_t begin = im.comm232.ch_begin_host[Sz(ch)];
    for (std::int32_t i = 0; i < r.s; ++i) {
      sorted_rows[Sz(i)] = orig[Sz(begin + i)];
      flat_rows[Sz(i)] = pt.ch_state_offset[ch] + i;
    }
    const std::vector<double> es = DenseE(pt, sorted_rows, r.n);
    const std::vector<double> ef = DenseE(pt, flat_rows, r.n);
    r.h = MatMul(es, bd, r.s, r.n, r.n);
    // W = E (B E^T): the other association than the kernels.
    const std::vector<double> bet =
        MatMul(bd, Transpose(ef, r.s, r.n), r.n, r.n, r.s);
    r.w = MatMul(es, bet, r.s, r.n, r.s);
  }
  return out;
}

/// H and W of all channels assembled from the tiles of a set of batches:
/// channel ch holds S x n (H, sorted rows, columns = lines) and S x S (W,
/// columns = PairThirdSpace channel-local states); positions no tile wrote
/// are NaN, `written` flags the others.
struct Assembled {
  std::vector<std::vector<double>> h, w;
  std::vector<std::vector<char>> h_written, w_written;
};

Assembled MakeAssembled(const ThreeBodyContext::Impl& im) {
  const op::PairThirdView pt = im.pair_third.host_view();
  const std::vector<std::int32_t> dim = im.ch_dim.ToVector();
  Assembled a;
  for (std::int32_t ch = 0; ch < pt.num_channels; ++ch) {
    const std::int64_t s = pt.ch_num_states[ch];
    a.h.emplace_back(Sz(s * dim[Sz(ch)]), kNaN);
    a.w.emplace_back(Sz(s * s), kNaN);
    a.h_written.emplace_back(Sz(s * dim[Sz(ch)]), 0);
    a.w_written.emplace_back(Sz(s * s), 0);
  }
  return a;
}

/// Runs `BuildH` and `BuildW` for each batch of `batches` on operand part
/// `op3` (of operator `x`) and scatters the results into `into`. Also checks
/// that every element of the batch buffers was written (the buffers are
/// pre-filled with NaN and no NaN may survive in the tiles).
void RunBatches(const ThreeBodyContext::Impl& im, const op::Operator& x,
                int transposed, const std::vector<LineBatch>& batches,
                Assembled& into) {
  const detail::N7Views v = im.N7(&x, nullptr);
  const op::PairThirdView pt = im.pair_third.device_view();
  const Comm232TablesView tab = im.comm232.View();
  const auto nch = static_cast<std::int32_t>(pt.num_channels);
  for (const LineBatch& batch : batches) {
    const LineBatchDevice dev(batch, nch);
    const LineBatchView view = dev.view();
    exec::DeviceBuffer<double> h(Sz(batch.h_doubles), "test_h");
    exec::DeviceBuffer<double> w(Sz(batch.w_doubles), "test_w");
    h.fill(kNaN);
    w.fill(kNaN);
    BuildH(v.sp, pt, tab, v.x3, transposed, view, h.device_data());
    BuildW(pt, view, h.device_data(), w.device_data());
    h.SyncToHost();
    w.SyncToHost();
    const std::span<const double> hs = h.host_span();
    const std::span<const double> ws = w.host_span();
    for (const LineTile& t : batch.tiles) {
      const std::int64_t s = im.pair_third.num_states(Channel3Index{t.channel});
      const std::int64_t n = im.ch_dim.ToVector()[Sz(t.channel)];
      for (std::int32_t k = t.row_begin; k < t.row_end; ++k) {
        for (std::int64_t i = 0; i < s; ++i) {
          const double val = hs[Sz(t.h_offset + i + (k - t.row_begin) * s)];
          ASSERT_FALSE(std::isnan(val)) << "H not written";
          into.h[Sz(t.channel)][Sz(i + k * s)] = val;
          into.h_written[Sz(t.channel)][Sz(i + k * s)] = 1;
        }
      }
      for (std::int32_t c = t.state_begin; c < t.state_end; ++c) {
        for (std::int64_t i = 0; i < s; ++i) {
          const double val = ws[Sz(t.w_offset + i + (c - t.state_begin) * s)];
          ASSERT_FALSE(std::isnan(val)) << "W not written";
          into.w[Sz(t.channel)][Sz(i + c * s)] = val;
          into.w_written[Sz(t.channel)][Sz(i + c * s)] = 1;
        }
      }
      (void)n;
    }
  }
}

/// Largest |a - b| over the written entries of `got` against `ref`; checks
/// that everything was written (`all` = true) and returns max|ref|.
double MaxDeviation(const std::vector<std::vector<double>>& got,
                    const std::vector<std::vector<char>>& written,
                    const std::vector<std::vector<double>>& ref,
                    double* max_ref) {
  double dev = 0.0;
  for (std::size_t c = 0; c < got.size(); ++c) {
    EXPECT_EQ(got[c].size(), ref[c].size());
    for (std::size_t i = 0; i < got[c].size(); ++i) {
      EXPECT_EQ(written[c][i], 1) << "ch " << c << " element " << i;
      *max_ref = std::max(*max_ref, std::fabs(ref[c][i]));
      dev = std::max(dev, std::fabs(got[c][i] - ref[c][i]));
    }
  }
  return dev;
}

/// True iff the two assembled results are bitwise equal (NaN patterns
/// included).
bool BitwiseEqual(const std::vector<std::vector<double>>& a,
                  const std::vector<std::vector<double>>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t c = 0; c < a.size(); ++c) {
    if (a[c].size() != b[c].size()) return false;
    if (!a[c].empty() &&
        std::memcmp(a[c].data(), b[c].data(), a[c].size() * 8) != 0) {
      return false;
    }
  }
  return true;
}

/// H and W of operand `x` (variant `transposed`) for all three budgets
/// against the dense reference; bitwise equal between the budgets.
void CheckOperand(const ThreeBodyContext& ctx,
                  const basis::ThreeBodySpace& space, op::Operator& x,
                  bool transposed, const std::string& tag) {
  SCOPED_TRACE(tag);
  const ThreeBodyContext::Impl& im = ctx.impl();
  const basis::RowSet whole = basis::RowSet::Whole(space);
  const std::vector<DenseRef> ref =
      MakeReference(im, *x.three_body(), transposed);
  std::vector<std::vector<double>> ref_h;
  std::vector<std::vector<double>> ref_w;
  for (const DenseRef& r : ref) {
    ref_h.push_back(r.h);
    ref_w.push_back(r.w);
  }
  const std::int64_t total = TotalBytes(PlanBatches(im, whole, kGiB, 1), 1);
  Assembled first;
  bool have_first = false;
  for (const std::int64_t budget :
       {std::int64_t{1}, std::max<std::int64_t>(1, total / 7), kGiB}) {
    SCOPED_TRACE("budget " + std::to_string(budget));
    Assembled got = MakeAssembled(im);
    RunBatches(im, x, transposed ? 1 : 0, PlanBatches(im, whole, budget, 1),
               got);
    double max_ref = 1.0;
    const double dh = MaxDeviation(got.h, got.h_written, ref_h, &max_ref);
    const double dw = MaxDeviation(got.w, got.w_written, ref_w, &max_ref);
    std::cout << "[ n7_batch ] " << tag << " budget " << budget << ": max|dH| "
              << dh << " max|dW| " << dw << " (scale " << max_ref << ")\n";
    EXPECT_LE(dh, 1.0e-12 * max_ref);
    EXPECT_LE(dw, 1.0e-12 * max_ref);
    if (!have_first) {
      first = std::move(got);
      have_first = true;
    } else {
      EXPECT_TRUE(BitwiseEqual(first.h, got.h)) << "H depends on the budget";
      EXPECT_TRUE(BitwiseEqual(first.w, got.w)) << "W depends on the budget";
    }
  }
}

void CheckNucleus(const std::string& ref_name, std::uint64_t seed) {
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  const auto ms3 = testing::MakeThreeBodySpace(2, ref_name, e3);
  const ThreeBodyContext ctx(ms3);
  op::Operator h =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, seed);
  op::Operator a = testing::RandomThreeBodyOperator(
      ctx, Hermiticity::kAntiHermitian, seed + 1);
  op::Operator n = testing::RandomThreeBodyOperator(
      ctx, Hermiticity::kNonHermitian, seed + 2);
  ASSERT_TRUE(h.three_body()->packed());
  ASSERT_FALSE(n.three_body()->packed());
  CheckOperand(ctx, *ms3->three_body_space(), h, false, ref_name + " H");
  CheckOperand(ctx, *ms3->three_body_space(), a, false, ref_name + " A");
  CheckOperand(ctx, *ms3->three_body_space(), n, false, ref_name + " N");
  CheckOperand(ctx, *ms3->three_body_space(), n, true,
               ref_name + " N transposed");
}

TEST(N7Batch, HAndWAgainstDenseHe4) { CheckNucleus("He4", 31); }
TEST(N7Batch, HAndWAgainstDenseO16) { CheckNucleus("O16", 37); }

// ---------------------------------------------------------- masked operands

/// For each part of the Balanced P = 4 partition and each batch of it, H and
/// W of the operand with NaN off the part's lines are bitwise those of the
/// unmasked operand (and finite).
void CheckMasked(const std::string& ref_name, Hermiticity herm,
                 std::uint64_t seed) {
  SCOPED_TRACE(ref_name);
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  const auto ms3 = testing::MakeThreeBodySpace(2, ref_name, e3);
  const ThreeBodyContext ctx(ms3);
  const ThreeBodyContext::Impl& im = ctx.impl();
  const basis::ThreeBodySpace& space = *ms3->three_body_space();
  op::Operator x = testing::RandomThreeBodyOperator(ctx, herm, seed);
  const basis::RowPartition part =
      basis::RowPartition::Balanced(space, N7LineWeights(*ms3), 4);
  const std::int64_t total =
      TotalBytes(PlanBatches(im, basis::RowSet::Whole(space), kGiB, 1), 1);
  for (std::int32_t p = 0; p < part.num_parts(); ++p) {
    SCOPED_TRACE("part " + std::to_string(p));
    const basis::RowSet& rows = part.part(p);
    const std::vector<LineBatch> batches =
        PlanBatches(im, rows, std::max<std::int64_t>(1, total / 11), 1);
    op::Operator masked = x;
    masked.three_body()->SyncToHost();
    testing::MaskOutsideLines(*masked.three_body(), rows);
    if (rows.NumRows() > 0) {  // the mask is not vacuous
      const std::span<const double> data =
          masked.three_body()->buffer().host_span();
      EXPECT_TRUE(std::any_of(data.begin(), data.end(),
                              [](double v) { return std::isnan(v); }));
    }
    for (const int transposed : {0, 1}) {
      if (transposed != 0 && masked.three_body()->packed()) continue;
      Assembled want = MakeAssembled(im);
      Assembled got = MakeAssembled(im);
      RunBatches(im, x, transposed, batches, want);
      RunBatches(im, masked, transposed, batches, got);
      EXPECT_TRUE(BitwiseEqual(want.h, got.h))
          << "H, transposed " << transposed;
      EXPECT_TRUE(BitwiseEqual(want.w, got.w))
          << "W, transposed " << transposed;
    }
  }
}

TEST(N7Batch, MaskedOperandHermitianHe4) {
  CheckMasked("He4", Hermiticity::kHermitian, 41);
}
TEST(N7Batch, MaskedOperandAntiHermitianO16) {
  CheckMasked("O16", Hermiticity::kAntiHermitian, 43);
}
TEST(N7Batch, MaskedOperandDenseHe4) {
  CheckMasked("He4", Hermiticity::kNonHermitian, 47);
}

// ----------------------------------------------------- device-side helpers

/// Host-checked helpers (the device pointers are host pointers on the host
/// backends; skipped otherwise): `FindSortedRow` finds exactly the states of
/// the tables, and `BatchColumn` agrees with the tiles.
TEST(N7Batch, FindSortedRowAndBatchColumn) {
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  const auto ms3 = testing::MakeThreeBodySpace(2, "O16", e3);
  const ThreeBodyContext ctx(ms3);
  const ThreeBodyContext::Impl& im = ctx.impl();
  if (!im.comm232.orig.host_is_device()) GTEST_SKIP() << "device backend";
  const op::PairThirdView pt = im.pair_third.device_view();
  const Comm232TablesView tab = im.comm232.View();
  const double* occ = ms3->reference().occupations().data();
  const std::int32_t norb = tab.norb;
  for (std::int32_t ch = 0; ch < tab.num_channels; ++ch) {
    const std::int32_t begin = tab.ch_begin[ch];
    const std::int32_t end = tab.ch_begin[ch + 1];
    std::vector<std::int32_t> expect(
        Sz(std::int64_t{norb} * norb * tab.nj * norb), -1);
    const auto at = [&](int p, int q, int jpq, int r) {
      return Sz(((std::int64_t{p} * norb + q) * tab.nj + jpq) * norb + r);
    };
    for (std::int32_t g = begin; g < end; ++g) {
      const int p = tab.key[g] / norb;
      const int q = tab.key[g] % norb;
      const int jpq = pt.st_two_jpq[tab.orig[g]] / 2;
      expect[at(p, q, jpq, im.comm232.third_host[Sz(g)])] = g - begin;
    }
    for (int p = 0; p < norb; ++p) {
      for (int q = p; q < norb; ++q) {
        for (int jpq = 0; jpq < tab.nj; ++jpq) {
          for (int r = 0; r < norb; ++r) {
            ASSERT_EQ(FindSortedRow(tab, occ, ch, p, q, jpq, r),
                      expect[at(p, q, jpq, r)])
                << "ch " << ch << " (" << p << " " << q << ") J" << jpq << " r"
                << r;
          }
        }
      }
    }
  }
  const basis::ThreeBodySpace& space = *ms3->three_body_space();
  const std::vector<LineBatch> batches =
      PlanBatches(im, basis::RowSet::Whole(space), 20000, 1);
  ASSERT_GT(batches.size(), 1u);
  for (const LineBatch& batch : batches) {
    const LineBatchDevice dev(batch, tab.num_channels);
    const LineBatchView view = dev.view();
    for (std::int32_t ch = 0; ch < tab.num_channels; ++ch) {
      const LineTile* tile = nullptr;
      for (const LineTile& t : batch.tiles) {
        if (t.channel == ch) tile = &t;
      }
      std::vector<std::int64_t> offsets;
      for (std::int32_t g = tab.ch_begin[ch]; g < tab.ch_begin[ch + 1]; ++g) {
        const std::int64_t col = BatchColumn(view, tab, pt, ch, g);
        const std::int32_t t = tab.orig[g] - pt.ch_state_offset[ch];
        if (tile == nullptr || t < tile->state_begin || t >= tile->state_end) {
          ASSERT_EQ(col, -1);
        } else {
          ASSERT_EQ(col, tile->w_offset + std::int64_t{t - tile->state_begin} *
                                              pt.ch_num_states[ch]);
          offsets.push_back(col);
        }
      }
      const std::int64_t expected_cols =
          tile == nullptr ? 0 : tile->state_end - tile->state_begin;
      std::sort(offsets.begin(), offsets.end());
      ASSERT_EQ(static_cast<std::int64_t>(offsets.size()), expected_cols);
      ASSERT_TRUE(std::adjacent_find(offsets.begin(), offsets.end()) ==
                  offsets.end());
    }
  }
}

}  // namespace
}  // namespace imsrg::commutator::detail
