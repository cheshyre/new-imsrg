// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Batch planner, device upload of the line tiles, the batch loop of the
// row-tiled IMSRG(3)-N7 terms and their fused driver `RunN7`
// (detail/n7_batch.h, detail/n7_terms.h). Host code.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/options.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

namespace {

/// Bytes of the intermediates of a tile of a channel with `s` states, `k`
/// rows and `m` states of its triples, for `blocks` operand blocks.
std::int64_t TileBytes(std::int64_t s, std::int64_t k, std::int64_t m,
                       std::int64_t blocks) {
  return 8 * blocks * s * (k + m);
}

/// Host tables the planner reads: the pair-third state ranges and the kept
/// row ranges of the triples.
struct PlanTables {
  op::PairThirdView pt;
  std::vector<std::int32_t> row_begin;  // per triple, channel-local
  std::vector<std::int32_t> row_end;
};

/// The batch under construction.
class BatchBuilder {
 public:
  BatchBuilder(std::int64_t budget, std::int64_t blocks, std::int32_t nch)
      : budget_(budget),
        blocks_(blocks),
        has_tile_(static_cast<std::size_t>(nch), false) {}

  /// True iff the current batch has no tile.
  bool empty() const { return cur_.tiles.empty(); }
  /// True iff the current batch already has a tile of `ch`.
  bool HasChannel(std::int32_t ch) const {
    return has_tile_[static_cast<std::size_t>(ch)];
  }
  /// Bytes still free in the current batch.
  std::int64_t Free() const {
    return budget_ - 8 * blocks_ * (cur_.h_doubles + cur_.w_doubles);
  }

  /// Appends `tile` (its offsets are set here) of a channel with `s` states.
  void Add(LineTile tile, std::int64_t s) {
    const std::int64_t k = tile.row_end - tile.row_begin;
    const std::int64_t m = tile.state_end - tile.state_begin;
    tile.h_offset = cur_.h_doubles;
    tile.w_offset = cur_.w_doubles;
    cur_.h_doubles += s * k;
    cur_.w_doubles += s * m;
    cur_.num_lines += k;
    has_tile_[static_cast<std::size_t>(tile.channel)] = true;
    cur_.tiles.push_back(tile);
  }

  /// Closes the current batch (if nonempty) and starts a new one.
  void Flush() {
    if (cur_.tiles.empty()) return;
    for (const LineTile& t : cur_.tiles) {
      has_tile_[static_cast<std::size_t>(t.channel)] = false;
    }
    done_.push_back(std::move(cur_));
    cur_ = LineBatch();
  }

  /// Closes the last batch and returns all batches.
  std::vector<LineBatch> Finish() {
    Flush();
    return std::move(done_);
  }

 private:
  std::int64_t budget_;
  std::int64_t blocks_;
  std::vector<bool> has_tile_;
  LineBatch cur_;
  std::vector<LineBatch> done_;
};

/// The largest `te` in [tb, end] such that the tile [tb, te) of a channel
/// with `s` states fits `free_bytes` (`tb` if not even one triple fits).
std::int32_t FitTriples(const PlanTables& pl, std::int32_t tb, std::int32_t end,
                        std::int64_t s, std::int64_t blocks,
                        std::int64_t free_bytes) {
  std::int32_t te = tb;
  while (te < end) {
    const auto last = static_cast<std::size_t>(te);
    const auto first = static_cast<std::size_t>(tb);
    const std::int64_t k = pl.row_end[last] - pl.row_begin[first];
    const std::int64_t m = pl.pt.tr_state_end[te] - pl.pt.tr_state_begin[tb];
    if (TileBytes(s, k, m, blocks) > free_bytes) break;
    ++te;
  }
  return te;
}

/// The tile of the triples [tb, te) of channel `ch`.
LineTile MakeTile(const PlanTables& pl, std::int32_t ch, std::int32_t tb,
                  std::int32_t te) {
  LineTile tile;
  tile.channel = ch;
  tile.triple_begin = tb;
  tile.triple_end = te;
  tile.row_begin = pl.row_begin[static_cast<std::size_t>(tb)];
  tile.row_end = pl.row_end[static_cast<std::size_t>(te - 1)];
  tile.state_begin = pl.pt.tr_state_begin[tb];
  tile.state_end = pl.pt.tr_state_end[te - 1];
  return tile;
}

}  // namespace

std::vector<LineBatch> PlanBatches(const ThreeBodyContext::Impl& im,
                                   const basis::RowSet& rows,
                                   std::int64_t budget_bytes,
                                   std::int32_t operand_blocks) {
  IMSRG_CHECK(operand_blocks >= 1, "PlanBatches: operand_blocks < 1");
  IMSRG_CHECK(budget_bytes >= 0, "PlanBatches: negative budget");
  PlanTables pl;
  pl.pt = im.pair_third.host_view();
  pl.row_begin = im.tr_row_begin.ToVector();
  pl.row_end = im.tr_row_end.ToVector();
  BatchBuilder bb(budget_bytes, operand_blocks, pl.pt.num_channels);
  for (const basis::RowRange& range : rows.ranges()) {
    const std::int32_t ch = range.channel.value();
    const std::int64_t s = pl.pt.ch_num_states[ch];
    std::int32_t tb = range.triple_begin.value();
    const std::int32_t end = range.triple_end.value();
    while (tb < end) {
      if (bb.HasChannel(ch)) bb.Flush();
      std::int32_t te = FitTriples(pl, tb, end, s, operand_blocks, bb.Free());
      if (te == tb) {
        if (!bb.empty()) {
          bb.Flush();
          continue;
        }
        te = tb + 1;  // a single triple over the budget: a batch of its own
      }
      bb.Add(MakeTile(pl, ch, tb, te), s);
      tb = te;
    }
  }
  return bb.Finish();
}

LineBatchDevice::LineBatchDevice(const LineBatch& batch,
                                 std::int32_t num_channels)
    : num_tiles_(static_cast<std::int32_t>(batch.tiles.size())),
      num_lines_(batch.num_lines),
      h_doubles_(batch.h_doubles),
      w_doubles_(batch.w_doubles) {
  std::vector<std::int32_t> of_channel(static_cast<std::size_t>(num_channels),
                                       -1);
  for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
    const std::int32_t ch = batch.tiles[i].channel;
    IMSRG_CHECK(ch >= 0 && ch < num_channels,
                "LineBatchDevice: channel out of range");
    of_channel[static_cast<std::size_t>(ch)] = static_cast<std::int32_t>(i);
  }
  tiles_ = exec::DeviceBuffer<LineTile>(batch.tiles.size(),
                                        "commutator::n7_batch_tiles");
  if (!batch.tiles.empty()) tiles_.CopyFromHost(batch.tiles);
  tile_of_channel_ = exec::DeviceBuffer<std::int32_t>(
      of_channel.size(), "commutator::n7_batch_tile_of_channel");
  if (!of_channel.empty()) tile_of_channel_.CopyFromHost(of_channel);
}

LineBatchDevice::LineBatchDevice(const LineBatchDevice& other) = default;
LineBatchDevice& LineBatchDevice::operator=(const LineBatchDevice& other) =
    default;
LineBatchDevice::LineBatchDevice(LineBatchDevice&& other) noexcept = default;
LineBatchDevice& LineBatchDevice::operator=(LineBatchDevice&& other) noexcept =
    default;
LineBatchDevice::~LineBatchDevice() = default;

LineBatchView LineBatchDevice::view() const noexcept {
  LineBatchView v;
  v.tiles = tiles_.device_data();
  v.tile_of_channel = tile_of_channel_.device_data();
  v.num_tiles = num_tiles_;
  v.num_lines = num_lines_;
  v.h_doubles = h_doubles_;
  v.w_doubles = w_doubles_;
  return v;
}

namespace {

/// The H and W buffers of one three-body operand (plain and, for a dense
/// part, transposed), allocated for the largest batch.
struct OperandBlocks {
  exec::DeviceBuffer<double> h;
  exec::DeviceBuffer<double> ht;
  exec::DeviceBuffer<double> w;
  exec::DeviceBuffer<double> wt;
};

/// Allocates the blocks of a part `b` for the sizes `h_max`, `w_max` (none
/// for a missing part or an intermediate that is not needed).
OperandBlocks AllocateBlocks(const ThreeBodyPartView& b, N7Needs needs,
                             std::int64_t h_max, std::int64_t w_max) {
  OperandBlocks o;
  if (b.local == nullptr || !(needs.h || needs.w)) return o;
  const bool dense = b.packed == 0;
  const auto hn = static_cast<std::size_t>(h_max);
  const auto wn = static_cast<std::size_t>(w_max);
  o.h = exec::DeviceBuffer<double>(hn, "commutator::n7_h");
  if (dense) o.ht = exec::DeviceBuffer<double>(hn, "commutator::n7_ht");
  if (needs.w) {
    o.w = exec::DeviceBuffer<double>(wn, "commutator::n7_w");
    if (dense) o.wt = exec::DeviceBuffer<double>(wn, "commutator::n7_wt");
  }
  return o;
}

/// Builds the intermediates of part `b` for `batch` into `o` and returns the
/// pointers {h, ht, w, wt} and the sign of the transposed blocks through the
/// out parameters (null for what is not built).
void BuildBlocks(const N7Views& v, const op::PairThirdView& pt,
                 const Comm232TablesView& tab, const ThreeBodyPartView& b,
                 N7Needs needs, const LineBatchView& batch, OperandBlocks& o,
                 const double*& h, const double*& ht, const double*& w,
                 const double*& wt, double& sign_t) {
  h = ht = w = wt = nullptr;
  sign_t = 1.0;
  if (b.local == nullptr || !(needs.h || needs.w)) return;
  const bool dense = b.packed == 0;
  BuildH(v.sp, pt, tab, b, 0, batch, o.h.device_data());
  if (dense) BuildH(v.sp, pt, tab, b, 1, batch, o.ht.device_data());
  h = o.h.device_data();
  ht = dense ? o.ht.device_data() : h;
  sign_t = dense ? 1.0 : static_cast<double>(b.sign);
  if (!needs.w) return;
  BuildW(pt, batch, o.h.device_data(), o.w.device_data());
  if (dense) BuildW(pt, batch, o.ht.device_data(), o.wt.device_data());
  w = o.w.device_data();
  wt = dense ? o.wt.device_data() : w;
}

}  // namespace

void ForEachN7Batch(const N7Views& v, const ThreeBodyContext& ctx,
                    const basis::RowSet& rows, std::int64_t budget_bytes,
                    N7Needs needs,
                    const std::function<void(const N7BatchInputs&)>& body) {
  const ThreeBodyContext::Impl& im = ctx.impl();
  if (needs.w) needs.h = true;
  const auto blocks_of = [](const ThreeBodyPartView& b) {
    return b.local == nullptr ? 0 : (b.packed == 0 ? 2 : 1);
  };
  const std::int32_t blocks = std::max(1, blocks_of(v.x3) + blocks_of(v.y3));
  const std::vector<LineBatch> batches =
      PlanBatches(im, rows, budget_bytes, blocks);
  std::int64_t h_max = 0;
  std::int64_t w_max = 0;
  for (const LineBatch& b : batches) {
    h_max = std::max(h_max, b.h_doubles);
    w_max = std::max(w_max, b.w_doubles);
  }
  OperandBlocks xb = AllocateBlocks(v.x3, needs, h_max, w_max);
  OperandBlocks yb = AllocateBlocks(v.y3, needs, h_max, w_max);
  N7BatchInputs in;
  in.v = v;
  in.pt = im.pair_third.device_view();
  in.tab = im.comm232.View();
  in.ctx = &ctx;
  for (const LineBatch& b : batches) {
    const LineBatchDevice dev(b, v.sp.num_channels);
    in.batch = dev.view();
    in.plan = &b;
    BuildBlocks(v, in.pt, in.tab, v.x3, needs, in.batch, xb, in.hx, in.hxt,
                in.wx, in.wxt, in.sxt);
    BuildBlocks(v, in.pt, in.tab, v.y3, needs, in.batch, yb, in.hy, in.hyt,
                in.wy, in.wyt, in.syt);
    body(in);
  }
}

namespace {

/// An N7 term and the name of its single-term function (check messages).
struct N7TermName {
  Term term;
  const char* name;
};

/// The N7 terms in imsrg++'s call order (Commutator.cc:403-430).
constexpr N7TermName kN7CallOrder[] = {
    {Term::kComm133ss, "Comm133ss"}, {Term::kComm330ss, "Comm330ss"},
    {Term::kComm223ss, "Comm223ss"}, {Term::kComm232ss, "Comm232ss"},
    {Term::kComm331ss, "Comm331ss"}, {Term::kComm231ss, "Comm231ss"},
    {Term::kComm132ss, "Comm132ss"}};

/// The name of the first term of `kN7CallOrder` in both `terms` and `among`
/// (the single-term function that would make a check first), or "RunN7".
const char* FirstTermName(TermMask terms, TermMask among) {
  for (const N7TermName& t : kN7CallOrder) {
    if (terms.test(t.term) && among.test(t.term)) return t.name;
  }
  return "RunN7";
}

/// The line offsets of the `TransposeOut` of `plan` (its layout): num_lines +
/// 1 entries, line l (row L) at [off[l], off[l] + L + 1), the last entry the
/// total size. O(lines).
std::vector<std::int64_t> TransposeLineOffsets(const LineBatch& plan) {
  std::vector<std::int64_t> off;
  off.reserve(static_cast<std::size_t>(plan.num_lines) + 1);
  std::int64_t n = 0;
  for (const LineTile& t : plan.tiles) {
    for (std::int32_t L = t.row_begin; L < t.row_end; ++L) {
      off.push_back(n);
      n += L + 1;
    }
  }
  off.push_back(n);
  return off;
}

/// `in` without the three-body part of X (`drop_x3`) and/or Y (`drop_y3`):
/// view, intermediates and sign, as they reach the kernel of a single-term
/// loop that removes the part from its views before the batch loop
/// (`Comm232Rows`, `Comm132Rows`). O(1).
N7BatchInputs WithoutParts(N7BatchInputs in, bool drop_x3, bool drop_y3) {
  if (drop_x3) {
    in.v.x3 = ThreeBodyPartView{};
    in.hx = in.hxt = in.wx = in.wxt = nullptr;
    in.sxt = 1.0;
  }
  if (drop_y3) {
    in.v.y3 = ThreeBodyPartView{};
    in.hy = in.hyt = in.wy = in.wyt = nullptr;
    in.syt = 1.0;
  }
  return in;
}

}  // namespace

void RunN7(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
           const ThreeBodyContext& ctx, TermMask terms,
           const basis::RowSet& rows, std::int64_t budget_bytes) {
  const TermMask n7_all = TermMask::Imsrg3N7().ThreeBodyTerms();
  if ((terms.ThreeBodyTerms() | n7_all) != n7_all) {
    throw NotImplementedError(
        "RunN7: only the N7 IMSRG(3) terms are implemented (the N8/N9 terms "
        "of the full IMSRG(3) are not)");
  }
  const TermMask n7 = terms & n7_all;
  if (n7.empty()) return;
  const auto selected = [n7](Term t) { return n7.test(t); };

  // The checks of the single-term functions, once and before any work.
  const N7Views v = PrepareN7(FirstTermName(n7, n7_all), X, Y, Z, ctx);
  const TermMask z3_terms =
      TermMask::Only(Term::kComm133ss) | TermMask::Only(Term::kComm223ss);
  const TermMask z2_terms =
      TermMask::Only(Term::kComm232ss) | TermMask::Only(Term::kComm132ss);
  ThreeBodyOutView z3;
  if (!(n7 & z3_terms).empty()) {
    z3 = ThreeBodyOut(FirstTermName(n7, z3_terms), Z, ctx);
  }
  TwoBodyOutView z2;
  if (!(n7 & z2_terms).empty()) {
    z2 = TwoBodyOut(FirstTermName(n7, z2_terms), Z, ctx);
  }
  const bool dense133 = selected(Term::kComm133ss) && z3.packed == 0;
  IMSRG_CHECK(
      !dense133 || rows.NumRows() ==
                       ctx.model_space_ptr()->three_body_space()->NumRows(),
      "RunN7: comm133ss into a dense (non-Hermitian) Z3 runs only on "
      "the whole kept space (Comm133ssReference)");

  // The terms that contribute: the early returns of the single-term loops.
  const bool x3 = v.x3.local != nullptr;
  const bool y3 = v.y3.local != nullptr;
  const bool do133 = selected(Term::kComm133ss) && !dense133 && (x3 || y3);
  // [X2, Y3] needs X2, [Y2, X3] needs Y2; [X1, Y3] needs X1, [X3, Y1] Y1.
  const bool drop232_x3 = v.y2.data == nullptr;
  const bool drop232_y3 = v.x2.data == nullptr;
  const bool do232 = selected(Term::kComm232ss) &&
                     ((x3 && !drop232_x3) || (y3 && !drop232_y3));
  const bool drop132_x3 = v.y1 == nullptr;
  const bool drop132_y3 = v.x1 == nullptr;
  const bool do132 = selected(Term::kComm132ss) &&
                     ((x3 && !drop132_x3) || (y3 && !drop132_y3));
  const bool do331 = selected(Term::kComm331ss) && x3 && y3 && v.norb > 0 &&
                     v.sp.num_channels > 0;
  const bool do231 = selected(Term::kComm231ss) && (x3 || y3) && v.norb > 0 &&
                     v.x2.tb.num_channels > 0;
  // The scalar part of an anti-Hermitian Z vanishes (the sum cancels
  // pairwise); it is skipped to keep Z0 exactly zero (as Comm330Rows).
  const bool do330 = selected(Term::kComm330ss) && x3 && y3 &&
                     Z.hermiticity() != Hermiticity::kAntiHermitian;
  const bool do223 = selected(Term::kComm223ss) && X.two_body().allocated() &&
                     Y.two_body().allocated();

  if (dense133) Comm133ssReference(X, Y, Z, ctx);
  double z0 = 0.0;
  if (do133 || do232 || do132 || do331 || do231 || do330) {
    N7Needs needs;
    needs.h = do133 || do331;
    needs.w = do232 || do231 || do132;
    double* z1 = (do331 || do231) ? OneBodyOut(Z) : nullptr;
    // One TransposeOut value buffer for all batches, grown on demand (every
    // value of a batch is written by Comm133Batch before it is read).
    exec::DeviceBuffer<double> tout_data;
    Comm133Cache c133;
    ForEachN7Batch(
        v, ctx, rows, budget_bytes, needs, [&](const N7BatchInputs& in) {
          if (do133) {
            const std::vector<std::int64_t> off =
                TransposeLineOffsets(*in.plan);
            const auto need = static_cast<std::size_t>(off.back());
            if (tout_data.size() < need) {
              tout_data = exec::DeviceBuffer<double>(
                  need, "commutator::n7_transpose_out");
            }
            exec::DeviceBuffer<std::int64_t> line_off(
                off.size(), "commutator::n7_transpose_line_off");
            line_off.CopyFromHost(off);
            TransposeOut tout;
            tout.data = tout_data.device_data();
            tout.line_off = line_off.device_data();
            Comm133Batch(in, z3, tout, c133);
            AddTransposeOut(in, tout, z3);
          }
          if (do232) Comm232Batch(WithoutParts(in, drop232_x3, drop232_y3), z2);
          if (do132) Comm132Batch(WithoutParts(in, drop132_x3, drop132_y3), z2);
          if (do331) Comm331Batch(in, z1);
          if (do231) Comm231Batch(in, z1);
          if (do330) z0 += Comm330Batch(in);
        });
  }
  if (do223) AddComm223Rows(X, Y, *Z.three_body(), ctx, rows);
  if (do330) Z.zero_body() += z0;
}

}  // namespace imsrg::commutator::detail
