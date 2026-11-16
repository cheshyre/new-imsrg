// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm331ss, [X3, Y3] -> Z1 (imsrg3.h). Kernel TU.
//
// J-scheme form: for the one-body channel (l, 2j, 2tz) of the orbits i, j,
//   Z_ij += 1 / (2 j_i + 1) sum_{J3 channel ch} (2 J3 + 1) / 12
//     sum_{a <= b, Jab} m_ab sum_K sym_K occ(P, K)
//       (X3[(P i); K] Y3[K; (P j)] - Y3[(P i); K] X3[K; (P j)]),
// P = (a b) Jab, m_ab = 1 (a == b) or 2, K the kept canonical rows of ch (sym_K
// = 6, 3 or 1 of its triple), occ(P, K) = n_a n_b nbar_c nbar_d nbar_e +
// nbar_a nbar_b n_c n_d n_e for K = ((c d e) Jcd), every three-body element
// in the imsrg++ GetME_pn convention (`detail::ThreeBodyElement`).
//
// Element-wise reference (`detail::Comm331ssReference`, the kernel before the
// row tiling): one work item per (output pair, channel) sums over every pair
// (a b), Jab and row K of the channel, each element read through
// `ThreeBodyElement`; a second pass reduces the channels in order.
//
// Batch form (`detail::Comm331Batch`, one batch = the lines K of its tiles).
// With H_B = E B the pair-third intermediate of the batch (`BuildH`, rows =
// the sorted pair-third states s = (P, third) of the channel, columns = the
// lines), X3[(P i); K] = H_X[s(P, i), K] and Y3[K; (P j)] =
// sign(Y) H_{Y^T}[s(P, j), K] (`N7BatchInputs::hyt`, `syt`; for a packed part
// H_{B^T} is H_B itself with the hermiticity sign). So
//   Z_ij += (2 J3 + 1) / (12 (2 j_i + 1)) sum_{s = (P, i)} m_P
//      sum_{lines K} sym_K (n_P nbar_K + nbar_P n_K) D_K(s, s'),
//   D_K = H_X[s, K] sign(Y) H_{Y^T}[s', K] - H_Y[s, K] sign(X) H_{X^T}[s', K],
// s' = the state (P, j) (same pair and Jab, third j; it exists iff (P, i)
// does). Work items: one per output pair (i, j) of a one-body channel, each
// summing over the tiles of the batch, the pair-third states of third i
// (contiguous in the sorted order, occupation classes 0..2 only) and the
// lines of the tile in a fixed order, so every Z1 element is written by one
// item (no atomics) and the result does not depend on the thread count. A
// small per-line table (sym_K nbar_K, sym_K n_K) is built per batch. Cost per
// batch: 2 (rows of the channels with third i) (lines) per output pair, i.e.
// ~ 2 k S n_1B for k lines, S states per channel and n_1B the largest
// one-body channel (next to the H build); the lines are the only attribution
// of the term (design/14 section 6.3), so the batches add up to the whole
// term (up to the order of the sums).

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator {

namespace {

/// True if orbits i and j are in the same one-body channel (l, 2j, 2tz).
IMSRG_INLINE_FUNCTION bool SameChannel(const detail::N7Views& v, int i,
                                       int j) noexcept {
  return v.ob.l[i] == v.ob.l[j] && v.ob.j2[i] == v.ob.j2[j] &&
         v.ob.tz2[i] == v.ob.tz2[j];
}

/// The part of Z_ij from the three-body channel `ch` (without the factor
/// 1 / (2 j_i + 1)): (2 J3 + 1) sum over a <= b, Jab and the kept rows K of
/// ch of ((2 - d_ab) sym_K / 12) occ (X3[(ab)Jab i; K] Y3[K; (ab)Jab j] -
/// Y3[(ab)Jab i; K] X3[K; (ab)Jab j]). Cost: O(orbit pairs * Jab * rows *
/// element read).
IMSRG_INLINE_FUNCTION double Comm331Channel(const detail::N7Views& v, int ch,
                                            int i, int j) noexcept {
  const int two_j3 = v.sp.ch_two_j[ch];
  const int ji = v.ob.j2[i];
  const std::int64_t base = v.sp.ch_row_offset[ch];
  const int nrows = v.sp.ch_dim[ch];
  double total = 0.0;
  for (int a = 0; a < v.norb; ++a) {
    const double na = v.ob.occupation[a];
    for (int b = a; b < v.norb; ++b) {
      if (detail::Channel3Of(v, two_j3, a, b, i) != ch) continue;
      const double nb = v.ob.occupation[b];
      const double nn = na * nb;
      const double nbnb = (1.0 - na) * (1.0 - nb);
      if (nn == 0.0 && nbnb == 0.0) continue;
      const int jlo = (v.ob.j2[a] > v.ob.j2[b] ? v.ob.j2[a] - v.ob.j2[b]
                                               : v.ob.j2[b] - v.ob.j2[a]) /
                      2;
      const int jhi = (v.ob.j2[a] + v.ob.j2[b]) / 2;
      const double mult = a == b ? 1.0 : 2.0;
      for (int jab = jlo; jab <= jhi; ++jab) {
        if (a == b && (jab & 1) != 0) continue;
        // Triangle of (Jab, j_i, J3) in twice-angular-momentum units.
        if (2 * jab + ji < two_j3 || two_j3 + ji < 2 * jab ||
            two_j3 + 2 * jab < ji) {
          continue;
        }
        double acc = 0.0;
        for (int k = 0; k < nrows; ++k) {
          const int t = v.row_triple[base + k];
          const int c = v.sp.tr_a[t];
          const int d = v.sp.tr_b[t];
          const int e = v.sp.tr_c[t];
          const double nc = v.ob.occupation[c];
          const double nd = v.ob.occupation[d];
          const double ne = v.ob.occupation[e];
          const double occ =
              nn * (1.0 - nc) * (1.0 - nd) * (1.0 - ne) + nbnb * nc * nd * ne;
          if (occ == 0.0) continue;
          const int jcd = v.row_two_jab[base + k] / 2;
          const double xik = detail::ThreeBodyElement(v, v.x3, two_j3, a, b, i,
                                                      jab, c, d, e, jcd);
          const double ykj = detail::ThreeBodyElement(v, v.y3, two_j3, c, d, e,
                                                      jcd, a, b, j, jab);
          const double yik = detail::ThreeBodyElement(v, v.y3, two_j3, a, b, i,
                                                      jab, c, d, e, jcd);
          const double xkj = detail::ThreeBodyElement(v, v.x3, two_j3, c, d, e,
                                                      jcd, a, b, j, jab);
          acc += v.sp.tr_sym[t] * occ * (xik * ykj - yik * xkj);
        }
        total += mult * acc;
      }
    }
  }
  return (two_j3 + 1) * total / 12.0;
}

}  // namespace

namespace detail {

void Comm331ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const detail::N7Views v = PrepareN7("Comm331ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr || v.y3.local == nullptr) return;
  const exec::Index norb = v.norb;
  const exec::Index nch = v.sp.num_channels;
  if (norb == 0 || nch == 0) return;
  // Pass 1: one partial sum per (output pair, 3B channel).
  exec::DeviceBuffer<double> partial(
      static_cast<std::size_t>(norb * norb * nch), "commutator::comm331_part");
  const exec::DeviceView<double> part = exec::View(partial);
  exec::ParallelFor(
      "commutator::comm331ss_reference_channels", norb * norb * nch,
      IMSRG_LAMBDA(const exec::Index w) {
        const int ch = static_cast<int>(w % nch);
        const exec::Index pair = w / nch;
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        double value = 0.0;
        if (SameChannel(v, i, j)) value = Comm331Channel(v, ch, i, j);
        part(w) = value;
      });
  // Pass 2: ordered reduction over the channels, added to Z1.
  double* z1 = OneBodyOut(Z);
  exec::ParallelFor(
      "commutator::comm331ss_reference_reduce", norb * norb,
      IMSRG_LAMBDA(const exec::Index pair) {
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        if (!SameChannel(v, i, j)) return;
        double sum = 0.0;
        for (exec::Index ch = 0; ch < nch; ++ch) sum += part(pair * nch + ch);
        z1[pair] += sum / (v.ob.j2[i] + 1);
      });
  exec::Fence("commutator::comm331ss_reference");
}

namespace {

/// The last index i in [0, n) with off[i] <= x (off non-decreasing, off[0] <=
/// x); of equal entries the last. O(log n).
IMSRG_INLINE_FUNCTION int LastAtOrBelow(const std::int64_t* off, int n,
                                        std::int64_t x) noexcept {
  int lo = 0;
  int hi = n - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (off[mid] <= x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/// Everything the kernels of one comm331 batch read (device pointers by
/// value; trivially copyable).
struct Comm331Args {
  N7Views v;
  Comm232TablesView tab;
  LineBatchView batch;
  const std::int32_t* ch_num_states = nullptr;  ///< Pair-third states per ch.
  const double* hx = nullptr;                   ///< H(X3).
  const double* hxt = nullptr;                  ///< H(X3^T) up to sxt.
  const double* hy = nullptr;                   ///< H(Y3).
  const double* hyt = nullptr;                  ///< H(Y3^T) up to syt.
  double sxt = 1.0;
  double syt = 1.0;
  const std::int64_t* line0 = nullptr;  ///< Per tile: first batch line.
  const double* lw = nullptr;           ///< Per line: sym nbar_K, sym n_K.
  double* z1 = nullptr;
};
static_assert(std::is_trivially_copyable_v<Comm331Args>);

/// Fills `lw` (2 doubles per batch line l in tile order, rows ascending):
/// lw[2 l] = sym_K nbar_c nbar_d nbar_e, lw[2 l + 1] = sym_K n_c n_d n_e for
/// the row K = (c d e) Jcd of the line. One work item per line; `a.line0`
/// holds the first line of each tile.
void BuildLineWeights(const Comm331Args& a, double* lw,
                      std::int64_t num_lines) {
  exec::ParallelFor(
      "commutator::comm331_line_weights", num_lines,
      IMSRG_LAMBDA(const exec::Index l) {
        const int ti = LastAtOrBelow(a.line0, a.batch.num_tiles, l);
        const LineTile& tile = a.batch.tiles[ti];
        const std::int64_t f = a.v.sp.ch_row_offset[tile.channel] +
                               tile.row_begin + (l - a.line0[ti]);
        const int t = a.v.row_triple[f];
        const double nc = a.v.ob.occupation[a.v.sp.tr_a[t]];
        const double nd = a.v.ob.occupation[a.v.sp.tr_b[t]];
        const double ne = a.v.ob.occupation[a.v.sp.tr_c[t]];
        const double sym = a.v.sp.tr_sym[t];
        lw[2 * l] = sym * (1.0 - nc) * (1.0 - nd) * (1.0 - ne);
        lw[2 * l + 1] = sym * nc * nd * ne;
      });
  exec::Fence("commutator::comm331_line_weights");
}

/// The batch's contribution to Z_ij (without 1 / (2 j_i + 1)) for one output
/// pair of a one-body channel (file comment). Fixed order: tiles, Jpq, states
/// of third i ascending, lines ascending. O(rows of third i * lines).
IMSRG_INLINE_FUNCTION double Comm331Pair(const Comm331Args& a, int i,
                                         int j) noexcept {
  const N7Views& v = a.v;
  const Comm232TablesView& tab = a.tab;
  double total = 0.0;
  for (int ti = 0; ti < a.batch.num_tiles; ++ti) {
    const LineTile& tile = a.batch.tiles[ti];
    const int ch = tile.channel;
    const std::int64_t stride = a.ch_num_states[ch];
    if (stride == 0) continue;
    const int nl = tile.row_end - tile.row_begin;
    const std::int64_t col0 = tile.h_offset;
    const double* lw = a.lw + 2 * a.line0[ti];
    const int chb = tab.ch_begin[ch];
    double tile_sum = 0.0;
    for (int jpq = 0; jpq < tab.nj; ++jpq) {
      const std::int64_t b =
          (static_cast<std::int64_t>(ch) * tab.norb + i) * tab.nj + jpq;
      const int r0 = tab.blk[kComm232Classes * b];
      const int r1 = tab.blk[kComm232Classes * b + 3];  // classes 0..2
      for (int t = r0; t < r1; ++t) {
        const int key = tab.key[t];
        const int p = key / tab.norb;
        const int q = key % tab.norb;
        const double np = v.ob.occupation[p];
        const double nq = v.ob.occupation[q];
        const double n_pair = np * nq;
        const double nbar_pair = (1.0 - np) * (1.0 - nq);
        const int sj = FindSortedRow(tab, v.ob.occupation, ch, p, q, jpq, j);
        if (sj < 0) continue;
        const std::int64_t si = t - chb;
        const double* hx_i = a.hx + col0 + si;
        const double* hy_i = a.hy + col0 + si;
        const double* hxt_j = a.hxt + col0 + sj;
        const double* hyt_j = a.hyt + col0 + sj;
        double acc_n = 0.0;
        double acc_nbar = 0.0;
        for (int l = 0; l < nl; ++l) {
          const std::int64_t o = l * stride;
          const double d =
              hx_i[o] * (a.syt * hyt_j[o]) - hy_i[o] * (a.sxt * hxt_j[o]);
          acc_nbar += lw[2 * l] * d;
          acc_n += lw[2 * l + 1] * d;
        }
        const double mult = p == q ? 1.0 : 2.0;
        tile_sum += mult * (n_pair * acc_nbar + nbar_pair * acc_n);
      }
    }
    total += (v.sp.ch_two_j[ch] + 1) * tile_sum;
  }
  return total / 12.0;
}

}  // namespace

void Comm331Batch(const N7BatchInputs& in, double* z1) {
  IMSRG_CHECK(in.plan != nullptr,
              "Comm331Batch: the batch inputs need the host plan");
  const LineBatch& plan = *in.plan;
  if (plan.tiles.empty() || plan.num_lines == 0) return;
  if (in.hx == nullptr || in.hy == nullptr || in.hxt == nullptr ||
      in.hyt == nullptr) {
    return;  // a missing three-body part: no contribution
  }
  const exec::Index norb = in.v.norb;
  if (norb == 0) return;
  std::vector<std::int64_t> line0 = {0};
  for (const LineTile& t : plan.tiles) {
    line0.push_back(line0.back() + (t.row_end - t.row_begin));
  }
  exec::DeviceBuffer<std::int64_t> dev_line0(line0.size(),
                                             "commutator::comm331_line0");
  dev_line0.CopyFromHost(line0);
  exec::DeviceBuffer<double> weights(
      static_cast<std::size_t>(2 * plan.num_lines), "commutator::comm331_lw");
  Comm331Args a;
  a.v = in.v;
  a.tab = in.tab;
  a.batch = in.batch;
  a.ch_num_states = in.pt.ch_num_states;
  a.hx = in.hx;
  a.hxt = in.hxt;
  a.hy = in.hy;
  a.hyt = in.hyt;
  a.sxt = in.sxt;
  a.syt = in.syt;
  a.line0 = dev_line0.device_data();
  a.lw = weights.device_data();
  a.z1 = z1;
  BuildLineWeights(a, weights.device_data(), plan.num_lines);
  exec::ParallelForDynamic(
      "commutator::comm331_batch", norb * norb,
      IMSRG_LAMBDA(const exec::Index w) {
        const int i = static_cast<int>(w % norb);
        const int j = static_cast<int>(w / norb);
        if (!SameChannel(a.v, i, j)) return;
        a.z1[w] += Comm331Pair(a, i, j) / (a.v.ob.j2[i] + 1);
      });
  exec::Fence("commutator::comm331_batch");
}

void Comm331Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  const N7Views v = PrepareN7("Comm331ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr || v.y3.local == nullptr) return;
  if (v.norb == 0 || v.sp.num_channels == 0) return;
  double* z1 = OneBodyOut(Z);
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{true, false},
                 [&](const N7BatchInputs& in) { Comm331Batch(in, z1); });
}

}  // namespace detail

namespace terms {

void Comm331ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm331Rows(
      X, Y, Z, ctx,
      basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
      detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
