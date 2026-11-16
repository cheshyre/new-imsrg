// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm133ss, [X1, Y3] + [X3, Y1] -> Z3 (imsrg3.h). Kernel TU.
//
// J-scheme form (m-scheme formula of imsrg3.h; bra K = |(p q) Jpq r; J>, ket K'
// = |(s t) Jst u; J>, a over the one-body channel of the replaced orbit):
//   Z3[K, K'] += T(X, Y) - T(Y, X),
//   T(X, Y) = sum_a { X_pa Y3[(a q) Jpq r; K'] + X_qa Y3[(p a) Jpq r; K']
//                     + X_ra Y3[(p q) Jpq a; K']
//                     - Y3[K; (a t) Jst u] X_as - Y3[K; (s a) Jst u] X_at
//                     - Y3[K; (s t) Jst a] X_au }.
//
// Element-wise reference (`detail::Comm133ssReference`, the kernel before the
// row tiling and the path of a dense Z): one parallel task per bra row
// computes the stored elements of its row (the packed lower triangle j <= i,
// or the dense row), every Y3 element read through `ThreeBodyElement`.
//
// Batch form (`detail::Comm133Batch`, packed Z). Every Y3 element above has
// one replaced orbit and keeps the pair J of its canonical ket, so it is,
// up to an exchange phase, an element of H_Y = E Y (`BuildH`: rows = sorted
// pair-third states, imsrg++ GetME_pn convention, unnormalized in the pair):
// for the canonical row m = (p q) J r (p <= q <= r) its three slots give
//   slot 1, a in ch(p): Y3[(a q) J r; k] = phi H_Y[((a, q) sorted, J, r), k],
//   slot 2, a in ch(q): Y3[(p a) J r; k] = phi H_Y[((p, a) sorted, J, r), k],
//   slot 3, a in ch(r): Y3[(p q) J a; k] = H_Y[((p, q), J, a), k],
// with phi = 1 for an ordered pair and the exchange phase -(-1)^(j1 + j2 - J)
// for a swapped one (exact: SortOrbits of the swapped ordering is the swapped
// permutation, and RecouplingCoefficient of the two differs by exactly this
// sign, the same 6j symbol); an equal pair with odd J and a missing pair-third
// state (cut triple, D34) read 0, as in `ThreeBodyElement`. So, for the stored
// element z(m, k), m >= k, of a channel,
//   z(m, k) += sum_{slot, a of m} phi (X_oa H_Y[s, k] - Y_oa H_X[s, k])
//              (bra side, from line k)
//            + sum_{slot, a of k} phi (-X_ao H_Y^T[s, m] + Y_ao H_X^T[s, m])
//              (ket side, from line m),
// o the slot's orbit, s its pair-third state, H_B^T = E B^T = sign(B) H_B for
// a packed B (`N7BatchInputs::hxt`, `sxt`). The batch form reads exactly the
// (ordered) elements the reference reads, so it reproduces the reference for
// ANY three-body operand (up to the |v| < 1e-13 drops of E and the summation
// order), not only for physical ones: the canonical kets of a triple with
// equal orbits are linearly dependent, and the element-wise atan generator of
// the IMSRG(3) flow leaves its three-body part slightly off the physical
// subspace there. (A formulation through the action of X on the third orbit
// of the pair-third states and the comm223 recoupling map R, with 1/sqrt(2)
// for equal pair orbits, agrees with the reference to 1e-15 on physical
// operands but not on such components: the IT-9 O16 emax-3 flow then
// developed a growing redundant mode, |eta_3b| 1.7e-7 instead of 3.1e-8 at
// step 173 and 174 steps instead of imsrg++'s 173.)
//
// Line attribution (design/14 section 6.3): line L of the batch adds its bra
// side z(m, L), m >= L, into its own row and writes its ket side D(r, L), r =
// 0..L, into the batch's `TransposeOut`; `AddTransposeOut` adds D(r, L) into
// the stored z(L, r) of row r. For an anti-Hermitian Z the diagonal stays 0
// (neither side is added there).
//
// Kernels of a batch: (1) the slot table of the canonical rows of the
// batch's channels: per row, slot and one-body orbit the sorted index of the
// pair-third state (`FindSortedRow`) and the coefficients phi X_oa, -phi Y_oa
// (bra) and -sign(Y) phi X_ao, sign(X) phi Y_ao (ket); (2) one work item per
// (batch line L, column j in [0, n]) of its channel (n rows): j <= L the ket
// value D(j, L), j > L the bra value z(j - 1, L), each a branch-free sum over
// the 3 n1max table entries of a row (n1max the largest one-body channel) of
// two H reads; (3) `AddTransposeOut`, one work item per (tile, target row r),
// contiguous in row r. No atomics, every sum in a fixed order: results do not
// depend on the thread count or the batching, up to the order in which the
// bra and ket parts of an element are added. Cost per line of a channel with
// n rows: n (3 n1max) (2 multiply-adds), next to the H build (`BuildH`).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator {

namespace {

/// T(X, Y) of the file comment for `x1` (X's one-body part) and `y3` (Y's
/// three-body part); `two_j` is 2J of the channel. O(sum of the six
/// one-body channel sizes * cost of `ThreeBodyElement`).
IMSRG_INLINE_FUNCTION double OneBodyOnThreeBody(
    const detail::N7Views& v, const double* x1,
    const detail::ThreeBodyPartView& y3, int two_j, int p, int q, int r,
    int jpq, int s, int t, int u, int jst) noexcept {
  if (x1 == nullptr || y3.local == nullptr) return 0.0;
  const int n = v.norb;
  double sum = 0.0;
  for (int k = v.ob_ch.offsets[p]; k < v.ob_ch.offsets[p + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, p, a);
    if (x != 0.0) {
      sum += x *
             detail::ThreeBodyElement(v, y3, two_j, a, q, r, jpq, s, t, u, jst);
    }
  }
  for (int k = v.ob_ch.offsets[q]; k < v.ob_ch.offsets[q + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, q, a);
    if (x != 0.0) {
      sum += x *
             detail::ThreeBodyElement(v, y3, two_j, p, a, r, jpq, s, t, u, jst);
    }
  }
  for (int k = v.ob_ch.offsets[r]; k < v.ob_ch.offsets[r + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, r, a);
    if (x != 0.0) {
      sum += x *
             detail::ThreeBodyElement(v, y3, two_j, p, q, a, jpq, s, t, u, jst);
    }
  }
  for (int k = v.ob_ch.offsets[s]; k < v.ob_ch.offsets[s + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, a, s);
    if (x != 0.0) {
      sum -= x *
             detail::ThreeBodyElement(v, y3, two_j, p, q, r, jpq, a, t, u, jst);
    }
  }
  for (int k = v.ob_ch.offsets[t]; k < v.ob_ch.offsets[t + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, a, t);
    if (x != 0.0) {
      sum -= x *
             detail::ThreeBodyElement(v, y3, two_j, p, q, r, jpq, s, a, u, jst);
    }
  }
  for (int k = v.ob_ch.offsets[u]; k < v.ob_ch.offsets[u + 1]; ++k) {
    const int a = v.ob_ch.orbits[k];
    const double x = detail::OneBodyElement(x1, n, a, u);
    if (x != 0.0) {
      sum -= x *
             detail::ThreeBodyElement(v, y3, two_j, p, q, r, jpq, s, t, a, jst);
    }
  }
  return sum;
}

}  // namespace

namespace detail {

void Comm133ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm133ss", X, Y, Z, ctx);
  const ThreeBodyOutView out = ThreeBodyOut("Comm133ss", Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  const exec::Index nrows = static_cast<exec::Index>(
      ctx.model_space_ptr()->three_body_space()->NumRows());
  exec::ParallelFor(
      "commutator::comm133ss_reference", nrows,
      IMSRG_LAMBDA(const exec::Index r) {
        const int tr = v.row_triple[r];
        const int ch = v.sp.tr_channel[tr];
        const std::int64_t base = v.sp.ch_row_offset[ch];
        const int i = static_cast<int>(r - base);
        const int n = v.sp.ch_dim[ch];
        const int two_j = v.sp.ch_two_j[ch];
        const int p = v.sp.tr_a[tr];
        const int q = v.sp.tr_b[tr];
        const int rr = v.sp.tr_c[tr];
        const int jpq = v.row_two_jab[r] / 2;
        const int jmax = out.packed != 0 ? i : n - 1;
        for (int j = 0; j <= jmax; ++j) {
          if (out.zero_diagonal != 0 && j == i) continue;
          const std::int64_t rk = base + j;
          const int tk = v.row_triple[rk];
          const int s = v.sp.tr_a[tk];
          const int t = v.sp.tr_b[tk];
          const int u = v.sp.tr_c[tk];
          const int jst = v.row_two_jab[rk] / 2;
          const double z = OneBodyOnThreeBody(v, v.x1, v.y3, two_j, p, q, rr,
                                              jpq, s, t, u, jst) -
                           OneBodyOnThreeBody(v, v.y1, v.x3, two_j, p, q, rr,
                                              jpq, s, t, u, jst);
          out.data[ThreeBodyOutIndex(out, base, i, j)] += z;
        }
      });
  exec::Fence("commutator::comm133ss_reference");
}

namespace {

/// Orbit slots of a canonical row (p, q, r): the replaced orbit of each of
/// the three bra (or ket) terms of the file comment.
constexpr int kSlots = 3;

/// Index of the last entry of the ascending `off[0, n)` that is <= `i`
/// (precondition: n >= 1 and off[0] <= i). Entries may repeat (empty
/// ranges); the last of equal entries is returned, i.e. the nonempty range
/// that contains i. O(log n).
IMSRG_INLINE_FUNCTION int LastAtOrBelow(const std::int64_t* off, int n,
                                        std::int64_t i) noexcept {
  int lo = 0;
  int hi = n - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (off[mid] <= i) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/// Host tables of one batch for the 133 kernels, prefix sums over the tiles
/// (num_tiles + 1 entries each): `line0[ti]` is the first batch line of tile
/// ti (lines numbered in tile order, rows ascending), `elem0[ti]` its first
/// (line, column) work item (lines x (n + 1) per tile, n the channel
/// dimension), `slot0[ti]` the start of its channel's slot table (n kSlots
/// n1max entries per tile).
struct BatchOffsets {
  std::vector<std::int64_t> line0;
  std::vector<std::int64_t> elem0;
  std::vector<std::int64_t> slot0;
};

/// The `BatchOffsets` of `plan` for the context `ctx` and `n1max` entries
/// per slot. O(tiles).
BatchOffsets MakeBatchOffsets(const LineBatch& plan,
                              const ThreeBodyContext& ctx, int n1max) {
  const basis::ThreeBodySpace& s = *ctx.model_space_ptr()->three_body_space();
  BatchOffsets o;
  o.line0.push_back(0);
  o.elem0.push_back(0);
  o.slot0.push_back(0);
  for (const LineTile& tile : plan.tiles) {
    const std::int64_t n = s.dim(Channel3Index{tile.channel});
    const std::int64_t lines = tile.row_end - tile.row_begin;
    o.line0.push_back(o.line0.back() + lines);
    o.elem0.push_back(o.elem0.back() + lines * (n + 1));
    o.slot0.push_back(o.slot0.back() + n * kSlots * n1max);
  }
  return o;
}

/// Largest one-body channel of the context (orbits of equal l, 2j, 2tz;
/// emax / 2 + 1 of them): the entries per slot of the slot table. Host,
/// O(orbits).
int MaxOneBodyChannel(const ThreeBodyContext& ctx) {
  const std::vector<std::int32_t> off = ctx.impl().ob_offsets.ToVector();
  int n1max = 1;
  for (std::size_t i = 0; i + 1 < off.size(); ++i) {
    n1max = std::max(n1max, static_cast<int>(off[i + 1] - off[i]));
  }
  return n1max;
}

/// The slot table of the tile channels `channels` (in tile order), kept
/// across the batches of a call while the channel list does not change.
struct SlotCacheEntry {
  bool valid = false;
  bool has_hy = false;
  bool has_hx = false;
  std::vector<std::int32_t> channels;
  exec::DeviceBuffer<std::int32_t> rows;
  exec::DeviceBuffer<double> coef;  ///< [bra | ket], 2 entries each.
};

/// Uploads the three tables of `o` as one device buffer [line0 | elem0 |
/// slot0] (num_tiles + 1 entries each).
exec::DeviceBuffer<std::int64_t> UploadOffsets(const BatchOffsets& o) {
  std::vector<std::int64_t> all;
  all.reserve(3 * o.line0.size());
  all.insert(all.end(), o.line0.begin(), o.line0.end());
  all.insert(all.end(), o.elem0.begin(), o.elem0.end());
  all.insert(all.end(), o.slot0.begin(), o.slot0.end());
  exec::DeviceBuffer<std::int64_t> buf(all.size(),
                                       "commutator::comm133_offsets");
  buf.CopyFromHost(all);
  return buf;
}

/// The signed table entry of the ordered state (x y) J z of channel `ch`: 0
/// if it reads 0 (equal pair orbits with odd J, or no pair-third state, i.e.
/// a cut triple), else phi (s + 1) with s the channel-local sorted index of
/// the pair-third state ((x, y) sorted, J, z) and phi the exchange phase of
/// a swapped pair (file comment). O(log block).
IMSRG_INLINE_FUNCTION std::int32_t SlotEntry(const Comm232TablesView& tab,
                                             const double* occ,
                                             const std::int32_t* j2, int ch,
                                             int x, int y, int J,
                                             int z) noexcept {
  if (x == y && (J & 1) != 0) return 0;
  const int lo = x < y ? x : y;
  const int hi = x < y ? y : x;
  const int s = FindSortedRow(tab, occ, ch, lo, hi, J, z);
  if (s < 0) return 0;
  const int phi =
      x > y ? basis::ExchangePhase(TwoJ{j2[x]}, TwoJ{j2[y]}, TwoJ{2 * J}) : 1;
  return phi * (s + 1);
}

/// The slot table of a batch (device pointers): per entry (row m of a
/// tile's channel, slot, one-body index i), at slot0[tile] + (m kSlots +
/// slot) n1max + i, the channel-local sorted row `s` of the pair-third state
/// read through H (0 for an empty entry) and the coefficients, phi times the
/// one-body element, of that H row: `bra[2 e]` multiplies H_Y and `bra[2 e +
/// 1]` H_X in the bra value, `ket[2 e]` the Y3^T block and `ket[2 e + 1]` the
/// X3^T block in the ket value (0 for an empty entry or a missing term).
/// Trivially copyable.
struct SlotTable {
  std::int32_t* s = nullptr;
  double* bra = nullptr;
  double* ket = nullptr;
};
static_assert(std::is_trivially_copyable_v<SlotTable>);

/// Fills the slot table `st` of the batch (`slot0` per tile, `total`
/// entries): for entry (row m, slot, i) the slot's orbit o of the canonical
/// row m = (p q) J r (p for slot 0, q for 1, r for 2) is replaced by the i-th
/// orbit a of its one-body channel (an empty entry beyond the channel);
/// `SlotEntry` gives phi and s, and the coefficients are phi X_oa and -phi
/// Y_oa (bra), -syt phi X_ao and sxt phi Y_ao (ket), where a term without
/// its H block (`has_hy`: H(Y3) present and X1 present; `has_hx` likewise)
/// gets 0. One work item per entry. Cost O(entries log block). Asynchronous.
void BuildSlots(const N7BatchInputs& in, const std::int64_t* slot0, int n1max,
                std::int64_t total, bool has_hy, bool has_hx,
                const SlotTable& st) {
  const LineBatchView batch = in.batch;
  const N7Views v = in.v;
  const Comm232TablesView tab = in.tab;
  const double sxt = in.sxt;
  const double syt = in.syt;
  exec::ParallelFor(
      "commutator::comm133_slots", total, IMSRG_LAMBDA(const exec::Index e) {
        const int ti = LastAtOrBelow(slot0, batch.num_tiles, e);
        const int ch = batch.tiles[ti].channel;
        const std::int64_t local = e - slot0[ti];
        const int i = static_cast<int>(local % n1max);
        const int slot = static_cast<int>((local / n1max) % kSlots);
        const int m = static_cast<int>(local / (n1max * kSlots));
        const std::int64_t f = v.sp.ch_row_offset[ch] + m;
        const int t = v.row_triple[f];
        const int p = v.sp.tr_a[t];
        const int q = v.sp.tr_b[t];
        const int r = v.sp.tr_c[t];
        const int J = v.row_two_jab[f] / 2;
        const int o = slot == 0 ? p : (slot == 1 ? q : r);
        const int k = v.ob_ch.offsets[o] + i;
        std::int32_t entry = 0;
        int a = o;
        if (k < v.ob_ch.offsets[o + 1]) {
          a = v.ob_ch.orbits[k];
          const double* occ = v.ob.occupation;
          const std::int32_t* j2 = v.ob.j2;
          if (slot == 0) {
            entry = SlotEntry(tab, occ, j2, ch, a, q, J, r);
          } else if (slot == 1) {
            entry = SlotEntry(tab, occ, j2, ch, p, a, J, r);
          } else {
            entry = SlotEntry(tab, occ, j2, ch, p, q, J, a);
          }
        }
        const double phi = entry > 0 ? 1.0 : (entry < 0 ? -1.0 : 0.0);
        const std::int64_t n = v.norb;
        const std::int64_t oa = o + a * n;  // (o, a), column-major
        const std::int64_t ao = a + o * n;  // (a, o)
        st.s[e] = entry == 0 ? 0 : (entry > 0 ? entry : -entry) - 1;
        st.bra[2 * e] = has_hy ? phi * v.x1[oa] : 0.0;
        st.bra[2 * e + 1] = has_hx ? -phi * v.y1[oa] : 0.0;
        st.ket[2 * e] = has_hy ? -syt * phi * v.x1[ao] : 0.0;
        st.ket[2 * e + 1] = has_hx ? sxt * phi * v.y1[ao] : 0.0;
      });
}

/// The operands of the main kernel (device pointers and scalars; trivially
/// copyable, captured by value). A missing H block points at the present one
/// (its coefficients are 0), so the element sums are branch-free.
struct Comm133Args {
  Space3View sp;
  LineBatchView batch;
  ThreeBodyOutView z3;
  TransposeOut tout;
  SlotTable st;
  /// Pair-third states per channel (the leading dimension of H).
  const std::int32_t* ch_num_states = nullptr;
  const double* hy = nullptr;           ///< H(Y3) (bra values).
  const double* hx = nullptr;           ///< H(X3) (bra values).
  const double* hyt = nullptr;          ///< H(Y3^T) / syt (ket values).
  const double* hxt = nullptr;          ///< H(X3^T) / sxt (ket values).
  const std::int64_t* line0 = nullptr;  ///< Per tile (num_tiles + 1).
  const std::int64_t* elem0 = nullptr;  ///< Per tile (num_tiles + 1).
  const std::int64_t* slot0 = nullptr;  ///< Per tile (num_tiles + 1).
  std::int32_t n1max = 1;
};
static_assert(std::is_trivially_copyable_v<Comm133Args>);

/// The value of one element of the main kernel: the sum over the kSlots
/// n1max slot-table entries from `e0` (one canonical row) of c_Y H_Y[hcol +
/// s] + c_X H_X[hcol + s], with the bra coefficients and blocks (`ket` false)
/// or the ket ones (`ket` true); `hcol` is the start of the line's H column.
/// O(3 n1max).
IMSRG_INLINE_FUNCTION double SlotSum(const Comm133Args& a, std::int64_t e0,
                                     std::int64_t hcol, bool ket) noexcept {
  const double* c = ket ? a.st.ket : a.st.bra;
  const double* h_y = (ket ? a.hyt : a.hy) + hcol;
  const double* h_x = (ket ? a.hxt : a.hx) + hcol;
  const std::int64_t e1 = e0 + kSlots * a.n1max;
  double sum = 0.0;
  for (std::int64_t e = e0; e < e1; ++e) {
    const std::int64_t s = a.st.s[e];
    sum += c[2 * e] * h_y[s] + c[2 * e + 1] * h_x[s];
  }
  return sum;
}

/// The main kernel of `Comm133Batch` (file comment, kernel 2): one work item
/// per (batch line L, column j in [0, n]) of the line's channel, `total`
/// items. Asynchronous.
void RunLines(const Comm133Args& a, std::int64_t total) {
  exec::ParallelFor(
      "commutator::comm133_batch", total, IMSRG_LAMBDA(const exec::Index it) {
        const int ti = LastAtOrBelow(a.elem0, a.batch.num_tiles, it);
        const LineTile& tile = a.batch.tiles[ti];
        const int ch = tile.channel;
        const std::int64_t width = a.sp.ch_dim[ch] + 1;
        const std::int64_t local = it - a.elem0[ti];
        const int ll = static_cast<int>(local / width);
        const int j = static_cast<int>(local % width);
        const int L = tile.row_begin + ll;
        const std::int64_t hcol =
            tile.h_offset + static_cast<std::int64_t>(ll) * a.ch_num_states[ch];
        const std::int64_t per_row = kSlots * a.n1max;
        const bool diag = a.z3.zero_diagonal != 0;
        if (j <= L) {  // ket value D(j, L) of the stored z(L, j)
          double* d = a.tout.data + a.tout.line_off[a.line0[ti] + ll];
          d[j] = (diag && j == L)
                     ? 0.0
                     : SlotSum(a, a.slot0[ti] + j * per_row, hcol, true);
        } else {  // bra value of the stored z(m, L), m = j - 1 >= L
          const int m = j - 1;
          if (diag && m == L) return;
          const std::int64_t row0 = a.sp.ch_row_offset[ch];
          a.z3.data[ThreeBodyOutIndex(a.z3, row0, m, L)] +=
              SlotSum(a, a.slot0[ti] + m * per_row, hcol, false);
        }
      });
}

/// The line offsets of the TransposeOut of `plan` (design: `TransposeOut`):
/// num_lines + 1 entries, line l (row L) at [off[l], off[l] + L + 1), the
/// last entry the total size. O(lines).
std::vector<std::int64_t> TransposeOutOffsets(const LineBatch& plan) {
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

}  // namespace

struct Comm133Cache::Impl {
  int n1max = 0;  ///< 0: not yet computed.
  SlotCacheEntry slots;
};

Comm133Cache::Comm133Cache() : impl_(std::make_unique<Impl>()) {}
Comm133Cache::~Comm133Cache() = default;
Comm133Cache::Comm133Cache(Comm133Cache&&) noexcept = default;
Comm133Cache& Comm133Cache::operator=(Comm133Cache&&) noexcept = default;

void Comm133Batch(const N7BatchInputs& in, const ThreeBodyOutView& z3,
                  const TransposeOut& tout, Comm133Cache& cache) {
  IMSRG_CHECK(in.plan != nullptr && in.ctx != nullptr,
              "Comm133Batch: the batch inputs need the host plan and context");
  IMSRG_CHECK(z3.packed != 0, "Comm133Batch: packed Z only");
  const LineBatch& plan = *in.plan;
  if (plan.tiles.empty()) return;
  Comm133Cache::Impl& ci = cache.impl();
  if (ci.n1max == 0) ci.n1max = MaxOneBodyChannel(*in.ctx);
  const int n1max = ci.n1max;
  // A term needs its one-body part and the other operand's H block.
  const bool has_hy = in.v.x1 != nullptr && in.hy != nullptr;
  const bool has_hx = in.v.y1 != nullptr && in.hx != nullptr;
  if (!has_hy && !has_hx) {  // nothing to add, but the ket values are read
    double* d = tout.data;
    exec::ParallelFor(
        "commutator::comm133_zero_transpose", TransposeOutOffsets(plan).back(),
        IMSRG_LAMBDA(const exec::Index e) { d[e] = 0.0; });
    exec::Fence("commutator::comm133_zero_transpose");
    return;
  }
  const BatchOffsets o = MakeBatchOffsets(plan, *in.ctx, n1max);
  const exec::DeviceBuffer<std::int64_t> off = UploadOffsets(o);
  const std::size_t nt1 = o.line0.size();
  const auto entries = static_cast<std::size_t>(o.slot0.back());
  // The slot table of a channel does not depend on the batch's row range:
  // rebuild it only if the tile channels (or the term set) changed.
  SlotCacheEntry& sc = ci.slots;
  std::vector<std::int32_t> chans;
  chans.reserve(plan.tiles.size());
  for (const LineTile& t : plan.tiles) chans.push_back(t.channel);
  const bool rebuild = !sc.valid || sc.has_hy != has_hy ||
                       sc.has_hx != has_hx || sc.channels != chans;
  if (rebuild) {
    sc.valid = false;
    sc.rows =
        exec::DeviceBuffer<std::int32_t>(entries, "commutator::comm133_slots");
    sc.coef =
        exec::DeviceBuffer<double>(4 * entries, "commutator::comm133_coef");
    sc.channels = chans;
    sc.has_hy = has_hy;
    sc.has_hx = has_hx;
  }
  Comm133Args a;
  a.sp = in.v.sp;
  a.batch = in.batch;
  a.z3 = z3;
  a.tout = tout;
  a.st.s = sc.rows.device_data();
  a.st.bra = sc.coef.device_data();
  a.st.ket = sc.coef.device_data() + 2 * entries;
  a.ch_num_states = in.pt.ch_num_states;
  // A missing block is replaced by the present one (coefficients 0).
  a.hy = has_hy ? in.hy : in.hx;
  a.hx = has_hx ? in.hx : in.hy;
  a.hyt = has_hy ? in.hyt : in.hxt;
  a.hxt = has_hx ? in.hxt : in.hyt;
  a.line0 = off.device_data();
  a.elem0 = off.device_data() + nt1;
  a.slot0 = off.device_data() + 2 * nt1;
  a.n1max = n1max;
  if (rebuild) {
    BuildSlots(in, a.slot0, n1max, o.slot0.back(), has_hy, has_hx, a.st);
    sc.valid = true;
  }
  RunLines(a, o.elem0.back());
  exec::Fence("commutator::comm133_batch");
}

void AddTransposeOut(const N7BatchInputs& in, const TransposeOut& tout,
                     const ThreeBodyOutView& z3) {
  IMSRG_CHECK(in.plan != nullptr,
              "AddTransposeOut: the batch inputs need the host plan");
  IMSRG_CHECK(z3.packed != 0, "AddTransposeOut: packed Z only");
  const LineBatch& plan = *in.plan;
  if (plan.tiles.empty()) return;
  // Per tile (prefix sums, num_tiles + 1 entries each): its first batch line
  // and its first work item; one work item per (tile, target row r < re).
  std::vector<std::int64_t> tables = {0};
  std::vector<std::int64_t> item0 = {0};
  for (const LineTile& t : plan.tiles) {
    tables.push_back(tables.back() + (t.row_end - t.row_begin));
    item0.push_back(item0.back() + t.row_end);
  }
  const std::size_t nt1 = tables.size();
  tables.insert(tables.end(), item0.begin(), item0.end());
  exec::DeviceBuffer<std::int64_t> dev(tables.size(),
                                       "commutator::comm133_add_offsets");
  dev.CopyFromHost(tables);
  const std::int64_t* line0 = dev.device_data();
  const std::int64_t* items = dev.device_data() + nt1;
  const LineBatchView batch = in.batch;
  const Space3View sp = in.v.sp;
  // Row r receives z(L, r) += D(r, L) for the tile's lines L >= r: contiguous
  // in row r (packed storage), read from the lines' TransposeOut segments.
  exec::ParallelForDynamic(
      "commutator::comm133_add_transpose", item0.back(),
      IMSRG_LAMBDA(const exec::Index it) {
        const int ti = LastAtOrBelow(items, batch.num_tiles, it);
        const LineTile& tile = batch.tiles[ti];
        const int r = static_cast<int>(it - items[ti]);
        const std::int64_t row0 = sp.ch_row_offset[tile.channel];
        const int first = r > tile.row_begin ? r : tile.row_begin;
        for (int L = first; L < tile.row_end; ++L) {
          if (z3.zero_diagonal != 0 && L == r) continue;
          const std::int64_t l = line0[ti] + (L - tile.row_begin);
          z3.data[ThreeBodyOutIndex(z3, row0, L, r)] +=
              tout.data[tout.line_off[l] + r];
        }
      });
  exec::Fence("commutator::comm133_add_transpose");
}

void Comm133Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  const N7Views v = PrepareN7("Comm133ss", X, Y, Z, ctx);
  const ThreeBodyOutView out = ThreeBodyOut("Comm133ss", Z, ctx);
  if (out.packed == 0) {
    throw NotImplementedError(
        "Comm133Rows: a dense (non-Hermitian) Z needs Comm133ssReference");
  }
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  // One TransposeOut value buffer for all batches, grown on demand (every
  // value of a batch is written by Comm133Batch before it is read).
  exec::DeviceBuffer<double> tout_data;
  Comm133Cache cache;
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{true, false},
                 [&](const N7BatchInputs& in) {
                   const std::vector<std::int64_t> off =
                       TransposeOutOffsets(*in.plan);
                   const auto need = static_cast<std::size_t>(off.back());
                   if (tout_data.size() < need) {
                     tout_data = exec::DeviceBuffer<double>(
                         need, "commutator::comm133_transpose_out");
                   }
                   exec::DeviceBuffer<std::int64_t> line_off(
                       off.size(), "commutator::comm133_line_off");
                   line_off.CopyFromHost(off);
                   TransposeOut tout;
                   tout.data = tout_data.device_data();
                   tout.line_off = line_off.device_data();
                   Comm133Batch(in, out, tout, cache);
                   AddTransposeOut(in, tout, out);
                 });
}

}  // namespace detail

namespace terms {

void Comm133ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  (void)detail::PrepareN7("Comm133ss", X, Y, Z, ctx);
  const detail::ThreeBodyOutView out =
      detail::ThreeBodyOut("Comm133ss", Z, ctx);
  if (out.packed == 0) {
    detail::Comm133ssReference(X, Y, Z, ctx);
    return;
  }
  detail::Comm133Rows(
      X, Y, Z, ctx,
      basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
      detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
