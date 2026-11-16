// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_N7_VIEWS_H_
#define IMSRG_COMMUTATOR_DETAIL_N7_VIEWS_H_

/// @file
/// Device views and device-callable element readers of the IMSRG(3)-N7 term
/// kernels (imsrg3.h). Internal to the commutator module; no Kokkos, so the
/// structs can be built in ordinary TUs and captured by value in kernels
/// (design/12 section 5).
///
/// Element conventions (identical to imsrg++, see imsrg3.h): one-body
/// elements X_pq of a dense column-major norb x norb block; two-body elements
/// through `TwoBodyView` (normalized storage, `get_unnormalized` for imsrg++'s
/// `GetTBME_J`); three-body elements are imsrg++'s unnormalized
/// antisymmetrized `GetME_pn(Jab, Jde, 2J, a, b, c, d, e, f)` for orbits in
/// ANY order, read from the canonical layout-A blocks of `op::ThreeBody`
/// through the recoupling of `angmom::RecouplingCoefficient`
/// (ThreeBodyStorage::GetKetIndex_withRecoupling).

#include <cstdint>
#include <type_traits>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/angmom/three_body_recoupling.h"
#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/device_tables.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::commutator::detail {

/// Largest number of two-body J values of an orbit pair the element readers
/// handle (2 emax + 2 <= 32 up to emax 15).
inline constexpr int kMaxKetTerms = 32;

/// The three-body part of an operand: layout-A rows addressed through
/// per-row offsets (design: basis/three_body_rows.h). Row k of channel ch is
/// the stored segment z(m, k) (packed: m >= k, dense: all m); with the flat
/// kept row f = `Space3View::ch_row_offset[ch] + k` it starts at
/// `buf[row_off[f]]` such that z(m, k) = `buf[row_off[f] + m]` (the offset
/// absorbs the packed start; it may be negative). `local == nullptr`
/// represents a missing part, which reads as zero.
struct ThreeBodyPartView {
  /// The operand's rows (serial: the whole `op::ThreeBody` buffer), or null.
  const double* local = nullptr;
  /// Rows fetched from other ranks for a batch (M12b); null in serial.
  const double* halo = nullptr;
  /// Per flat kept row f: z(m, k) of that row is `buf[row_off[f] + m]`.
  const std::int64_t* row_off = nullptr;
  /// Per flat kept row: 0 local, 1 halo (-1 not resident). May be null when
  /// `halo` is null (it is never read then).
  const std::int8_t* row_buf = nullptr;
  std::int32_t packed = 1;  ///< 1: packed lower triangle, 0: dense blocks.
  std::int32_t sign = 1;    ///< Packed: element (i, j), i < j, is sign (j, i).
};
static_assert(std::is_trivially_copyable_v<ThreeBodyPartView>);

/// Everything an N7 kernel reads (raw device pointers, captured by value).
struct N7Views {
  Space3View sp;                              ///< Kept canonical space.
  const std::int32_t* row_two_jab = nullptr;  ///< 2 Jab per flat row.
  const std::int32_t* row_triple = nullptr;   ///< Triple per flat row.
  angmom::SixJ3View sj;                       ///< 3B 6j table.
  basis::OrbitTableView ob;                   ///< Orbits and occupations.
  OneBodyChannelView ob_ch;                   ///< One-body channels.
  TwoBodyView x2;                             ///< X's two-body part.
  TwoBodyView y2;                             ///< Y's two-body part.
  const double* x1 = nullptr;  ///< X's one-body part (norb x norb, col-major).
  const double* y1 = nullptr;  ///< Y's one-body part.
  ThreeBodyPartView x3;        ///< X's three-body part (or none).
  ThreeBodyPartView y3;        ///< Y's three-body part (or none).
  std::int32_t norb = 0;       ///< Number of orbits.
};
static_assert(std::is_trivially_copyable_v<N7Views>);

/// The output rows of a term that writes Z's three-body part.
struct ThreeBodyOutView {
  double* data = nullptr;  ///< Device data of Z's `op::ThreeBody`.
  /// Per flat kept row f: element (i, j) of channel ch (f = `ch_row_offset
  /// [ch]`) is `data[row_off[f + j] + i]` (i >= j if packed).
  const std::int64_t* row_off = nullptr;
  std::int32_t packed = 1;  ///< 1: packed lower triangle (i >= j), 0: dense.
  std::int32_t zero_diagonal = 0;  ///< 1: anti-Hermitian Z, diagonal is 0.
};
static_assert(std::is_trivially_copyable_v<ThreeBodyOutView>);

/// Index into `out.data` of element (i, j) (channel-local rows; i >= j if
/// packed) of the channel whose first flat kept row is `ch_row0`. O(1).
IMSRG_INLINE_FUNCTION std::int64_t ThreeBodyOutIndex(
    const ThreeBodyOutView& out, std::int64_t ch_row0, int i, int j) noexcept {
  return out.row_off[ch_row0 + j] + i;
}

/// One-body element (p, q) of a dense column-major n x n block, 0 for a null
/// block. O(1).
IMSRG_INLINE_FUNCTION double OneBodyElement(const double* d, int n, int p,
                                            int q) noexcept {
  return d == nullptr ? 0.0 : d[p + static_cast<std::int64_t>(q) * n];
}

/// Element (i, j) (channel-local rows) of channel `ch` of a three-body part,
/// with the mirror sign for i < j in packed mode. Rows living in the halo
/// buffer are read from there (the halo test comes first, so serial code
/// never reads `row_buf`). O(1).
IMSRG_INLINE_FUNCTION double ThreeBodyLocal(const ThreeBodyPartView& t,
                                            const Space3View& sp, int ch, int i,
                                            int j) noexcept {
  if (t.local == nullptr) return 0.0;
  const std::int64_t f = sp.ch_row_offset[ch];
  if (t.packed != 0 && i < j) {
    const std::int64_t r = f + i;
    const double* buf =
        (t.halo == nullptr || t.row_buf[r] == 0) ? t.local : t.halo;
    return t.sign * buf[t.row_off[r] + j];
  }
  const std::int64_t r = f + j;
  const double* buf =
      (t.halo == nullptr || t.row_buf[r] == 0) ? t.local : t.halo;
  return buf[t.row_off[r] + i];
}

/// Channel of the three-body kets with orbits (a, b, c) and total 2J, or -1
/// if 2J is beyond the channels of the space. O(1).
IMSRG_INLINE_FUNCTION int Channel3Of(const N7Views& v, int two_j, int a, int b,
                                     int c) noexcept {
  if (two_j < 1 || (two_j & 1) == 0) return -1;
  const int parity = (v.ob.l[a] + v.ob.l[b] + v.ob.l[c]) & 1;
  const int two_tz = v.ob.tz2[a] + v.ob.tz2[b] + v.ob.tz2[c];
  const int ch = basis::ThreeBodyChannels::channel_index(
                     TwoJ{two_j}, static_cast<Parity>(parity), TwoTz{two_tz})
                     .value();
  return ch < v.sp.num_channels ? ch : -1;
}

/// Triple index of the canonical orbits a <= b <= c in channel `ch`, or -1
/// if the triple has no kept row there. Binary search over the channel's
/// triples (lexicographic in (a, b, c)), O(log triples).
IMSRG_INLINE_FUNCTION int FindTriple(const Space3View& sp, int ch, int a, int b,
                                     int c) noexcept {
  int lo = sp.ch_triple_begin[ch];
  int hi = sp.ch_triple_end[ch] - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    const int ta = sp.tr_a[mid];
    const int tb = sp.tr_b[mid];
    const int tc = sp.tr_c[mid];
    const bool less = ta < a || (ta == a && (tb < b || (tb == b && tc < c)));
    if (ta == a && tb == b && tc == c) return mid;
    if (less) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

/// Channel-local row of the kept canonical ket |(a b) Jab, c; ch> (a <= b <=
/// c, `two_jab` = 2 Jab), or -1 if it is not kept. O(log triples + rows of
/// the triple).
IMSRG_INLINE_FUNCTION int FindRow(const N7Views& v, int ch, int a, int b, int c,
                                  int two_jab) noexcept {
  const int t = FindTriple(v.sp, ch, a, b, c);
  if (t < 0) return -1;
  const std::int64_t base = v.sp.ch_row_offset[ch];
  for (int r = v.sp.tr_row_begin[t]; r < v.sp.tr_row_end[t]; ++r) {
    if (v.row_two_jab[base + r] == two_jab) return r;
  }
  return -1;
}

/// imsrg++ `GetME_pn(Jab, Jde, 2J, a, b, c, d, e, f)` of a three-body part
/// for orbits in ANY order: bra |(a b) Jab, c; J>, ket |(d e) Jde, f; J>
/// (Jab, Jde integer two-body J, `two_j` = 2J). Each side is expanded into
/// the kept canonical kets of its sorted triple with PermutationPhase *
/// RecouplingCoefficient; 0 if a side has no kept ket, if bra and ket lie in
/// different channels, or if the part is missing. O(J range^2 log triples).
IMSRG_INLINE_FUNCTION double ThreeBodyElement(const N7Views& v,
                                              const ThreeBodyPartView& t,
                                              int two_j, int a, int b, int c,
                                              int jab, int d, int e, int f,
                                              int jde) noexcept {
  if (t.local == nullptr) return 0.0;
  const int ch = Channel3Of(v, two_j, a, b, c);
  if (ch < 0 || ch != Channel3Of(v, two_j, d, e, f)) return 0.0;
  if (v.sp.ch_dim[ch] == 0) return 0.0;
  int sa = 0;
  int sb = 0;
  int sc = 0;
  const angmom::ThreeBodyPerm pb = angmom::SortOrbits(a, b, c, sa, sb, sc);
  int sd = 0;
  int se = 0;
  int sf = 0;
  const angmom::ThreeBodyPerm pk = angmom::SortOrbits(d, e, f, sd, se, sf);
  const TwoJ ja{v.ob.j2[sa]};
  const TwoJ jb{v.ob.j2[sb]};
  const TwoJ jc{v.ob.j2[sc]};
  const TwoJ jd{v.ob.j2[sd]};
  const TwoJ je{v.ob.j2[se]};
  const TwoJ jf{v.ob.j2[sf]};
  const TwoJ two_J{two_j};
  const int phase = angmom::PermutationPhase(pb) * angmom::PermutationPhase(pk);
  const int jab_lo =
      (ja.twice > jb.twice ? ja.twice - jb.twice : jb.twice - ja.twice) / 2;
  const int jab_hi = (ja.twice + jb.twice) / 2;
  const int jde_lo =
      (jd.twice > je.twice ? jd.twice - je.twice : je.twice - jd.twice) / 2;
  const int jde_hi = (jd.twice + je.twice) / 2;
  // The ket expansion once (rows and coefficients); a two-body J range has at
  // most j_d + j_e - |j_d - j_e| + 1 <= 2 emax + 2 values.
  int ket_row[kMaxKetTerms];
  double ket_coef[kMaxKetTerms];
  int nk = 0;
  for (int j2 = jde_lo; j2 <= jde_hi && nk < kMaxKetTerms; ++j2) {
    const double ck =
        angmom::RecouplingCoefficient(pk, jd, je, jf, jde, j2, two_J, v.sj);
    if (ck == 0.0) continue;
    const int rk = FindRow(v, ch, sd, se, sf, 2 * j2);
    if (rk < 0) continue;
    ket_row[nk] = rk;
    ket_coef[nk] = ck;
    ++nk;
  }
  if (nk == 0) return 0.0;
  double sum = 0.0;
  for (int j1 = jab_lo; j1 <= jab_hi; ++j1) {
    const double cb =
        angmom::RecouplingCoefficient(pb, ja, jb, jc, jab, j1, two_J, v.sj);
    if (cb == 0.0) continue;
    const int rb = FindRow(v, ch, sa, sb, sc, 2 * j1);
    if (rb < 0) continue;
    double inner = 0.0;
    for (int k = 0; k < nk; ++k) {
      inner += ket_coef[k] * ThreeBodyLocal(t, v.sp, ch, rb, ket_row[k]);
    }
    sum += cb * inner;
  }
  return phase * sum;
}

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_N7_VIEWS_H_
