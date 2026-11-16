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
// One parallel task per bra row computes the stored elements of its row (the
// packed lower triangle j <= i, or the dense row), so there are no atomics.

#include <cstdint>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator::terms {

namespace {

/// T(X, Y) of the file comment for `x1` (X's one-body part) and `y3` (Y's
/// three-body part); `two_j` is 2J of the channel. O(sum of the six
/// one-body channel sizes * cost of `ThreeBodyElement`).
IMSRG_INLINE_FUNCTION double OneBodyOnThreeBody(
    const detail::N7Views& v, const double* x1,
    const detail::ThreeBodyPartView& y3, int two_j, int p, int q, int r,
    int jpq, int s, int t, int u, int jst) noexcept {
  if (x1 == nullptr || y3.data == nullptr) return 0.0;
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

void Comm133ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  const detail::N7Views v = detail::PrepareN7("Comm133ss", X, Y, Z, ctx);
  const detail::ThreeBodyOutView out =
      detail::ThreeBodyOut("Comm133ss", Z, ctx);
  if (v.x3.data == nullptr && v.y3.data == nullptr) return;
  const exec::Index nrows = static_cast<exec::Index>(
      ctx.model_space_ptr()->three_body_space()->NumRows());
  exec::ParallelFor(
      "commutator::comm133ss", nrows, IMSRG_LAMBDA(const exec::Index r) {
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
          const std::int64_t idx = out.packed != 0
                                       ? v.sp.ch_packed_offset[ch] +
                                             op::ThreeBody::PackedIndex(n, i, j)
                                       : v.sp.ch_dense_offset[ch] + i +
                                             static_cast<std::int64_t>(j) * n;
          out.data[idx] += z;
        }
      });
  exec::Fence("commutator::comm133ss");
}

}  // namespace imsrg::commutator::terms
