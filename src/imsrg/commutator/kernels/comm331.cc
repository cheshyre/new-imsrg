// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm331ss, [X3, Y3] -> Z1 (imsrg3.h). Kernel TU.

#include <cstddef>
#include <cstdint>

#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
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

}  // namespace detail

namespace terms {

void Comm331ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm331ssReference(X, Y, Z, ctx);
}

}  // namespace terms

}  // namespace imsrg::commutator
