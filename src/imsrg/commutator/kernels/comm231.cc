// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm231ss, [X2, Y3] + [X3, Y2] -> Z1 (imsrg3.h). Kernel TU.

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

/// sum_J3 (2 J3 + 1) [A2^{J2}_{ab,cd} B3^{J3}((cd)J2 i; (ab)J2 j) -
/// B3^{J3}((ab)J2 i; (cd)J2 j) A2^{J2}_{cd,ab}] for fixed orbits. O(J3
/// range * element read).
IMSRG_INLINE_FUNCTION double Weighted3(const detail::N7Views& v,
                                       const detail::TwoBodyView& a2,
                                       const detail::ThreeBodyPartView& b3,
                                       int two_j2, int a, int b, int c, int d,
                                       int i, int j) noexcept {
  const double ab_cd = a2.get_unnormalized(a, b, c, d, two_j2);
  const double cd_ab = a2.get_unnormalized(c, d, a, b, two_j2);
  if (ab_cd == 0.0 && cd_ab == 0.0) return 0.0;
  const int ji = v.ob.j2[i];
  const int lo = two_j2 > ji ? two_j2 - ji : ji - two_j2;
  const int hi = two_j2 + ji;
  const int j2 = two_j2 / 2;
  double sum = 0.0;
  for (int two_j3 = lo; two_j3 <= hi; two_j3 += 2) {
    const double t1 =
        ab_cd == 0.0 ? 0.0
                     : ab_cd * detail::ThreeBodyElement(v, b3, two_j3, c, d, i,
                                                        j2, a, b, j, j2);
    const double t2 =
        cd_ab == 0.0 ? 0.0
                     : cd_ab * detail::ThreeBodyElement(v, b3, two_j3, a, b, i,
                                                        j2, c, d, j, j2);
    sum += (two_j3 + 1) * (t1 - t2);
  }
  return sum;
}

/// The part of Z_ij (without the factor 1 / (2 j_i + 1)) from the two-body
/// channel `ch2`: sum over kets (ab), (cd) of the channel, w_ab w_cd n_a n_b
/// nbar_c nbar_d (Weighted3(X2, Y3) - Weighted3(Y2, X3)). Cost O(kets^2 * J3
/// range * element read).
IMSRG_INLINE_FUNCTION double Comm231Channel(const detail::N7Views& v, int ch2,
                                            int i, int j) noexcept {
  const basis::TwoBodyTableView& tb = v.x2.tb;
  const int two_j2 = tb.channel_two_j[ch2];
  const int nk = tb.channel_num_kets[ch2];
  const std::int32_t* kets = tb.channel_kets + tb.channel_ket_offset[ch2];
  double total = 0.0;
  for (int ka = 0; ka < nk; ++ka) {
    const int a = tb.ket_p[kets[ka]];
    const int b = tb.ket_q[kets[ka]];
    const double nab = v.ob.occupation[a] * v.ob.occupation[b];
    if (nab == 0.0) continue;
    const double wab = a == b ? 0.5 : 1.0;
    for (int kc = 0; kc < nk; ++kc) {
      const int c = tb.ket_p[kets[kc]];
      const int d = tb.ket_q[kets[kc]];
      const double ncd =
          (1.0 - v.ob.occupation[c]) * (1.0 - v.ob.occupation[d]);
      if (ncd == 0.0) continue;
      const double w = wab * (c == d ? 0.5 : 1.0) * nab * ncd;
      total += w * (Weighted3(v, v.x2, v.y3, two_j2, a, b, c, d, i, j) -
                    Weighted3(v, v.y2, v.x3, two_j2, a, b, c, d, i, j));
    }
  }
  return total;
}

}  // namespace

namespace detail {

void Comm231ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const detail::N7Views v = PrepareN7("Comm231ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  const exec::Index norb = v.norb;
  const exec::Index nch = v.x2.tb.num_channels;
  if (norb == 0 || nch == 0) return;
  // Pass 1: one partial sum per (output pair, two-body channel).
  exec::DeviceBuffer<double> partial(
      static_cast<std::size_t>(norb * norb * nch), "commutator::comm231_part");
  const exec::DeviceView<double> part = exec::View(partial);
  exec::ParallelFor(
      "commutator::comm231ss_reference_channels", norb * norb * nch,
      IMSRG_LAMBDA(const exec::Index w) {
        const int ch2 = static_cast<int>(w % nch);
        const exec::Index pair = w / nch;
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        double value = 0.0;
        if (SameChannel(v, i, j)) value = Comm231Channel(v, ch2, i, j);
        part(w) = value;
      });
  // Pass 2: ordered reduction over the channels, added to Z1.
  double* z1 = OneBodyOut(Z);
  exec::ParallelFor(
      "commutator::comm231ss_reference_reduce", norb * norb,
      IMSRG_LAMBDA(const exec::Index pair) {
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        if (!SameChannel(v, i, j)) return;
        double sum = 0.0;
        for (exec::Index ch = 0; ch < nch; ++ch) sum += part(pair * nch + ch);
        z1[pair] += sum / (v.ob.j2[i] + 1);
      });
  exec::Fence("commutator::comm231ss_reference");
}

}  // namespace detail

namespace terms {

void Comm231ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm231ssReference(X, Y, Z, ctx);
}

}  // namespace terms

}  // namespace imsrg::commutator
