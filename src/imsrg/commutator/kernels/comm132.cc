// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm132ss, [X1, Y3] + [X3, Y1] -> Z2 (imsrg3.h). Kernel TU.
//
// J-scheme form (bra |(i j) J>, ket |(k l) J>, a and b in the same one-body
// channel, 2J3 = |2J - 2 j_b| .. 2J + 2 j_b):
//   Z^J_ijkl (unnormalized) = sum_{ab} (n_a - n_b) sum_{J3} (2 J3 + 1)/(2 J +
//   1)
//       [ X_ab Y3^{J3}((ij)J b; (kl)J a) - Y_ab X3^{J3}((ij)J b; (kl)J a) ],
// added to the stored normalized element divided by sqrt((1 + d_ij)(1 +
// d_kl)). One parallel task per stored element of Z's two-body part (both
// triangles of every block), so there are no atomics.

#include <cstdint>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator::terms {

void Comm132ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  const detail::N7Views v = detail::PrepareN7("Comm132ss", X, Y, Z, ctx);
  const detail::TwoBodyOutView out = detail::TwoBodyOut("Comm132ss", Z, ctx);
  if (v.x3.data == nullptr && v.y3.data == nullptr) return;
  const std::int64_t total = Z.two_body().matrix().total_size();
  const basis::TwoBodyTableView tb = v.x2.tb;
  exec::ParallelFor(
      "commutator::comm132ss", static_cast<exec::Index>(total),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = detail::FindBlock(out.block_offset, tb.num_channels, e);
        const std::int64_t n = tb.channel_num_kets[ch];
        const std::int64_t idx = e - out.block_offset[ch];
        const int ib = static_cast<int>(idx % n);
        const int ik = static_cast<int>(idx / n);
        const int two_J = tb.channel_two_j[ch];
        const int J = two_J / 2;
        const int koff = tb.channel_ket_offset[ch];
        const int bra = tb.channel_kets[koff + ib];
        const int ket = tb.channel_kets[koff + ik];
        const int i = tb.ket_p[bra];
        const int j = tb.ket_q[bra];
        const int k = tb.ket_p[ket];
        const int l = tb.ket_q[ket];
        double z = 0.0;
        for (int a = 0; a < v.norb; ++a) {
          const double na = v.ob.occupation[a];
          for (int m = v.ob_ch.offsets[a]; m < v.ob_ch.offsets[a + 1]; ++m) {
            const int b = v.ob_ch.orbits[m];
            const double dn = na - v.ob.occupation[b];
            if (dn == 0.0) continue;
            const double xab = detail::OneBodyElement(v.x1, v.norb, a, b);
            const double yab = detail::OneBodyElement(v.y1, v.norb, a, b);
            if (xab == 0.0 && yab == 0.0) continue;
            const int j2b = v.ob.j2[b];
            const int lo = two_J > j2b ? two_J - j2b : j2b - two_J;
            double acc = 0.0;
            for (int two_j3 = lo; two_j3 <= two_J + j2b; two_j3 += 2) {
              double t = 0.0;
              if (xab != 0.0) {
                t += xab * detail::ThreeBodyElement(v, v.y3, two_j3, i, j, b, J,
                                                    k, l, a, J);
              }
              if (yab != 0.0) {
                t -= yab * detail::ThreeBodyElement(v, v.x3, two_j3, i, j, b, J,
                                                    k, l, a, J);
              }
              acc += (two_j3 + 1) * t;
            }
            z += dn * acc;
          }
        }
        z /= (two_J + 1);
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        out.data[e] += z;
      });
  exec::Fence("commutator::comm132ss");
}

}  // namespace imsrg::commutator::terms
