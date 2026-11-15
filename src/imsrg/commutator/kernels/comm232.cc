// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The N7 term comm232ss, [X2, Y3] + [X3, Y2] -> Z2 (imsrg3.h). Kernel TU.
//
// m-scheme (three contractions; o_abc = n_a n_b nbar_c + nbar_a nbar_b n_c):
//   [X2, Y3]_{ij,kl} = (s/2) sum_{abc} o_abc
//       { X_{ic,ab} Y_{abj,ckl} - X_{jc,ab} Y_{abi,ckl}
//         - X_{ab,kc} Y_{cij,abl} + X_{ab,lc} Y_{cij,abk} },
//   Z_{ij,kl} += [X2, Y3]_{ij,kl} - [Y2, X3]_{ij,kl}.
//
// J-coupled form (unnormalized elements; bra |(i j) J>, ket |(k l) J>):
//   [X2, Y3]^J_{ijkl} = s { F[X, Y](i, j; k l) - p_ij F[X, Y](j, i; k l)
//                           - F[X^T, Y^T](k, l; i j)
//                           + p_kl F[X^T, Y^T](l, k; i j) },
// p_ij = (-1)^(j_i + j_j - J), with the creator-side half sum
//   F[A, B](p, q; r s) = sum_c sum_{J'} sum_{a <= b} w_ab o_abc
//       sum_{J3} C(p, q, c, J', J, J3) A^{J'}_{(p c),(a b)}
//       B^{J3}((a b) J' q; (r s) J c),
//   C = (2 J3 + 1) Hat(J') / Hat(J) (-1)^(j_q + j_c + J + J')
//       {j_q j_p J; j_c J3 J'},
// w_ab = 1 for a < b and 1/2 for a == b (the m-scheme 1/2 sum over all a,
// b), A^T_{(p c),(a b)} = A_{(a b),(p c)} and B^T(bra; ket) = B(ket; bra).
// The coefficient is (2 J3 + 1) / (2 J + 1) times the recoupling
// <((p q) J, c) J3 | ((p c) J', q) J3> (each m-scheme pair (a b) and (r s)
// keeps its coupling). The stored element is the sum divided by
// sqrt((1 + d_ij)(1 + d_kl)). The annihilator-side terms (with X^T, Y^T) are
// computed explicitly, so no hermiticity of X or Y is assumed (imsrg++
// replaces them by h_X h_Y times the creator side,
// IMSRG3Commutators.cc:2047).

#include "imsrg/commutator/detail/comm232.h"

#include <cstdint>

#include "imsrg/base/constants.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator {

namespace {

/// The overall sign s of the m-scheme expression (file comment), fixed by
/// the m-scheme brute force (commutator_ref::mscheme).
constexpr double kSign = -1.0;

/// (-1)^n for an integer n.
IMSRG_INLINE_FUNCTION int Phase(int n) noexcept { return (n & 1) ? -1 : 1; }

/// The creator-side half sum F[A, B](p, q; r s) at pair J (file comment),
/// reference version: three-body elements through `ThreeBodyElement`.
/// `trans` != 0 evaluates F[A^T, B^T]. Zero if A or B is missing. Cost
/// O(norb * J'-range * kets * J3-range) element reads.
IMSRG_INLINE_FUNCTION double HalfSumReference(
    const detail::N7Views& v, const detail::TwoBodyView& a2,
    const detail::ThreeBodyPartView& b3, int trans, int p, int q, int r, int s,
    int J) noexcept {
  if (a2.data == nullptr || b3.data == nullptr) return 0.0;
  const basis::TwoBodyTableView& tb = a2.tb;
  const int jp = v.ob.j2[p];
  const int jq = v.ob.j2[q];
  const int two_J = 2 * J;
  double total = 0.0;
  for (int c = 0; c < v.norb; ++c) {
    const int jc = v.ob.j2[c];
    const double nc = v.ob.occupation[c];
    const int lo2 = (jp > jc ? jp - jc : jc - jp) / 2;
    const int hi2 = (jp + jc) / 2;
    for (int Jp = lo2; Jp <= hi2; ++Jp) {
      const int ch2 =
          tb.channel_of(OrbitIndex{p}, OrbitIndex{c}, TwoJ{2 * Jp}).value();
      if (ch2 >= tb.num_channels) continue;
      const int d1 = 2 * Jp > jq ? 2 * Jp - jq : jq - 2 * Jp;
      const int d2 = two_J > jc ? two_J - jc : jc - two_J;
      const int lo3 = d1 > d2 ? d1 : d2;
      const int hi3 = 2 * Jp + jq < two_J + jc ? 2 * Jp + jq : two_J + jc;
      if (lo3 > hi3) continue;
      const double pref =
          Phase((jq + jc) / 2 + J + Jp) * v.sj.Hat(Jp) / v.sj.Hat(J);
      const int nk = tb.channel_num_kets[ch2];
      const std::int32_t* kets = tb.channel_kets + tb.channel_ket_offset[ch2];
      for (int ka = 0; ka < nk; ++ka) {
        const int a = tb.ket_p[kets[ka]];
        const int b = tb.ket_q[kets[ka]];
        const double na = v.ob.occupation[a];
        const double nb = v.ob.occupation[b];
        double occ = na * nb * (1.0 - nc) + (1.0 - na) * (1.0 - nb) * nc;
        if (occ == 0.0) continue;
        if (a == b) occ *= 0.5;
        const double x = trans != 0 ? a2.get_unnormalized(a, b, p, c, 2 * Jp)
                                    : a2.get_unnormalized(p, c, a, b, 2 * Jp);
        if (x == 0.0) continue;
        double acc = 0.0;
        for (int two_j3 = lo3; two_j3 <= hi3; two_j3 += 2) {
          const double six = v.sj(TwoJ{jq}, TwoJ{jp}, TwoJ{two_J}, TwoJ{jc},
                                  TwoJ{two_j3}, TwoJ{2 * Jp});
          if (six == 0.0) continue;
          const double y = trans != 0
                               ? detail::ThreeBodyElement(v, b3, two_j3, r, s,
                                                          c, J, a, b, q, Jp)
                               : detail::ThreeBodyElement(v, b3, two_j3, a, b,
                                                          q, Jp, r, s, c, J);
          acc += (two_j3 + 1) * six * y;
        }
        total += pref * occ * x * acc;
      }
    }
  }
  return total;
}

/// The four bracket terms of [A2, B3]^J_{ijkl} without the sign s (file
/// comment), reference version. O(4 HalfSumReference).
IMSRG_INLINE_FUNCTION double BracketReference(
    const detail::N7Views& v, const detail::TwoBodyView& a2,
    const detail::ThreeBodyPartView& b3, int i, int j, int k, int l,
    int J) noexcept {
  const int p_ij = Phase((v.ob.j2[i] + v.ob.j2[j]) / 2 - J);
  const int p_kl = Phase((v.ob.j2[k] + v.ob.j2[l]) / 2 - J);
  return HalfSumReference(v, a2, b3, 0, i, j, k, l, J) -
         p_ij * HalfSumReference(v, a2, b3, 0, j, i, k, l, J) -
         HalfSumReference(v, a2, b3, 1, k, l, i, j, J) +
         p_kl * HalfSumReference(v, a2, b3, 1, l, k, i, j, J);
}

}  // namespace

namespace detail {

void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm232ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm232ss", Z, ctx);
  if (v.x3.data == nullptr && v.y3.data == nullptr) return;
  const std::int64_t total = Z.two_body().matrix().total_size();
  const basis::TwoBodyTableView tb = v.x2.tb;
  exec::ParallelFor(
      "commutator::comm232ss_reference", static_cast<exec::Index>(total),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = FindBlock(out.block_offset, tb.num_channels, e);
        const std::int64_t n = tb.channel_num_kets[ch];
        const std::int64_t idx = e - out.block_offset[ch];
        const int ib = static_cast<int>(idx % n);
        const int ik = static_cast<int>(idx / n);
        const int J = tb.channel_two_j[ch] / 2;
        const int koff = tb.channel_ket_offset[ch];
        const int bra = tb.channel_kets[koff + ib];
        const int ket = tb.channel_kets[koff + ik];
        const int i = tb.ket_p[bra];
        const int j = tb.ket_q[bra];
        const int k = tb.ket_p[ket];
        const int l = tb.ket_q[ket];
        double z = BracketReference(v, v.x2, v.y3, i, j, k, l, J) -
                   BracketReference(v, v.y2, v.x3, i, j, k, l, J);
        z *= kSign;
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        out.data[e] += z;
      });
  exec::Fence("commutator::comm232ss_reference");
}

}  // namespace detail

namespace terms {

void Comm232ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm232ssReference(X, Y, Z, ctx);
}

}  // namespace terms

}  // namespace imsrg::commutator
