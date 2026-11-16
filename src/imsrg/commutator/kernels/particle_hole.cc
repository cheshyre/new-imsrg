// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Particle-hole (Pandya) term comm222_phss (design/12 section 4.2; imsrg++
// `Comm222Phss`, reference/02 P-12). Kernel TU: includes the Kokkos
// vocabulary.
//
// Algorithm. For every cross-coupled channel c (J', parity, Tz) with Ncc kets,
// build
//   XK(alpha, kappa) = D_kappa xbar(i, l, a, b; J')
//   YK(kappa, beta)  = ybar(a, b, k, j; J')
// (alpha = ket (i, l), kappa = (a, b), beta = ordered pair (k, j)),
// GEMM W = XK YK, combine W in place into the Pandya-transformed result (ZB
// in imsrg++), and transform back to the coupled two-body matrix element.
// Only rows alpha (kets, first orbit <= second) are computed: the remaining
// rows follow from hermiticity of X, Y and from the flip symmetry of the
// ordered-pair basis; the inverse transform reads only the stored half (with
// an explicit sign for flipped bra pairs). The ket-pair inverse relies on the
// local ket order of every two-body channel being the global ket order
// (`basis::TwoBodyChannels`), so that b >= a is the upper triangle.
//
// Performance (M12a): the fill, flip and combine kernels work on whole rows
// or columns (one item per cc ket, dynamic scheduling) and the inverse on
// pairs of kets (all J at once) instead of one item per element, so the block
// search and index divisions run once per item; the 6j symbols of a J' (or
// J'') sum come from one contiguous table row; the fill evaluates the X and Y
// transforms in one pass over the shared indices; only half of YK is stored
// (second GEMM after a row-pair flip). The arithmetic on every value is that
// of the per-element kernels, except that the GEMMs now have n x n instead of
// n x 2n shapes (round-off level differences where BLAS changes its path).

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/pandya.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::commutator::terms {

namespace {

using detail::PandyaView;
using detail::TwoBodyView;

/// POD bundle of everything the particle-hole kernels read.
struct PhArgs {
  PandyaView pv;
  TwoBodyView xv;
  TwoBodyView yv;
  TwoBodyView zv;
  angmom::SixJView sj;
  double hx =
      1.0;  ///< Hermiticity sign of X (+1 Hermitian, -1 anti-Hermitian).
  double hy = 1.0;    ///< Hermiticity sign of Y.
  double hxhy = 1.0;  ///< hX hY.
};
static_assert(std::is_trivially_copyable_v<PhArgs>);

/// (-1)^n for integer n (any sign).
IMSRG_INLINE_FUNCTION double ParitySign(int n) noexcept {
  return (n & 1) ? -1.0 : 1.0;
}

/// Capacity of the per-pair J' buffers of the inverse transform (checked by
/// `BuildPandyaData`). On GPUs these 1 KiB per thread arrays would live in
/// local memory; team scratch is the M14 alternative.
constexpr int kMaxJp = detail::kPandyaMaxJp;

/// Pandya transform xbar(p, q, r, s; J') = - sum_{J''} (2J''+1) {jp jq J'; jr
/// js J''} X~^{J''}_{psrq} with X~ the unnormalized coupled element. `two_Jp`
/// is 2J'. O(J'' range).
IMSRG_INLINE_FUNCTION double PandyaBar(const TwoBodyView& x,
                                       const angmom::SixJView& sj, int p, int q,
                                       int r, int s, int two_Jp) noexcept {
  const int* j2 = x.tb.orbit_j2;
  const int jp = j2[p], jq = j2[q], jr = j2[r], js = j2[s];
  int lo = jp > js ? jp - js : js - jp;
  const int lo2 = jr > jq ? jr - jq : jq - jr;
  if (lo2 > lo) lo = lo2;
  const int hi1 = jp + js;
  const int hi2 = jr + jq;
  const int hi = hi1 < hi2 ? hi1 : hi2;
  double sum = 0.0;
  for (int two_Jpp = lo; two_Jpp <= hi; two_Jpp += 2) {
    const double sixj =
        sj(TwoJ{jp}, TwoJ{jq}, TwoJ{two_Jp}, TwoJ{jr}, TwoJ{js}, TwoJ{two_Jpp});
    if (sixj == 0.0) continue;
    sum += (two_Jpp + 1) * sixj * x.get_unnormalized(p, s, r, q, two_Jpp);
  }
  return -sum;
}

/// Local index of the ket (a <= b) at J' in its cc channel, -1 if absent. O(1).
IMSRG_INLINE_FUNCTION int CcLocal(const PhArgs& a, int oa, int ob,
                                  int Jp) noexcept {
  const auto& tb = a.zv.tb;
  const int ja = tb.orbit_j2[oa], jb = tb.orbit_j2[ob];
  const int jmin = (ja > jb ? ja - jb : jb - ja) / 2;
  if (Jp < jmin || Jp > (ja + jb) / 2) return -1;
  const int k = tb.ket_index(OrbitIndex{oa}, OrbitIndex{ob}).value();
  return a.pv.local_index_table[a.pv.ket_J_offsets[k] + Jp - jmin];
}

/// cc channel index of the pair (p, q) at J'.
IMSRG_INLINE_FUNCTION int CcChannel(const PhArgs& a, int p, int q,
                                    int Jp) noexcept {
  const auto& tb = a.zv.tb;
  const int dtz = tb.orbit_tz2[p] - tb.orbit_tz2[q];
  return 3 * (2 * Jp) + (dtz < 0 ? -dtz : dtz) + 2 +
         ((tb.orbit_l[p] + tb.orbit_l[q]) & 1);
}

/// Both Pandya transforms of `PandyaBar` for the same orbits at once, xbar for
/// X and ybar for Y: X and Y share every index computation. The 6j symbols
/// come from one contiguous table row (`SixJView::row`), the kets of the pairs
/// (p, s) and (r, q) and their channel class (parity, Tz) are resolved once,
/// and only the local indices (table reads), the channel and the exchange
/// phase remain per J''. The operations on each value are those of
/// `PandyaBar` and `TwoBodyView::get_unnormalized` in the same order, so the
/// results are bitwise identical. O(J'' range).
IMSRG_FORCEINLINE_FUNCTION void PandyaBarXY(const PhArgs& a, int p, int q,
                                            int r, int s, int two_Jp,
                                            double& xbar,
                                            double& ybar) noexcept {
  const auto& tb = a.xv.tb;
  const int* j2 = tb.orbit_j2;
  const int jp = j2[p], jq = j2[q], jr = j2[r], js = j2[s];
  int lo = jp > js ? jp - js : js - jp;
  const int lo2 = jr > jq ? jr - jq : jq - jr;
  if (lo2 > lo) lo = lo2;
  const int hi1 = jp + js;
  const int hi2 = jr + jq;
  const int hi = hi1 < hi2 ? hi1 : hi2;
  double sx = 0.0;
  double sy = 0.0;
  // X~^{J''}_{psrq}: bra pair (p, s), ket pair (r, q), each stored as the ket
  // (min, max) with an exchange phase if swapped. Both kets must lie in the
  // same channel class, else every term vanishes (TwoBodyView returns 0).
  const bool swap_bra = p > s;
  const bool swap_ket = r > q;
  const int a1 = swap_bra ? s : p;
  const int b1 = swap_bra ? p : s;
  const int a2 = swap_ket ? q : r;
  const int b2 = swap_ket ? r : q;
  const int par1 = (tb.orbit_l[a1] + tb.orbit_l[b1]) % 2;
  const int par2 = (tb.orbit_l[a2] + tb.orbit_l[b2]) % 2;
  const int tz1 = tb.orbit_tz2[a1] + tb.orbit_tz2[b1];
  if (lo <= hi && par1 == par2 && tz1 == tb.orbit_tz2[a2] + tb.orbit_tz2[b2]) {
    const int k1 = tb.ket_index(OrbitIndex{a1}, OrbitIndex{b1}).value();
    const int k2 = tb.ket_index(OrbitIndex{a2}, OrbitIndex{b2}).value();
    const int ja1 = j2[a1], jb1 = j2[b1], ja2 = j2[a2], jb2 = j2[b2];
    // Local index tables of the two kets, indexed by J'' (valid on the
    // J'' range, which lies inside both triangles).
    const std::int32_t* lit1 = tb.local_index_table + tb.ket_J_offsets[k1] -
                               (ja1 > jb1 ? ja1 - jb1 : jb1 - ja1) / 2;
    const std::int32_t* lit2 = tb.local_index_table + tb.ket_J_offsets[k2] -
                               (ja2 > jb2 ? ja2 - jb2 : jb2 - ja2) / 2;
    const int cls = tz1 + 2 + par1;  // channel = 3 (2J'') + cls
    const double* row =
        a.sj.row(TwoJ{jp}, TwoJ{jq}, TwoJ{two_Jp}, TwoJ{jr}, TwoJ{js});
    const double* xd = a.xv.data;
    const double* yd = a.yv.data;
    const std::int64_t* block_offset = a.xv.block_offset;
    for (int two_Jpp = lo; two_Jpp <= hi; two_Jpp += 2) {
      const int Jpp = two_Jpp / 2;
      const double sixj = row[Jpp];
      if (sixj == 0.0) continue;
      const std::int32_t ib = lit1[Jpp];
      if (ib < 0) continue;
      const std::int32_t ik = lit2[Jpp];
      if (ik < 0) continue;
      const int ch = 3 * two_Jpp + cls;
      const std::int64_t idx =
          block_offset[ch] + ib +
          static_cast<std::int64_t>(ik) * tb.channel_num_kets[ch];
      int phase = 1;
      if (swap_bra)
        phase *= basis::ExchangePhase(TwoJ{jp}, TwoJ{js}, TwoJ{two_Jpp});
      if (swap_ket)
        phase *= basis::ExchangePhase(TwoJ{jr}, TwoJ{jq}, TwoJ{two_Jpp});
      double vx = phase * xd[idx];
      double vy = phase * yd[idx];
      if (p == s) {
        vx *= constants::kSqrt2;
        vy *= constants::kSqrt2;
      }
      if (r == q) {
        vx *= constants::kSqrt2;
        vy *= constants::kSqrt2;
      }
      const double w = (two_Jpp + 1) * sixj;
      sx += w * vx;
      sy += w * vy;
    }
  }
  xbar = -sx;
  ybar = -sy;
}

/// One of the two Pandya-transformed elements zbar(p, q, r, s; J') that an
/// element of Z needs, with everything that does not depend on J' resolved
/// once (see `MakeCcTerm`; read by `CcValue`). Trivially copyable.
struct CcTerm {
  const std::int32_t* lit_bra = nullptr;  ///< cc local index of the bra pair
                                          ///< (min, max), indexed by J'.
  const std::int32_t* lit_ket = nullptr;  ///< Same for the ket pair.
  int lo = 1;             ///< First J' at which both cc kets exist.
  int hi = 0;             ///< Last such J' (lo > hi: the term vanishes).
  int cls = 0;            ///< cc channel = 6 J' + cls.
  bool ket_flip = false;  ///< Column = ket local index + Ncc (flipped pair).
  double sign = 1.0;      ///< 1, or -hX hY phi for a flipped bra pair.
};
static_assert(std::is_trivially_copyable_v<CcTerm>);

/// Resolves zbar(p, q, r, s; J') of the stored half of the Pandya-transformed
/// result (rows = cc kets, columns = ordered pairs, see detail/pandya.h): for
/// p <= q the element sits at row local(p, q), column local(min(r, s), max(r,
/// s)) (+ Ncc if r > s); for p > q, Hermiticity and the flip symmetry give
/// -hX hY phi times the element of row local(q, p), column local(min, max) (+
/// Ncc if s > r), phi = (-1)^((jp + jq)/2 + (jr + js)/2). Pairs of different
/// cc channel classes give an empty J' range. O(1).
IMSRG_FORCEINLINE_FUNCTION CcTerm MakeCcTerm(const PhArgs& a, int p, int q,
                                             int r, int s) noexcept {
  // Precomputed descriptors of the cc kets of (p, q) and (r, s)
  // (`PandyaView::cc_pair`): local-index-table offset, J' range, class.
  const int norb = a.zv.tb.num_orbits;
  const std::int32_t* d1 = a.pv.cc_pair + 4 * (p * norb + q);
  const std::int32_t* d2 = a.pv.cc_pair + 4 * (r * norb + s);
  CcTerm t;
  const int cls = d1[3] & 255;
  if (cls != (d2[3] & 255)) return t;
  t.cls = cls;
  if (p <= q) {
    t.ket_flip = r > s;
  } else {
    t.ket_flip = s > r;
    // phi = (-1)^((jp + jq)/2 + (jr + js)/2).
    const double phi = ParitySign((d1[3] >> 8) + (d2[3] >> 8));
    t.sign = -a.hxhy * phi;
  }
  t.lit_bra = a.pv.local_index_table + d1[0];
  t.lit_ket = a.pv.local_index_table + d2[0];
  t.lo = d1[1] > d2[1] ? d1[1] : d2[1];
  t.hi = d1[2] < d2[2] ? d1[2] : d2[2];
  return t;
}

/// zbar(p, q, r, s; J') for the term `t` made by `MakeCcTerm`, read from the
/// combined W (0 if a cc ket does not exist at J'). O(1).
IMSRG_FORCEINLINE_FUNCTION double CcValue(const PhArgs& a, const CcTerm& t,
                                          int Jp) noexcept {
  if (Jp < t.lo || Jp > t.hi) return 0.0;
  const int c = 6 * Jp + t.cls;
  if (c >= a.pv.num_cc) return 0.0;
  const int n = a.pv.ncc[c];
  if (n == 0) return 0.0;
  const std::int32_t alpha = t.lit_bra[Jp];
  const std::int32_t lb = t.lit_ket[Jp];
  if (alpha < 0 || lb < 0) return 0.0;
  const int beta = lb + (t.ket_flip ? n : 0);
  return t.sign *
         a.pv.w[a.pv.w_off[c] + alpha + static_cast<std::int64_t>(beta) * n];
}

/// POD bundle of the general-path kernels (PhArgs plus the general scratch).
struct PhGeneralArgs {
  PhArgs ph;
  const std::int64_t* g_off = nullptr;   ///< Offsets of the A and B blocks.
  const std::int64_t* gz_off = nullptr;  ///< Offsets of the Z blocks.
  double* a = nullptr;
  double* b = nullptr;
  double* z = nullptr;
};
static_assert(std::is_trivially_copyable_v<PhGeneralArgs>);

/// Element zbar(p, q, r, s; J') of the general Pandya-transformed result:
/// rows and columns are the ordered pairs (index local(min, max) + Ncc for a
/// flipped pair) of the cc channel. Zero if a pair does not exist at J'.
IMSRG_INLINE_FUNCTION double ZbarGeneral(const PhGeneralArgs& g, int p, int q,
                                         int r, int s, int Jp) noexcept {
  const PhArgs& a = g.ph;
  const int c = CcChannel(a, p, q, Jp);
  if (c < 0 || c >= a.pv.num_cc || c != CcChannel(a, r, s, Jp)) return 0.0;
  const int n = a.pv.ncc[c];
  if (n == 0) return 0.0;
  const int la = p <= q ? CcLocal(a, p, q, Jp) : CcLocal(a, q, p, Jp);
  const int lb = r <= s ? CcLocal(a, r, s, Jp) : CcLocal(a, s, r, Jp);
  if (la < 0 || lb < 0) return 0.0;
  const int alpha = la + (p > q ? n : 0);
  const int beta = lb + (r > s ? n : 0);
  return g.z[g.gz_off[c] + alpha + static_cast<std::int64_t>(beta) * 2 * n];
}

/// The particle-hole term for general X or Y (see `EnsureGeneralScratch`).
void Comm222PhssGeneral(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, Workspace::Impl& impl) {
  detail::PandyaData& pd = impl.pandya;
  detail::EnsureGeneralScratch(pd);
  PhGeneralArgs g;
  g.ph.pv = pd.view();
  g.ph.xv = impl.two_body_view(X);
  g.ph.yv = impl.two_body_view(Y);
  g.ph.zv = impl.two_body_view(Z);
  g.ph.sj = impl.sixj.device_view();
  g.g_off = pd.general.g_off.device_data();
  g.gz_off = pd.general.gz_off.device_data();
  g.a = pd.general.a.device_data();
  g.b = pd.general.b.device_data();
  g.z = pd.general.z.device_data();

  const int num_cc = pd.num_cc;
  const auto g_total = static_cast<exec::Index>(pd.general.g_off_host.back());
  // Host BLAS only; device backends need a device gemm (design/12 section 3).
  IMSRG_CHECK(pd.general.z.host_is_device(),
              "comm222_phss: host BLAS on device memory; needs a device gemm "
              "(GPU backends)");
  // Two passes share the A and B scratch (each 2 Ncc x nk): Z = A0 B0 + A1 B1
  // with, for kappa = (a, b) a pair of K and ordered cc pairs (p, q),
  //   pass 0: A0(alpha, kappa) =  D_kappa Xbar(alpha-pair, kappa),
  //           B0(kappa, beta)  =  Ybar(kappa, beta-pair),
  //   pass 1: A1(alpha, kappa) = -D_kappa Ybar(alpha-pair, kappa),
  //           B1(kappa, beta)  =  Xbar(kappa, beta-pair),
  // the second GEMM accumulating (beta = 1) into Z.
  for (int pass = 0; pass < 2; ++pass) {
    if (g_total > 0) {
      exec::ParallelFor(
          "commutator::comm222_phss_general::fill", g_total,
          IMSRG_LAMBDA(const exec::Index e) {
            const PhArgs& a = g.ph;
            const int c = detail::FindBlock(g.g_off, num_cc, e);
            const std::int64_t local = e - g.g_off[c];
            const int n = a.pv.ncc[c];
            const int nk = a.pv.nk[c];
            if (nk == 0) return;
            const int two_Jp = 2 * (c / 6);
            const int* kp = a.pv.ket_p + a.pv.ket_off[c];
            const int* kq = a.pv.ket_q + a.pv.ket_off[c];
            const TwoBodyView& first = pass == 0 ? a.xv : a.yv;
            const TwoBodyView& second = pass == 0 ? a.yv : a.xv;
            // A element (row alpha < 2n, column kappa < nk).
            {
              const int alpha = static_cast<int>(local % (2 * n));
              const int kappa = static_cast<int>(local / (2 * n));
              const int ka = a.pv.k_off[c] + kappa;
              const int pa = alpha < n ? kp[alpha] : kq[alpha - n];
              const int qa = alpha < n ? kq[alpha] : kp[alpha - n];
              const double d = pass == 0 ? a.pv.k_d[ka] : -a.pv.k_d[ka];
              g.a[e] = d * PandyaBar(first, a.sj, pa, qa, a.pv.k_a[ka],
                                     a.pv.k_b[ka], two_Jp);
            }
            // B element (row kappa < nk, column beta < 2n).
            {
              const int kappa = static_cast<int>(local % nk);
              const int beta = static_cast<int>(local / nk);
              const int ka = a.pv.k_off[c] + kappa;
              const int rb = beta < n ? kp[beta] : kq[beta - n];
              const int sb = beta < n ? kq[beta] : kp[beta - n];
              g.b[e] = PandyaBar(second, a.sj, a.pv.k_a[ka], a.pv.k_b[ka], rb,
                                 sb, two_Jp);
            }
          });
    }
    exec::Fence("commutator::comm222_phss_general::fill");
    for (auto& gemm : pd.general.gemms) gemm.beta = pass == 0 ? 0.0 : 1.0;
    linalg::GroupedGemm(pd.general.gemms);
  }

  double* z2 = Z.two_body().matrix().buffer().device_data();
  const std::int64_t* block_offset = impl.block_offsets.device_data();
  const int nch = g.ph.zv.tb.num_channels;
  const auto total = static_cast<exec::Index>(impl.two_body_size);
  // Inverse transform of every stored element (both triangles, no mirroring).
  exec::ParallelFor(
      "commutator::comm222_phss_general::inverse", total,
      IMSRG_LAMBDA(const exec::Index e) {
        const auto& tb = g.ph.zv.tb;
        const int ch = detail::FindBlock(block_offset, nch, e);
        const std::int64_t local = e - block_offset[ch];
        const int n = tb.channel_num_kets[ch];
        const int ib = static_cast<int>(local % n);
        const int ik = static_cast<int>(local / n);
        const int* kets = tb.channel_kets + tb.channel_ket_offset[ch];
        const int i = tb.ket_p[kets[ib]];
        const int j = tb.ket_q[kets[ib]];
        const int k = tb.ket_p[kets[ik]];
        const int l = tb.ket_q[kets[ik]];
        const int two_J = tb.channel_two_j[ch];
        const int* j2 = tb.orbit_j2;
        const int ji = j2[i], jj = j2[j], jk = j2[k], jl = j2[l];
        const auto iabs = [](int v) { return v < 0 ? -v : v; };
        const auto imax = [](int u, int v) { return u > v ? u : v; };
        const auto imin = [](int u, int v) { return u < v ? u : v; };
        const int lo2 = imin(imax(iabs(ji - jl), iabs(jj - jk)),
                             imax(iabs(jj - jl), iabs(ji - jk)));
        const int hi2 = imax(imin(ji + jl, jj + jk), imin(jj + jl, ji + jk));
        const double ph_ij = ParitySign((ji + jj) / 2 - two_J / 2);
        double z = 0.0;
        for (int Jp = lo2 / 2; Jp <= hi2 / 2; ++Jp) {
          const double s1 = g.ph.sj(TwoJ{ji}, TwoJ{jj}, TwoJ{two_J}, TwoJ{jk},
                                    TwoJ{jl}, TwoJ{2 * Jp});
          const double s2 = g.ph.sj(TwoJ{jj}, TwoJ{ji}, TwoJ{two_J}, TwoJ{jk},
                                    TwoJ{jl}, TwoJ{2 * Jp});
          double t = 0.0;
          if (s1 != 0.0) t += s1 * ZbarGeneral(g, i, l, k, j, Jp);
          if (s2 != 0.0) t -= ph_ij * s2 * ZbarGeneral(g, j, l, k, i, Jp);
          z += (2 * Jp + 1) * t;
        }
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        z2[e] += z;
      });
  exec::Fence("commutator::comm222_phss_general");
}

}  // namespace

void Comm222Phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  if (!X.two_body().allocated() || !Y.two_body().allocated() ||
      !Z.two_body().allocated())
    return;
  auto& impl = ws.impl();
  const detail::PandyaData& pd = impl.pandya;
  if (pd.num_cc == 0) return;
  if (detail::IsGeneral(X, Y)) {
    Comm222PhssGeneral(X, Y, Z, impl);
    return;
  }

  PhArgs a;
  a.pv = pd.view();
  a.xv = impl.two_body_view(X);
  a.yv = impl.two_body_view(Y);
  a.zv = impl.two_body_view(Z);
  a.sj = impl.sixj.device_view();
  const double hx = X.hermiticity() == Hermiticity::kHermitian ? 1.0 : -1.0;
  const double hy = Y.hermiticity() == Hermiticity::kHermitian ? 1.0 : -1.0;
  a.hx = hx;
  a.hy = hy;
  a.hxhy = hx * hy;

  const int num_cc = pd.num_cc;
  // One work item per cc ket of every channel (rows of XK, columns of W).
  exec::Index cc_kets = 0;
  for (const std::int32_t v : pd.ncc_host) cc_kets += v;
  const std::int32_t* ket_off = pd.ket_off.device_data();

  if (pd.xk_off_host.back() > 0) {
    // Item (c, beta), beta = ket (k, j) of cc channel c: for every pair kappa
    // = (a, b) of K one fused evaluation xbar = Xbar(kappa, beta), ybar =
    // Ybar(kappa, beta) (`PandyaBarXY`) fills
    //   XK(beta, kappa) = D_kappa hX xbar  (Xbar^T = hX Xbar),
    //   YK(kappa, beta) = ybar.
    // (The flipped-pair half YK(kappa ^ 1, beta + Ncc) = hY phi_kappa phi_beta
    // ybar of F Ybar F = hY Phi Ybar Phi is not stored: see the flip below.)
    // Columns of YK are contiguous; the row of XK is strided by Ncc but reused
    // by the neighbouring items.
    exec::ParallelForDynamic(
        "commutator::comm222_phss::fill", cc_kets,
        IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::FindBlock(ket_off, num_cc, e);
          const int nk = a.pv.nk[c];
          if (nk == 0) return;
          const int n = a.pv.ncc[c];
          const int beta = static_cast<int>(e - ket_off[c]);
          const int k = a.pv.ket_p[e];
          const int j = a.pv.ket_q[e];
          const int two_Jp = 2 * (c / 6);
          double* xk = a.pv.xk + a.pv.xk_off[c] + beta;
          double* yk =
              a.pv.yk + a.pv.yk_off[c] + static_cast<std::int64_t>(beta) * nk;
          const int koff = a.pv.k_off[c];
          for (int kappa = 0; kappa < nk; ++kappa) {
            const int ka = koff + kappa;
            const int ia = a.pv.k_a[ka];
            const int ib = a.pv.k_b[ka];
            double xbar = 0.0;
            double ybar = 0.0;
            PandyaBarXY(a, ia, ib, k, j, two_Jp, xbar, ybar);
            xk[static_cast<std::int64_t>(kappa) * n] =
                a.pv.k_d[ka] * a.hx * xbar;
            yk[kappa] = ybar;
          }
        });
  }
  exec::Fence("commutator::comm222_phss::fill");
  // Host BLAS only; device backends need a device gemm (design/12 section 3).
  IMSRG_CHECK(pd.w.host_is_device(),
              "comm222_phss: host BLAS on device memory; needs a device gemm "
              "(GPU backends)");
  linalg::GroupedGemm(pd.gemms);  // left halves: W(:, 0 .. Ncc) = XK YK

  // The right half of W needs YK_R(kappa ^ 1, beta) = hY phi_kappa phi_beta
  // YK(kappa, beta). With YK'(kappa, beta) = phi_kappa YK(kappa ^ 1, beta)
  // (row pairs swapped in place, phi_kappa = phi_{kappa ^ 1}),
  //   [XK YK_R](alpha, beta) = hY phi_beta [XK YK'](alpha, beta),
  // so a second GEMM writes W'(:, Ncc + beta) = [XK YK'](:, beta) and the
  // combine below applies hY phi_beta. Item (c, beta) flips column beta.
  exec::ParallelForDynamic(
      "commutator::comm222_phss::flip", cc_kets,
      IMSRG_LAMBDA(const exec::Index e) {
        const int c = detail::FindBlock(ket_off, num_cc, e);
        const int nk = a.pv.nk[c];
        if (nk == 0) return;
        const std::int64_t beta = e - ket_off[c];
        double* yk = a.pv.yk + a.pv.yk_off[c] + beta * nk;
        const int* ka = a.pv.k_a + a.pv.k_off[c];
        const int* kb = a.pv.k_b + a.pv.k_off[c];
        const int* j2 = a.zv.tb.orbit_j2;
        for (int kappa = 0; kappa < nk; kappa += 2) {
          const double phi = ParitySign((j2[ka[kappa]] + j2[kb[kappa]]) / 2);
          const double y0 = yk[kappa];
          yk[kappa] = phi * yk[kappa + 1];
          yk[kappa + 1] = phi * y0;
        }
      });
  exec::Fence("commutator::comm222_phss::flip");
  linalg::GroupedGemm(pd.gemms_right);  // W'(:, Ncc .. 2 Ncc) = XK YK'

  // Combine W in place into the stored half of the Pandya-transformed result:
  //   left block  (beta < Ncc):  W(alpha, beta) - hX hY W(beta, alpha),
  //   right block (Ncc + bb):    W_R(alpha, bb) + phi_alpha phi_bb W_R(bb,
  //                              alpha) = hY phi_bb (W'(alpha, Ncc + bb) +
  //                              W'(bb, Ncc + alpha)) with W_R = hY phi W'.
  // Item (c, beta) owns the pairs {(alpha, beta), (beta, alpha)} and
  // {(alpha, Ncc + beta), (beta, Ncc + alpha)} for alpha <= beta, so every
  // element is read and written by exactly one item (no separate ZB buffer).
  exec::ParallelForDynamic(
      "commutator::comm222_phss::combine", cc_kets,
      IMSRG_LAMBDA(const exec::Index e) {
        const int c = detail::FindBlock(ket_off, num_cc, e);
        if (a.pv.nk[c] == 0) return;  // W stays zero
        const std::int64_t n = a.pv.ncc[c];
        const std::int64_t beta = e - ket_off[c];
        double* w = a.pv.w + a.pv.w_off[c];
        const int* kp = a.pv.ket_p + ket_off[c];
        const int* kq = a.pv.ket_q + ket_off[c];
        const int* j2 = a.zv.tb.orbit_j2;
        const double phi_b = ParitySign((j2[kp[beta]] + j2[kq[beta]]) / 2);
        for (std::int64_t alpha = 0; alpha <= beta; ++alpha) {
          const std::int64_t ab = alpha + beta * n;
          const std::int64_t ba = beta + alpha * n;
          const double x = w[ab];
          const double y = w[ba];
          w[ab] = x - a.hxhy * y;
          if (alpha != beta) w[ba] = y - a.hxhy * x;
          const double phi_a = ParitySign((j2[kp[alpha]] + j2[kq[alpha]]) / 2);
          const std::int64_t ar = alpha + (n + beta) * n;
          const std::int64_t br = beta + (n + alpha) * n;
          const double u = w[ar];
          const double v = w[br];
          w[ar] = a.hy * phi_b * (u + v);
          if (alpha != beta) w[br] = a.hy * phi_a * (v + u);
        }
      });
  exec::Fence("commutator::comm222_phss::combine");

  // Inverse transform over pairs of kets: item = ket a = (i, j) of a class
  // (parity, Tz), partners b >= a of the same class, b = (k, l). The two zbar
  // values per J' depend only on the four orbits, not on J, so they are
  // gathered once per pair and reused for every J the two kets share: one
  // element Z(ij, kl; J) of the upper triangle per J (inside a channel the
  // local order is the global ket order, so b >= a means column >= row), plus
  // its mirror hZ Z. The arithmetic per element is that of the per-element
  // kernel it replaced (bitwise identical). Writes are race free: item a
  // writes row a (columns >= a) and column a (rows > a) of every channel.
  // (2 emax + 2 <= kMaxJp is checked when the Pandya data are built.)
  double* z2 = Z.two_body().matrix().buffer().device_data();
  const std::int64_t* block_offset = impl.block_offsets.device_data();
  exec::ParallelForDynamic(
      "commutator::comm222_phss::inverse",
      static_cast<exec::Index>(pd.num_class_kets),
      IMSRG_LAMBDA(const exec::Index e) {
        const auto& tb = a.zv.tb;
        const int* j2 = tb.orbit_j2;
        const int cl =
            detail::FindBlock(a.pv.class_offset, a.pv.num_classes, e);
        const int last = a.pv.class_offset[cl + 1];
        const int ka = a.pv.class_kets[e];
        const int i = tb.ket_p[ka];
        const int j = tb.ket_q[ka];
        const int ji = j2[i], jj = j2[j];
        const int jmin_a = (ji > jj ? ji - jj : jj - ji) / 2;
        const int jmax_a = (ji + jj) / 2;
        const std::int32_t* lit_a =
            tb.local_index_table + tb.ket_J_offsets[ka] - jmin_a;
        // Two-body channel of the class at J: 3 (2J) + cls.
        const int cls = tb.orbit_tz2[i] + tb.orbit_tz2[j] + 2 +
                        (tb.orbit_l[i] + tb.orbit_l[j]) % 2;
        const auto iabs = [](int v) { return v < 0 ? -v : v; };
        const auto imax = [](int u, int v) { return u > v ? u : v; };
        const auto imin = [](int u, int v) { return u < v ? u : v; };
        for (int pb = static_cast<int>(e); pb < last; ++pb) {
          const int kb = a.pv.class_kets[pb];
          const int k = tb.ket_p[kb];
          const int l = tb.ket_q[kb];
          const int jk = j2[k], jl = j2[l];
          const int jmin_b = iabs(jk - jl) / 2;
          const int jlo = imax(jmin_a, jmin_b);
          const int jhi = imin(jmax_a, (jk + jl) / 2);
          if (jlo > jhi) continue;
          const std::int32_t* lit_b =
              tb.local_index_table + tb.ket_J_offsets[kb] - jmin_b;
          const int lo2 = imin(imax(iabs(ji - jl), iabs(jj - jk)),
                               imax(iabs(jj - jl), iabs(ji - jk)));
          const int hi2 = imax(imin(ji + jl, jj + jk), imin(jj + jl, ji + jk));
          const int jp0 = lo2 / 2;
          const int jp1 = hi2 / 2;
          const CcTerm t1 = MakeCcTerm(a, i, l, k, j);
          const CcTerm t2 = MakeCcTerm(a, j, l, k, i);
          double zbar1[kMaxJp];
          double zbar2[kMaxJp];
          for (int Jp = jp0; Jp <= jp1; ++Jp) {
            zbar1[Jp - jp0] = CcValue(a, t1, Jp);
            zbar2[Jp - jp0] = CcValue(a, t2, Jp);
          }
          for (int J = jlo; J <= jhi; ++J) {
            const std::int32_t ib = lit_a[J];
            if (ib < 0) continue;
            const std::int32_t ik = lit_b[J];
            if (ik < 0) continue;
            const int ch = 6 * J + cls;
            const std::int64_t n = tb.channel_num_kets[ch];
            double* zb = z2 + block_offset[ch];
            const double ph_ij = ParitySign((ji + jj) / 2 - J);
            const double* row1 =
                a.sj.row(TwoJ{ji}, TwoJ{jj}, TwoJ{2 * J}, TwoJ{jk}, TwoJ{jl});
            const double* row2 =
                a.sj.row(TwoJ{jj}, TwoJ{ji}, TwoJ{2 * J}, TwoJ{jk}, TwoJ{jl});
            double z = 0.0;
            for (int Jp = jp0; Jp <= jp1; ++Jp) {
              const double s1 = row1[Jp];
              const double s2 = row2[Jp];
              double t = 0.0;
              if (s1 != 0.0) t += s1 * zbar1[Jp - jp0];
              if (s2 != 0.0) t -= ph_ij * s2 * zbar2[Jp - jp0];
              z += (2 * Jp + 1) * t;
            }
            // imsrg++ normalization: divide by sqrt(2) per equal-orbit pair
            // (exact constants::kSqrt2).
            if (i == j) z /= constants::kSqrt2;
            if (k == l) z /= constants::kSqrt2;
            zb[ib + ik * n] += z;
            // Z is (anti-)Hermitian with sign hZ = -hX hY.
            if (ib != ik) zb[ik + ib * n] -= a.hxhy * z;
          }
        }
      });
  exec::Fence("commutator::comm222_phss");
}

}  // namespace imsrg::commutator::terms
