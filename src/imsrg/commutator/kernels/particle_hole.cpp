// Particle-hole (Pandya) term comm222_phss (design/12 section 4.2; imsrg++ `comm222_phss`,
// reference/02 P-12). Kernel TU: includes the Kokkos vocabulary.
//
// Algorithm. For every cross-coupled channel c (J', parity, Tz) with Ncc kets, build
//   XK(alpha, kappa) = D_kappa xbar(i, l, a, b; J')        (alpha = ket (i, l), kappa = (a, b))
//   YK(kappa, beta)  = ybar(a, b, k, j; J')                (beta = ordered pair (k, j))
// GEMM W = XK YK, combine W into the Pandya-transformed result ZB, and transform back to the
// coupled two-body matrix element by element. Only rows alpha (kets, first orbit <= second) are
// computed: the remaining rows follow from hermiticity of X, Y (ZB(beta, alpha) block) and from
// the flip symmetry of the ordered-pair basis; the inverse transform reads only the stored half
// (with an explicit sign for flipped bra pairs), so it needs no lexicographic ket order.

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/constants.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/device_views.hpp"
#include "imsrg/commutator/detail/pandya.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"
#include "imsrg/exec/kokkos_backend.hpp"
#include "imsrg/linalg/blas.hpp"

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
  double hxhy = 1.0;  ///< hX hY (hermiticity signs).
};
static_assert(std::is_trivially_copyable_v<PhArgs>);

/// (-1)^n for integer n (any sign).
IMSRG_INLINE_FUNCTION double parity_sign(int n) noexcept { return (n & 1) ? -1.0 : 1.0; }

/// Pandya transform xbar(p, q, r, s; J') = - sum_{J''} (2J''+1) {jp jq J'; jr js J''}
/// X~^{J''}_{psrq} with X~ the unnormalized coupled element. `two_Jp` is 2J'. O(J'' range).
IMSRG_INLINE_FUNCTION double pandya_bar(const TwoBodyView& x, const angmom::SixJView& sj, int p,
                                        int q, int r, int s, int two_Jp) noexcept {
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
    const double sixj = sj(TwoJ{jp}, TwoJ{jq}, TwoJ{two_Jp}, TwoJ{jr}, TwoJ{js}, TwoJ{two_Jpp});
    if (sixj == 0.0) continue;
    sum += (two_Jpp + 1) * sixj * x.get_unnormalized(p, s, r, q, two_Jpp);
  }
  return -sum;
}

/// Local index of the ket (a <= b) at J' in its cc channel, -1 if absent. O(1).
IMSRG_INLINE_FUNCTION int cc_local(const PhArgs& a, int oa, int ob, int Jp) noexcept {
  const auto& tb = a.zv.tb;
  const int ja = tb.orbit_j2[oa], jb = tb.orbit_j2[ob];
  const int jmin = (ja > jb ? ja - jb : jb - ja) / 2;
  if (Jp < jmin || Jp > (ja + jb) / 2) return -1;
  const int k = tb.ket_index(OrbitIndex{oa}, OrbitIndex{ob}).value();
  return a.pv.local_index_table[a.pv.ket_J_offsets[k] + Jp - jmin];
}

/// cc channel index of the pair (p, q) at J'.
IMSRG_INLINE_FUNCTION int cc_channel(const PhArgs& a, int p, int q, int Jp) noexcept {
  const auto& tb = a.zv.tb;
  const int dtz = tb.orbit_tz2[p] - tb.orbit_tz2[q];
  return 3 * (2 * Jp) + (dtz < 0 ? -dtz : dtz) + 2 + ((tb.orbit_l[p] + tb.orbit_l[q]) & 1);
}

/// Element of the Pandya-transformed result, zbar(p, q, r, s; J'), from the stored half of ZB.
IMSRG_INLINE_FUNCTION double zbar(const PhArgs& a, int p, int q, int r, int s, int Jp) noexcept {
  const auto& tb = a.zv.tb;
  const int c = cc_channel(a, p, q, Jp);
  if (c < 0 || c >= a.pv.num_cc || c != cc_channel(a, r, s, Jp)) return 0.0;
  const int n = a.pv.ncc[c];
  if (n == 0) return 0.0;
  const double* zb = a.pv.zb + a.pv.w_off[c];
  if (p <= q) {
    const int alpha = cc_local(a, p, q, Jp);
    const int lb = r <= s ? cc_local(a, r, s, Jp) : cc_local(a, s, r, Jp);
    if (alpha < 0 || lb < 0) return 0.0;
    const int beta = lb + (r > s ? n : 0);
    return zb[alpha + static_cast<std::int64_t>(beta) * n];
  }
  const int alpha = cc_local(a, q, p, Jp);
  const int lb = s <= r ? cc_local(a, s, r, Jp) : cc_local(a, r, s, Jp);
  if (alpha < 0 || lb < 0) return 0.0;
  const int beta = lb + (s > r ? n : 0);
  const double phi =
      parity_sign((tb.orbit_j2[p] + tb.orbit_j2[q]) / 2 + (tb.orbit_j2[r] + tb.orbit_j2[s]) / 2);
  return -a.hxhy * phi * zb[alpha + static_cast<std::int64_t>(beta) * n];
}

}  // namespace

void comm222_phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  if (!X.two_body().allocated() || !Y.two_body().allocated() || !Z.two_body().allocated()) return;
  auto& impl = ws.impl();
  const detail::PandyaData& pd = impl.pandya;
  if (pd.num_cc == 0) return;

  PhArgs a;
  a.pv = pd.view();
  a.xv = impl.two_body_view(X);
  a.yv = impl.two_body_view(Y);
  a.zv = impl.two_body_view(Z);
  a.sj = impl.sixj.device_view();
  const double hx = X.hermiticity() == Hermiticity::Hermitian ? 1.0 : -1.0;
  const double hy = Y.hermiticity() == Hermiticity::Hermitian ? 1.0 : -1.0;
  a.hxhy = hx * hy;

  const int num_cc = pd.num_cc;
  const auto xk_total = static_cast<exec::Index>(pd.xk_off_host.back());
  const auto yk_total = static_cast<exec::Index>(pd.yk_off_host.back());
  const auto w_total = static_cast<exec::Index>(pd.w_off_host.back());

  if (xk_total > 0) {
    exec::parallel_for(
        "commutator::comm222_phss::XK", xk_total, IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::find_block(a.pv.xk_off, num_cc, e);
          const std::int64_t local = e - a.pv.xk_off[c];
          const int n = a.pv.ncc[c];
          const int alpha = static_cast<int>(local % n);
          const int kappa = static_cast<int>(local / n);
          const int i = a.pv.ket_p[a.pv.ket_off[c] + alpha];
          const int l = a.pv.ket_q[a.pv.ket_off[c] + alpha];
          const int ka = a.pv.k_off[c] + kappa;
          const double v =
              a.pv.k_d[ka] * pandya_bar(a.xv, a.sj, i, l, a.pv.k_a[ka], a.pv.k_b[ka], 2 * (c / 6));
          a.pv.xk[e] = v;
        });
    exec::parallel_for(
        "commutator::comm222_phss::YK", yk_total, IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::find_block(a.pv.yk_off, num_cc, e);
          const std::int64_t local = e - a.pv.yk_off[c];
          const int nk = a.pv.nk[c];
          const int n = a.pv.ncc[c];
          const int kappa = static_cast<int>(local % nk);
          const int beta = static_cast<int>(local / nk);
          const int ka = a.pv.k_off[c] + kappa;
          const int kb = beta < n ? beta : beta - n;
          const int p = a.pv.ket_p[a.pv.ket_off[c] + kb];
          const int q = a.pv.ket_q[a.pv.ket_off[c] + kb];
          const int k = beta < n ? p : q;
          const int j = beta < n ? q : p;
          a.pv.yk[e] = pandya_bar(a.yv, a.sj, a.pv.k_a[ka], a.pv.k_b[ka], k, j, 2 * (c / 6));
        });
  }
  exec::fence("commutator::comm222_phss::fill");
  // Host BLAS only; device backends need a device gemm (design/12 section 3).
  IMSRG_CHECK(pd.w.host_is_device(),
              "comm222_phss: host BLAS on device memory; needs a device gemm (GPU backends)");
  linalg::grouped_gemm(pd.gemms);

  exec::parallel_for(
      "commutator::comm222_phss::ZB", w_total, IMSRG_LAMBDA(const exec::Index e) {
        const int c = detail::find_block(a.pv.w_off, num_cc, e);
        const std::int64_t local = e - a.pv.w_off[c];
        const int n = a.pv.ncc[c];
        const int alpha = static_cast<int>(local % n);
        const int beta = static_cast<int>(local / n);
        const double* w = a.pv.w + a.pv.w_off[c];
        const int* kp = a.pv.ket_p + a.pv.ket_off[c];
        const int* kq = a.pv.ket_q + a.pv.ket_off[c];
        const int* j2 = a.zv.tb.orbit_j2;
        double v = w[alpha + static_cast<std::int64_t>(beta) * n];
        if (beta < n) {
          v -= a.hxhy * w[beta + static_cast<std::int64_t>(alpha) * n];
        } else {
          const int bb = beta - n;
          const double phi_a = parity_sign((j2[kp[alpha]] + j2[kq[alpha]]) / 2);
          const double phi_b = parity_sign((j2[kp[bb]] + j2[kq[bb]]) / 2);
          v += phi_a * phi_b * w[bb + static_cast<std::int64_t>(alpha + n) * n];
        }
        a.pv.zb[e] = v;
      });
  exec::fence("commutator::comm222_phss::ZB");

  double* z2 = Z.two_body().matrix().buffer().device_data();
  const std::int64_t* block_offset = impl.block_offsets.device_data();
  const int nch = a.zv.tb.num_channels;
  const auto total = static_cast<exec::Index>(impl.two_body_size);
  exec::parallel_for(
      "commutator::comm222_phss::inverse", total, IMSRG_LAMBDA(const exec::Index e) {
        const auto& tb = a.zv.tb;
        const int ch = detail::find_block(block_offset, nch, e);
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
        const int lo2 =
            imin(imax(iabs(ji - jl), iabs(jj - jk)), imax(iabs(jj - jl), iabs(ji - jk)));
        const int hi2 = imax(imin(ji + jl, jj + jk), imin(jj + jl, ji + jk));
        const double ph_ij = parity_sign((ji + jj) / 2 - two_J / 2);
        double z = 0.0;
        for (int Jp = lo2 / 2; Jp <= hi2 / 2; ++Jp) {
          const double s1 = a.sj(TwoJ{ji}, TwoJ{jj}, TwoJ{two_J}, TwoJ{jk}, TwoJ{jl}, TwoJ{2 * Jp});
          const double s2 = a.sj(TwoJ{jj}, TwoJ{ji}, TwoJ{two_J}, TwoJ{jk}, TwoJ{jl}, TwoJ{2 * Jp});
          double t = 0.0;
          if (s1 != 0.0) t += s1 * zbar(a, i, l, k, j, Jp);
          if (s2 != 0.0) t -= ph_ij * s2 * zbar(a, j, l, k, i, Jp);
          z += (2 * Jp + 1) * t;
        }
        // imsrg++ normalization: divide by sqrt(2) per equal-orbit pair (exact constants::sqrt2).
        if (i == j) z /= constants::sqrt2;
        if (k == l) z /= constants::sqrt2;
        z2[e] += z;
      });
  exec::fence("commutator::comm222_phss");
}

}  // namespace imsrg::commutator::terms
