#include "imsrg/mbpt/energies.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "imsrg/angmom/wigner.hpp"
#include "imsrg/linalg/matrix_view.hpp"

namespace imsrg::mbpt {

namespace {
double diag(const op::Operator& H, OrbitIndex a) { return H.one_body().get(a, a); }
}  // namespace

double mp2_energy(op::Operator& H) {
  H.sync_to_host();
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  const auto& ref = ms.reference();
  const auto& two = H.two_body();
  double e2 = 0.0;
  for (const OrbitIndex i : ref.particles()) {
    const auto& oi = basis.orbit(i);
    const double ei = diag(H, i);
    for (const OrbitIndex a : ref.holes()) {
      const auto& oa = basis.orbit(a);
      const double ea = diag(H, a);
      const double fia = H.one_body().get(i, a);
      if (std::abs(fia) > 1e-9) e2 += (oa.j2.twice + 1) * ref.occupation(a) * fia * fia / (ea - ei);
      for (const OrbitIndex j : ref.particles()) {
        if (j.value() < i.value()) continue;
        const auto& oj = basis.orbit(j);
        const double ej = diag(H, j);
        for (const OrbitIndex b : ref.holes()) {
          if (b.value() < a.value()) continue;
          const auto& ob = basis.orbit(b);
          if ((oi.l + oj.l + oa.l + ob.l) % 2 > 0) continue;
          if (oi.tz2.twice + oj.tz2.twice != oa.tz2.twice + ob.tz2.twice) continue;
          const double denom = ea + diag(H, b) - ei - ej;
          int jmin = std::max(std::abs(oi.j2.twice - oj.j2.twice),
                              std::abs(oa.j2.twice - ob.j2.twice)) / 2;
          const int jmax = std::min(oi.j2.twice + oj.j2.twice, oa.j2.twice + ob.j2.twice) / 2;
          int dj = 1;
          if (a == b || i == j) {
            jmin += jmin % 2;
            dj = 2;
          }
          for (int J = jmin; J <= jmax; J += dj) {
            const double t = two.get_normalized(a, b, i, j, TwoJ{2 * J});
            if (std::abs(t) > 1e-9)
              e2 += (2 * J + 1) * ref.occupation(a) * ref.occupation(b) * t * t / denom;
          }
        }
      }
    }
  }
  return e2;
}

Mp3Energy mp3_energy(op::Operator& H) {
  H.sync_to_host();
  const auto& ms = H.model_space();
  const auto& basis = ms.basis();
  const auto& ref = ms.reference();
  const auto& tb = ms.two_body();
  const auto& sub = ms.two_body_subsets();
  Mp3Energy out;

  // pp and hh ladders.
  for (std::int32_t c = 0; c < tb.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const auto& tbc = tb.channel(ch);
    const auto hh = sub.kets(ch, basis::KetClass::HH);
    const auto pp = sub.kets(ch, basis::KetClass::PP);
    if (hh.empty() || pp.empty()) continue;
    const auto V = H.two_body().host_block(ch);
    const auto kets = tb.channel_kets(ch);
    const std::size_t nh = hh.size(), np = pp.size();
    linalg::Matrix<double> M(static_cast<std::int64_t>(nh), static_cast<std::int64_t>(np));
    for (std::size_t I = 0; I < nh; ++I) {
      const auto& kh = tb.ket(kets[hh[I].value()]);
      const double eh = diag(H, kh.p) + diag(H, kh.q);
      for (std::size_t A = 0; A < np; ++A) {
        const auto& kp = tb.ket(kets[pp[A].value()]);
        const double ep = diag(H, kp.p) + diag(H, kp.q);
        M(I, A) = V(hh[I].value(), pp[A].value()) / (eh - ep);
      }
    }
    const double w = 2 * tbc.J() + 1;
    // Epp = (2J+1) tr(M Vpp M^T), Ehh = (2J+1) tr(M^T Vhh M)
    for (std::size_t A = 0; A < np; ++A)
      for (std::size_t B = 0; B < np; ++B) {
        double s = 0.0;
        for (std::size_t I = 0; I < nh; ++I) s += M(I, A) * M(I, B);
        out.pp += w * s * V(pp[A].value(), pp[B].value());
      }
    for (std::size_t I = 0; I < nh; ++I)
      for (std::size_t K = 0; K < nh; ++K) {
        double s = 0.0;
        for (std::size_t A = 0; A < np; ++A) s += M(I, A) * M(K, A);
        out.hh += w * s * V(hh[I].value(), hh[K].value());
      }
  }

  // ph ring term in the cross-coupled channels.
  const auto& ph = ms.particle_hole();
  struct PhKet {
    OrbitIndex a, i;
    int ja2, ji2;
    double phase_ai, phase_ia;
  };
  for (std::int32_t c = 0; c < ph.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const int Jph = ph.channel(ch).J();
    std::vector<PhKet> pk;
    for (const KetIndex k : ph.channel_kets(ch)) {
      const auto& ket = tb.ket(k);
      const bool hp = ref.occupation(ket.p) >= basis::kOccupationCut;
      const bool hq = ref.occupation(ket.q) >= basis::kOccupationCut;
      if (hp == hq) continue;
      PhKet x{ket.p, ket.q, basis.orbit(ket.p).j2.twice, basis.orbit(ket.q).j2.twice, 1.0, 0.0};
      const int e = (x.ja2 + x.ji2) / 2 - Jph;
      x.phase_ia = -((e % 2 == 0) ? 1.0 : -1.0);
      if (ref.occupation(ket.p) < ref.occupation(ket.q)) {
        std::swap(x.a, x.i);
        std::swap(x.ja2, x.ji2);
        std::swap(x.phase_ai, x.phase_ia);
      }
      pk.push_back(x);
    }
    const std::size_t n = pk.size();
    if (n == 0) continue;
    linalg::Matrix<double> V1(static_cast<std::int64_t>(n), static_cast<std::int64_t>(n));
    linalg::Matrix<double> V2(static_cast<std::int64_t>(n), static_cast<std::int64_t>(n));
    const TwoJ tJph{2 * Jph};
    const auto& two = H.two_body();
    for (std::size_t I = 0; I < n; ++I)
      for (std::size_t II = 0; II < n; ++II) {
        const auto& x = pk[I];
        const auto& y = pk[II];  // (b, j)
        const double delta = diag(H, x.i) + diag(H, y.i) - diag(H, x.a) - diag(H, y.a);
        V1(I, II) = 0.0;
        V2(II, I) = 0.0;
        const TwoJ ja{x.ja2}, ji{x.ji2}, jb{y.ja2}, jj{y.ji2};
        if (!angmom::triangle(jj, jb, tJph) || !angmom::triangle(ji, ja, tJph)) continue;
        double t1 = 0.0;
        for (int J1 = std::max(std::abs(ja.twice - jb.twice), std::abs(ji.twice - jj.twice)) / 2;
             J1 <= std::min(ja.twice + jb.twice, ji.twice + jj.twice) / 2; ++J1)
          t1 -= angmom::wigner_6j(ja, ji, tJph, jj, jb, TwoJ{2 * J1}) * (2 * J1 + 1) *
                two.get_unnormalized(x.i, y.i, y.a, x.a, TwoJ{2 * J1});
        double t2 = 0.0;
        for (int J1 = std::max(std::abs(ji.twice - jb.twice), std::abs(ja.twice - jj.twice)) / 2;
             J1 <= std::min(ji.twice + jb.twice, ja.twice + jj.twice) / 2; ++J1)
          t2 -= angmom::wigner_6j(jb, jj, tJph, ja, ji, TwoJ{2 * J1}) * (2 * J1 + 1) *
                two.get_unnormalized(y.a, x.i, x.a, y.i, TwoJ{2 * J1});
        V1(I, II) = t1 * x.phase_ia * y.phase_ai / delta;
        V2(II, I) = t2 * y.phase_ai * x.phase_ai;
      }
    // Eph += (2Jph+1) tr(V1 V2 V1^T)
    double tr = 0.0;
    for (std::size_t I = 0; I < n; ++I)
      for (std::size_t K = 0; K < n; ++K) {
        double s = 0.0;
        for (std::size_t II = 0; II < n; ++II) s += V1(I, II) * V2(II, K);
        tr += s * V1(I, K);
      }
    out.ph += (2 * Jph + 1) * tr;
  }
  out.total = out.pp + out.hh + out.ph;
  return out;
}

}  // namespace imsrg::mbpt
