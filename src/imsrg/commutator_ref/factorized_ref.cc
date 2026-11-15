// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Reference implementation of the factorized double commutator with one-body
// intermediates (see factorized_ref.h): a transcription of imsrg++
// `FactorizedDoubleCommutator.cc` `comm223_231_chi1b` (lines 133-436) and
// `comm223_232_chi1b` (lines 1090-1394, scalar path).
//
// Deliberate differences from imsrg++:
//  (1) no occupation skip thresholds (imsrg++ skips |occfactor| < 1e-6,
//      nbar_a < 1e-6 and products < 1e-7, and sums the CHI_I/CHI_II hole loops
//      over `holes`): every term is summed over all orbits, only exact zeros
//      are skipped;
//  (2) like the rest of commutator_ref, EVERY stored element of Z is computed
//      directly (all (p, q) of each one-body channel, both triangles of each
//      two-body block) instead of computing q <= p / ibra <= iket and
//      mirroring with hZ; the formulas are hZ-consistent, so the results agree
//      with imsrg++ to round-off. The same holds for the one-body
//      intermediates, which are computed for all (d, e);
//  (3) the ket loops of the 232 term run over the one-body channel of the
//      orbit (imsrg++ uses Eta's channels; identical for scalars);
//  (4) imsrg++'s normalized-element lookups GetTBME_norm(ibra_bq, iket) with
//      the explicit sqrt(2) and exchange-phase factors are replaced by the
//      unnormalized getter (`TwoBody::get_unnormalized`, which applies the same
//      factors); the net factor 1/sqrt((1 + d_pq)(1 + d_rs)) is applied by
//      `AddToEveryTwoBodyElement`.
//
// Memory model as in commutator_ref.cc: element accessors read and write the
// host mirror; Z is synchronized to the device at the end.
#include "imsrg/commutator_ref/factorized_ref.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"

namespace imsrg::commutator_ref {
namespace {

using op::Operator;

OrbitIndex Oi(int a) { return OrbitIndex{a}; }

int J2Of(const Operator& O, int a) {
  return O.model_space().basis().orbit(Oi(a)).j2.twice;
}

/// Unnormalized <ab; J|V|cd; J>; a missing two-body part reads as zero.
double Tbme(const Operator& O, int J, int a, int b, int c, int d) {
  if (!O.two_body().allocated()) return 0.0;
  return O.two_body().get_unnormalized(Oi(a), Oi(b), Oi(c), Oi(d), TwoJ{2 * J});
}

/// Orbits with the same (l, 2j, 2tz) as orbit a (imsrg++ GetOneBodyChannel).
std::vector<int> Channel(const Operator& O, int a) {
  std::vector<int> out;
  for (const OrbitIndex b : O.model_space().basis().SameLjtz(Oi(a))) {
    out.push_back(b.value());
  }
  return out;
}

/// Dense norb x norb matrix of one-body intermediates (zero outside channels).
class Mat {
 public:
  explicit Mat(int n) : n_(static_cast<std::size_t>(n)), v_(n_ * n_, 0.0) {}
  double& operator()(int a, int b) {
    return v_[static_cast<std::size_t>(a) * n_ + static_cast<std::size_t>(b)];
  }
  double operator()(int a, int b) const {
    return v_[static_cast<std::size_t>(a) * n_ + static_cast<std::size_t>(b)];
  }

 private:
  std::size_t n_;
  std::vector<double> v_;
};

/// Occupations n_a of the reference, indexed by plain int.
class Occupations {
 public:
  explicit Occupations(const Operator& O) {
    const auto occ = O.model_space().reference().occupations();
    values_.assign(occ.begin(), occ.end());
  }
  double operator[](int a) const {
    return values_[static_cast<std::size_t>(a)];
  }

 private:
  std::vector<double> values_;
};

int HermSign(Hermiticity h) { return h == Hermiticity::kHermitian ? 1 : -1; }

/// Validates the operands; returns hGamma through `h_gamma`.
void CheckOperands(const Operator& eta, const Operator& gamma,
                   const Operator& Z, int& h_gamma) {
  for (const Operator* o : {&eta, &gamma, &Z}) {
    if (!o->spec().is_scalar() || !o->spec().is_number_conserving()) {
      throw NotImplementedError(
          "factorized_ref: only scalar number-conserving operators");
    }
    if (o->spec().particle_rank > 2) {
      throw NotImplementedError("factorized_ref: particle rank 3");
    }
  }
  if (eta.spec().hermiticity != Hermiticity::kAntiHermitian) {
    throw NotImplementedError(
        "factorized_ref: eta must be anti-Hermitian (as in imsrg++)");
  }
  if (gamma.spec().hermiticity == Hermiticity::kNonHermitian) {
    throw NotImplementedError(
        "factorized_ref: gamma must be Hermitian or anti-Hermitian");
  }
  if (Z.spec().hermiticity != gamma.spec().hermiticity) {
    throw InvariantError("factorized_ref: Z must carry gamma's hermiticity");
  }
  if (&eta.model_space() != &gamma.model_space() ||
      &eta.model_space() != &Z.model_space()) {
    throw InvariantError("factorized_ref: operands on different model spaces");
  }
  h_gamma = HermSign(gamma.spec().hermiticity);
}

/// Adds f(J, i, j, k, l) / sqrt((1 + d_ij)(1 + d_kl)) to every stored element
/// of Z (f is an unnormalized value; no mirroring).
template <class F>
void AddToEveryTwoBodyElement(Operator& Z, F&& f) {
  const auto& chans = Z.model_space().two_body();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const auto& channel = chans.channel(ch);
    const int J = channel.J();
    const int nk = channel.num_kets;
    for (int ibra = 0; ibra < nk; ++ibra) {
      const auto& bra = chans.ket(chans.global_ket(ch, LocalKetIndex{ibra}));
      const int i = bra.p.value();
      const int j = bra.q.value();
      for (int iket = 0; iket < nk; ++iket) {
        const auto& ket = chans.ket(chans.global_ket(ch, LocalKetIndex{iket}));
        const int k = ket.p.value();
        const int l = ket.q.value();
        double z = f(J, i, j, k, l);
        if (i == j) z /= std::sqrt(2.0);
        if (k == l) z /= std::sqrt(2.0);
        auto& tb = Z.two_body();
        tb.set(ch, ch, LocalKetIndex{ibra}, LocalKetIndex{iket},
               tb.get(ch, ch, LocalKetIndex{ibra}, LocalKetIndex{iket}) + z);
      }
    }
  }
}

}  // namespace

void Factorized231Chi1b(const Operator& eta, const Operator& gamma,
                        Operator& Z) {
  int h_gamma = 0;
  CheckOperands(eta, gamma, Z, h_gamma);
  if (!eta.two_body().allocated() || !gamma.two_body().allocated()) return;
  const int hEta = -1;
  const int hZ = h_gamma;
  const int n = Z.model_space().num_orbits();
  const Occupations occ(Z);
  const auto& chans = Z.model_space().two_body();

  // Diagram I. Per channel: Nnnn = Eta with the occupation factor applied as
  // imsrg++ does (ibra <= iket: *occfactor, then iket, ibra: *(-occfactor));
  // intermediate = tmp + tmp^T with tmp = 2 (2J+1) Nnnn * Eta (normalized
  // matrices), stored in a scratch two-body operator.
  Operator inter(Z.model_space_ptr(),
                 op::OperatorSpec::Scalar(2, Hermiticity::kNonHermitian));
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const int J0 = chans.channel(ch).J();
    const int nk = chans.channel(ch).num_kets;
    const auto unk = static_cast<std::size_t>(nk);
    std::vector<double> E(unk * unk);
    auto at = [unk](std::vector<double>& m, int i, int j) -> double& {
      return m[static_cast<std::size_t>(i) * unk + static_cast<std::size_t>(j)];
    };
    for (int i = 0; i < nk; ++i) {
      for (int j = 0; j < nk; ++j) {
        at(E, i, j) =
            eta.two_body().get(ch, ch, LocalKetIndex{i}, LocalKetIndex{j});
      }
    }
    std::vector<double> N = E;
    for (int ibra = 0; ibra < nk; ++ibra) {
      const auto& bra = chans.ket(chans.global_ket(ch, LocalKetIndex{ibra}));
      const double n_i = occ[bra.p.value()];
      const double n_j = occ[bra.q.value()];
      for (int iket = ibra; iket < nk; ++iket) {
        const auto& ket = chans.ket(chans.global_ket(ch, LocalKetIndex{iket}));
        const double n_k = occ[ket.p.value()];
        const double n_l = occ[ket.q.value()];
        const double occfactor = n_i * n_j * (1 - n_k) * (1 - n_l) -
                                 (1 - n_i) * (1 - n_j) * n_k * n_l;
        at(N, ibra, iket) *= occfactor;
        at(N, iket, ibra) *= -occfactor;
      }
    }
    std::vector<double> tmp(unk * unk, 0.0);
    for (int i = 0; i < nk; ++i) {
      for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nk; ++j) {
          at(tmp, i, j) += 2 * (2 * J0 + 1) * at(N, i, k) * at(E, k, j);
        }
      }
    }
    for (int i = 0; i < nk; ++i) {
      for (int j = 0; j < nk; ++j) {
        inter.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j},
                             at(tmp, i, j) + at(tmp, j, i));
      }
    }
  }

  // Chi_221_a(d, e) = 1/(2 j_d + 1) sum_b sum_J0 inter^J0_{bd,be}.
  Mat chi_a(n);
  for (int d = 0; d < n; ++d) {
    const int jd = J2Of(Z, d);
    for (const int e : Channel(Z, d)) {
      double eta_de = 0.0;
      for (int b = 0; b < n; ++b) {
        int Jmin = std::abs(jd - J2Of(Z, b)) / 2;
        const int Jmax = (jd + J2Of(Z, b)) / 2;
        if (d == b || e == b) Jmin += Jmin % 2;
        for (int J0 = Jmin; J0 <= Jmax; ++J0) {
          eta_de += Tbme(inter, J0, b, d, b, e);
        }
      }
      chi_a(d, e) = eta_de / (jd + 1.0);
    }
  }

  // Chi_221_b(d, e) = 1/(2 j_d + 1) sum_{J0, b<=c, a} (2J0+1) occfactor
  //   eta^J0_{bc,ae} gamma^J0_{ad,bc} (counted twice for b != c).
  Mat chi_b(n);
  for (int d = 0; d < n; ++d) {
    for (const int e : Channel(Z, d)) {
      double eta_de = 0.0;
      for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
        const ChannelIndex ch{c};
        const int J0 = chans.channel(ch).J();
        const int nk = chans.channel(ch).num_kets;
        for (int ibra = 0; ibra < nk; ++ibra) {
          const auto& bra =
              chans.ket(chans.global_ket(ch, LocalKetIndex{ibra}));
          const int b = bra.p.value();
          const int cc = bra.q.value();
          for (int a = 0; a < n; ++a) {
            const double occfactor =
                ((1.0 - occ[a]) * (1.0 - occ[e]) * occ[b] * occ[cc] -
                 (1.0 - occ[b]) * (1.0 - occ[cc]) * occ[a] * occ[e]);
            if (occfactor == 0.0) continue;
            const double MEs = (2 * J0 + 1) * occfactor *
                               Tbme(eta, J0, b, cc, a, e) *
                               Tbme(gamma, J0, a, d, b, cc);
            eta_de += MEs;
            if (b != cc) eta_de += MEs;
          }
        }
      }
      chi_b(d, e) = eta_de / (J2Of(Z, d) + 1.0);
    }
  }

  // Contractions, every (p, q) of each one-body channel.
  for (int p = 0; p < n; ++p) {
    const int jp = J2Of(Z, p);
    for (const int q : Channel(Z, p)) {
      const int jq = J2Of(Z, q);
      double zij_I = 0.0;
      double zij_a = 0.0;
      double zij_b = 0.0;
      for (int d = 0; d < n; ++d) {
        const int jd = J2Of(Z, d);
        for (const int e : Channel(Z, d)) {
          const int J1min = std::abs(jd - jq) / 2;
          const int J1max = (jd + jq) / 2;
          for (int J1 = J1min; J1 <= J1max; ++J1) {
            zij_I += (2 * J1 + 1) * chi_a(d, e) * Tbme(gamma, J1, e, p, d, q);
          }
          const int K1min = std::abs(J2Of(Z, e) - jp) / 2;
          const int K1max = (J2Of(Z, e) + jp) / 2;
          for (int J1 = K1min; J1 <= K1max; ++J1) {
            const double etaME = (2 * J1 + 1) * Tbme(eta, J1, e, p, d, q);
            zij_a += chi_b(d, e) * etaME;
            zij_b -= hEta * hZ * chi_b(e, d) * etaME;
          }
        }
      }
      Z.one_body().set_element(Oi(p), Oi(q),
                               Z.one_body().get(Oi(p), Oi(q)) +
                                   0.5 * zij_I / (jp + 1.0) +
                                   0.5 * (zij_a - zij_b) / (jp + 1.0));
    }
  }
  Z.SyncToDevice();
}

void Factorized232Chi1b(const Operator& eta, const Operator& gamma,
                        Operator& Z) {
  int h_gamma = 0;
  CheckOperands(eta, gamma, Z, h_gamma);
  if (!eta.two_body().allocated() || !gamma.two_body().allocated() ||
      !Z.two_body().allocated()) {
    return;
  }
  const int hEta = -1;
  const int hZ = h_gamma;
  const int n = Z.model_space().num_orbits();
  const Occupations occ(Z);

  // CHI_I_pq  = 1/2 sum_{a,i,j,J2} (2J2+1)/(2 j_q + 1) nbar_a n_i n_j
  //               eta^J2_{ap,ij} eta^J2_{ij,aq}
  //           + 1/2 sum_{a,b,i,J2} (2J2+1)/(2 j_q + 1) nbar_a nbar_b n_i
  //               eta^J2_{ip,ab} eta^J2_{ab,iq};
  // CHI_II_pq is the same with the first factor of each product from gamma.
  Mat chi_I(n);
  Mat chi_II(n);
  for (int p = 0; p < n; ++p) {
    const int jp = J2Of(Z, p);
    for (const int q : Channel(Z, p)) {
      const int jq = J2Of(Z, q);
      double chi = 0.0;
      double chiY = 0.0;
      for (int a = 0; a < n; ++a) {
        const int ja = J2Of(Z, a);
        const double nbar_a = 1.0 - occ[a];
        if (nbar_a == 0.0) continue;
        for (int i = 0; i < n; ++i) {
          const int ji = J2Of(Z, i);
          const double n_i = occ[i];
          if (n_i == 0.0) continue;
          for (int j = 0; j < n; ++j) {
            const int jj = J2Of(Z, j);
            const double occfactor = nbar_a * n_i * occ[j];
            if (occfactor == 0.0) continue;
            const int J2min =
                std::max(std::abs(ja - jq), std::abs(ji - jj)) / 2;
            const int J2max = std::min(ja + jq, ji + jj) / 2;
            for (int J2 = J2min; J2 <= J2max; ++J2) {
              const double xijaq = Tbme(eta, J2, i, j, a, q);
              const double xapij = Tbme(eta, J2, a, p, i, j);
              const double yapij = Tbme(gamma, J2, a, p, i, j);
              chi += 0.5 * occfactor * (2 * J2 + 1) / (jq + 1) * xapij * xijaq;
              chiY += 0.5 * occfactor * (2 * J2 + 1) / (jq + 1) * yapij * xijaq;
            }
          }
          for (int b = 0; b < n; ++b) {
            const int jb = J2Of(Z, b);
            const double occfactor = nbar_a * (1.0 - occ[b]) * n_i;
            if (occfactor == 0.0) continue;
            const int J2min = std::max({std::abs(ja - jb), std::abs(ji - jq),
                                        std::abs(ji - jp)}) /
                              2;
            const int J2max = std::min({ja + jb, ji + jq, ji + jp}) / 2;
            for (int J2 = J2min; J2 <= J2max; ++J2) {
              const double xabiq = Tbme(eta, J2, a, b, i, q);
              const double xipab = Tbme(eta, J2, i, p, a, b);
              const double yipab = Tbme(gamma, J2, i, p, a, b);
              chi += 0.5 * occfactor * (2 * J2 + 1) / (jq + 1) * xipab * xabiq;
              chiY += 0.5 * occfactor * (2 * J2 + 1) / (jq + 1) * yipab * xabiq;
            }
          }
        }
      }
      chi_I(p, q) = chi;
      chi_II(p, q) = chiY;
    }
  }

  // Z^J_pqrs: CHI_I on gamma and CHI_II on eta, on each of the four legs.
  AddToEveryTwoBodyElement(Z, [&](int J, int p, int q, int r, int s) {
    double z = 0.0;
    for (const int b : Channel(Z, p)) {
      z += chi_I(p, b) * Tbme(gamma, J, b, q, r, s);
      z -= hEta * hZ * chi_II(b, p) * Tbme(eta, J, b, q, r, s);
    }
    for (const int b : Channel(Z, q)) {
      z += chi_I(q, b) * Tbme(gamma, J, p, b, r, s);
      z -= hEta * hZ * chi_II(b, q) * Tbme(eta, J, p, b, r, s);
    }
    for (const int b : Channel(Z, r)) {
      z += Tbme(gamma, J, p, q, b, s) * chi_I(b, r);
      z -= Tbme(eta, J, p, q, b, s) * chi_II(b, r);
    }
    for (const int b : Channel(Z, s)) {
      z += Tbme(gamma, J, p, q, r, b) * chi_I(b, s);
      z -= Tbme(eta, J, p, q, r, b) * chi_II(b, s);
    }
    return z;
  });
  Z.SyncToDevice();
}

void FactorizedDoubleCommutator(const Operator& eta, const Operator& gamma,
                                Operator& Z) {
  Factorized231Chi1b(eta, gamma, Z);
  Factorized232Chi1b(eta, gamma, Z);
}

}  // namespace imsrg::commutator_ref
