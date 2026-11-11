// Reference implementations of the scalar IMSRG(2) commutator terms (see commutator_ref.hpp).
//
// Everything is written as the plain loops of imsrg++ `ReferenceImplementations/
// IMSRG2Commutators.cc`, with unnormalized matrix elements and explicit 6j symbols. Deviations
// from imsrg++ are deliberate and listed here:
//  * every stored element of Z is computed (both triangles of every block, all (i, j) of the
//    one-body channels) and written with `TwoBody::set` / `OneBody::set_element`, without
//    mirroring, so non-Hermitian inputs give the right result;
//  * the particle-hole term precomputes the Pandya-transformed matrix elements once per call
//    (a dense table) instead of recomputing them inside the innermost loops; the expressions are
//    unchanged.
//
// Memory model: all element accessors read and write the host mirror. The inputs X and Y are
// const and are assumed to be synchronized with their host mirrors (true on the OpenMP and Serial
// backends, where host and device memory coincide, and after any host-side construction); Z is
// synchronized to the device at the end of every function.
#include "imsrg/commutator_ref/commutator_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/angmom/wigner.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/base/types.hpp"

namespace imsrg::commutator_ref {
namespace {

using op::Operator;

/// Occupations n_a of the reference, one per orbit, indexed by plain int.
class Occupations {
 public:
  explicit Occupations(const Operator& X) {
    const auto occ = X.model_space().reference().occupations();
    values_.assign(occ.begin(), occ.end());
  }
  double operator[](int a) const { return values_[static_cast<std::size_t>(a)]; }

 private:
  std::vector<double> values_;
};

Occupations occupations_of(const Operator& X) { return Occupations(X); }

/// (-1)^n for integer n (any sign).
int phase_of(int n) { return (n % 2 == 0) ? 1 : -1; }

OrbitIndex oi(int a) { return OrbitIndex{a}; }

/// Unnormalized <ab; J|V|cd; J> for integer J.
/// A missing two-body part (particle_rank 1) reads as zero.
double tbme(const Operator& O, int J, int a, int b, int c, int d) {
  if (!O.two_body().allocated()) return 0.0;
  return O.two_body().get_unnormalized(oi(a), oi(b), oi(c), oi(d), TwoJ{2 * J});
}

/// Adds f(J, i, j, k, l) (an unnormalized two-body value) to EVERY stored element of Z: divides
/// by sqrt(2) for i == j and for k == l (imsrg++ normalization before AddToTBME), no mirroring.
template <class F>
void add_to_every_two_body_element(Operator& Z, F&& f) {
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

/// Adds f(i, j) to every one-body element (i, j) of equal (l, 2j, 2tz).
template <class F>
void add_to_every_one_body_element(Operator& Z, F&& f) {
  const auto& basis = Z.model_space().basis();
  const int n = Z.model_space().num_orbits();
  for (int i = 0; i < n; ++i) {
    for (const OrbitIndex jo : basis.same_ljtz(oi(i))) {
      const int j = jo.value();
      Z.one_body().set_element(oi(i), jo, Z.one_body().get(oi(i), jo) + f(i, j));
    }
  }
}

int j2_of(const Operator& O, int a) { return O.model_space().basis().orbit(oi(a)).j2.twice; }

}  // namespace

op::OperatorSpec commutator_spec(const op::OperatorSpec& x, const op::OperatorSpec& y) {
  if (!x.is_scalar() || !y.is_scalar()) {
    throw NotImplementedError("commutator_ref: only scalar operators are supported");
  }
  if (!x.is_number_conserving() || !y.is_number_conserving()) {
    throw NotImplementedError("commutator_ref: only number-conserving operators are supported");
  }
  if (x.particle_rank > 2 || y.particle_rank > 2) {
    throw NotImplementedError("commutator_ref: particle rank 3 is not supported");
  }
  op::OperatorSpec z = op::OperatorSpec::scalar(std::max(x.particle_rank, y.particle_rank));
  if (x.hermiticity == Hermiticity::NonHermitian || y.hermiticity == Hermiticity::NonHermitian) {
    z.hermiticity = Hermiticity::NonHermitian;
  } else {
    z.hermiticity =
        (x.hermiticity == y.hermiticity) ? Hermiticity::AntiHermitian : Hermiticity::Hermitian;
  }
  return z;
}

void comm110ss(const Operator& X, const Operator& Y, Operator& Z) {
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  double z0 = 0.0;
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      const double xab = X.one_body().get(oi(a), oi(b));
      const double yba = Y.one_body().get(oi(b), oi(a));
      const double yab = Y.one_body().get(oi(a), oi(b));
      const double xba = X.one_body().get(oi(b), oi(a));
      z0 += (j2_of(X, a) + 1) * occ[a] * (1.0 - occ[b]) * (xab * yba - yab * xba);
    }
  }
  Z.zero_body() += z0;
  Z.sync_to_device();
}

void comm220ss(const Operator& X, const Operator& Y, Operator& Z) {
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  double z0 = 0.0;
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      if (occ[a] * occ[b] == 0.0) continue;
      for (int c = 0; c < n; ++c) {
        for (int d = 0; d < n; ++d) {
          const double w = occ[a] * occ[b] * (1.0 - occ[c]) * (1.0 - occ[d]);
          if (w == 0.0) continue;
          const int Jmin =
              std::max(std::abs(j2_of(X, a) - j2_of(X, b)), std::abs(j2_of(X, c) - j2_of(X, d))) /
              2;
          const int Jmax = std::min(j2_of(X, a) + j2_of(X, b), j2_of(X, c) + j2_of(X, d)) / 2;
          for (int J = Jmin; J <= Jmax; ++J) {
            z0 += 0.25 * (2 * J + 1) * w *
                  (tbme(X, J, a, b, c, d) * tbme(Y, J, c, d, a, b) -
                   tbme(Y, J, a, b, c, d) * tbme(X, J, c, d, a, b));
          }
        }
      }
    }
  }
  Z.zero_body() += z0;
  Z.sync_to_device();
}

void comm111ss(const Operator& X, const Operator& Y, Operator& Z) {
  const int n = X.model_space().num_orbits();
  add_to_every_one_body_element(Z, [&](int i, int j) {
    double z = 0.0;
    for (int a = 0; a < n; ++a) {
      z += X.one_body().get(oi(i), oi(a)) * Y.one_body().get(oi(a), oi(j)) -
           Y.one_body().get(oi(i), oi(a)) * X.one_body().get(oi(a), oi(j));
    }
    return z;
  });
  Z.sync_to_device();
}

void comm121ss(const Operator& X, const Operator& Y, Operator& Z) {
  // [X1,Y2] and [X2,Y1] each need only one two-body operand: skip only if both are missing.
  if (!X.two_body().allocated() && !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  add_to_every_one_body_element(Z, [&](int i, int j) {
    double z = 0.0;
    const int ji = j2_of(X, i);
    const int jj = j2_of(X, j);
    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < n; ++b) {
        if (occ[a] == occ[b]) continue;
        const int ja = j2_of(X, a);
        const int jb = j2_of(X, b);
        const int Jmin = std::max(std::abs(jb - ji), std::abs(ja - jj)) / 2;
        const int Jmax = std::min(jb + ji, ja + jj) / 2;
        for (int J = Jmin; J <= Jmax; ++J) {
          z += (2 * J + 1) / static_cast<double>(ji + 1) * (occ[a] - occ[b]) *
               (X.one_body().get(oi(a), oi(b)) * tbme(Y, J, b, i, a, j) -
                Y.one_body().get(oi(a), oi(b)) * tbme(X, J, b, i, a, j));
        }
      }
    }
    return z;
  });
  Z.sync_to_device();
}

void comm221ss(const Operator& X, const Operator& Y, Operator& Z) {
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  add_to_every_one_body_element(Z, [&](int i, int j) {
    double z = 0.0;
    const int ji = j2_of(X, i);
    for (int c = 0; c < n; ++c) {
      for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
          const double w =
              occ[a] * occ[b] * (1.0 - occ[c]) + (1.0 - occ[a]) * (1.0 - occ[b]) * occ[c];
          if (w == 0.0) continue;
          const int ja = j2_of(X, a);
          const int jb = j2_of(X, b);
          const int jc = j2_of(X, c);
          const int Jmin = std::max(std::abs(jb - ja), std::abs(jc - ji)) / 2;
          const int Jmax = std::min(jb + ja, jc + ji) / 2;
          for (int J = Jmin; J <= Jmax; ++J) {
            z += 0.5 * (2 * J + 1) / static_cast<double>(ji + 1) * w *
                 (tbme(X, J, c, i, a, b) * tbme(Y, J, a, b, c, j) -
                  tbme(Y, J, c, i, a, b) * tbme(X, J, a, b, c, j));
          }
        }
      }
    }
    return z;
  });
  Z.sync_to_device();
}

void comm122ss(const Operator& X, const Operator& Y, Operator& Z) {
  // One factor may be a pure one-body operator (its unallocated two-body part reads as zero).
  if (!Z.two_body().allocated()) return;
  if (!X.two_body().allocated() && !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  add_to_every_two_body_element(Z, [&](int J, int i, int j, int k, int l) {
    double zx = 0.0;
    double zy = 0.0;
    for (int a = 0; a < n; ++a) {
      zx += X.one_body().get(oi(i), oi(a)) * tbme(Y, J, a, j, k, l) +
            X.one_body().get(oi(j), oi(a)) * tbme(Y, J, i, a, k, l) -
            tbme(Y, J, i, j, a, l) * X.one_body().get(oi(a), oi(k)) -
            tbme(Y, J, i, j, k, a) * X.one_body().get(oi(a), oi(l));
      zy += Y.one_body().get(oi(i), oi(a)) * tbme(X, J, a, j, k, l) +
            Y.one_body().get(oi(j), oi(a)) * tbme(X, J, i, a, k, l) -
            tbme(X, J, i, j, a, l) * Y.one_body().get(oi(a), oi(k)) -
            tbme(X, J, i, j, k, a) * Y.one_body().get(oi(a), oi(l));
    }
    return zx - zy;
  });
  Z.sync_to_device();
}

void comm222_pp_hhss(const Operator& X, const Operator& Y, Operator& Z) {
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  add_to_every_two_body_element(Z, [&](int J, int i, int j, int k, int l) {
    double z = 0.0;
    for (int a = 0; a < n; ++a) {
      for (int b = a; b < n; ++b) {
        const double flip = (a == b) ? 1.0 : 2.0;
        const double w = (1.0 - occ[a]) * (1.0 - occ[b]) - occ[a] * occ[b];
        if (w == 0.0) continue;
        z += 0.5 * flip * w *
             (tbme(X, J, i, j, a, b) * tbme(Y, J, a, b, k, l) -
              tbme(Y, J, i, j, a, b) * tbme(X, J, a, b, k, l));
      }
    }
    return z;
  });
  Z.sync_to_device();
}

void comm222_phss(const Operator& X, const Operator& Y, Operator& Z) {
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const int n = X.model_space().num_orbits();
  const auto occ = occupations_of(X);
  const int Jbig = X.model_space().basis().max_two_body_J();

  // Pandya table F[Jp][p][q][r][s] = - sum_Jpp (2 Jpp + 1) {j_p j_q Jp; j_r j_s Jpp}
  //                                          <p s; Jpp| V |r q; Jpp>   (unnormalized),
  // so xbar_ilab = F[Jp][i][l][a][b] and xbar_abkj = F[Jp][a][b][k][j].
  const std::size_t n2 = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t n4 = n2 * n2;
  auto idx = [&](int Jp, int p, int q, int r, int s) {
    const auto un = static_cast<std::size_t>(n);
    return static_cast<std::size_t>(Jp) * n4 +
           ((static_cast<std::size_t>(p) * un + static_cast<std::size_t>(q)) * un +
            static_cast<std::size_t>(r)) *
               un +
           static_cast<std::size_t>(s);
  };
  auto build = [&](const Operator& O) {
    std::vector<double> F(static_cast<std::size_t>(Jbig + 1) * n4, 0.0);
    for (int Jp = 0; Jp <= Jbig; ++Jp) {
      for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
          for (int r = 0; r < n; ++r) {
            for (int s = 0; s < n; ++s) {
              const int jp = j2_of(O, p);
              const int jq = j2_of(O, q);
              const int jr = j2_of(O, r);
              const int js = j2_of(O, s);
              const int Jmin = std::max(std::abs(jp - js), std::abs(jr - jq)) / 2;
              const int Jmax = std::min(jp + js, jr + jq) / 2;
              double f = 0.0;
              for (int Jpp = Jmin; Jpp <= Jmax; ++Jpp) {
                const double sixj = angmom::wigner_6j(TwoJ{jp}, TwoJ{jq}, TwoJ{2 * Jp}, TwoJ{jr},
                                                      TwoJ{js}, TwoJ{2 * Jpp});
                f -= (2 * Jpp + 1) * sixj * tbme(O, Jpp, p, s, r, q);
              }
              F[idx(Jp, p, q, r, s)] = f;
            }
          }
        }
      }
    }
    return F;
  };
  const std::vector<double> FX = build(X);
  const std::vector<double> FY = build(Y);

  add_to_every_two_body_element(Z, [&](int J, int i, int j, int k, int l) {
    const int ji = j2_of(X, i);
    const int jj = j2_of(X, j);
    const int jk = j2_of(X, k);
    const int jl = j2_of(X, l);
    const int Jpmin = std::min(std::max(std::abs(ji - jl), std::abs(jj - jk)),
                               std::max(std::abs(jj - jl), std::abs(ji - jk))) /
                      2;
    const int Jpmax = std::max(std::min(ji + jl, jj + jk), std::min(jj + jl, ji + jk)) / 2;
    double z = 0.0;
    for (int Jp = Jpmin; Jp <= Jpmax; ++Jp) {
      double zbar_ilkj = 0.0;
      double zbar_jlki = 0.0;
      for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
          const double w = occ[a] - occ[b];
          if (w == 0.0) continue;
          zbar_ilkj += w * (FX[idx(Jp, i, l, a, b)] * FY[idx(Jp, a, b, k, j)] -
                            FY[idx(Jp, i, l, a, b)] * FX[idx(Jp, a, b, k, j)]);
          zbar_jlki += w * (FX[idx(Jp, j, l, a, b)] * FY[idx(Jp, a, b, k, i)] -
                            FY[idx(Jp, j, l, a, b)] * FX[idx(Jp, a, b, k, i)]);
        }
      }
      const double sixj_ijkl =
          angmom::wigner_6j(TwoJ{ji}, TwoJ{jj}, TwoJ{2 * J}, TwoJ{jk}, TwoJ{jl}, TwoJ{2 * Jp});
      const double sixj_jikl =
          angmom::wigner_6j(TwoJ{jj}, TwoJ{ji}, TwoJ{2 * J}, TwoJ{jk}, TwoJ{jl}, TwoJ{2 * Jp});
      const int phase_ij = phase_of((ji + jj - 2 * J) / 2);
      z += (2 * Jp + 1) * sixj_ijkl * zbar_ilkj;
      z -= (2 * Jp + 1) * sixj_jikl * zbar_jlki * phase_ij;
    }
    return z;
  });
  Z.sync_to_device();
}

void comm222_pp_hh_221ss(const Operator& X, const Operator& Y, Operator& Z) {
  comm222_pp_hhss(X, Y, Z);
  comm221ss(X, Y, Z);
}

Operator commutator(const Operator& X, const Operator& Y) {
  Operator Z(X.model_space_ptr(), commutator_spec(X.spec(), Y.spec()));
  comm110ss(X, Y, Z);
  comm220ss(X, Y, Z);
  comm111ss(X, Y, Z);
  comm121ss(X, Y, Z);
  comm221ss(X, Y, Z);
  comm122ss(X, Y, Z);
  comm222_pp_hhss(X, Y, Z);
  comm222_phss(X, Y, Z);
  return Z;
}

}  // namespace imsrg::commutator_ref
