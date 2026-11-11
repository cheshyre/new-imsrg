#include "imsrg/hf/hartree_fock.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/hf/diis.hpp"
#include "imsrg/hf/monopole.hpp"
#include "imsrg/linalg/blas.hpp"

namespace imsrg::hf {

namespace {

using linalg::Matrix;

/// Everything the iteration needs, built once.
class Solver {
 public:
  Solver(const op::Operator& H, const op::No2bThreeBody* v3, const HFOptions& opt)
      : H_(H), opt_(opt), norb_(H.model_space().num_orbits()) {
    H_.sync_to_host();
    const auto& basis = H_.model_space().basis();
    const auto n = static_cast<std::size_t>(norb_);
    channels_ = detail::one_body_channels(basis);
    two_j_.resize(n);
    tz2_.resize(n);
    for (std::int32_t i = 0; i < norb_; ++i) {
      const auto& o = basis.orbit(OrbitIndex{i});
      two_j_[static_cast<std::size_t>(i)] = o.j2.twice;
      tz2_[static_cast<std::size_t>(i)] = o.tz2.twice;
    }
    KE_ = Matrix<double>(norb_, norb_);
    for (std::int32_t i = 0; i < norb_; ++i) {
      for (std::int32_t j = 0; j < norb_; ++j) {
        KE_(i, j) = H_.one_body().get(OrbitIndex{i}, OrbitIndex{j});
      }
    }
    mono2_ = std::make_unique<detail::Monopole2B>(H_);
    if (v3 != nullptr) mono3_ = std::make_unique<detail::Monopole3B>(H_, *v3, opt.e3max);
  }

  HFResult run() {
    const auto& ref = H_.model_space().reference();
    z_ref_ = ref.Z();
    n_ref_ = ref.N();
    const auto n = static_cast<std::size_t>(norb_);

    HFResult res;
    C_ = Matrix<double>(norb_, norb_);
    for (std::int32_t i = 0; i < norb_; ++i) C_(i, i) = 1.0;
    occ_.assign(n, 0.0);
    for (std::int32_t i = 0; i < norb_; ++i) {
      occ_[static_cast<std::size_t>(i)] = ref.occupation(OrbitIndex{i});
    }
    rho_ = density();
    build_fock(rho_, F_);

    std::vector<double> energies(n), prev(n, 0.0);
    for (std::int32_t i = 0; i < norb_; ++i) energies[static_cast<std::size_t>(i)] = KE_(i, i);

    std::vector<double> diis_weights(n);
    for (std::size_t i = 0; i < n; ++i) diis_weights[i] = two_j_[i] + 1.0;
    detail::PulayDiis diis(diis_weights, static_cast<std::size_t>(opt_.diis_subspace));

    bool use_diis = opt_.mixing == Mixing::Diis;
    bool freeze = opt_.freeze_occupations;
    int fill_modulus = 3;
    std::int32_t it = 0;
    for (; it < opt_.max_iterations; ++it) {
      prev = energies;
      diagonalize(F_, energies);

      if (it > 10 && it % fill_modulus == 0 && !freeze) {
        if (fill_lowest_orbits()) diis.clear();
        fill_modulus = (fill_modulus % 5) + 1;
      }
      if (it == 800) freeze = true;  // imsrg++: stop re-filling a run that does not settle

      Matrix<double> rho_new = density();
      if (use_diis) {
        rho_ = std::move(rho_new);
        Matrix<double> Fk;
        build_fock(rho_, Fk);
        if (it >= opt_.diis_start) {
          diis.push(Fk, commutator(Fk, rho_));
          F_ = diis.extrapolate();
        } else {
          F_ = std::move(Fk);
        }
      } else {
        const double w = opt_.density_mixing;
        for (std::int32_t j = 0; j < norb_; ++j) {
          for (std::int32_t i = 0; i < norb_; ++i) {
            rho_(i, j) = (1.0 - w) * rho_new(i, j) + w * rho_(i, j);
          }
        }
        build_fock(rho_, F_);
      }
      energy(res);

      double diff = 0.0;
      for (std::size_t i = 0; i < n; ++i) diff += (energies[i] - prev[i]) * (energies[i] - prev[i]);
      diff = std::sqrt(diff) / static_cast<double>(norb_);
      res.ediff_history.push_back(diff);
      if (diff < opt_.tolerance) {
        res.converged = true;
        break;
      }
      // imsrg++ falls back to DIIS when linear mixing stalls (its own variant, after
      // iteration 100); here the Pulay DIIS of this module takes over with a fresh history.
      if (!use_diis && it == 100) {
        use_diis = true;
        diis.clear();
      }
    }
    res.iterations = it;

    // Final state: F consistent with the final rho, final energies.
    build_fock(rho_, F_);
    energy(res);
    res.spe = energies;
    res.C = C_;
    res.rho = rho_;
    res.F = F_;
    res.occupations = occ_;
    return res;
  }

 private:
  /// rho(a,b) = sum_alpha n_alpha C(a,alpha) C(b,alpha) over the holes alpha.
  Matrix<double> density() const {
    Matrix<double> rho(norb_, norb_);
    for (std::int32_t alpha = 0; alpha < norb_; ++alpha) {
      const double n = occ_[static_cast<std::size_t>(alpha)];
      if (n < basis::kOccupationCut) continue;
      for (std::int32_t b = 0; b < norb_; ++b) {
        const double cb = n * C_(b, alpha);
        if (cb == 0.0) continue;
        for (std::int32_t a = 0; a < norb_; ++a) rho(a, b) += C_(a, alpha) * cb;
      }
    }
    return rho;
  }

  /// F = KE + V2[rho] + 1/2 V3[rho]; keeps V2, V3 for the energy.
  void build_fock(const Matrix<double>& rho, Matrix<double>& F) {
    mono2_->apply(rho, Vij_);
    F = KE_;
    for (std::int32_t j = 0; j < norb_; ++j) {
      for (std::int32_t i = 0; i < norb_; ++i) F(i, j) += Vij_(i, j);
    }
    if (mono3_) {
      mono3_->apply(rho, V3ij_);
      for (std::int32_t j = 0; j < norb_; ++j) {
        for (std::int32_t i = 0; i < norb_; ++i) F(i, j) += 0.5 * V3ij_(i, j);
      }
    }
  }

  /// e1, e2, e3 from the current rho and the V2, V3 of the last `build_fock(rho_, .)`.
  void energy(HFResult& res) const {
    double e1 = 0.0, e2 = 0.0, e3 = 0.0;
    for (const auto& ch : channels_) {
      for (std::int32_t i : ch) {
        const double w = two_j_[static_cast<std::size_t>(i)] + 1.0;
        for (std::int32_t j : ch) {
          const double r = rho_(i, j) * w;
          e1 += r * KE_(i, j);
          e2 += 0.5 * r * Vij_(i, j);
          if (mono3_) e3 += r * V3ij_(i, j) / 6.0;
        }
      }
    }
    res.e1 = e1;
    res.e2 = e2;
    res.e3 = e3;
    res.e_hf = e1 + e2 + e3;
  }

  /// Channel-wise diagonalization of F: C columns = eigenvectors (ascending energy), the
  /// eigenvalue of rank k in a channel is stored at the orbit index of rank k.
  void diagonalize(const Matrix<double>& F, std::vector<double>& energies) {
    C_ = Matrix<double>(norb_, norb_);
    for (const auto& ch : channels_) {
      const auto m = static_cast<std::int64_t>(ch.size());
      Matrix<double> Fc(m, m);
      for (std::int64_t q = 0; q < m; ++q) {
        for (std::int64_t p = 0; p < m; ++p) {
          Fc(p, q) = F(ch[static_cast<std::size_t>(p)], ch[static_cast<std::size_t>(q)]);
        }
      }
      auto eig = linalg::eigh(Fc.const_view());
      for (std::int64_t q = 0; q < m; ++q) {
        energies[static_cast<std::size_t>(ch[static_cast<std::size_t>(q)])] =
            eig.eigenvalues[static_cast<std::size_t>(q)];
        for (std::int64_t p = 0; p < m; ++p) {
          C_(ch[static_cast<std::size_t>(p)], ch[static_cast<std::size_t>(q)]) =
              eig.eigenvectors(p, q);
        }
      }
    }
    // imsrg++ phase: the diagonal element of every column is made non-negative.
    for (std::int32_t i = 0; i < norb_; ++i) {
      if (C_(i, i) < 0.0) {
        for (std::int32_t a = 0; a < norb_; ++a) C_(a, i) = -C_(a, i);
      }
    }
  }

  /// imsrg++ `FillLowestOrbits`: occupy the lowest HF orbits of each species up to Z and N.
  /// Returns true if the occupations changed.
  bool fill_lowest_orbits() {
    const auto n = static_cast<std::size_t>(norb_);
    // diag(C^T F C)
    std::vector<double> e(n, 0.0);
    for (std::int32_t alpha = 0; alpha < norb_; ++alpha) {
      double s = 0.0;
      for (std::int32_t b = 0; b < norb_; ++b) {
        double t = 0.0;
        for (std::int32_t a = 0; a < norb_; ++a) t += C_(a, alpha) * F_(a, b);
        s += t * C_(b, alpha);
      }
      e[static_cast<std::size_t>(alpha)] = s;
    }
    std::vector<std::int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int32_t x, std::int32_t y) {
      return e[static_cast<std::size_t>(x)] < e[static_cast<std::size_t>(y)];
    });
    std::vector<double> occ(n, 0.0);
    double placed_p = 0.0, placed_n = 0.0;
    for (std::int32_t alpha : order) {
      const auto a = static_cast<std::size_t>(alpha);
      const bool proton = tz2_[a] < 0;
      double& placed = proton ? placed_p : placed_n;
      const double target = proton ? z_ref_ : n_ref_;
      const double g = two_j_[a] + 1.0;
      if (placed < target) {
        occ[a] = std::min(1.0, (target - placed) / g);
        placed = std::min(placed + g, target);
      }
      if (placed_p >= z_ref_ && placed_n >= n_ref_) break;
    }
    const bool changed = occ != occ_;
    occ_ = std::move(occ);
    return changed;
  }

  /// e = F rho - rho F.
  static Matrix<double> commutator(const Matrix<double>& F, const Matrix<double>& rho) {
    const auto n = F.rows();
    Matrix<double> e(n, n);
    linalg::gemm(1.0, F.const_view(), linalg::Op::NoTrans, rho.const_view(), linalg::Op::NoTrans,
                 0.0, e.view());
    linalg::gemm(-1.0, rho.const_view(), linalg::Op::NoTrans, F.const_view(), linalg::Op::NoTrans,
                 1.0, e.view());
    return e;
  }

  op::Operator H_;  // host-synced deep copy
  HFOptions opt_;
  std::int32_t norb_;
  std::vector<std::vector<std::int32_t>> channels_;
  std::vector<std::int32_t> two_j_, tz2_;
  Matrix<double> KE_, Vij_, V3ij_, C_, rho_, F_;
  std::vector<double> occ_;
  double z_ref_ = 0.0, n_ref_ = 0.0;
  std::unique_ptr<detail::Monopole2B> mono2_;
  std::unique_ptr<detail::Monopole3B> mono3_;
};

}  // namespace

HFResult solve(const op::Operator& H, const op::No2bThreeBody* v3, const HFOptions& options) {
  IMSRG_CHECK(!H.empty(), "hf::solve needs a non-empty Hamiltonian");
  IMSRG_CHECK(options.max_iterations >= 1 && options.tolerance > 0.0,
              "invalid Hartree-Fock options");
  Solver solver(H, v3, options);
  return solver.run();
}

}  // namespace imsrg::hf
