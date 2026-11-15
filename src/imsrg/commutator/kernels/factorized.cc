// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The factorized double commutator of IMSRG(3f2) with one-body intermediates
// (factorized.h; imsrg++ FactorizedDoubleCommutator.cc, comm223_231_chi1b at
// 133-436 and comm223_232_chi1b at 1090-1394). Kernel TU. Steps:
//   1. W_x = eta P_x eta (detail::LadderProducts), contracted into chi1, chia
//      (`FactorizedEta`, reusable for several gamma);
//   2. V_x = gamma P_x eta, contracted into chi2, chib (workspace scratch);
//   3. Z1, one thread per one-body channel entry (the comm121ss shape);
//   4. Z2, one thread per upper-triangle element (the comm122ss shape, over
//      the lifted one-body lists of the workspace).
// From imsrg++'s loops to the ladder products (N_x = n n, Nbar_x = nbar nbar
// over the orbits of ket x): diagram I builds per channel tmp = 2(2J+1)
// (O o eta) eta with O_xy = N_x Nbar_y - Nbar_x N_y (`Eta_matrix_nnnn`), so
// tmp + tmp^T = 2(2J+1) [(N_x + N_z) W_pp - (Nbar_x + Nbar_z) W_hh]_xz; for
// x = (b d), z = (b e) its sum over b and J is chia. Chi_221_b sums
// occfactor eta~_bcae gamma~_adbc with occfactor = nbar_a nbar_e N_bc -
// Nbar_bc n_a n_e over all (b, c); an unrestricted sum over (b, c) is twice
// the sum over normalized kets, so it is 2 nbar_e B_hh - 2 n_e B_pp = chib.
// CHI_I and CHI_II of 232 are chi1 and chi2 (the i, j sums are the hole-hole
// halves, the a, b sums the particle-particle halves).

#include "imsrg/commutator/factorized.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/ladder.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator {

namespace {

using detail::OneBodyChannelView;
using detail::TwoBodyView;
using linalg::MatrixView;

/// View of a dense num_orbits x num_orbits scratch matrix.
MatrixView<double> SquareView(exec::DeviceBuffer<double>& buf, int norb) {
  return MatrixView<double>{buf.device_data(), norb, norb, norb};
}

/// Read-only view of a dense num_orbits x num_orbits matrix.
MatrixView<const double> ConstSquareView(const exec::DeviceBuffer<double>& buf,
                                         int norb) {
  return MatrixView<const double>{buf.device_data(), norb, norb, norb};
}

/// POD bundle of the contraction kernel.
struct ChiArgs {
  TwoBodyView mpp;  ///< W_pp or V_pp (ladder scratch, two-body layout).
  TwoBodyView mhh;  ///< W_hh or V_hh.
  const double* occ = nullptr;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* owner = nullptr;
  const std::int32_t* orbits = nullptr;
  MatrixView<double> sum;       ///< chi1 or chi2.
  MatrixView<double> weighted;  ///< chia or chib.
  int num_orbits = 0;
  bool eta_eta = true;  ///< W (eta P eta) or V (gamma P eta) in the scratch.
};
static_assert(std::is_trivially_copyable_v<ChiArgs>);

/// From the products in `impl.ladder.{pp,hh}.product`, for every one-body
/// channel entry (d, e): A_pp = C[M_pp; n], A_hh = C[M_hh; nbar] (C as in
/// factorized.h), then sum = A_pp + A_hh and weighted = 2 (n_d + n_e) A_pp -
/// 2 (nbar_d + nbar_e) A_hh (`eta_eta`: chi1, chia) or 2 nbar_e A_hh - 2 n_e
/// A_pp (chi2, chib). `like` supplies the two-body tables. One thread per
/// entry, O(E norb Jr). Fenced.
void ContractChi(const op::Operator& like, bool eta_eta, Workspace::Impl& impl,
                 MatrixView<double> sum, MatrixView<double> weighted) {
  ChiArgs a;
  a.mpp = impl.two_body_view(like);
  a.mpp.data = impl.ladder.pp.product.device_data();
  a.mhh = a.mpp;
  a.mhh.data = impl.ladder.hh.product.device_data();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.owner = impl.ob_owner.device_data();
  a.orbits = impl.ob_orbits.device_data();
  a.sum = sum;
  a.weighted = weighted;
  a.num_orbits = impl.tables.orbits().num_orbits;
  a.eta_eta = eta_eta;
  exec::ParallelFor(
      "commutator::factorized_chi",
      static_cast<exec::Index>(impl.ob_orbits.size()),
      IMSRG_LAMBDA(const exec::Index k) {
        const int d = a.owner[k];
        const int e = a.orbits[k];
        const int jd = a.j2[d];  // = j_e
        double app = 0.0;
        double ahh = 0.0;
        for (int c = 0; c < a.num_orbits; ++c) {
          const double wp = a.occ[c];
          const double wh = 1.0 - a.occ[c];
          const int jc = a.j2[c];
          const int lo = jc > jd ? jc - jd : jd - jc;
          const int hi = jc + jd;
          for (int two_J = lo; two_J <= hi; two_J += 2) {
            if (wp != 0.0)
              app +=
                  (two_J + 1) * wp * a.mpp.get_unnormalized(c, d, c, e, two_J);
            if (wh != 0.0)
              ahh +=
                  (two_J + 1) * wh * a.mhh.get_unnormalized(c, d, c, e, two_J);
          }
        }
        app /= jd + 1;
        ahh /= jd + 1;
        const double nd = a.occ[d];
        const double ne = a.occ[e];
        a.sum(d, e) = app + ahh;
        a.weighted(d, e) = a.eta_eta ? 2.0 * (nd + ne) * app -
                                           2.0 * ((1.0 - nd) + (1.0 - ne)) * ahh
                                     : 2.0 * (1.0 - ne) * ahh - 2.0 * ne * app;
      });
  exec::Fence("commutator::factorized_chi");
}

/// POD bundle of the one-body output kernel.
struct OneBodyOutArgs {
  TwoBodyView eta;
  TwoBodyView gamma;
  MatrixView<const double> chia;
  MatrixView<const double> chib;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* owner = nullptr;
  const std::int32_t* orbits = nullptr;
  OneBodyChannelView ch1;
  MatrixView<double> z1;
  double s = 0.0;  ///< h_eta h_Z.
  int num_orbits = 0;
};
static_assert(std::is_trivially_copyable_v<OneBodyOutArgs>);

/// Z_pq += 1 / (2 jhat_p^2) sum_de sum_J (2J+1) (chia_de gamma~^J_epdq +
/// chic_de eta~^J_epdq), chic_de = chib_de + s chib_ed (231: I, IIIa, IIIb).
/// One thread per one-body channel entry (p, q); O(E norb c Jr). Fenced.
void AddOneBody(const op::Operator& eta, const op::Operator& gamma,
                MatrixView<const double> chia, op::Operator& Z, double s,
                Workspace::Impl& impl) {
  const int norb = impl.tables.orbits().num_orbits;
  OneBodyOutArgs a;
  a.eta = impl.two_body_view(eta);
  a.gamma = impl.two_body_view(gamma);
  a.chia = chia;
  a.chib = SquareView(impl.factorized.chib, norb);
  a.j2 = impl.tables.orbits().j2;
  a.owner = impl.ob_owner.device_data();
  a.orbits = impl.ob_orbits.device_data();
  a.ch1 = impl.one_body_channels();
  a.z1 = Z.one_body().device_view();
  a.s = s;
  a.num_orbits = norb;
  exec::ParallelFor(
      "commutator::factorized_231",
      static_cast<exec::Index>(impl.ob_orbits.size()),
      IMSRG_LAMBDA(const exec::Index k) {
        const int p = a.owner[k];
        const int q = a.orbits[k];
        const int jp = a.j2[p];  // = j_q
        double z = 0.0;
        for (int d = 0; d < a.num_orbits; ++d) {
          const int jd = a.j2[d];  // = j_e
          const int lo = jd > jp ? jd - jp : jp - jd;
          const int hi = jd + jp;
          for (int t = a.ch1.offsets[d]; t < a.ch1.offsets[d + 1]; ++t) {
            const int e = a.ch1.orbits[t];
            const double ca = a.chia(d, e);
            const double cc = a.chib(d, e) + a.s * a.chib(e, d);
            if (ca == 0.0 && cc == 0.0) continue;
            double sg = 0.0;
            double se = 0.0;
            for (int two_J = lo; two_J <= hi; two_J += 2) {
              sg += (two_J + 1) * a.gamma.get_unnormalized(e, p, d, q, two_J);
              se += (two_J + 1) * a.eta.get_unnormalized(e, p, d, q, two_J);
            }
            z += ca * sg + cc * se;
          }
        }
        a.z1(p, q) += 0.5 * z / (jp + 1);
      });
  exec::Fence("commutator::factorized_231");
}

/// POD bundle of the two-body output kernel.
struct TwoBodyOutArgs {
  const double* eta = nullptr;    ///< eta_2 (normalized blocks).
  const double* gamma = nullptr;  ///< gamma_2 (normalized blocks).
  double* z2 = nullptr;
  MatrixView<const double> chi1;
  MatrixView<const double> chi2;
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* num_kets = nullptr;
  const std::int32_t* row_base = nullptr;
  const std::int32_t* entry_offset = nullptr;
  const std::int32_t* lm = nullptr;
  const std::int32_t* lfrom = nullptr;
  const std::int32_t* lto = nullptr;
  const double* lcoef = nullptr;
  double hz = 1.0;  ///< h_Z = h_gamma.
  double s = 0.0;   ///< h_eta h_Z.
  int num_channels = 0;
};
static_assert(std::is_trivially_copyable_v<TwoBodyOutArgs>);

/// Z~_pqrs += sum_b (chi1_pb gamma~_bqrs + chi1_qb gamma~_pbrs + gamma~_pqbs
/// chi1_br + gamma~_pqrb chi1_bs) - sum_b (s chi2_bp eta~_bqrs + s chi2_bq
/// eta~_pbrs + eta~_pqbs chi2_br + eta~_pqrb chi2_bs) (232: Ia, Ib, IVa, IVb).
/// As comm122ss: one thread per element (r, c) with r <= c of every block; the
/// bra part B(r, c) runs over the lifted list of row r with chi1(from, to) and
/// -s chi2(to, from); the ket part is h_Z B'(c, r), B' the same with chi1
/// transposed (exact for the round-off asymmetry of chi1); Z(c, r) = h_Z
/// Z(r, c). O(sum over channels of N^2 x orbits per one-body channel). Fenced.
void AddTwoBody(const op::Operator& eta, const op::Operator& gamma,
                MatrixView<const double> chi1, op::Operator& Z, double hz,
                double s, Workspace::Impl& impl) {
  const int norb = impl.tables.orbits().num_orbits;
  const TwoBodyView ev = impl.two_body_view(eta);
  TwoBodyOutArgs a;
  a.eta = ev.data;
  a.gamma = impl.two_body_view(gamma).data;
  a.z2 = Z.two_body().matrix().buffer().device_data();
  a.chi1 = chi1;
  a.chi2 = SquareView(impl.factorized.chi2, norb);
  a.block_offset = impl.block_offsets.device_data();
  a.num_kets = ev.tb.channel_num_kets;
  a.row_base = impl.lifted.row_base.device_data();
  a.entry_offset = impl.lifted.entry_offset.device_data();
  a.lm = impl.lifted.m.device_data();
  a.lfrom = impl.lifted.from.device_data();
  a.lto = impl.lifted.to.device_data();
  a.lcoef = impl.lifted.coef.device_data();
  a.hz = hz;
  a.s = s;
  a.num_channels = ev.tb.num_channels;
  exec::ParallelFor(
      "commutator::factorized_232",
      static_cast<exec::Index>(impl.two_body_size),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = detail::FindBlock(a.block_offset, a.num_channels, e);
        const std::int64_t off = a.block_offset[ch];
        const std::int64_t local = e - off;
        const int n = a.num_kets[ch];
        const int r = static_cast<int>(local % n);
        const int c = static_cast<int>(local / n);
        if (r > c) return;
        const int rb = a.row_base[ch];
        double b_rc = 0.0;
        double b_cr = 0.0;
        for (int side = 0; side < 2; ++side) {
          const int row = side == 0 ? r : c;
          const std::int64_t col = side == 0 ? c : r;
          double acc = 0.0;
          for (int t = a.entry_offset[rb + row];
               t < a.entry_offset[rb + row + 1]; ++t) {
            const std::int64_t idx = off + a.lm[t] + col * n;
            const int from = a.lfrom[t];
            const int to = a.lto[t];
            const double c1 = side == 0 ? a.chi1(from, to) : a.chi1(to, from);
            acc += a.lcoef[t] *
                   (c1 * a.gamma[idx] - a.s * a.chi2(to, from) * a.eta[idx]);
          }
          if (side == 0)
            b_rc = acc;
          else
            b_cr = acc;
        }
        const double z = b_rc + a.hz * b_cr;
        a.z2[off + r + static_cast<std::int64_t>(c) * n] += z;
        if (r < c) a.z2[off + c + static_cast<std::int64_t>(r) * n] += a.hz * z;
      });
  exec::Fence("commutator::factorized_232");
}

/// NotImplementedError unless `o` is scalar, number conserving and of
/// particle_rank <= 2.
void CheckOperand(const op::Operator& o, const char* name) {
  const op::OperatorSpec& sp = o.spec();
  if (!sp.is_scalar() || !sp.is_number_conserving() || sp.particle_rank > 2)
    throw NotImplementedError(
        std::string("factorized double commutator: ") + name +
        " must be scalar, number conserving and of particle rank <= 2");
}

/// The checks of eta (see factorized.h).
void CheckEta(const op::Operator& eta, const Workspace& ws) {
  IMSRG_CHECK(ws.Compatible(eta),
              "factorized: eta is not on the workspace's model space");
  CheckOperand(eta, "eta");
  if (eta.hermiticity() != Hermiticity::kAntiHermitian)
    throw NotImplementedError(
        "factorized double commutator: eta must be anti-Hermitian (imsrg++ "
        "assumes it)");
}

/// The checks of gamma, Z and the options (see factorized.h).
void CheckGamma(const op::Operator& eta, const op::Operator& gamma,
                const op::Operator& Z, const FactorizedOptions& opts,
                const Workspace& ws) {
  IMSRG_CHECK(ws.Compatible(gamma),
              "factorized: gamma is not on the workspace's model space");
  IMSRG_CHECK(&Z != &eta && &Z != &gamma,
              "factorized: Z must not alias eta or gamma");
  CheckOperand(gamma, "gamma");
  if (gamma.hermiticity() == Hermiticity::kNonHermitian)
    throw NotImplementedError(
        "factorized double commutator: NonHermitian gamma is not supported");
  if (opts.two_body_intermediates)
    throw NotImplementedError(
        "factorized double commutator: two-body intermediates are not "
        "implemented (the IMSRG(3f2) recipe, D32, switches them off)");
}

/// Resets Z to a zero scalar rank-2 operator with gamma's hermiticity, reusing
/// its storage when the spec already matches.
void ResetZ(const op::Operator& gamma, op::Operator& Z, const Workspace& ws) {
  const op::OperatorSpec spec =
      op::OperatorSpec::Scalar(2, gamma.hermiticity());
  if (!Z.empty() && Z.spec() == spec && ws.Compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(ws.model_space_ptr(), spec);
  }
}

/// True if F vanishes identically: no topology selected, or eta or gamma
/// without a two-body part.
bool Vanishes(const op::Operator& eta, const op::Operator& gamma,
              const FactorizedOptions& opts) {
  return !opts.one_body_intermediates || !eta.two_body().allocated() ||
         !gamma.two_body().allocated();
}

}  // namespace

FactorizedEta::FactorizedEta(const op::Operator& eta, Workspace& ws)
    : eta_(&eta) {
  CheckEta(eta, ws);
  auto& impl = ws.impl();
  const int norb = impl.tables.orbits().num_orbits;
  const auto n =
      static_cast<std::size_t>(norb) * static_cast<std::size_t>(norb);
  chi1_ = exec::DeviceBuffer<double>(n, "commutator::factorized_chi1");
  chia_ = exec::DeviceBuffer<double>(n, "commutator::factorized_chia");
  if (!eta.two_body().allocated()) return;
  detail::LadderProducts(eta, eta, impl);  // W_x = eta P_x eta
  ContractChi(eta, true, impl, SquareView(chi1_, norb),
              SquareView(chia_, norb));
}

void FactorizedDoubleCommutator(const FactorizedEta& pre,
                                const op::Operator& gamma, op::Operator& Z,
                                const FactorizedOptions& opts, Workspace& ws) {
  const op::Operator& eta = pre.eta();
  IMSRG_CHECK(ws.Compatible(eta),
              "factorized: eta is not on the workspace's model space");
  CheckGamma(eta, gamma, Z, opts, ws);
  ResetZ(gamma, Z, ws);
  if (Vanishes(eta, gamma, opts)) return;

  auto& impl = ws.impl();
  const int norb = impl.tables.orbits().num_orbits;
  const double hz =
      static_cast<double>(op::HermiticitySign(gamma.hermiticity()));
  const double s =
      static_cast<double>(op::HermiticitySign(eta.hermiticity())) * hz;
  detail::LadderProducts(gamma, eta, impl);  // V_x = gamma P_x eta
  ContractChi(eta, false, impl, SquareView(impl.factorized.chi2, norb),
              SquareView(impl.factorized.chib, norb));
  AddOneBody(eta, gamma, ConstSquareView(pre.chia(), norb), Z, s, impl);
  AddTwoBody(eta, gamma, ConstSquareView(pre.chi1(), norb), Z, hz, s, impl);
}

void FactorizedDoubleCommutator(const op::Operator& eta,
                                const op::Operator& gamma, op::Operator& Z,
                                const FactorizedOptions& opts, Workspace& ws) {
  CheckEta(eta, ws);
  CheckGamma(eta, gamma, Z, opts, ws);
  if (Vanishes(eta, gamma, opts)) {
    ResetZ(gamma, Z, ws);
    return;
  }
  const FactorizedEta pre(eta, ws);
  FactorizedDoubleCommutator(pre, gamma, Z, opts, ws);
}

}  // namespace imsrg::commutator
