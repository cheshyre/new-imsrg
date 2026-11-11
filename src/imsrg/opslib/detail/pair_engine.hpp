#pragma once
/// @file
/// Internal helpers of the opslib module (not public API): the two-body matrix-element engine of
/// the operators p1.p2 and r1.r2 (imsrg++ `Calculate_p1p2` / `Calculate_r1r2`) and the host fills
/// of one-body and two-body parts.

#include <cmath>
#include <cstdint>
#include <memory>

#include "imsrg/angmom/moshinsky.hpp"
#include "imsrg/angmom/tables.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::opslib::detail {

/// Which two-body operator the engine evaluates.
enum class PairKind {
  Momentum,  ///< p1.p2/m = 0.5 hbar_omega x (oscillator sums), MeV
  Position,  ///< r1.r2 = 0.5 b^2 x (oscillator sums), fm^2, b^2 = hbar^2/(m_nucleon hbar_omega)
};

/// Evaluates normalized antisymmetrized J-coupled matrix elements <ab;J| p1.p2/m |cd;J> and
/// <ab;J| r1.r2 |cd;J> in the harmonic-oscillator basis (Moshinsky transformation, LS-jj
/// recoupling). Construction builds the Moshinsky and 9j tables once; `element` is const.
class PairEngine {
 public:
  /// Tables for orbits with e = 2n + l <= `emax` and l <= `lmax`; `hbar_omega` in MeV (> 0).
  /// Cost: a few ms at emax 6 (Moshinsky table up to E = 2 emax).
  PairEngine(int emax, int lmax, double hbar_omega);

  /// Reproduces imsrg++ `Calculate_p1p2` (Momentum) / `Calculate_r1r2` (Position) for the bra
  /// |a b; J> and ket |c d; J>: LS coupling, Moshinsky transformation to (N Lambda, n lambda),
  /// the (T_cm - T_rel)-type sum, and the final factor 0.5 hbar_omega (or 0.5 b^2) times
  /// 1/sqrt((1 + delta_ab)(1 + delta_cd)), where delta_ab = 1 if a == b. The result is the
  /// normalized antisymmetrized element of the operator without division by A. Zero unless the
  /// pair energies e_a+e_b and e_c+e_d differ by 0 or 2. Cost: O(E^3) loop iterations with O(1)
  /// table lookups (E = e_a + e_b).
  double element(PairKind kind, const basis::Orbit& a, const basis::Orbit& b, const basis::Orbit& c,
                 const basis::Orbit& d, TwoJ two_j) const;

 private:
  double hbar_omega_;
  angmom::MoshinskyTable mosh_;
  angmom::NineJLSjjTable nj_;
};

/// Largest shell e = 2n + l of the model space's basis (equals its emax in untruncated bases).
/// O(num_orbits).
int max_orbit_shell(const basis::ModelSpace& ms);

/// Fills the scalar one-body part with scale * [ (2n + l + 3/2) on the diagonal, off_sign *
/// sqrt(n_big (n_big + l + 1/2)) between n and n + 1 ] for every orbit with `select(orbit)` true
/// (imsrg++ KineticEnergy_Op, TCM_Op (1B), RSquaredOp, R2_1body_Op; `off_sign` is +1 for
/// kinetic-like and -1 for r^2-like operators). Adds to the existing values; host mirror only.
/// Cost: O(num_orbits^2).
template <class Select>
void add_oscillator_one_body(op::Operator& op, double scale, double off_sign, Select select) {
  const auto& basis = op.model_space().basis();
  const std::int32_t norb = basis.num_orbits();
  for (std::int32_t ia = 0; ia < norb; ++ia) {
    const basis::Orbit& oa = basis.orbit(OrbitIndex{ia});
    if (!select(oa)) {
      continue;
    }
    op.one_body().add(OrbitIndex{ia}, OrbitIndex{ia}, scale * (2 * oa.n + oa.l + 1.5));
    for (std::int32_t ib = ia + 1; ib < norb; ++ib) {
      const basis::Orbit& ob = basis.orbit(OrbitIndex{ib});
      if (ob.l != oa.l || ob.j2 != oa.j2 || ob.tz2 != oa.tz2) {
        continue;
      }
      if (oa.n + 1 == ob.n) {
        op.one_body().add(OrbitIndex{ia}, OrbitIndex{ib},
                          scale * off_sign * std::sqrt(ob.n * (ob.n + ob.l + 0.5)));
      } else if (ob.n + 1 == oa.n) {
        op.one_body().add(OrbitIndex{ia}, OrbitIndex{ib},
                          scale * off_sign * std::sqrt(oa.n * (oa.n + oa.l + 0.5)));
      }
    }
  }
}

/// Adds `value(channel, bra_orbits, ket_orbits)` to the two-body part for every channel, and every
/// pair of kets bra <= ket (local indices), writing both triangles (Hermitian operator). `value`
/// has the signature `double(const TwoBodyChannel&, const Orbit& a, const Orbit& b, const Orbit&
/// c, const Orbit& d)`; entries with |value| <= `threshold` are skipped (imsrg++ TCM_Op drops
/// |v| <= 1e-7; pass 0 to keep all non-zero entries). Host mirror only. Cost: O(sum over
/// channels of num_kets^2) calls of `value`.
template <class Value>
void add_two_body(op::Operator& op, double threshold, Value value) {
  const auto& chans = op.model_space().two_body();
  const auto& basis = op.model_space().basis();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const basis::TwoBodyChannel& tbc = chans.channel(ch);
    const auto kets = chans.channel_kets(ch);
    auto block = op.two_body().host_block(ch);
    const std::int32_t nk = static_cast<std::int32_t>(kets.size());
    for (std::int32_t i = 0; i < nk; ++i) {
      const basis::Ket& bra = chans.ket(kets[static_cast<std::size_t>(i)]);
      const basis::Orbit& oa = basis.orbit(bra.p);
      const basis::Orbit& ob = basis.orbit(bra.q);
      for (std::int32_t j = i; j < nk; ++j) {
        const basis::Ket& ket = chans.ket(kets[static_cast<std::size_t>(j)]);
        const double v = value(tbc, oa, ob, basis.orbit(ket.p), basis.orbit(ket.q));
        if (std::abs(v) <= threshold || v == 0.0) {
          continue;
        }
        block(i, j) += v;
        if (j != i) {
          block(j, i) += v;
        }
      }
    }
  }
}

}  // namespace imsrg::opslib::detail
