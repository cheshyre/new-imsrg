#pragma once
/// @file
/// Talmi-Moshinsky brackets and Talmi coefficients for harmonic-oscillator (HO) states
/// (design/10 section 2, layer 1; replaces imsrg++ `AngMom::Moshinsky`, `AngMom::TalmiB` and
/// `imsrg_util::TalmiI`).
///
/// Conventions (identical to imsrg++, so results compare element by element): equal masses
/// (imsrg++'s default `mosh_beta_1` = pi/4); HO radial quantum number n >= 0 and orbital l >= 0;
/// brackets in Moshinsky's phase convention, computed with the algorithm of Buck et al., NPA 600
/// (1996) 387, and converted by (-1)^(l + L + lambda) exactly as `AngMom.cc:163-241`.
/// Equivalently (verified with an independent exact construction, tests/unit/angmom/reference/
/// gen_moshinsky_exact.py): R = (r1 + r2)/sqrt2, r = (r1 - r2)/sqrt2, HO radial functions
/// positive at the origin, Condon-Shortley coupling, and the relative motion coupled FIRST,
/// i.e. the value is <n l, N L; lambda | n1 l1, n2 l2; lambda> although imsrg++'s argument order
/// lists (N, L) first. Host-only; kernels read `MoshinskyTable` (tables.hpp).

namespace imsrg::angmom {

/// HO quantum numbers (n, l) of one particle or of a relative/center-of-mass motion.
/// The HO energy quantum is e = 2n + l.
struct HoNL {
  int n = 0;  ///< radial quantum number (number of nodes), >= 0
  int l = 0;  ///< orbital angular momentum, >= 0
};

/// Largest HO energy E = 2n1 + l1 + 2n2 + l2 for which `moshinsky` is implemented (size of its
/// double-double factorial tables; emax 20). Beyond it `moshinsky` returns NaN.
inline constexpr int moshinsky_max_energy = 40;

/// Moshinsky bracket <cm, rel; lambda | a, b; lambda> = <N L, n l; lambda | n1 l1, n2 l2; lambda>:
/// overlap of the state with center-of-mass motion `cm` = (N, L) and relative motion `rel` =
/// (n, l) coupled to total orbital angular momentum `lambda` with the two-particle state
/// (n1 l1, n2 l2) coupled to `lambda`. Same argument order and value as imsrg++
/// `AngMom::Moshinsky(N, L, n, l, n1, l1, n2, l2, lambda)`. Zero unless energy is conserved
/// (2N + L + 2n + l = 2n1 + l1 + 2n2 + l2), parities agree and both pairs couple to lambda.
/// Evaluated in double-double arithmetic and rounded once, so the result is the double nearest
/// to the exact bracket in practice (validated for all 405,537 brackets with E <= 16); the plain
/// double algorithm of imsrg++ loses up to ~1e-10 relative on some high-l brackets. Returns NaN
/// for E > `moshinsky_max_energy`. Cost grows like O(E^5); use `MoshinskyTable` in loops.
double moshinsky(HoNL cm, HoNL rel, HoNL a, HoNL b, int lambda) noexcept;

/// Talmi coefficient B(n_a l_a, n_b l_b; p) of Brody and Moshinsky ("Tables of Transformation
/// Brackets", 1967), with which <a| V(r) |b> = sum_p B(a, b; p) I_p for a central potential;
/// identical to imsrg++ `AngMom::TalmiB` (`AngMom.cc:246-261`). Zero unless l_a + l_b is even
/// and (l_a + l_b)/2 <= p <= (l_a + l_b)/2 + n_a + n_b.
double talmi_b(HoNL a, HoNL b, int p) noexcept;

/// Talmi integral of the power potential r^k in oscillator units (b = 1):
/// I_p = (2 / Gamma(p + 3/2)) * integral dr r^(2p+2) r^k exp(-r^2) = Gamma(p + 3/2 + k/2) /
/// Gamma(p + 3/2), evaluated as exp(lgamma - lgamma) like imsrg++ `imsrg_util::TalmiI`
/// (`imsrg_util.cc:2340-2346`). Precondition: 2p + 3 + k > 0. For another oscillator length b,
/// multiply by b^k.
double talmi_integral(int p, double k) noexcept;

}  // namespace imsrg::angmom
