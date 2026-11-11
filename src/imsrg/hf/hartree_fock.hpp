#pragma once
/// @file
/// Spherical Hartree-Fock in the harmonic-oscillator basis (imsrg++ `HartreeFock`). Given a
/// normal-ordered-free Hamiltonian H (one-body kinetic part, two-body interaction in the
/// operator, optional NO2B three-body interaction) it finds the self-consistent orbitals
/// |alpha> = sum_i C(i, alpha) |i> that diagonalize the Fock operator channel by channel
/// ((l, 2j, tz) is conserved), and the HF energy of the reference state.
///
/// Conventions: matrices are column-major `linalg::Matrix<double>` of size num_orbits x
/// num_orbits indexed by imsrg++ orbit index; energies in MeV. HF orbit alpha lives in the channel
/// of oscillator orbits it is built from and is numbered by the position of its eigenvalue in
/// that channel (ascending), mapped to the orbit index of the same rank, as in imsrg++.
/// Host only.

#include <cstdint>
#include <vector>

#include "imsrg/linalg/matrix_view.hpp"
#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::hf {

/// How successive densities are combined.
enum class Mixing : std::uint8_t {
  Linear,  ///< imsrg++ scheme: rho <- (1 - w) rho_new + w rho_old with w = `density_mixing`.
  Diis     ///< Pulay DIIS on the Fock matrix (no density mixing).
};

/// Options of `solve`.
struct HFOptions {
  Mixing mixing = Mixing::Linear;
  /// Convergence threshold on ||eps - eps_prev||_F / num_orbits (eps: single-particle energies).
  double tolerance = 1e-8;
  /// Maximum number of iterations.
  std::int32_t max_iterations = 1000;
  /// Linear mixing: weight of the OLD density (imsrg++ 0.2 for the density mixing).
  double density_mixing = 0.2;
  /// Keep the occupations of the reference fixed (no `FillLowestOrbits`). With false the lowest
  /// orbits are re-filled periodically from iteration 11 on (imsrg++ `freeze_occupations`;
  /// the constructor default there is true, the golden data were produced with freezing on).
  bool freeze_occupations = true;
  /// DIIS: first iteration whose Fock matrix enters the subspace.
  std::int32_t diis_start = 2;
  /// DIIS: number of (Fock, error) pairs kept.
  std::int32_t diis_subspace = 8;
  /// Three-body cut: only three-body monopole terms with e_a + e_b + e_c <= e3max are used.
  std::int32_t e3max = 12;
};

/// Result of `solve`.
struct HFResult {
  double e_hf = 0.0;  ///< E_HF = e1 + e2 + e3 (MeV).
  double e1 = 0.0;    ///< One-body (kinetic) contribution.
  double e2 = 0.0;    ///< Two-body contribution.
  double e3 = 0.0;    ///< Three-body (NO2B) contribution (0 without v3).
  /// Loop index at which convergence was detected (imsrg++ `iterations`); equals max_iterations
  /// if not converged.
  std::int32_t iterations = 0;
  bool converged = false;
  /// HF single-particle energies, indexed by HF orbit alpha (MeV).
  std::vector<double> spe;
  /// Coefficients C(ho orbit, hf orbit), imsrg++ sign convention (diagonal entries >= 0).
  linalg::Matrix<double> C;
  /// Density matrix rho(i,j) = sum_alpha n_alpha C(i,alpha) C(j,alpha) in the oscillator basis.
  linalg::Matrix<double> rho;
  /// Fock matrix F[rho] (oscillator basis) consistent with the final `rho`.
  linalg::Matrix<double> F;
  /// Occupations n_alpha of the HF orbits.
  std::vector<double> occupations;
  /// ||eps - eps_prev||_F / num_orbits of every iteration.
  std::vector<double> ediff_history;
};

/// Runs spherical Hartree-Fock for the Hamiltonian `H`.
///
/// The reference (occupations, Z, N) is the one of H's model space. `H` is deep-copied and the
/// copy is synced to the host (`Operator::sync_to_host` is non-const and `H` is device
/// authoritative); `H` itself is not modified. Only the one-body part and the two-body part of
/// `H` are used (no zero-body term). `v3` may be null (2N only); otherwise its monopole is used
/// with the `options.e3max` cut. The Fock operator is F = T + V2[rho] + 1/2 V3[rho].
///
/// Complexity: dominated by the monopole contractions, O(N^2 c^2 N) per iteration for the 2N
/// part and O(#keys) for the 3N part, plus a one-off construction of the monopoles.
/// @throws InputError / InvariantError if H has no two-body part or `v3` does not match its
///         model space.
HFResult solve(const op::Operator& H, const op::No2bThreeBody* v3, const HFOptions& options = {});

}  // namespace imsrg::hf
