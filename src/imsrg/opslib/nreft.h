// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPSLIB_NREFT_H_
#define IMSRG_OPSLIB_NREFT_H_
/// @file
/// Dark-matter NREFT one-body response operators (Anand, Fitzpatrick, Haxton
/// conventions) in the harmonic-oscillator basis: a port of the `DM_NREFT`
/// namespace of the owner's imsrg++ fork (DarkMatterNREFT.cc at 79e9330,
/// itself a port of D. Gazda's Fortran) and of its operator-name shorthands
/// `F<family><L><p|n>_<q>` (imsrg_util.cc:199-453). Design:
/// design/50-al27-responses.md C2; reference/06-al27-fork-pipeline.md
/// section 2; provenance README section 5.
///
/// Conventions (identical to the fork, so tables compare element by element):
/// * q is the momentum transfer in MeV; y = q^2 b^2 / 4 with
///   b^2 = 1 / (hw m_N) in MeV^-2 (= (hbar c)^2 / (m_N hw) in fm^2),
///   m_N = (m_p + m_n) / 2 (`constants::kMNucleon`).
/// * One-body elements are reduced matrix elements <a||O_J||b> in the
///   Wigner-Eckart convention <j m|O_JM|j' m'> = (-1)^(j-m)
///   (j J j'; -m M m') <j||O_J||j'>, EXCEPT that a building block with J = 0
///   and even parity is stored as a plain (non-reduced) matrix element,
///   i.e. divided by sqrt(2 j_a + 1) (imsrg++ scalar storage).
/// * Isospin selection multiplies the element by a factor of the bra
///   orbit's species: isoscalar {p 1, n 1}, isovector {1, -1}, proton
///   {1, 0}, neutron {0, 1}. All operators conserve Tz.
/// * Elements are only formed inside the fork's one-body channels (same tz,
///   (l_a + l_b + parity) even, |l_a - l_b| <= J + 1, |j_a - j_b| <= J);
///   all others are zero.
/// Host-only construction; complexity O(num_orbits^2 n_max^2) 1F1
/// evaluations per table.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/operator/operator.h"

namespace imsrg::opslib::nreft {

/// Isospin selection of a one-body NREFT operator (fork "+", "-", "p", "n").
enum class Isospin : std::int8_t { kIsoscalar, kIsovector, kProton, kNeutron };

/// The NREFT operator families of the fork (`DM_NREFT::<name>`), rank J.
/// Parity: (-1)^J for M, Sigma, Phip, Phipp, Phitp and Deltap (normal; Deltap
/// is built from Mg(J, J -+ 1)); (-1)^(J+1) for Sigmap, Sigmapp, Delta
/// (abnormal).
enum class Family : std::int8_t {
  kM,        ///< M_J (charge), Hermitian.
  kSigma,    ///< Sigma_J = Ms(J, J), anti-Hermitian.
  kSigmap,   ///< Sigma'_J (transverse spin), Hermitian.
  kSigmapp,  ///< Sigma''_J (longitudinal spin), Hermitian.
  kDelta,    ///< Delta_J = Mg(J, J), Hermitian (master's flag is a bug).
  kDeltap,   ///< Delta'_J, anti-Hermitian.
  kPhip,     ///< Phi'_J, neither Hermitian nor anti-Hermitian.
  kPhipp,    ///< Phi''_J, Hermitian.
  kPhitp,    ///< tilde-Phi'_J = Phi'_J + Sigma_J / 2, Hermitian.
};

/// Parity of a family's rank-J operator.
Parity FamilyParity(Family family, std::int32_t J);

/// The (numerically verified, reference/06 section 3) Hermiticity of a
/// family in the tensor sense <b||O||a> = (-1)^(j_a - j_b) <a||O||b>.
Hermiticity FamilyHermiticity(Family family);

/// y = q^2 / (4 hw m_N), computed exactly as the fork does
/// (`q*q*b2/4.0` with `b2 = 1.0/(hw*M_NUCLEON)`).
double OscillatorY(double q_MeV, double hw_MeV);

/// <a||Ms_JL||b> = <a|| [Y_L(r) j_L(qr) x sigma]_J ||b> (fork `Ms`, parity
/// (-1)^L), isospin factor 1. Precondition: a and b in one channel of a
/// rank-J, parity-(-1)^L operator (else the result is meaningless).
double MsElement(std::int32_t J, std::int32_t L, const basis::Orbit& a,
                 const basis::Orbit& b, double y);

/// <a||Mg_JL||b> = <a|| M_JL . (1/q) grad ||b> (fork `Mg`, parity
/// (-1)^(L+1)), isospin factor 1; zero if l_a = 0 or l_b = 0. Precondition
/// as `MsElement` for parity (-1)^(L+1).
double MgElement(std::int32_t J, std::int32_t L, const basis::Orbit& a,
                 const basis::Orbit& b, double y);

/// <a||O_J||b> of a family with isospin factor 1 (J = 0 even-parity blocks
/// non-reduced, see the file comment). Linear combinations follow the fork's
/// order of operations.
double FamilyElement(Family family, std::int32_t J, const basis::Orbit& a,
                     const basis::Orbit& b, double y);

/// True if the fork's rank-J, parity-`parity` operator stores an element
/// between a and b (its `OneBodyChannels`, see the file comment).
bool InOneBodyChannel(std::int32_t J, Parity parity, const basis::Orbit& a,
                      const basis::Orbit& b);

/// Dense one-body table of a (generally tensor) operator: reduced matrix
/// elements, row-major, value(a, b) = <a||O_J||b>.
struct OneBodyTable {
  std::int32_t num_orbits = 0;
  std::int32_t rank_J = 0;
  Parity parity = Parity::kEven;
  Hermiticity hermiticity = Hermiticity::kHermitian;
  std::vector<double> values;  ///< num_orbits^2 entries, row-major.

  /// <a||O||b>. Precondition: valid indices.
  double value(OrbitIndex a, OrbitIndex b) const {
    return values[static_cast<std::size_t>(a.value()) * num_orbits +
                  static_cast<std::size_t>(b.value())];
  }
};

/// The table of `family` with rank J and isospin selection `isospin` at
/// momentum transfer q (MeV) in `basis` (hw from `basis.hbar_omega()`), times
/// `prefactor`, flagged with `FamilyHermiticity(family)`.
OneBodyTable BuildTable(const basis::SingleParticleBasis& basis, Family family,
                        std::int32_t J, Isospin isospin, double q_MeV,
                        double prefactor = 1.0);

/// sqrt(4 pi), the prefactor of the charge-like shorthands FM and FPhipp.
double SqrtFourPi();

/// Prefactor of the `FPhitp<L>` shorthands: sqrt(4 pi), like FPhipp (the
/// normal-parity, even-J family; reference/06 section 2.4).
/// PENDING the owner's confirmation (interview 2026-11-26, 20:00 EST).
double PhitpShorthandPrefactor();

/// A parsed operator-name shorthand `F<family><L><p|n>_<q>`.
struct Shorthand {
  Family family = Family::kM;
  std::int32_t L = 0;
  Isospin isospin = Isospin::kProton;
  double q_MeV = 0.0;
  double prefactor = 1.0;  ///< sqrt(4 pi) for FM, FPhipp, FPhitp; else 1.
};

/// Parses `FM<L>`, `FPhipp<L>` (L = 0, 2, 4), `FSigmap<L>`, `FSigmapp<L>`,
/// `FDelta<L>` (L = 1, 3, 5) and `FPhitp<L>` (L = 2, 4), each followed by
/// `p` or `n`, `_`, and q in MeV (a decimal number, as the fork's
/// `istringstream >> double`). Returns nullopt for any other name (the
/// fork's L values only: these are the names it defines, plus FPhitp).
std::optional<Shorthand> ParseShorthand(std::string_view name);

/// `BuildTable` for a shorthand (Hermitian for all shorthands).
OneBodyTable BuildShorthandTable(const basis::SingleParticleBasis& basis,
                                 const Shorthand& s);

/// The scalar (L = 0: FM0, FPhipp0) shorthand as a Hermitian particle-rank-2
/// operator with zero two-body part; elements are plain matrix elements.
/// @throws InputError if `s.L != 0` (rank L > 0 needs tensor operators,
///         milestone M15).
op::Operator BuildScalarShorthand(
    const std::shared_ptr<const basis::ModelSpace>& ms, const Shorthand& s);

}  // namespace imsrg::opslib::nreft

#endif  // IMSRG_OPSLIB_NREFT_H_
