// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_GENERATOR_H_
#define IMSRG_FLOW_GENERATOR_H_
/// @file
/// IMSRG generators eta(H) (design/13 section 7, design/reference/03 section 4,
/// imsrg++ `Generator.cc`). A generator is an enum `GeneratorKind` plus
/// options, evaluated by a free function (design/10 section 3.4); there is no
/// generator class hierarchy.
///
/// Every generator builds eta element by element from the "off-diagonal" part
/// of H and an energy denominator Delta:
///   eta_od = f(H_od, Delta),  f = white: H/Delta, atan: atan(2 H/Delta)/2,
///   imaginary-time: H sign(Delta), wegner: H Delta, qtransfer-atan_n:
///   (|Delta| M_N / (hbar c)^2)^(n/2) atan(2 H/Delta)/2,
/// and eta is anti-Hermitian. Blocks (core c, valence v, q-space q; partition
/// of H's model space):
/// * single reference (white, atan, imaginary-time, wegner, qtransfer-atan):
///   one-body eta(i,a) = f(H(i,a), Delta(i,a)), eta(a,i) = f(H(a,i),
///   -Delta(i,a)) for a in core, i in valence u q; two-body eta(bra,ket) for
///   ket in cc and bra in qq u vv u qv, and eta(ket,bra) = -eta(bra,ket)
///   (Generator.cc:277-340);
/// * shell model (shell-model, -atan, -wegner, -imaginary-time): one-body for a
///   in core u valence, i in valence u q, i != a; two-body for (bra in vv u qv
///   u qq, ket in cc u vc) and for (bra in qv u qq, ket in vv)
///   (Generator.cc:432-519);
/// * hartree-fock: one-body only, eta(i,j) = H(i,j)/Delta(i,j) for all j < i
///   (white form for any options), eta(j,i) = -eta(i,j); eta has particle_rank
///   1 (Generator.cc:662-680).
/// All other elements of eta are zero.
///
/// Three-body part (H and eta of particle_rank 3, single-reference kinds only;
/// imsrg++ `ConstructGenerator_SingleRef_3body`, Generator.cc:328-427): for
/// every channel, bra row (a b c; Jab) whose three orbits are all non-core
/// (valence or q-space) and ket row (i j k; Jij) whose three orbits are all
/// core: eta(bra, ket) = f(H(bra, ket), Delta3) and eta(ket, bra) =
/// -eta(bra, ket), with Delta3 = f_aa + f_bb + f_cc - f_ii - f_jj - f_kk from
/// the one-body diagonal of H alone (`Get3bDenominator`: no Epstein-Nesbet
/// terms, and, as in imsrg++, NO denominator floor, no `denominator_delta`: a
/// vanishing Delta3 gives inf/NaN for white). Kets dropped by the three-body
/// cuts do not exist.
///
/// Denominators (Generator.cc:122-230), f = the one-body part of H, n =
/// reference occupations, V-bar_abab = the J-averaged unnormalized monopole
///   V-bar_abab = sum_J (2J+1) V^J_abab (unnormalized)
///                / ((2 j_a + 1)(2 j_b + 1)):
/// * one body: Delta(i,j) = f_ii - f_jj (+ (n_i - n_j) V-bar_ijij for
///   Epstein-Nesbet; MP-isospin: (Delta + f_i'i' - f_j'j') / 2 with the isospin
///   partners i', j');
/// * two body, bra |ij>, ket |kl>: Delta = f_ii + f_jj - f_kk - f_ll
///   (MP-isospin averaged as above), plus for Epstein-Nesbet (1-n_i-n_j)
///   V-bar_ijij - (1-n_k-n_l) V-bar_klkl + (n_i-n_k) V-bar_ikik + (n_i-n_l)
///   V-bar_ilil + (n_j-n_k) V-bar_jkjk + (n_j-n_l) V-bar_jljl;
/// * `denominator_delta` is added to the one-body denominator if it targets all
///   orbits or i or j, to the two-body denominator only if it targets all
///   orbits;
/// * floor: |Delta| < denominator_cutoff -> Delta = sign(Delta) * cutoff (sign
///   of 0 is +). **Deviation (design/00 D21, design/30 section 6):** imsrg++
///   sets Delta = +cutoff, losing the sign of small negative denominators.
#include <memory>
#include <optional>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::flow {

/// Generator types (imsrg++ `generator_type` strings in the comments; the
/// driver maps its `driver::Generator` enum 1:1 onto this one).
enum class GeneratorKind : std::uint8_t {
  kWhite,           ///< "white"
  kAtan,            ///< "atan" (imsrg++ default core generator)
  kImaginaryTime,   ///< "imaginary-time"
  kWegner,          ///< "wegner" (imsrg++: "never tested")
  kQtransferAtan,   ///< "qtransfer-atan" / "qtransfer-atan_<n>"
  kShellModel,      ///< "shell-model" (white function, shell-model blocks)
  kShellModelAtan,  ///< "shell-model-atan" (imsrg++ default valence generator)
  kShellModelWegner,         ///< "shell-model-wegner"
  kShellModelImaginaryTime,  ///< "shell-model-imaginary-time"
  kShellModelAtanNpnh,       ///< "shell-model-atan-npnh" (planned:
                             ///< NotImplementedError)
  kHartreeFock,              ///< "hartree-fock"
  kOnePa                     ///< "1PA" (planned: NotImplementedError)
};

/// Energy-denominator partitioning (imsrg++ `denominator_partitioning`).
enum class Partitioning : std::uint8_t {
  kEpsteinNesbet,  ///< "Epstein_Nesbet" (default): SPE differences plus
                   ///< monopole terms.
  kMollerPlesset,  ///< "Moller_Plesset": SPE differences only.
  kMpIsospin  ///< "MP_isospin": SPE differences averaged over isospin partners.
};

/// Options of `ComputeGenerator`. Defaults are the imsrg++ defaults.
struct GeneratorOptions {
  GeneratorKind kind = GeneratorKind::kAtan;
  Partitioning partitioning = Partitioning::kEpsteinNesbet;
  /// Floor of |Delta| (imsrg++ `denominator_cutoff`, default 1e-6); must be
  /// positive and finite (`ComputeGenerator` throws InputError otherwise).
  double denominator_cutoff = 1.0e-6;
  /// Constant added to denominators (imsrg++ `denominator_delta`); active only
  /// if `denominator_delta_all` or `denominator_delta_orbit` is set (imsrg++
  /// quirk P5).
  double denominator_delta = 0.0;
  /// The delta applies to all one- and two-body denominators (imsrg++ index
  /// -12345, "all").
  bool denominator_delta_all = false;
  /// The delta applies to one-body denominators involving this orbit.
  std::optional<OrbitIndex> denominator_delta_orbit;
  /// Exponent index n of qtransfer-atan (weight (|Delta| M_N/(hbar
  /// c)^2)^(n/2)); imsrg++ "qtransfer-atan" is n = 1.
  int qtransfer_n = 1;
  /// Only the one-body part of eta (imsrg++ `only_1b_eta`).
  bool only_1b_eta = false;
  /// Keep eta to its one- and two-body parts even for a rank-3 H (imsrg++
  /// `only_2b_eta`).
  bool only_2b_eta = false;
};

/// The generator function f(H_od, Delta) of `kind` (see the file comment). For
/// the shell-model kinds the function of the corresponding single-reference
/// kind (shell-model = white).
/// @throws NotImplementedError for HartreeFock (it has no function argument),
///         ShellModelAtanNpnh and OnePa.
double GeneratorFunction(GeneratorKind kind, double h_od, double denominator,
                         int qtransfer_n = 1);

/// The denominator floor: returns `delta` if |delta| >= cutoff, else
/// sign(delta) * cutoff with sign(0) = +1 (D21; imsrg++ returns +cutoff).
double FloorDenominator(double delta, double cutoff) noexcept;

/// The (floored) energy denominators of a Hamiltonian, precomputed once per
/// generator update: the diagonal of the one-body part and the monopole table
/// V-bar_abab for all orbit pairs. Read-only after construction (design/00
/// D10).
class Denominators {
 public:
  /// Precomputes the tables from H (scalar, particle_rank >= 2) and the
  /// occupations of H's model space. Host computation, O(norb^2 * J range).
  /// Syncs H's host mirror (no copy of H), so it is valid right after library
  /// calls.
  Denominators(const op::Operator& H, const GeneratorOptions& opts);

  Denominators(const Denominators&) = default;
  Denominators& operator=(const Denominators&) = default;
  Denominators(Denominators&&) = default;
  Denominators& operator=(Denominators&&) = default;
  /// One-body denominator Delta(i, j) (floored).
  double one_body(OrbitIndex i, OrbitIndex j) const;
  /// Two-body denominator of bra = local ket `ibra`, ket = local ket `iket` of
  /// channel `ch` (floored).
  double two_body(ChannelIndex ch, LocalKetIndex ibra,
                  LocalKetIndex iket) const;
  /// Three-body denominator f_aa + f_bb + f_cc - f_ii - f_jj - f_kk of bra
  /// orbits (a, b, c) and ket orbits (i, j, k): one-body diagonal only, NOT
  /// floored (imsrg++ `Get3bDenominator`).
  double three_body(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex i,
                    OrbitIndex j, OrbitIndex k) const;
  /// The unnormalized J-averaged monopole V-bar_abab (imsrg++
  /// `GetTBMEmonopole(a,b,a,b)`).
  double monopole(OrbitIndex a, OrbitIndex b) const;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  GeneratorOptions opts_;
  std::int32_t norb_ = 0;
  std::vector<double> f_diag_;    // f_aa
  std::vector<double> monopole_;  // V-bar_abab at a + b * norb
  std::vector<OrbitIndex> isospin_partner_;
};

/// The spec of eta(H): scalar, anti-Hermitian, particle_rank of H (1 for
/// HartreeFock).
op::OperatorSpec GeneratorSpec(const op::OperatorSpec& h, GeneratorKind kind);

/// Overwrites `eta` with eta(H) (imsrg++ `Generator::Update`: erase, then fill
/// the blocks of the file comment). If `eta` is empty or its spec or model
/// space differs from (GeneratorSpec(H.spec(), kind), H's model space) it is
/// replaced by a new operator; otherwise it is reused without allocation (the
/// flow-loop case). Host computation on copies of H's device data; eta's device
/// data are current on return. Preconditions: H scalar, number conserving,
/// particle_rank 2 or 3 (InvariantError; rank 3 needs a three-body part).
/// @throws NotImplementedError for ShellModelAtanNpnh and OnePa, and for the
///         shell-model kinds and hartree-fock with a rank-3 H.
/// Complexity: O(norb^2 J) for the tables plus O(#decoupled elements).
void ComputeGenerator(const op::Operator& H, op::Operator& eta,
                      const GeneratorOptions& opts);

/// Allocating form of `ComputeGenerator`.
op::Operator ComputeGenerator(const op::Operator& H,
                              const GeneratorOptions& opts);

}  // namespace imsrg::flow

#endif  // IMSRG_FLOW_GENERATOR_H_
