#pragma once
/// @file
/// IMSRG generators eta(H) (design/13 section 7, design/reference/03 section 4, imsrg++
/// `Generator.cc`). A generator is an enum `GeneratorKind` plus options, evaluated by a free
/// function (design/10 section 3.4); there is no generator class hierarchy.
///
/// Every generator builds eta element by element from the "off-diagonal" part of H and an
/// energy denominator Delta:
///   eta_od = f(H_od, Delta),  f = white: H/Delta, atan: atan(2 H/Delta)/2,
///   imaginary-time: H sign(Delta), wegner: H Delta, qtransfer-atan_n:
///   (|Delta| M_N / (hbar c)^2)^(n/2) atan(2 H/Delta)/2,
/// and eta is anti-Hermitian. Blocks (core c, valence v, q-space q; partition of H's model space):
/// * single reference (white, atan, imaginary-time, wegner, qtransfer-atan): one-body
///   eta(i,a) = f(H(i,a), Delta(i,a)), eta(a,i) = f(H(a,i), -Delta(i,a)) for a in core, i in
///   valence u q; two-body eta(bra,ket) for ket in cc and bra in qq u vv u qv, and
///   eta(ket,bra) = -eta(bra,ket) (Generator.cc:277-340);
/// * shell model (shell-model, -atan, -wegner, -imaginary-time): one-body for a in core u valence,
///   i in valence u q, i != a; two-body for (bra in vv u qv u qq, ket in cc u vc) and for
///   (bra in qv u qq, ket in vv) (Generator.cc:432-519);
/// * hartree-fock: one-body only, eta(i,j) = H(i,j)/Delta(i,j) for all j < i (white form for any
///   options), eta(j,i) = -eta(i,j); eta has particle_rank 1 (Generator.cc:662-680).
/// All other elements of eta are zero.
///
/// Denominators (Generator.cc:122-230), f = the one-body part of H, n = reference occupations,
/// V-bar_abab = the J-averaged unnormalized monopole
///   V-bar_abab = sum_J (2J+1) V^J_abab (unnormalized) / ((2 j_a + 1)(2 j_b + 1)):
/// * one body: Delta(i,j) = f_ii - f_jj (+ (n_i - n_j) V-bar_ijij for Epstein-Nesbet;
///   MP-isospin: (Delta + f_i'i' - f_j'j') / 2 with the isospin partners i', j');
/// * two body, bra |ij>, ket |kl>: Delta = f_ii + f_jj - f_kk - f_ll (MP-isospin averaged as
///   above), plus for Epstein-Nesbet
///   (1-n_i-n_j) V-bar_ijij - (1-n_k-n_l) V-bar_klkl + (n_i-n_k) V-bar_ikik + (n_i-n_l) V-bar_ilil
///   + (n_j-n_k) V-bar_jkjk + (n_j-n_l) V-bar_jljl;
/// * `denominator_delta` is added to the one-body denominator if it targets all orbits or i or j,
///   to the two-body denominator only if it targets all orbits;
/// * floor: |Delta| < denominator_cutoff -> Delta = sign(Delta) * cutoff (sign of 0 is +).
///   **Deviation (design/00 D21, design/30 section 6):** imsrg++ sets Delta = +cutoff, losing the
///   sign of small negative denominators.
#include <optional>

#include "imsrg/base/types.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::flow {

/// Generator types (imsrg++ `generator_type` strings in the comments; the driver maps its
/// `driver::Generator` enum 1:1 onto this one).
enum class GeneratorKind : std::uint8_t {
  White,                    ///< "white"
  Atan,                     ///< "atan" (imsrg++ default core generator)
  ImaginaryTime,            ///< "imaginary-time"
  Wegner,                   ///< "wegner" (imsrg++: "never tested")
  QtransferAtan,            ///< "qtransfer-atan" / "qtransfer-atan_<n>"
  ShellModel,               ///< "shell-model" (white function, shell-model blocks)
  ShellModelAtan,           ///< "shell-model-atan" (imsrg++ default valence generator)
  ShellModelWegner,         ///< "shell-model-wegner"
  ShellModelImaginaryTime,  ///< "shell-model-imaginary-time"
  ShellModelAtanNpnh,       ///< "shell-model-atan-npnh" (planned: NotImplementedError)
  HartreeFock,              ///< "hartree-fock"
  OnePa                     ///< "1PA" (planned: NotImplementedError)
};

/// Energy-denominator partitioning (imsrg++ `denominator_partitioning`).
enum class Partitioning : std::uint8_t {
  EpsteinNesbet,  ///< "Epstein_Nesbet" (default): SPE differences plus monopole terms.
  MollerPlesset,  ///< "Moller_Plesset": SPE differences only.
  MpIsospin       ///< "MP_isospin": SPE differences averaged over isospin partners.
};

/// Options of `compute_generator`. Defaults are the imsrg++ defaults.
struct GeneratorOptions {
  GeneratorKind kind = GeneratorKind::Atan;
  Partitioning partitioning = Partitioning::EpsteinNesbet;
  /// Floor of |Delta| (imsrg++ `denominator_cutoff`, default 1e-6).
  double denominator_cutoff = 1e-6;
  /// Constant added to denominators (imsrg++ `denominator_delta`); active only if
  /// `denominator_delta_all` or `denominator_delta_orbit` is set (imsrg++ quirk P5).
  double denominator_delta = 0.0;
  /// The delta applies to all one- and two-body denominators (imsrg++ index -12345, "all").
  bool denominator_delta_all = false;
  /// The delta applies to one-body denominators involving this orbit.
  std::optional<OrbitIndex> denominator_delta_orbit;
  /// Exponent index n of qtransfer-atan (weight (|Delta| M_N/(hbar c)^2)^(n/2)); imsrg++
  /// "qtransfer-atan" is n = 1.
  int qtransfer_n = 1;
  /// Only the one-body part of eta (imsrg++ `only_1b_eta`).
  bool only_1b_eta = false;
};

/// The generator function f(H_od, Delta) of `kind` (see the file comment). For the shell-model
/// kinds the function of the corresponding single-reference kind (shell-model = white).
/// @throws NotImplementedError for HartreeFock (it has no function argument), ShellModelAtanNpnh
///         and OnePa.
double generator_function(GeneratorKind kind, double h_od, double denominator, int qtransfer_n = 1);

/// The denominator floor: returns `delta` if |delta| >= cutoff, else sign(delta) * cutoff with
/// sign(0) = +1 (D21; imsrg++ returns +cutoff).
double floor_denominator(double delta, double cutoff) noexcept;

/// The (floored) energy denominators of a Hamiltonian, precomputed once per generator update:
/// the diagonal of the one-body part and the monopole table V-bar_abab for all orbit pairs.
/// Read-only after construction (design/00 D10).
class Denominators {
 public:
  /// Precomputes the tables from H (scalar, particle_rank >= 2) and the occupations of H's
  /// model space. Host computation, O(norb^2 * J range). Reads H's device data (const-correct
  /// copies), so it is valid right after library calls.
  Denominators(const op::Operator& H, const GeneratorOptions& opts);
  /// One-body denominator Delta(i, j) (floored).
  double one_body(OrbitIndex i, OrbitIndex j) const;
  /// Two-body denominator of bra = local ket `ibra`, ket = local ket `iket` of channel `ch`
  /// (floored).
  double two_body(ChannelIndex ch, LocalKetIndex ibra, LocalKetIndex iket) const;
  /// The unnormalized J-averaged monopole V-bar_abab (imsrg++ `GetTBMEmonopole(a,b,a,b)`).
  double monopole(OrbitIndex a, OrbitIndex b) const;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  GeneratorOptions opts_;
  std::int32_t norb_ = 0;
  std::vector<double> f_diag_;    // f_aa
  std::vector<double> monopole_;  // V-bar_abab at a + b * norb
  std::vector<OrbitIndex> isospin_partner_;
};

/// The spec of eta(H): scalar, anti-Hermitian, particle_rank of H (1 for HartreeFock).
op::OperatorSpec generator_spec(const op::OperatorSpec& h, GeneratorKind kind);

/// Overwrites `eta` with eta(H) (imsrg++ `Generator::Update`: erase, then fill the blocks of the
/// file comment). If `eta` is empty or its spec or model space differs from
/// (generator_spec(H.spec(), kind), H's model space) it is replaced by a new operator; otherwise
/// it is reused without allocation (the flow-loop case). Host computation on copies of H's
/// device data; eta's device data are current on return.
/// Preconditions: H scalar, number conserving, particle_rank 2 (InvariantError).
/// @throws NotImplementedError for ShellModelAtanNpnh and OnePa.
/// Complexity: O(norb^2 J) for the tables plus O(#decoupled elements).
void compute_generator(const op::Operator& H, op::Operator& eta, const GeneratorOptions& opts);

/// Allocating form of `compute_generator`.
op::Operator compute_generator(const op::Operator& H, const GeneratorOptions& opts);

}  // namespace imsrg::flow
