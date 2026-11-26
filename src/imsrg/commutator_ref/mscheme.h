// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_REF_MSCHEME_H_
#define IMSRG_COMMUTATOR_REF_MSCHEME_H_

/// @file
/// Brute-force m-scheme commutator reference: the validation oracle of the
/// IMSRG(3) terms (design/14 section 3, decision D34). Test-only and host-only
/// (design/10 section 2).
///
/// An operator of particle rank n is written as
///
///   A = 1/(n!)^2 sum A_{p1..pn; q1..qn} a+_p1 .. a+_pn a_qn .. a_q1,
///
/// normal ordered with respect to the reference, with A antisymmetric in the
/// p's and in the q's (n = 2: 1/4 sum A_{pq;rs} a+p a+q a_s a_r; n = 3: 1/36).
/// The commutator parts [A^(nA), B^(nB)]^(nC) are evaluated by explicit sums
/// over single-particle m-states with a generic Wick engine (all cross
/// contractions, with the reference densities n for a creator-annihilator and
/// 1 - n for an annihilator-creator contraction); no term-specific formula is
/// used, so the result is an independent check of the J-coupled code. The
/// J-scheme operators of the library are converted with `Expand` (J -> m) and
/// `Contract` (m -> J).
///
/// Conventions:
/// * m-states are ordered by orbit index, then 2m ascending; an m-state index
///   is a plain `int` (this module is a test oracle, not a kernel interface);
/// * a three-body m-state triple is KEPT iff its orbit triple has a kept row in
///   the three-body space of the model space (any Jab). Commutator results are
///   computed for kept output triples only, matching the J-scheme semantics
///   (the cut defines the space; the inputs are zero outside it);
/// * the values of `Expand` are the unnormalized antisymmetrized elements of
///   imsrg++ (`op::TwoBody::get_unnormalized`, `op::ThreeBody::get`) expanded
///   with Clebsch-Gordan coefficients.
///
/// Cost: a rank-3 operator on N m-states stores (N choose 3)^2 doubles (560^2
/// at emax 1, N = 16); the [3,3]->3 part is O(10^3) times slower than the
/// others and is meant for emax 1 only.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator_ref::mscheme {

/// One single-particle m-state |orbit, m>.
struct MState {
  std::int32_t orbit = 0;    ///< Orbit index (`OrbitIndex::value()`).
  std::int32_t two_m = 0;    ///< 2m, in -2j..2j step 2.
  std::int32_t l = 0;        ///< Orbital angular momentum of the orbit.
  std::int32_t two_j = 1;    ///< 2j of the orbit.
  std::int32_t two_tz = -1;  ///< 2tz: -1 proton, +1 neutron.
  double n = 0.0;            ///< Reference occupation of the orbit.
  double nbar = 1.0;         ///< 1 - n.
};

/// The m-scheme single-particle basis of a model space, with the projection of
/// the three-body space (see the file comment). Value type.
class MBasis {
 public:
  /// Builds the basis: for every orbit of `ms`, the 2j + 1 m-states in
  /// ascending m; the occupations are the reference occupations of `ms`. If
  /// `ms` has a three-body space, the kept orbit triples are recorded. O(N^3)
  /// for N m-states.
  explicit MBasis(const basis::ModelSpace& ms);

  MBasis(const MBasis&) = default;
  MBasis& operator=(const MBasis&) = default;
  MBasis(MBasis&&) = default;
  MBasis& operator=(MBasis&&) = default;

  /// Number of m-states N.
  int size() const noexcept { return static_cast<int>(states_.size()); }
  /// m-state `i`. Precondition: 0 <= i < size().
  const MState& StateAt(int i) const { return states_[static_cast<size_t>(i)]; }
  /// Index of the m-state (`orbit`, `two_m`). Precondition: it exists.
  int Index(int orbit, int two_m) const;
  /// First m-state of orbit `orbit` (its 2j + 1 states are contiguous).
  int FirstOfOrbit(int orbit) const {
    return first_of_orbit_[static_cast<size_t>(orbit)];
  }
  /// True iff the (sorted or unsorted) m-states `a`, `b`, `c` form a kept
  /// three-body triple: their orbit triple, sorted, has a kept row. False for
  /// repeated m-states and if the model space has no three-body space. O(1).
  bool IsKeptTriple(int a, int b, int c) const;
  /// True if the model space has a three-body space.
  bool HasThreeBody() const noexcept { return !kept_orbit_triple_.empty(); }
  /// Number of orbits.
  int num_orbits() const noexcept {
    return static_cast<int>(first_of_orbit_.size());
  }

 private:
  std::vector<MState> states_;
  std::vector<int> first_of_orbit_;
  std::vector<char> kept_orbit_triple_;  // [a * no * no + b * no + c], a<=b<=c
};

/// An operator of ranks 0..3 in the m-scheme (see the file comment). Stores
/// the zero-body value, the one-body matrix (N x N) and the antisymmetric
/// coefficients of the two- and three-body parts for SORTED label sets only
/// (pairs p < q, triples p < q < r; dense over bra x ket). Value type.
class MOperator {
 public:
  /// Zero operator on `num_states` m-states (0 for an empty operator). Only
  /// the parts of rank <= `max_rank` (0..3) are allocated: the rank-3 part
  /// needs (N choose 3)^2 doubles (790 MB at N = 40, emax 2), so tensor
  /// checks beyond emax 1 pass `max_rank` 2. `Get` of an unallocated part is
  /// 0; `Add`/`Set` of one is a precondition violation.
  explicit MOperator(int num_states = 0, int max_rank = 3);

  MOperator(const MOperator&) = default;
  MOperator& operator=(const MOperator&) = default;
  MOperator(MOperator&&) = default;
  MOperator& operator=(MOperator&&) = default;

  /// Number of m-states.
  int num_states() const noexcept { return n_; }
  /// The zero-body part.
  double& zero_body() noexcept { return zero_; }
  /// The zero-body part.
  double zero_body() const noexcept { return zero_; }

  /// Coefficient A_{out; in} of the rank-`rank` part (0..3) for labels in ANY
  /// order: the value of the sorted element times the sign of the sorting
  /// permutations of `out` and of `in`, 0 if a label is repeated in `out` or in
  /// `in`. `out` and `in` have `rank` entries (m-state indices). O(1).
  double Get(int rank, std::span<const int> out, std::span<const int> in) const;
  /// Adds `v` to the coefficient A_{out; in} (labels in any order: `v` times
  /// the sorting sign is added to the sorted element; ignored for a repeated
  /// label). O(1).
  void Add(int rank, std::span<const int> out, std::span<const int> in,
           double v);
  /// Sets the coefficient A_{out; in} (labels in any order, as `Add`).
  void Set(int rank, std::span<const int> out, std::span<const int> in,
           double v);

  /// Number of sorted label sets of rank `rank` (1, N, N(N-1)/2,
  /// N(N-1)(N-2)/6 for rank 0..3).
  std::int64_t NumLabelSets(int rank) const;
  /// Largest absolute value of the rank-`rank` part.
  double MaxAbs(int rank) const;
  /// Largest absolute difference of the rank-`rank` part to that of `other`
  /// (same number of m-states).
  double MaxAbsDifference(const MOperator& other, int rank) const;
  /// this += f * other (all parts; same number of m-states).
  void Axpy(double f, const MOperator& other);

 private:
  int n_ = 0;
  int max_rank_ = 3;
  int n2_ = 0;  // number of pairs
  int n3_ = 0;  // number of triples
  double zero_ = 0.0;
  std::vector<double> one_;
  std::vector<double> two_;
  std::vector<double> three_;
  std::vector<std::int32_t> pair_index_;    // N * N, -1 unless p < q
  std::vector<std::int32_t> triple_index_;  // N^3, -1 unless p < q < r
};

/// J -> m expansion of `X` (zero-, one-, two- and, if present, three-body
/// part) on `basis` (built from X's model space), with the formulas of the
/// file comment: 1B A_ab = X_{orb a, orb b} for equal 2m; 2B A_{ab;cd} = sum_J
/// <ja ma jb mb|J M><jc mc jd md|J M> X^J_unnormalized; 3B for sorted
/// triples sum_{J, Jab, Jde} CG CG CG CG X3^J (0 for a ket that is not kept).
/// Syncs `X` to the host. Cost O(N^6) for the three-body part.
MOperator Expand(const op::Operator& X, const MBasis& basis);

/// m -> J contraction: the operator on `ms` with symmetry and hermiticity of
/// `spec` (scalar, rank spec.particle_rank) whose J-coupled elements are the
/// angular-momentum projections of `A`: 0B copied; 1B X_ab = A_{(a, m), (b,
/// m)} for 2m = 2 j_a; 2B the normalized element sum_m CG CG A / sqrt((1 +
/// d_ab)(1 + d_cd)) at M = J; 3B the canonical unnormalized element sum_m CG
/// CG CG CG A at 2M = 1 (every stored element, both triangles of a
/// non-Hermitian part; packed storage via `op::ThreeBody::set`, the
/// anti-Hermitian diagonal is 0). Parts above spec.particle_rank are ignored.
/// The result is synchronized to the device.
/// @throws InputError if spec.particle_rank is 3 and `ms` has no three-body
///         space.
op::Operator Contract(const MOperator& A,
                      std::shared_ptr<const basis::ModelSpace> ms,
                      const op::OperatorSpec& spec);

/// Quantum-number shift of the OUTPUT of a commutator whose factors are not
/// all scalars: the coefficient A_{out; in} may be nonzero only if
/// 2M_out = 2M_in + two_mu, the parities differ by `parity`, and the sums of
/// 2tz differ by +-`two_dtz` (both signs; a delta_Tz > 0 tensor is stored as O
/// + O^dagger, so both directions occur). The default (all zero) is the scalar
/// selection rule. Value type.
struct TensorShift {
  int two_mu = 0;   ///< 2 mu of the tensor component (M_out - M_in, doubled).
  int parity = 0;   ///< 0 even, 1 odd operator.
  int two_dtz = 0;  ///< 2 delta_Tz = |change of the sum of 2tz|.
};

/// Optional extra phases between imsrg++'s reduced elements and the
/// Wigner-Eckart theorem of `ExpandTensor`. The DEFAULT (all false) is the
/// convention that reproduces the imsrg-devel goldens (the test
/// commutator_ref_mscheme_tensor_golden sweeps all eight combinations: only
/// the default matches, to 1e-15; every other one deviates by O(1)). The knobs
/// exist so that this statement stays checkable. Value type.
struct TensorGauge {
  /// One-body elements carry the extra phase (-1)^(j_bra - j_ket).
  bool one_body_phase = false;
  /// Two-body elements carry the extra phase (-1)^(J_bra - J_ket).
  bool two_body_phase = false;
  /// Two-body elements carry the extra phase (-1)^rank_J relative to the
  /// one-body ones.
  bool two_body_rank_phase = false;
};

/// m-scheme operator of the component `mu` (2 mu = `two_mu.twice`) of a tensor
/// operator `T` of rank lambda = `T.spec().rank_J` (scalars included, with mu =
/// 0), by the Wigner-Eckart theorem with the reduced elements of `T`:
///
///   <j' m'| T_mu |j m> = <j m lambda mu | j' m'> / sqrt(2 j' + 1) R(j', j),
///
/// R the stored reduced element and <j m lambda mu | j' m'> a Clebsch-Gordan
/// coefficient (the ket coupled to the tensor gives the bra). This is exactly
/// Edmonds' definition with (-1)^(j'-m') (j' lambda j; -m' mu m) <j'||T||j>
/// (checked with sympy for several (j', j, lambda, mu, m): the ratio is 1).
/// One-body: R = `T.one_body().get(a', a)`. Two-body: the bra pair is coupled
/// to J_bra, the ket pair to J_ket, and R is the UNNORMALIZED antisymmetrized
/// reduced element `T.two_body().get_unnormalized(p q r s, J_bra, J_ket)`, so
///   A_{ab;cd} = sum_{J_bra, J_ket} CG(ja ma jb mb | Jb Mb) CG(jc mc jd md |
///               Jk Mk) <Jb Mb|T_mu|Jk Mk>,    sorted m-state pairs a < b, c <
///               d.
///
/// THE CONVENTION OF IMSRG++ (found by comparison with the tensor goldens of
/// all four (X, Y) combinations of tests/golden/tensor/emax1_He4, total and
/// per output rank and term, max deviation 1e-15, independent of mu): imsrg++
/// stores reduced elements in EDMONDS' convention for the one-body part and
/// for J-coupled pairs ALIKE: bra j' on the left, no extra phase, the same
/// 1/sqrt(2 J_bra + 1) for pairs as for single particles, a pair |pq; J> being
/// sum CG(jp mp jq mq | J M) a+_p a+_q |0> with p before q. Extra phases
/// `TensorGauge` ((-1)^(j_bra - j_ket) on the one-body part, or (-1)^(J_bra -
/// J_ket) on the pairs, or (-1)^lambda between them) are NOT consistent with
/// the commutator (relative deviations 1.0 to 1.3), so the gauge remark of
/// design/15 (imsrg++ one-body closed forms equal (-1)^(j_bra - j_ket) times
/// the Edmonds element) concerns the physical operators' definitions, not the
/// storage convention of the reduced elements of the code. Cost O(N^4 J^2).
/// Syncs `T` to the host. Returns an operator with ranks up to 2.
MOperator ExpandTensor(const op::Operator& T, const MBasis& basis, TwoM two_mu,
                       const TensorGauge& gauge = {});

/// Inverse of `ExpandTensor` for one component: the tensor operator on `ms`
/// with spec `spec` (rank_J, parity, delta_Tz, hermiticity, particle_rank 1 or
/// 2) whose reduced elements are R = A_coupled / (g CG / sqrt(2 j_bra + 1)),
/// where A_coupled is the m-scheme matrix element (one-body: A_{a m'; b m};
/// two- body: the CG-coupled pair element <Jb Mb|A|Jk Mk>) and the coefficient
/// is taken in the LEAST-SQUARES sense over all allowed m of the ket,
///   R = sum_m w_m A_m / sum_m w_m^2,  w_m = g CG / sqrt(2 j_bra + 1),
/// so a deviation from the Wigner-Eckart structure (an operator that is not a
/// pure component mu of a rank-lambda tensor) shows as a mismatch in the round
/// trip, never as a silently picked configuration. Zero-body part: not set (0).
/// Two-body elements are the normalized stored ones (division by sqrt((1 +
/// delta_pq)(1 + delta_rs))); the stored blocks ch_bra <= ch_ket are filled
/// completely (both triangles of diagonal blocks). The result is synchronized
/// to the device. Cost O(N^4 J^2).
op::Operator ContractTensor(const MOperator& A,
                            std::shared_ptr<const basis::ModelSpace> ms,
                            const op::OperatorSpec& spec, TwoM two_mu,
                            const TensorGauge& gauge = {});

/// One part of the commutator, [A^(nA), B^(nB)]^(nC) = engine(A, B) -
/// engine(B, A), as an `MOperator` holding only the rank-`nC` part (the
/// others are zero): a generic Wick engine over all numbers k1 (contractions
/// of an annihilator of the first factor with a creator of the second, weight
/// 1 - n) and k2 (creator with annihilator, weight n) with k1 + k2 = nA + nB -
/// nC. Evaluated for sorted output labels only; outputs violating the
/// conservation of sum 2m, parity or 2Tz are skipped
/// (for scalars; `shift` replaces the selection rule for a tensor factor, see
/// `TensorShift`), and rank-3 outputs outside the kept three-body space of
/// `basis` are left zero. nA, nB, nC in
/// 0..3 (the rank-nA part of `A` and rank-nB part of `B` are used).
/// Complexity: O(#outputs * #shuffles * N^(k1 + k2)) element lookups.
MOperator CommutatorPart(const MOperator& A, int nA, const MOperator& B, int nB,
                         int nC, const MBasis& basis,
                         const TensorShift& shift = {});

/// [A, B] = sum over nA, nB in 0..3 and nC in 0..`max_rank` of
/// `CommutatorPart` (a rank-3 part needs the slow [3,3]->3 pieces; pass
/// `max_rank` 2 to skip it).
MOperator Commutator(const MOperator& A, const MOperator& B,
                     const MBasis& basis, int max_rank = 3,
                     const TensorShift& shift = {});

}  // namespace imsrg::commutator_ref::mscheme

#endif  // IMSRG_COMMUTATOR_REF_MSCHEME_H_
