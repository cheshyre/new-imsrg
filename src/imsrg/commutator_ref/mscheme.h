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
  /// Zero operator on `num_states` m-states (0 for an empty operator).
  explicit MOperator(int num_states = 0);

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

/// One part of the commutator, [A^(nA), B^(nB)]^(nC) = engine(A, B) -
/// engine(B, A), as an `MOperator` holding only the rank-`nC` part (the
/// others are zero): a generic Wick engine over all numbers k1 (contractions
/// of an annihilator of the first factor with a creator of the second, weight
/// 1 - n) and k2 (creator with annihilator, weight n) with k1 + k2 = nA + nB -
/// nC. Evaluated for sorted output labels only; outputs violating the
/// conservation of sum 2m, parity or 2Tz are skipped, and rank-3 outputs
/// outside the kept three-body space of `basis` are left zero. nA, nB, nC in
/// 0..3 (the rank-nA part of `A` and rank-nB part of `B` are used).
/// Complexity: O(#outputs * #shuffles * N^(k1 + k2)) element lookups.
MOperator CommutatorPart(const MOperator& A, int nA, const MOperator& B, int nB,
                         int nC, const MBasis& basis);

/// [A, B] = sum over nA, nB in 0..3 and nC in 0..`max_rank` of
/// `CommutatorPart` (a rank-3 part needs the slow [3,3]->3 pieces; pass
/// `max_rank` 2 to skip it).
MOperator Commutator(const MOperator& A, const MOperator& B,
                     const MBasis& basis, int max_rank = 3);

}  // namespace imsrg::commutator_ref::mscheme

#endif  // IMSRG_COMMUTATOR_REF_MSCHEME_H_
