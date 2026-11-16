// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_PAIR_THIRD_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_PAIR_THIRD_H_
/// @file
/// Pair-third states and the recoupling maps between them and the canonical
/// kets (benchmark prototype, design/14 section 2).
///
/// A pair-third state |(p q) Jpq, r; J> has a two-body ket p <= q (Pauli: Jpq
/// even if p == q) coupled to Jpq and a third orbit r in any position
/// relative to p and q. For each canonical triple T = (a <= b <= c) of a
/// channel, its pair-third states are the distinct splittings of T, taken
/// in the order of first appearance among imsrg++'s permutations ABC, CBA,
/// ACB of comm223ss (IMSRG3Commutators.cc:5034):
///   ABC -> pair (a, b), third c;  CBA -> pair (b, c), third a;
///   ACB -> pair (a, c), third b,
/// each with every Jpq in [|jp - jq|, jp + jq] that satisfies the Pauli rule
/// and triangle(Jpq, jr, J), in ascending Jpq. They are the rows of layout C
/// and the index space of the [2,2]->3 intermediate W.
///
/// Index conventions: pair-third states have a flat index in [0, NumStates());
/// channel c owns [pt_offset(c), pt_offset(c) + num_pt(c)); triple t owns the
/// channel-local range [pt_begin(t), pt_end(t)). Triples are those of the
/// `CanonicalSpace` (same flat triple index).
///
/// Maps (CSR, values exact up to |v| < 1e-13, which are dropped):
/// * R (canonical row -> pair-third states of the same triple): the
///   recoupling of comm223ss, so that z_A = R W R^T for the canonical [2,2]->3
///   block. For row (T, J1) and perm p in {ABC, CBA, ACB} with (I1, I2, I3) =
///   Permute(p, a, b, c), the column is the state (min(I1, I2), max(I1, I2)),
///   third I3, J1', for every J1' (only J1' = J1 for ABC), with value
///     PermutationPhase(p) * RecouplingCoefficient(p, ja, jb, jc, J1', J1, 2J)
///     * (I1 > I2 ? -(-1)^((j1 + j2)/2 - J1') : 1) * (I1 == I2 ? sqrt(2) : 1)
///   (ja, jb, jc of the canonical a, b, c), summed when two perms give the
///   same state (IMSRG3Commutators.cc:5181-5210).
/// * E (pair-third state -> canonical rows of the same triple): the expansion
///   of the antisymmetrized state, GetKetIndex_withRecoupling
///   (ThreeBodyStorage.cc:236-291): with p_case = SortOrbits(p, q, r),
///   E = PermutationPhase(p_case) * RecouplingCoefficient(p_case, ja, jb, jc,
///   Jpq, Jab, 2J) for each kept canonical row (T, Jab). Fully antisymmetrized
///   pair-third blocks are z_C = E z_A E^T.

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/exec/device_buffer.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {

/// Compressed sparse rows: row i has entries [ptr[i], ptr[i + 1]) of `col`
/// and `val`. Trivially copyable view.
struct CsrView {
  const std::int64_t* ptr = nullptr;
  const std::int32_t* col = nullptr;
  const double* val = nullptr;
};
static_assert(std::is_trivially_copyable_v<CsrView>);

/// Owning CSR matrix in device buffers (host mirror filled first).
struct Csr {
  exec::DeviceBuffer<std::int64_t> ptr;  ///< rows + 1 entries.
  exec::DeviceBuffer<std::int32_t> col;  ///< Column per entry.
  exec::DeviceBuffer<double> val;        ///< Value per entry.

  /// Host view.
  CsrView host_view() const noexcept;
  /// Device view.
  CsrView device_view() const noexcept;
  /// Number of stored entries.
  std::int64_t Nonzeros() const noexcept {
    return static_cast<std::int64_t>(val.size());
  }
  /// Bytes of the three arrays.
  std::int64_t Bytes() const noexcept;
  /// Builds the buffers from host arrays (ptr.size() = rows + 1).
  static Csr FromHost(const std::vector<std::int64_t>& ptr,
                      const std::vector<std::int32_t>& col,
                      const std::vector<double>& val);
};

/// Device view of a `PairThirdSpace` (struct of arrays).
struct PairThirdView {
  const std::int32_t* ch_pt_offset = nullptr;  ///< Per canonical channel.
  const std::int32_t* ch_num_pt = nullptr;     ///< Per canonical channel.
  const std::int32_t* tr_pt_begin = nullptr;   ///< Per triple, channel-local.
  const std::int32_t* tr_pt_end = nullptr;     ///< Per triple, channel-local.
  const std::int32_t* pt_p = nullptr;          ///< Per flat state: pair p.
  const std::int32_t* pt_q = nullptr;          ///< Pair q (q >= p).
  const std::int32_t* pt_r = nullptr;          ///< Third orbit.
  const std::int32_t* pt_two_jpq = nullptr;    ///< 2 Jpq.
  CsrView r;  ///< R: flat canonical row -> channel-local states.
  std::int32_t num_channels = 0;
  std::int64_t num_states = 0;
};
static_assert(std::is_trivially_copyable_v<PairThirdView>);

/// Pair-third states of a canonical space with the maps R and E.
/// Needs an initialized `exec::Runtime`. Value type.
class PairThirdSpace {
 public:
  /// Builds the states and the maps. `sixj` must cover the model space's
  /// emax. Cost O(states * J range).
  PairThirdSpace(const CanonicalSpace& space, const SixJ3Table& sixj);

  PairThirdSpace(const PairThirdSpace&) = default;
  PairThirdSpace& operator=(const PairThirdSpace&) = default;
  PairThirdSpace(PairThirdSpace&&) = default;
  PairThirdSpace& operator=(PairThirdSpace&&) = default;
  ~PairThirdSpace() = default;

  /// Number of pair-third states (sum of the layout-C channel dimensions).
  std::int64_t NumStates() const noexcept { return num_states_; }
  /// ΣN_C, ΣN_C², packed size, max N_C per channel.
  SpaceCounts Counts() const;
  /// The R map (rows: flat canonical rows; columns: channel-local states).
  const Csr& RMap() const noexcept { return r_; }
  /// The E map (rows: flat states; columns: channel-local canonical rows).
  const Csr& EMap() const noexcept { return e_; }
  /// Bytes of the state tables (without R and E).
  std::int64_t StateBytes() const noexcept;

  /// Host view.
  PairThirdView host_view() const noexcept;
  /// Device view.
  PairThirdView device_view() const noexcept;

  /// Counts of the pair-third states of `space` without building maps (for
  /// the extrapolation to large emax; same enumeration as the constructor).
  static SpaceCounts CountOnly(const CanonicalSpace& space);

 private:
  std::int64_t num_states_ = 0;
  std::int32_t num_channels_ = 0;
  exec::DeviceBuffer<std::int32_t> ch_pt_offset_, ch_num_pt_;
  exec::DeviceBuffer<std::int32_t> tr_pt_begin_, tr_pt_end_;
  exec::DeviceBuffer<std::int32_t> pt_p_, pt_q_, pt_r_, pt_two_jpq_;
  Csr r_;
  Csr e_;
};

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_PAIR_THIRD_H_
