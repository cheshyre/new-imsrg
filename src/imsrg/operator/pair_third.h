// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_PAIR_THIRD_H_
#define IMSRG_OPERATOR_PAIR_THIRD_H_
/// @file
/// Pair-third states and the recoupling maps between them and the canonical
/// kets (production port of the benchmark prototype, design/14 section 4
/// item 0, design/benchmarks/imsrg3-layout.md section 3).
///
/// A pair-third state |(p q) Jpq, r; J> has a two-body ket p <= q (Pauli: Jpq
/// even if p == q) coupled to Jpq and a third orbit r in any position
/// relative to p and q. For each triple T = (a <= b <= c) of the kept
/// canonical space (`basis::ThreeBodySpace`), its pair-third states are the
/// distinct splittings of T, taken in the order of first appearance among
/// imsrg++'s permutations ABC, CBA, ACB of comm223ss
/// (IMSRG3Commutators.cc:5034):
///   ABC -> pair (a, b), third c;  CBA -> pair (b, c), third a;
///   ACB -> pair (a, c), third b,
/// each with every Jpq in [|jp - jq|, jp + jq] that satisfies the Pauli rule
/// and triangle(Jpq, jr, J), in ascending Jpq. They are the index space of the
/// [2,2]->3 intermediate W.
///
/// Index conventions (the contract of the kernels):
/// * pair-third states have a flat index in [0, num_states()); channel `ch`
///   (a `Channel3Index` of the `ThreeBodySpace`; channels of kept
///   dimension 0 own no states) owns the flat range [state_offset(ch),
///   state_offset(ch) + num_states(ch)), in channel order; the
///   channel-local state index is the flat index minus state_offset(ch);
/// * triple t (a `basis::Triple3Index` of the `ThreeBodySpace`) owns the
///   channel-local state range [state_begin(t), state_end(t)) of its channel;
///   triples follow in `ThreeBodySpace` order, states of a triple are
///   contiguous in the order above;
/// * R has one row per FLAT kept row (`ThreeBodySpace` flat row order) and
///   its columns are CHANNEL-LOCAL state indices; E has one row per FLAT
///   state and its columns are CHANNEL-LOCAL kept rows. Entries are sorted by
///   column, duplicates are summed, and entries with |v| < 1e-13 are dropped.
///
/// Maps (values exact up to that drop):
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
///
/// Host-side construction; the arrays live in `exec::DeviceBuffer`s (host
/// mirror plus device copy), so an initialized `exec::Runtime` is required.
/// No Kokkos in this header.

#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::op {

/// Compressed sparse rows: row i has entries [ptr[i], ptr[i + 1]) of `col`
/// and `val`. Trivially copyable; captured by value in kernels.
struct CsrView {
  const std::int64_t* ptr = nullptr;  ///< rows + 1 offsets.
  const std::int32_t* col = nullptr;  ///< Column per entry.
  const double* val = nullptr;        ///< Value per entry.
};
static_assert(std::is_trivially_copyable_v<CsrView>);

/// Owning CSR matrix in device buffers (host mirror filled first). Value
/// type (copies are deep).
class PairThirdCsr {
 public:
  /// Empty matrix (no rows).
  PairThirdCsr() = default;
  PairThirdCsr(const PairThirdCsr&) = default;
  PairThirdCsr& operator=(const PairThirdCsr&) = default;
  PairThirdCsr(PairThirdCsr&&) = default;
  PairThirdCsr& operator=(PairThirdCsr&&) = default;
  ~PairThirdCsr() = default;

  /// Builds the buffers from host arrays (`ptr.size()` = rows + 1, `col` and
  /// `val` of equal size). Needs an initialized runtime.
  static PairThirdCsr FromHost(const std::vector<std::int64_t>& ptr,
                               const std::vector<std::int32_t>& col,
                               const std::vector<double>& val);

  /// View of the host mirror (raw pointers, valid while `*this` lives).
  CsrView host_view() const noexcept;
  /// View of the device data.
  CsrView device_view() const noexcept;
  /// Number of stored entries.
  std::int64_t nonzeros() const noexcept {
    return static_cast<std::int64_t>(val_.size());
  }
  /// Number of rows.
  std::int64_t num_rows() const noexcept {
    return ptr_.empty() ? 0 : static_cast<std::int64_t>(ptr_.size()) - 1;
  }
  /// Bytes of the three arrays.
  std::int64_t Bytes() const noexcept;

 private:
  exec::DeviceBuffer<std::int64_t> ptr_;
  exec::DeviceBuffer<std::int32_t> col_;
  exec::DeviceBuffer<double> val_;
};

/// Device-capturable view of a `PairThirdSpace` (struct of arrays; index
/// conventions in the file comment). Trivially copyable.
struct PairThirdView {
  const std::int32_t* ch_state_offset = nullptr;  ///< Per channel, flat.
  const std::int32_t* ch_num_states = nullptr;    ///< Per channel.
  const std::int32_t* tr_state_begin = nullptr;   ///< Per triple, ch-local.
  const std::int32_t* tr_state_end = nullptr;     ///< Per triple, ch-local.
  const std::int32_t* st_p = nullptr;             ///< Per flat state: pair p.
  const std::int32_t* st_q = nullptr;             ///< Pair q (q >= p).
  const std::int32_t* st_r = nullptr;             ///< Third orbit r.
  const std::int32_t* st_two_jpq = nullptr;       ///< 2 Jpq.
  CsrView r;  ///< R: flat kept row -> channel-local states.
  CsrView e;  ///< E: flat state -> channel-local kept rows.
  std::int32_t num_channels = 0;
  std::int32_t num_triples = 0;
  std::int64_t num_states = 0;
};
static_assert(std::is_trivially_copyable_v<PairThirdView>);

/// Pair-third states of a `ThreeBodySpace` with the maps R and E. Immutable
/// value type; needs an initialized `exec::Runtime`.
class PairThirdSpace {
 public:
  /// Builds the states and the maps. Cost O(states * J range).
  /// Preconditions: `sixj.emax() >= basis.emax()` (IMSRG_CHECK) and `space`
  /// was built over `basis`.
  PairThirdSpace(const basis::ThreeBodySpace& space,
                 const basis::SingleParticleBasis& basis,
                 const angmom::SixJ3Table& sixj);

  PairThirdSpace(const PairThirdSpace&) = default;
  PairThirdSpace& operator=(const PairThirdSpace&) = default;
  PairThirdSpace(PairThirdSpace&&) = default;
  PairThirdSpace& operator=(PairThirdSpace&&) = default;
  ~PairThirdSpace() = default;

  /// Total number of pair-third states.
  std::int64_t num_states() const noexcept { return num_states_; }
  /// Number of states of channel `ch` (0 for empty channels). Precondition:
  /// valid channel index (IMSRG_ASSERT).
  std::int32_t num_states(Channel3Index ch) const;
  /// Flat index of the first state of channel `ch`.
  std::int32_t state_offset(Channel3Index ch) const;
  /// First channel-local state of triple `t`.
  std::int32_t state_begin(basis::Triple3Index t) const;
  /// One past the last channel-local state of triple `t`.
  std::int32_t state_end(basis::Triple3Index t) const;
  /// Number of channels (those of the `ThreeBodySpace`).
  std::int32_t num_channels() const noexcept { return num_channels_; }
  /// Number of triples (those of the `ThreeBodySpace`).
  std::int32_t num_triples() const noexcept { return num_triples_; }
  /// Largest channel state count.
  std::int32_t max_channel_states() const noexcept { return max_states_; }
  /// Sum over channels of the squared state counts.
  std::int64_t sum_states2() const noexcept { return sum_states2_; }

  /// The R map (rows: flat kept rows; columns: channel-local states).
  const PairThirdCsr& r_map() const noexcept { return r_; }
  /// The E map (rows: flat states; columns: channel-local kept rows).
  const PairThirdCsr& e_map() const noexcept { return e_; }
  /// Bytes of the state tables (without R and E).
  std::int64_t StateBytes() const noexcept;
  /// Bytes of the state tables, R and E.
  std::int64_t Bytes() const noexcept;

  /// View of the host mirror.
  PairThirdView host_view() const noexcept;
  /// View of the device data (equal to the host view after a round trip).
  PairThirdView device_view() const noexcept;

 private:
  std::int64_t num_states_ = 0;
  std::int64_t sum_states2_ = 0;
  std::int32_t num_channels_ = 0;
  std::int32_t num_triples_ = 0;
  std::int32_t max_states_ = 0;
  exec::DeviceBuffer<std::int32_t> ch_state_offset_, ch_num_states_;
  exec::DeviceBuffer<std::int32_t> tr_state_begin_, tr_state_end_;
  exec::DeviceBuffer<std::int32_t> st_p_, st_q_, st_r_, st_two_jpq_;
  PairThirdCsr r_;
  PairThirdCsr e_;
};

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_PAIR_THIRD_H_
