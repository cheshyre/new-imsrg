// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_COMM232_H_
#define IMSRG_COMMUTATOR_DETAIL_COMM232_H_

/// @file
/// Internal pieces of the N7 term comm232ss ([X2, Y3] + [X3, Y2] -> Z2,
/// imsrg3.h; implementation and conventions in kernels/comm232.cc): the
/// sorted pair-third tables (built once per `ThreeBodyContext`) and the entry
/// points used by `terms::Comm232ss` and the tests (design/12 section 4.2:
/// every kernel phase passes the same tests). Internal to the commutator
/// module and its tests; no Kokkos.

#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Scratch budget of `terms::Comm232ss` for the pair-third blocks (bytes, all
/// block sets of one batch together): 512 MiB, so emax 3 at E3max 9 runs in
/// one batch (two block sets for an anti-Hermitian X and a Hermitian Y:
/// 365 MB for O16, 505 MB for He4; 8.8 MB for O16 at emax 2, E3max 6).
inline constexpr std::int64_t kComm232MaxBlockBytes = std::int64_t{1} << 29;

/// Number of occupation classes of a pair (kernels/comm232.cc `PairClass`):
/// hh-only, mixed, pp-only, inactive.
inline constexpr std::int32_t kComm232Classes = 4;

/// Device view of `Comm232Tables`: the pair-third states of the context's
/// kept space (`op::PairThirdSpace`) in the order of the comm232ss pair-third
/// path, per channel sorted by (third r, Jpq, occupation class of the pair,
/// p, q). "Sorted index" t: the position in that order, flat over the
/// channels (channel ch owns [ch_begin[ch], ch_begin[ch + 1]), the flat
/// range of its states in `op::PairThirdSpace`). Block b = (ch norb + r) nj +
/// Jpq holds the states of channel ch with third r and pair J Jpq; its class
/// k starts at blk[kComm232Classes b + k] and the block ends at
/// blk[kComm232Classes (b + 1)]. The active rows (classes 0..2) of a channel
/// have compact indices 0..A_ch - 1 in sorted order; those of block b start
/// at arow0[b]. W blocks: channel ch's dense column-major A_ch x S_ch block of
/// active rows sits at w_off[ch] of a buffer holding all channels, so column
/// t (a sorted index) starts at w_off[ch] + (t - ch_begin[ch]) A_ch and the
/// columns of all channels are consecutive in t. Raw device pointers;
/// trivially copyable, captured by value.
struct Comm232TablesView {
  const std::int32_t* blk = nullptr;        ///< kComm232Classes nblocks + 1.
  const std::int32_t* arow0 = nullptr;      ///< Per block.
  const std::int32_t* ch_begin = nullptr;   ///< num_channels + 1 entries.
  const std::int32_t* ch_active = nullptr;  ///< A_ch per channel.
  /// Active rows of all channels in compact order (channel ch from
  /// act_begin[ch]): the sorted index of each.
  const std::int32_t* act_row = nullptr;
  const std::int32_t* act_begin = nullptr;  ///< num_channels entries.
  const std::int32_t* orig = nullptr;       ///< Sorted -> PairThirdSpace flat.
  const std::int32_t* key = nullptr;    ///< p norb + q of the pair (p <= q).
  const std::int32_t* local = nullptr;  ///< Local index of (p, q) at Jpq.
  const double* occ_hh = nullptr;       ///< w n_p n_q, w = 1/sqrt 2 for p == q.
  const double* occ_pp = nullptr;       ///< w nbar_p nbar_q.
  const std::int64_t* w_off = nullptr;  ///< num_channels + 1 entries.
  std::int32_t norb = 0;                ///< Orbits of the basis.
  std::int32_t nj = 0;                  ///< Two-body J values (max J + 1).
  std::int32_t num_channels = 0;        ///< Three-body channels.
};
static_assert(std::is_trivially_copyable_v<Comm232TablesView>);

/// The sorted pair-third tables of comm232ss (see `Comm232TablesView`) in
/// device buffers, with the host copies the batching needs. Built once per
/// `ThreeBodyContext` (`ThreeBodyContext::Impl::comm232`); read-only
/// afterwards. Value type (copies are deep).
struct Comm232Tables {
  exec::DeviceBuffer<std::int32_t> blk;
  exec::DeviceBuffer<std::int32_t> arow0;
  exec::DeviceBuffer<std::int32_t> ch_begin;
  exec::DeviceBuffer<std::int32_t> ch_active;
  exec::DeviceBuffer<std::int32_t> act_row;
  exec::DeviceBuffer<std::int32_t> act_begin;
  exec::DeviceBuffer<std::int32_t> orig;
  exec::DeviceBuffer<std::int32_t> key;
  exec::DeviceBuffer<std::int32_t> local;
  exec::DeviceBuffer<double> occ_hh;
  exec::DeviceBuffer<double> occ_pp;
  exec::DeviceBuffer<std::int64_t> w_off;
  std::vector<std::int64_t> w_off_host;      ///< Host copy of w_off.
  std::vector<std::int32_t> ch_begin_host;   ///< Host copy of ch_begin.
  std::vector<std::int32_t> ch_active_host;  ///< Host copy of ch_active.
  std::vector<std::int32_t> ch_two_j_host;   ///< 2 J3 per channel.
  std::vector<std::int32_t> third_host;      ///< Third orbit per sorted t.
  std::int32_t norb = 0;                     ///< Orbits of the basis.
  std::int32_t nj = 0;                       ///< Two-body J values.
  std::int32_t max_two_jc = 0;               ///< Largest 2j of an orbit.

  /// The device view (pointers valid while *this lives unchanged). O(1).
  Comm232TablesView View() const noexcept;
  /// Bytes of the device buffers (for `ThreeBodyContext::TableBytes`).
  std::int64_t Bytes() const noexcept;
};

/// Builds the comm232ss tables of the pair-third states `pair_third` of the
/// kept space of `ms3` (classes from the occupations of its reference).
/// Needs an initialized `exec::Runtime`. Cost O(states log states) on the
/// host plus the copies.
/// @throws InvariantError if a pair-third state has no two-body ket or the
///         channels' states are not contiguous.
Comm232Tables BuildComm232Tables(const op::PairThirdSpace& pair_third,
                                 const basis::ModelSpace& ms3);

/// comm232ss evaluated element by element: one work item per stored Z2
/// element, three-body elements read with `ThreeBodyElement` (imsrg++
/// `GetME_pn`, recoupled from the canonical kets) inside the sums over c, J',
/// (a b) and J3. Same contract as `terms::Comm232ss` (adds into Z, any
/// hermiticity). Complexity per element O(norb * J'-range * kets *
/// J3-range) element reads of cost O(J range^2 log triples).
void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);

/// comm232ss through the pair-third blocks of the three-body operands (the
/// path of `terms::Comm232ss`, which passes `kComm232MaxBlockBytes`): the
/// active rows of W = E B3 E^T, built per call for batches of consecutive
/// columns (flat sorted states, so a batch may hold several channels or part
/// of one) whose blocks (one set per operand, two for a non-Hermitian
/// three-body part) fit `max_block_bytes` together; a batch holds at least
/// one column, so the budget bounds the scratch unless a single column
/// (A_ch doubles per set) exceeds it. The element pass runs once per batch,
/// over the Z2 blocks whose J can reach the batch's 2 J3 range, with the c,
/// J' and J3 loops clamped to it (same result for any budget up to rounding).
/// Same contract as `terms::Comm232ss`. Complexity: the block build O(sum
/// over channels A S nnz(E row)^2) plus, per batch, O(elements in the J range
/// * norb * (J', J3 in range) * (rows + log S)) (S pair-third states of a
/// channel, A of them with an active pair).
void Comm232ssPairThird(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx,
                        std::int64_t max_block_bytes);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_COMM232_H_
