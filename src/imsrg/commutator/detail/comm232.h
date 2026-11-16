// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_COMM232_H_
#define IMSRG_COMMUTATOR_DETAIL_COMM232_H_

/// @file
/// Internal pieces of the N7 term comm232ss ([X2, Y3] + [X3, Y2] -> Z2,
/// imsrg3.h; implementation and conventions in kernels/comm232.cc): the
/// sorted pair-third tables (built once per `ThreeBodyContext`) and the entry
/// points used by `terms::Comm232ss` and the tests (design/12 section 4.2:
/// every kernel phase passes the same tests). The batch kernel
/// `Comm232Batch` and the row loop `Comm232Rows` are declared with the other
/// N7 batch kernels in detail/n7_terms.h. Internal to the commutator module
/// and its tests; no Kokkos.

#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Number of occupation classes of a pair (detail/n7_batch.h `PairClass`):
/// hh-only, mixed, pp-only, inactive.
inline constexpr std::int32_t kComm232Classes = 4;

/// Device view of `Comm232Tables`: the pair-third states of the context's
/// kept space (`op::PairThirdSpace`) in the row order of the pair-third
/// intermediates H and W (detail/n7_batch.h), per channel sorted by (third
/// r, Jpq, occupation class of the pair, p, q). "Sorted index" t: the
/// position in that order, flat over the channels (channel ch owns
/// [ch_begin[ch], ch_begin[ch + 1]), the flat range of its states in
/// `op::PairThirdSpace`). Block b = (ch norb + r) nj + Jpq holds the states
/// of channel ch with third r and pair J Jpq; its class k starts at
/// blk[kComm232Classes b + k] and the block ends at blk[kComm232Classes (b +
/// 1)], so the rows of a block and of a class range are contiguous. Raw
/// device pointers; trivially copyable, captured by value.
struct Comm232TablesView {
  const std::int32_t* blk = nullptr;       ///< kComm232Classes nblocks + 1.
  const std::int32_t* ch_begin = nullptr;  ///< num_channels + 1 entries.
  const std::int32_t* orig = nullptr;      ///< Sorted -> PairThirdSpace flat.
  const std::int32_t* key = nullptr;       ///< p norb + q of the pair (p <= q).
  const std::int32_t* local = nullptr;     ///< Local index of (p, q) at Jpq.
  const double* occ_hh = nullptr;  ///< w n_p n_q, w = 1/sqrt 2 for p == q.
  const double* occ_pp = nullptr;  ///< w nbar_p nbar_q.
  std::int32_t norb = 0;           ///< Orbits of the basis.
  std::int32_t nj = 0;             ///< Two-body J values (max J + 1).
  std::int32_t num_channels = 0;   ///< Three-body channels.
};
static_assert(std::is_trivially_copyable_v<Comm232TablesView>);

/// The sorted pair-third tables of comm232ss (see `Comm232TablesView`) in
/// device buffers, with host copies for host code and tests. Built once per
/// `ThreeBodyContext` (`ThreeBodyContext::Impl::comm232`); read-only
/// afterwards. Value type (copies are deep).
struct Comm232Tables {
  exec::DeviceBuffer<std::int32_t> blk;
  exec::DeviceBuffer<std::int32_t> ch_begin;
  exec::DeviceBuffer<std::int32_t> orig;
  exec::DeviceBuffer<std::int32_t> key;
  exec::DeviceBuffer<std::int32_t> local;
  exec::DeviceBuffer<double> occ_hh;
  exec::DeviceBuffer<double> occ_pp;
  std::vector<std::int32_t> ch_begin_host;  ///< Host copy of ch_begin.
  /// Number of active states per channel (pair class 0..2: n_p n_q != 0 or
  /// nbar_p nbar_q != 0).
  std::vector<std::int32_t> ch_active_host;
  std::vector<std::int32_t> third_host;  ///< Third orbit per sorted t.
  std::int32_t norb = 0;                 ///< Orbits of the basis.
  std::int32_t nj = 0;                   ///< Two-body J values.

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

/// comm232ss through the pair-third blocks W = E B3 E^T of the three-body
/// operands: `detail::Comm232Rows` (detail/n7_terms.h) on the whole space
/// with the batch budget `max_block_bytes` (the H and W blocks of all
/// operand blocks of one batch; `terms::Comm232ss` passes `kN7BatchBytes`).
/// A batch holds at least one triple, so the budget bounds the scratch
/// unless a single triple exceeds it; any budget gives the same result up to
/// rounding. Same contract as `terms::Comm232ss`. Complexity: per batch the
/// H/W build (`BuildH`, `BuildW`) plus `Comm232Batch`, i.e. over all batches
/// O(sum over Z2 elements of the (column, J') pairs that reach them * rows).
void Comm232ssPairThird(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx,
                        std::int64_t max_block_bytes);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_COMM232_H_
