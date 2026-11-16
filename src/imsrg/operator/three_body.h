// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_THREE_BODY_H_
#define IMSRG_OPERATOR_THREE_BODY_H_
/// @file
/// `ThreeBody`: the three-body part of an operator in the canonical layout A
/// (design/00 D43, design/benchmarks/imsrg3-layout.md section 1).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::op {

/// The three-body part of a scalar `Operator`: one block per channel
/// (2J, parity, 2Tz) of the kept canonical space `basis::ThreeBodySpace` of the
/// model space, whose rows are the kets |(ab) Jab, c; J> with a <= b <= c
/// (imsrg++'s canonical pn kets, kets failing the configuration cuts dropped),
/// all blocks in one `exec::DeviceBuffer<double>`; block `ch` starts at
/// `block_offsets()[ch]`, the offsets have num_channels + 1 entries. Values are
/// imsrg++'s unnormalized antisymmetrized `GetME_pn(Jab, Jde, 2J, ...)`.
///
/// Storage by hermiticity, for a block of dimension n:
/// * Hermitian and anti-Hermitian: the PACKED LOWER TRIANGLE (LAPACK 'L'
///   packed, column-major), element (i, j) with i >= j at
///   `PackedIndex(n, i, j)`: imsrg++'s stored z(ibra >= iket). The other
///   triangle is h z(j, i) with h = +1 (Hermitian) or -1 (anti-Hermitian); the
///   anti-Hermitian diagonal is zero and stored as 0.
/// * Non-Hermitian: DENSE n x n blocks, column-major, element (i, j) at
///   `i + j n`.
///
/// Memory model (design/12 section 2, see `Operator`): the host accessors
/// (`get`, `set`, `Add`, `host_span`) use the host mirror; device data are
/// authoritative between library calls. Call `SyncToHost()` before host reads
/// that follow a library call and `SyncToDevice()` after host writes.
///
/// Distributed storage (M12b): if the model space carries a three-body row
/// distribution (`ModelSpace::three_body_distribution() != nullptr`), the part
/// stores only the rows of `model_space().three_body_rows()` (this rank's
/// share), concatenated in set order (rows of a range ascending), row k of a
/// block of dimension n having length n - k (packed, starting at z(k, k)) or n
/// (dense, column k). For the undistributed model space (`RowSet::Whole`) this
/// is byte for byte the layout above. A distributed part has no
/// `block_offsets()`; `get`/`set`/`Add` of an element whose stored row is not
/// local fail an `IMSRG_CHECK`. Elementwise operations (axpy, scale, ...) act
/// on the equal local buffers; collective operations (`ImsrgppNorm`,
/// `GatherToRoot`) must be called by every rank.
///
/// Value semantics (deep copies, cheap moves); needs an initialized
/// `exec::Runtime` unless empty. The part keeps the model space alive.
class ThreeBody {
 public:
  /// Empty part (no model space, no blocks).
  ThreeBody() = default;
  /// The zero part with hermiticity `h` on the kept three-body space of `ms`.
  /// @throws InputError if `ms` is null or has no three-body space (build it
  ///         with `ModelSpace::WithThreeBody`).
  ThreeBody(std::shared_ptr<const basis::ModelSpace> ms, Hermiticity h);

  ThreeBody(const ThreeBody&) = default;
  ThreeBody& operator=(const ThreeBody&) = default;
  ThreeBody(ThreeBody&&) = default;
  ThreeBody& operator=(ThreeBody&&) = default;

  /// Index of element (i, j), i >= j, of an n x n packed lower triangle
  /// (column-major): i + j (2 n - j - 1) / 2. Preconditions: 0 <= j <= i < n.
  /// O(1), usable in kernels.
  static IMSRG_INLINE_FUNCTION constexpr std::int64_t PackedIndex(
      std::int64_t n, std::int64_t i, std::int64_t j) noexcept {
    return i + j * (2 * n - j - 1) / 2;
  }
  /// Number of stored doubles of an n x n block: n (n + 1) / 2 if `packed`,
  /// else n^2. O(1), usable in kernels.
  static IMSRG_INLINE_FUNCTION constexpr std::int64_t BlockSize(
      std::int64_t n, bool packed) noexcept {
    return packed ? n * (n + 1) / 2 : n * n;
  }

  /// True for a default-constructed (or moved-from) part.
  bool empty() const noexcept { return ms_ == nullptr; }
  /// The model space (null for an empty part).
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr()
      const noexcept {
    return ms_;
  }
  /// The kept three-body space. Precondition: !empty() (IMSRG_CHECK).
  const basis::ThreeBodySpace& space() const;
  /// The hermiticity (storage mode).
  Hermiticity hermiticity() const noexcept { return hermiticity_; }
  /// True for the packed lower-triangle storage (Hermitian, anti-Hermitian).
  bool packed() const noexcept {
    return hermiticity_ != Hermiticity::kNonHermitian;
  }
  /// Changes the hermiticity. A distributed part supports packed <->
  /// packed (Hermitian <-> anti-Hermitian) only; converting it to or from
  /// non-Hermitian (dense) throws `NotImplementedError` (M12b-Q8). Hermitian
  /// <-> anti-Hermitian only flips the flag: the stored (lower) values are
  /// kept, so the upper triangle changes sign (as `Operator::set_hermiticity`);
  /// a nonzero diagonal turning anti-Hermitian is zeroed. To non-Hermitian: the
  /// storage becomes dense with the OLD symmetry applied to the upper triangle
  /// (the represented matrix is unchanged). From non-Hermitian: the lower
  /// triangle (diagonal included, zeroed for anti-Hermitian) is kept and the
  /// upper one dropped. Conversions go through the host mirror (after a
  /// device-to-host sync) and leave host and device data equal. O(size).
  void set_hermiticity(Hermiticity h);

  /// Number of channels of the space (0 for an empty part).
  std::int32_t num_channels() const noexcept {
    return offsets_.empty() ? 0
                            : static_cast<std::int32_t>(offsets_.size()) - 1;
  }
  /// Dimension n of block `ch`: the number of kept rows. Precondition: valid
  /// channel (IMSRG_ASSERT).
  std::int32_t dim(Channel3Index ch) const;
  /// True iff the model space distributes the three-body rows
  /// (`three_body_distribution() != nullptr`), i.e. the part holds this
  /// rank's rows only.
  bool distributed() const noexcept;
  /// Start of every block in the buffer; num_channels() + 1 entries, the last
  /// is `size()`. Precondition: the part is not distributed (IMSRG_CHECK: a
  /// distributed buffer does not hold whole blocks).
  std::span<const std::int64_t> block_offsets() const;
  /// Number of stored doubles (of the local rows if distributed).
  std::size_t size() const noexcept { return data_.size(); }

  /// Element (i, j) of the full matrix of channel `ch` (rows `KeptRow3Index`)
  /// from the host mirror, with the hermiticity sign for i < j in packed mode
  /// (0 on the anti-Hermitian diagonal). Preconditions: valid indices
  /// (IMSRG_ASSERT); if distributed, the stored row of the element (row
  /// min(i, j) packed, column j dense) is local (IMSRG_CHECK). O(1) (O(log
  /// ranges) if distributed).
  double get(Channel3Index ch, basis::KeptRow3Index i,
             basis::KeptRow3Index j) const;
  /// Sets element (i, j) = v (host mirror). Packed mode: for i < j this
  /// stores h v at (j, i), so that get(i, j) == v afterwards; a nonzero value
  /// on the anti-Hermitian diagonal is an IMSRG_CHECK failure. Dense mode:
  /// writes (i, j) only. O(1).
  void set(Channel3Index ch, basis::KeptRow3Index i, basis::KeptRow3Index j,
           double v);
  /// Adds v to element (i, j) (host mirror), with the conventions of `set`.
  /// O(1).
  void Add(Channel3Index ch, basis::KeptRow3Index i, basis::KeptRow3Index j,
           double v);
  /// Sets every stored element to zero (device kernel).
  void set_zero();

  /// Host mirror of the whole buffer (see the memory model).
  std::span<double> host_span() { return data_.host_span(); }
  /// Host mirror, read-only.
  std::span<const double> host_span() const { return data_.host_span(); }
  /// Device data, for kernels.
  double* device_data() noexcept { return data_.device_data(); }
  /// Device data, read-only.
  const double* device_data() const noexcept { return data_.device_data(); }
  /// The underlying buffer.
  exec::DeviceBuffer<double>& buffer() noexcept { return data_; }
  /// The underlying buffer, read-only.
  const exec::DeviceBuffer<double>& buffer() const noexcept { return data_; }
  /// Copies the device data to the host mirror.
  void SyncToHost() const { data_.SyncToHost(); }
  /// Copies the host mirror to the device data.
  void SyncToDevice() { data_.SyncToDevice(); }

 private:
  /// Buffer index of element (i, j) and the sign of the mirror relation
  /// (0: a structurally zero element, the anti-Hermitian diagonal).
  struct Location {
    std::int64_t index;
    int sign;
  };
  Location Locate(Channel3Index ch, basis::KeptRow3Index i,
                  basis::KeptRow3Index j) const;
  /// Start of the stored row `k` of block `ch` (distributed parts only; an
  /// IMSRG_CHECK failure if the row is not local).
  std::int64_t LocalRow(Channel3Index ch, std::int64_t k) const;
  /// Offsets for the storage mode `packed` (num_channels + 1 entries).
  std::vector<std::int64_t> OffsetsFor(bool packed) const;

  std::shared_ptr<const basis::ModelSpace> ms_;
  Hermiticity hermiticity_ = Hermiticity::kHermitian;
  std::vector<std::int64_t> offsets_;
  exec::DeviceBuffer<double> data_;
};

/// imsrg++'s three-body norm (design/00 D12): sqrt of the sum of squares of
/// the stored lower triangle (i >= j, diagonal once) of every block, exactly
/// `ThreeBodyStorage_pn::Norm` for Hermitian and anti-Hermitian parts (whose
/// storage is that triangle). For a non-Hermitian (dense) part the same sum
/// over the lower triangle: this is OUR definition, imsrg++ has no
/// non-Hermitian three-body storage. 0 for an empty part. One device
/// reduction (packed) or a host loop (dense); O(size).
///
/// COLLECTIVE for a distributed part (hidden collective, M12b review note
/// M4): every rank of the distribution's communicator must call it. Each rank
/// sums the squares of its rows, `ReduceBcastSum` adds the one double over
/// the ranks and the square root follows, so the result has identical bits
/// on all ranks and equals the serial value up to the reassociation of the
/// sum (which depends on the number of ranks).
double ImsrgppNorm(const ThreeBody& t);

/// Collective for a distributed part (every rank calls it): assembles the
/// whole three-body part on rank 0 of the distribution's communicator. Rank 0
/// receives a part on `model_space().WithoutThreeBodyDistribution()` (all
/// rows, same hermiticity, `RowSet::Whole` layout, device synchronized); every
/// other rank receives an EMPTY part (`empty()`), to keep the memory of the
/// gather on one rank. For a part that is not distributed it is a local copy
/// on every rank (no communication). For tests and dumps: the whole part must
/// fit on rank 0, and each per-rank message must hold fewer than 2^31 doubles
/// (checked by `Communicator::Alltoallv`). Host mirrors are used (the local
/// part is synchronized to the host first). O(size).
ThreeBody GatherToRoot(const ThreeBody& t);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_THREE_BODY_H_
