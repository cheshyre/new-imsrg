// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_CANONICAL_SPACE_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_CANONICAL_SPACE_H_
/// @file
/// The kept canonical three-body space shared by all layouts (benchmark
/// prototype, design/14 section 2): the imsrg++ kets |(ab) Jab, c; J> with a
/// <= b <= c of `basis::ThreeBodyChannels` that pass the configuration cut
/// (dE3max, OccNat3Cut). Kets failing the cut are dropped at enumeration
/// (owner, D34), so the cut defines the space identically for every term.
///
/// Index conventions (every table of the prototype uses them):
/// * canonical channel c in [0, num_channels): the 3B channels (2J, parity,
///   2Tz) that keep at least one ket, in imsrg++ channel order;
/// * canonical row: flat index in [0, NumRows()); channel c owns rows
///   [row_offset(c), row_offset(c) + dim(c)); inside a channel the rows keep
///   the imsrg++ local order (ascending global ket index, i.e. lexicographic
///   in (a, b, c, Jab)), so the rows of one orbit triple are contiguous;
/// * triple: flat index t in [0, num_triples); channel c owns triples
///   [triple_begin(c), triple_end(c)) in row order; a triple stores its
///   channel-local row range.
/// Canonical operator blocks are dense dim x dim, column-major, at
/// `block_offset(c)` of one buffer (total `total_size()` doubles); values are
/// imsrg++'s (unnormalized, antisymmetrized) canonical matrix elements.

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

#include "imsrg/base/device.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::layout3 {

/// Device view of a `CanonicalSpace` (struct of arrays, raw pointers;
/// trivially copyable, captured by value in kernels).
struct CanonicalView {
  // Per canonical channel (num_channels entries).
  const std::int32_t* ch_two_j = nullptr;         ///< 2J (odd).
  const std::int32_t* ch_parity = nullptr;        ///< 0 even, 1 odd.
  const std::int32_t* ch_two_tz = nullptr;        ///< 2Tz.
  const std::int32_t* ch_dim = nullptr;           ///< Kept rows.
  const std::int32_t* ch_row_offset = nullptr;    ///< First flat row.
  const std::int32_t* ch_triple_begin = nullptr;  ///< First flat triple.
  const std::int32_t* ch_triple_end = nullptr;    ///< One past the last.
  const std::int64_t* ch_block_offset =
      nullptr;  ///< Dense block offsets, num_channels + 1 entries.
  // Per triple (num_triples entries).
  const std::int32_t* tr_a = nullptr;  ///< Orbits a <= b <= c.
  const std::int32_t* tr_b = nullptr;
  const std::int32_t* tr_c = nullptr;
  const std::int32_t* tr_channel = nullptr;    ///< Canonical channel.
  const std::int32_t* tr_row_begin = nullptr;  ///< Channel-local rows.
  const std::int32_t* tr_row_end = nullptr;
  const std::int32_t* tr_sym = nullptr;  ///< 6, 3 or 1 (imsrg++ symm_abc).
  // Per flat row (NumRows() entries).
  const std::int32_t* row_two_jab = nullptr;  ///< 2 Jab.
  const std::int32_t* row_triple = nullptr;   ///< Flat triple index.
  std::int32_t num_channels = 0;
  std::int32_t num_triples = 0;
  std::int64_t num_rows = 0;
};
static_assert(std::is_trivially_copyable_v<CanonicalView>);

/// Size summary of a layout or space.
struct SpaceCounts {
  std::int64_t sum_n = 0;       ///< Sum over channels of the dimension N.
  std::int64_t sum_n2 = 0;      ///< Sum of N^2 (dense doubles per operator).
  std::int64_t sum_packed = 0;  ///< Sum of N (N + 1) / 2 (packed triangle).
  std::int32_t max_n = 0;       ///< Largest N.
  std::int32_t num_channels = 0;
  std::int64_t num_triples = 0;
};

/// Immutable kept canonical three-body space. Value type (copies are deep).
/// Needs an initialized `exec::Runtime`.
class CanonicalSpace {
 public:
  /// Builds the space from `ms->three_body()` (precondition: `ms` was built
  /// `WithThreeBody`, IMSRG_CHECK) keeping the kets for which
  /// `ThreeBodyChannels::PassesConfigurationCut(k, basis, reference, de3max,
  /// occnat3_cut)` holds (reference occupations). Channels without kept kets
  /// are skipped. Cost O(sum of channel kets).
  CanonicalSpace(std::shared_ptr<const basis::ModelSpace> ms, double de3max,
                 double occnat3_cut);

  CanonicalSpace(const CanonicalSpace&) = default;
  CanonicalSpace& operator=(const CanonicalSpace&) = default;
  CanonicalSpace(CanonicalSpace&&) = default;
  CanonicalSpace& operator=(CanonicalSpace&&) = default;
  ~CanonicalSpace() = default;

  /// The model space (with three-body channels).
  const basis::ModelSpace& model_space() const noexcept { return *ms_; }
  /// Shared pointer to the model space.
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr()
      const noexcept {
    return ms_;
  }
  /// Number of canonical channels.
  std::int32_t num_channels() const noexcept { return num_channels_; }
  /// Number of kept rows (sum of the channel dimensions).
  std::int64_t NumRows() const noexcept { return num_rows_; }
  /// Number of triples.
  std::int32_t NumTriples() const noexcept { return num_triples_; }
  /// Doubles of all dense channel blocks (sum of dim^2).
  std::int64_t total_size() const noexcept;
  /// ΣN, ΣN², packed size, max N, channel and triple counts.
  SpaceCounts Counts() const;
  /// Bytes of all index tables (the layout-A metadata).
  std::int64_t MetadataBytes() const noexcept;

  /// View of the host mirror (host code reads every table through it).
  CanonicalView host_view() const noexcept;
  /// View of the device data (kernels).
  CanonicalView device_view() const noexcept;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::int32_t num_channels_ = 0;
  std::int32_t num_triples_ = 0;
  std::int64_t num_rows_ = 0;
  // All int32 tables back to back (see CanonicalView), and the int64 block
  // offsets. Field offsets into `i32_` are fixed at construction.
  exec::DeviceBuffer<std::int32_t> i32_;
  exec::DeviceBuffer<std::int64_t> block_offset_;
  std::int64_t field_offset_[16] = {};
};

/// imsrg++'s three-body norm (D12; ThreeBodyStorage_pn::Norm): sqrt of the
/// sum over channels of z_ij^2 over the packed triangle i <= j of the
/// canonical blocks `canonical` (layout: see the file comment; size
/// `space.total_size()`). Dropped kets count as zeros, as imsrg++ stores them
/// but its kernels never fill them. O(total_size).
double ImsrgppNorm(const CanonicalSpace& space,
                   std::span<const double> canonical);

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_CANONICAL_SPACE_H_
