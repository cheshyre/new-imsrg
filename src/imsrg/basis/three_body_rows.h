// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASIS_THREE_BODY_ROWS_H_
#define IMSRG_BASIS_THREE_BODY_ROWS_H_
/// @file
/// Row ranges of the kept three-body space: the vocabulary shared by the
/// IMSRG(3) N7 kernels and the distribution of three-body blocks over MPI
/// ranks. Host only.
///
/// * Block of channel ch: n = dim(ch) rows; the rows of a triple are
///   contiguous (`ThreeBodySpace`).
/// * Row k of a block is its k-th stored segment. Packed storage: the values
///   z(m, k), m = k..n-1, contiguous at `PackedIndex(n, k, k)`, length n - k.
///   Dense storage: column k, z(m, k) for all m, length n. The storage format
///   is unchanged: the rows of all channels, concatenated in channel order and
///   ascending, are the buffer of `op::ThreeBody` byte for byte.
/// * Line k = the whole matrix row k = row k plus its transpose part (entries
///   z(k, m') stored in rows m' < k).
/// * A range holds whole triples [triple_begin, triple_end) of ONE channel; a
///   `RowSet` is a sorted list of disjoint ranges (one rank's share); a
///   `RowPartition` is P `RowSet`s covering the kept space exactly once.

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/basis/three_body_space.h"

namespace imsrg::basis {

/// Whole triples of one channel and the channel-local rows they span.
struct RowRange {
  Channel3Index channel;      ///< The channel.
  Triple3Index triple_begin;  ///< First triple.
  Triple3Index triple_end;    ///< One past the last triple.
  KeptRow3Index row_begin;    ///< `row_begin` of triple `triple_begin`.
  KeptRow3Index row_end;      ///< `row_end` of triple `triple_end - 1`.
};
static_assert(std::is_trivially_copyable_v<RowRange>);

/// A sorted list of disjoint, nonempty, triple-aligned `RowRange`s (the share
/// of one rank). Immutable value type.
class RowSet {
 public:
  /// The empty set.
  RowSet() = default;

  RowSet(const RowSet&) = default;
  RowSet& operator=(const RowSet&) = default;
  RowSet(RowSet&&) = default;
  RowSet& operator=(RowSet&&) = default;

  /// One range per nonempty channel of `s` (all its triples). O(channels).
  static RowSet Whole(const ThreeBodySpace& s);
  /// Validates and sorts `r` by (channel, row_begin).
  /// @throws InputError if a range has an invalid channel, is empty, is not
  ///         triple-aligned (triple bounds outside the channel or row fields
  ///         different from those of the triples), or two ranges overlap.
  /// O(r log r).
  static RowSet FromRanges(const ThreeBodySpace& s, std::vector<RowRange> r);

  /// The ranges, sorted by (channel, row_begin).
  std::span<const RowRange> ranges() const noexcept { return ranges_; }
  /// Number of rows in the set.
  std::int64_t NumRows() const noexcept { return num_rows_; }
  /// Number of stored doubles of the rows of the set: n - k per row k if
  /// `packed`, n per row if dense. O(1).
  std::int64_t StoredSize(bool packed) const noexcept;
  /// True iff row `k` of channel `ch` belongs to the set. O(log ranges).
  bool Contains(Channel3Index ch, KeptRow3Index k) const;
  /// Start of row `k` of channel `ch` in the LOCAL storage (the rows of the
  /// set concatenated in set order, the rows of a range ascending), or -1 if
  /// the row is absent. O(log ranges).
  std::int64_t LocalRowOffset(Channel3Index ch, KeptRow3Index k,
                              bool packed) const;

 private:
  /// Index of the range containing (ch, k), or -1.
  std::int64_t FindRange(Channel3Index ch, KeptRow3Index k) const;

  std::vector<RowRange> ranges_;
  std::vector<std::int32_t> dims_;           // channel dimension per range
  std::vector<std::int64_t> packed_prefix_;  // stored size before each range
  std::vector<std::int64_t> dense_prefix_;   // (ranges + 1 entries each)
  std::int64_t num_rows_ = 0;
};

/// Per-triple weights of a partition: one entry per `Triple3Index`.
struct LineWeights {
  std::vector<double> compute;  ///< Work of the lines of the triple.
  std::vector<double> memory;   ///< Stored doubles of the triple's lines.
};

/// P `RowSet`s that cover the kept space exactly once. Immutable value type.
class RowPartition {
 public:
  RowPartition(const RowPartition&) = default;
  RowPartition& operator=(const RowPartition&) = default;
  RowPartition(RowPartition&&) = default;
  RowPartition& operator=(RowPartition&&) = default;

  /// P = 1: `RowSet::Whole`.
  static RowPartition Whole(const ThreeBodySpace& s);
  /// Balanced partition into `parts` shares, deterministic. (1) Per channel a
  /// zipper fold: its triples are walked from both ends, at each step the
  /// front or the back triple is taken, whichever keeps (taken compute /
  /// channel compute) - (taken memory / channel memory) closer to 0 (ties:
  /// front; a zero total counts as ratio 0), so that any contiguous piece of
  /// the sequence is at most two ranges. (2) Channels by decreasing dimension
  /// (ties: lower index), their sequences concatenated, with w_t = 1/2
  /// compute_t / C + 1/2 memory_t / M (global totals, a zero total counts as
  /// 0), part p gets the triples whose midpoint (cumulative w before t plus
  /// w_t / 2) lies in [p / P, (p + 1) / P) (clamped to P - 1). Parts may be
  /// empty. O(triples log triples).
  /// @throws InputError if `parts < 1` or the weights do not have one finite
  ///         non-negative entry per triple.
  static RowPartition Balanced(const ThreeBodySpace& s, const LineWeights& w,
                               std::int32_t parts);
  /// Whole channels only (longest processing time first: channels by
  /// decreasing total compute, ties by index, each to the part with the least
  /// compute so far, ties lowest part). For comparison with `Balanced`. Same
  /// errors as `Balanced`. O(triples + channels log channels).
  static RowPartition WholeChannels(const ThreeBodySpace& s,
                                    const LineWeights& w, std::int32_t parts);

  /// Number of parts.
  std::int32_t num_parts() const noexcept {
    return static_cast<std::int32_t>(parts_.size());
  }
  /// Part `p`. Precondition: 0 <= p < num_parts() (IMSRG_ASSERT).
  const RowSet& part(std::int32_t p) const;
  /// The part owning row `k` of channel `ch`, or -1 for an invalid row.
  /// O(log ranges).
  std::int32_t Owner(Channel3Index ch, KeptRow3Index k) const;

 private:
  struct Entry {
    Channel3Index channel;
    KeptRow3Index row_begin;
    KeptRow3Index row_end;
    std::int32_t part = 0;
  };
  explicit RowPartition(std::vector<RowSet> parts);

  std::vector<RowSet> parts_;
  std::vector<Entry> index_;  // all ranges sorted by (channel, row_begin)
};

/// Load imbalance of a partition: max / mean - 1 of the per-part sums.
struct PartitionImbalanceResult {
  double compute = 0.0;  ///< Of the compute weights.
  double memory = 0.0;   ///< Of the memory weights.
};

/// max / mean - 1 over the parts of the summed compute and memory weights (0
/// if a total is 0). Precondition: `w` has one entry per triple of the space
/// the partition was built on (the caller's space; only sizes checked via
/// IMSRG_CHECK against the largest triple used). O(triples).
PartitionImbalanceResult PartitionImbalance(const RowPartition& p,
                                            const LineWeights& w);

}  // namespace imsrg::basis

#endif  // IMSRG_BASIS_THREE_BODY_ROWS_H_
