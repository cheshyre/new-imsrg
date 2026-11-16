// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_SUPPORT_N7_PARTITION_SUPPORT_H_
#define TESTS_SUPPORT_N7_PARTITION_SUPPORT_H_

/// @file
/// Partition-invariance and masked-operand checks of the row-tiled N7 terms
/// (design/14 section 6.5). A term is run through a callback on the lines of
/// a `basis::RowSet`; the checks compare the sum over the parts of standard
/// partitions with the whole-space run (bitwise or within
/// max|dZ| <= tol max(1, max|Z|)), and, for every part, the run on operands
/// whose three-body entries off the part's lines are NaN with the run on the
/// unmasked operands (bitwise: a kernel that reads outside its batch's lines
/// fails). Report through GoogleTest expectations.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/operator/operator.h"

namespace imsrg::testing {

/// Adds a term of [X, Y], restricted to the lines of `rows`, into Z.
using N7RowsFn =
    std::function<void(const op::Operator& X, const op::Operator& Y,
                       op::Operator& Z, const basis::RowSet& rows)>;

/// `parts` contiguous pieces of the triple sequence of `s` (by triple index),
/// each as a `RowSet` (a piece crossing a channel boundary gives one range
/// per channel).
std::vector<basis::RowSet> ContiguousParts(const basis::ThreeBodySpace& s,
                                           std::int32_t parts);

/// The parts of `RowPartition::Balanced` with `commutator::N7LineWeights`.
/// Precondition: `ms` has a three-body space.
std::vector<basis::RowSet> BalancedParts(const basis::ModelSpace& ms,
                                         std::int32_t parts);

/// The partitions of the N7 tests, labelled: contiguous P = 2, 3, 7 and
/// balanced P = 4, 16.
std::vector<std::pair<std::string, std::vector<basis::RowSet>>>
StandardPartitions(const basis::ModelSpace& ms);

/// Deviation of `a` from `ref` over every part (0B, 1B, 2B and 3B, host
/// mirrors after a sync): the largest |a - ref|, the largest |ref|, and
/// whether all values are bitwise equal (NaN != NaN).
struct OperatorDiff {
  double max_abs_diff = 0.0;  ///< max |a - ref| (inf if a NaN appears).
  double max_abs_ref = 0.0;   ///< max |ref|.
  bool bitwise_equal = true;  ///< Every value has the same bits.
};
OperatorDiff CompareOperators(const op::Operator& a, const op::Operator& ref);

/// A copy of `x` whose three-body entries off the lines of `rows` are NaN
/// (`MaskOutsideLines`); 0B, 1B and 2B parts unchanged. A missing three-body
/// part stays missing.
op::Operator MaskedCopy(const op::Operator& x, const basis::RowSet& rows);

/// Single-triple `RowSet`s (a single triple is always one batch) of `s`: all
/// triples if there are at most `max_triples`, else a deterministic sample of
/// about `max_triples` triples spread over all channels (the first and the
/// last triple of every nonempty channel are always included). O(triples).
std::vector<basis::RowSet> SingleTripleSets(const basis::ThreeBodySpace& s,
                                            std::int32_t max_triples);

/// Which checks `ExpectPartitionInvariance` runs besides the partition sums.
struct PartitionChecks {
  /// Masked operands per single triple (`SingleTripleSets`, at most
  /// `max_batch_triples` triples): a part holds several batches, and a read
  /// of another batch's lines in the same part is only caught at this
  /// granularity (in M12b only the current batch's data is resident).
  bool per_batch = true;
  std::int32_t max_batch_triples = 64;  ///< Sample size of the per-batch check.
  /// Output ownership (design/14 section 6.5): a part run into a zeroed Z
  /// writes Z3 only in the rows of the part (`OnlyRowsNonzero`). Off for a
  /// term that places output outside its rows (133 through `Comm133Rows`).
  bool ownership = false;
};

/// The partition-invariance and masked-operand checks (file comment) for
/// `run`, with `zero()` creating a zeroed output Z. For every standard
/// partition: Z_parts = sum over the parts of run(part) (into one Z) against
/// Z_whole = run(Whole): bitwise if `bitwise`, else max|dZ| <= tol max(1,
/// max|Z|); for every part p: run(MaskedCopy(X, p), MaskedCopy(Y, p), p)
/// bitwise equal to run(X, Y, p) and free of NaN; per `checks`, the same for
/// single triples and the output-ownership check. Prints the worst deviation
/// with `tag`. Precondition: `ms` is the model space of X, Y and Z.
void ExpectPartitionInvariance(const basis::ModelSpace& ms,
                               const op::Operator& X, const op::Operator& Y,
                               const std::function<op::Operator()>& zero,
                               const N7RowsFn& run, double tol, bool bitwise,
                               const std::string& tag,
                               const PartitionChecks& checks = {});

}  // namespace imsrg::testing

#endif  // TESTS_SUPPORT_N7_PARTITION_SUPPORT_H_
