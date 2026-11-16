// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASIS_THREE_BODY_DISTRIBUTION_H_
#define IMSRG_BASIS_THREE_BODY_DISTRIBUTION_H_
/// @file
/// How the rows of the kept three-body space are distributed over the ranks
/// of a communicator (M12b plan section 4.1). A `ModelSpace` carries one
/// distribution (`ModelSpace::WithThreeBodyDistribution`), so that all
/// three-body operators of the model space store the same rows.

#include <memory>

#include "imsrg/basis/three_body_rows.h"
#include "imsrg/comm/communicator.h"

namespace imsrg::basis {

/// A row partition of the kept three-body space and the communicator over
/// which it is distributed: this rank owns `partition->part(comm.rank())`.
/// Invariant (checked by `ModelSpace::WithThreeBodyDistribution`): the
/// partition is non-null and `partition->num_parts() == comm.size()`; the
/// caller guarantees that it covers the kept space of the model space it is
/// attached to. Cheap to copy (shared partition, shared communicator handle).
struct ThreeBodyDistribution {
  /// The partition; part p belongs to the rank p of `comm`.
  std::shared_ptr<const RowPartition> partition;
  /// The communicator (parts = comm.size(), this rank's part = comm.rank()).
  comm::Communicator comm;
};

}  // namespace imsrg::basis

#endif  // IMSRG_BASIS_THREE_BODY_DISTRIBUTION_H_
