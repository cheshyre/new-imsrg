// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
#define IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
/// @file
/// Making replicated operators bitwise identical across the MPI ranks (M12b
/// phase 1, design/12 section 6).

#include "imsrg/comm/communicator.h"
#include "imsrg/operator/operator.h"

namespace imsrg::driver::pipeline {

/// Overwrites `op` on every rank with the value it has on rank 0: the zero-,
/// one-, two- and, if present, three-body parts, bit for bit. Needed because
/// every rank builds the starting operators itself, but HF and the normal
/// ordering use parallel reductions, so the ranks agree only to rounding.
///
/// Collective over `comm`. Precondition: `op` has the same spec and the same
/// (structurally equal) model space on every rank, so the buffers have equal
/// sizes. Memory model: syncs the device data to the host mirrors, broadcasts
/// the mirrors, and makes the device data current again (`SyncToDevice`) on
/// every rank. A no-op for a serial communicator. Complexity: one broadcast
/// per part, O(size of the operator) bytes.
void BroadcastOperator(op::Operator& op, const comm::Communicator& comm);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
