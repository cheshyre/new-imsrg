// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
#define IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
/// @file
/// Making replicated operators bitwise identical across the MPI ranks (M12b
/// phase 1, design/12 section 6).

#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/operator/operator.h"

namespace imsrg::driver::pipeline {

/// Verifies that every rank of `comm` runs the same replicated setup, and
/// throws `RuntimeError` on every rank if not. Compared: a 64-bit FNV-1a hash
/// of the orbit quantum numbers and the reference occupations (bits of the
/// doubles), the number of orbits, a hash of the two-body channel ket counts,
/// whether a three-body space exists, and `concurrency` (the thread count of
/// the default execution space: Kokkos reductions are deterministic only for
/// a fixed thread count, and a last-bit difference in a norm can make one rank
/// run one more commutator and hang in a collective). The decision uses the
/// reduced (max and min over the ranks) values, so all ranks throw together
/// and none is left in a collective. The message names what differs.
///
/// Collective over `comm`; a no-op for a serial communicator. Complexity:
/// O(orbits + channels) per rank plus two small allreduces.
/// @throws RuntimeError on all ranks if the setups differ.
void CheckReplicatedSetup(const basis::ModelSpace& ms,
                          const comm::Communicator& comm, int concurrency);

/// `CheckReplicatedSetup` with the concurrency of the current Kokkos runtime
/// (`exec::Runtime::Concurrency()`, which needs an initialized runtime).
void CheckReplicatedSetup(const basis::ModelSpace& ms,
                          const comm::Communicator& comm);

/// Overwrites `op` on every rank with the value it has on rank 0: the zero-,
/// one-, two- and, if present, three-body parts, bit for bit. Needed because
/// every rank builds the starting operators itself, but HF and the normal
/// ordering use parallel reductions, so the ranks agree only to rounding.
///
/// Collective over `comm`; it starts with `CheckReplicatedSetup` (model space
/// and thread count), so a rank-dependent setup throws on every rank instead
/// of corrupting the distributed work. Precondition: `op` has the same spec and
/// the same (structurally equal) model space on every rank, so the buffers have
/// equal sizes. Memory model: syncs the device data to the host mirrors,
/// broadcasts the mirrors, and makes the device data current again
/// (`SyncToDevice`) on every rank. A no-op for a serial communicator.
/// Complexity: one broadcast per part, O(size of the operator) bytes.
void BroadcastOperator(op::Operator& op, const comm::Communicator& comm);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_MPI_SYNC_H_
