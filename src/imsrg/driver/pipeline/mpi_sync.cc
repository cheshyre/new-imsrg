// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/mpi_sync.h"

#include <span>

#include "imsrg/exec/device_buffer.h"

namespace imsrg::driver::pipeline {

namespace {

/// Broadcasts the host mirror of `buffer` from rank 0 (empty buffers are
/// skipped on every rank, since the sizes agree).
void BroadcastBuffer(exec::DeviceBuffer<double>& buffer,
                     const comm::Communicator& comm) {
  if (buffer.empty()) return;
  comm.Broadcast(buffer.host_span(), 0);
}

}  // namespace

void BroadcastOperator(op::Operator& op, const comm::Communicator& comm) {
  if (comm.is_serial()) return;
  op.SyncToHost();
  comm.Broadcast(std::span<double>(&op.zero_body(), 1), 0);
  BroadcastBuffer(op.one_body().buffer(), comm);
  BroadcastBuffer(op.two_body().matrix().buffer(), comm);
  if (op.three_body().has_value()) {
    BroadcastBuffer(op.three_body()->buffer(), comm);
  }
  op.SyncToDevice();
}

}  // namespace imsrg::driver::pipeline
