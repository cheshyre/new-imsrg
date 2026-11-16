// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/driver/pipeline/mpi_sync.h"

#include <bit>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/runtime.h"

namespace imsrg::driver::pipeline {

namespace {

/// Broadcasts the host mirror of `buffer` from rank 0 (empty buffers are
/// skipped on every rank, since the sizes agree).
void BroadcastBuffer(exec::DeviceBuffer<double>& buffer,
                     const comm::Communicator& comm) {
  if (buffer.empty()) return;
  comm.Broadcast(buffer.host_span(), 0);
}

/// 64-bit FNV-1a accumulator over 64-bit words.
class Fnv1a {
 public:
  void Add(std::uint64_t word) {
    for (int b = 0; b < 8; ++b) {
      hash_ ^= (word >> (8 * b)) & 0xffu;
      hash_ *= 0x100000001b3ull;
    }
  }
  void AddSigned(std::int64_t v) { Add(static_cast<std::uint64_t>(v)); }
  std::int64_t value() const { return static_cast<std::int64_t>(hash_); }

 private:
  std::uint64_t hash_ = 0xcbf29ce484222325ull;
};

}  // namespace

void CheckReplicatedSetup(const basis::ModelSpace& ms,
                          const comm::Communicator& comm, int concurrency) {
  if (comm.is_serial()) return;
  static const char* const kNames[] = {
      "orbits and occupations", "number of orbits",
      "two-body channel ket counts", "three-body space presence",
      "thread count"};
  Fnv1a orbits;
  const auto& basis = ms.basis();
  const auto occ = ms.reference().occupations();
  for (const basis::Orbit& o : basis.orbits()) {
    orbits.AddSigned(o.n);
    orbits.AddSigned(o.l);
    orbits.AddSigned(o.j2.twice);
    orbits.AddSigned(o.tz2.twice);
  }
  for (const double x : occ) orbits.Add(std::bit_cast<std::uint64_t>(x));
  Fnv1a kets;
  const auto& two_body = ms.two_body();
  kets.AddSigned(two_body.num_channels());
  for (const auto& ch : two_body.channels()) kets.AddSigned(ch.num_kets);
  std::vector<std::int64_t> local = {
      orbits.value(), static_cast<std::int64_t>(ms.num_orbits()), kets.value(),
      ms.three_body_space() != nullptr ? 1 : 0,
      static_cast<std::int64_t>(concurrency)};
  std::vector<std::int64_t> hi = local, lo = local;
  comm.AllreduceMax(std::span<std::int64_t>(hi));
  comm.AllreduceMin(std::span<std::int64_t>(lo));
  std::string what;
  for (std::size_t i = 0; i < local.size(); ++i) {
    if (hi[i] != lo[i]) {
      if (!what.empty()) what += ", ";
      what += kNames[i];
    }
  }
  if (!what.empty()) {
    throw RuntimeError(
        "MPI ranks do not run the same replicated setup; they differ in: " +
        what + " (rank " + std::to_string(comm.rank()) + " of " +
        std::to_string(comm.size()) + ")");
  }
}

void CheckReplicatedSetup(const basis::ModelSpace& ms,
                          const comm::Communicator& comm) {
  if (comm.is_serial()) return;
  CheckReplicatedSetup(ms, comm, exec::Runtime::Concurrency());
}

void BroadcastOperator(op::Operator& op, const comm::Communicator& comm) {
  if (comm.is_serial()) return;
  CheckReplicatedSetup(op.model_space(), comm);
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
