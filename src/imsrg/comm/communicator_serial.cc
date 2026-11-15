// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Communicator in a build without MPI: only the one-rank communicator exists.
#include <algorithm>
#include <cstring>

#include "imsrg/base/assert.h"
#include "imsrg/comm/communicator.h"

namespace imsrg::comm {

struct Communicator::Impl {};

Communicator Communicator::Serial() {
  static const auto kImpl = std::make_shared<const Impl>();
  return Communicator(kImpl);
}

Communicator Communicator::World() { return Serial(); }

int Communicator::rank() const noexcept { return 0; }
int Communicator::size() const noexcept { return 1; }
bool Communicator::is_serial() const noexcept { return true; }
void Communicator::Barrier() const {}

#define IMSRG_COMM_SERIAL_REDUCE(T)                      \
  void Communicator::AllreduceSum(std::span<T>) const {} \
  void Communicator::AllreduceMax(std::span<T>) const {} \
  void Communicator::AllreduceMin(std::span<T>) const {}
IMSRG_COMM_SERIAL_REDUCE(double)
IMSRG_COMM_SERIAL_REDUCE(float)
IMSRG_COMM_SERIAL_REDUCE(std::int32_t)
IMSRG_COMM_SERIAL_REDUCE(std::int64_t)
#undef IMSRG_COMM_SERIAL_REDUCE

double Communicator::AllreduceSum(double x) const { return x; }
std::int64_t Communicator::AllreduceSum(std::int64_t x) const { return x; }
double Communicator::AllreduceMax(double x) const { return x; }
std::int64_t Communicator::AllreduceMax(std::int64_t x) const { return x; }
double Communicator::AllreduceMin(double x) const { return x; }
std::int64_t Communicator::AllreduceMin(std::int64_t x) const { return x; }

#define IMSRG_COMM_SERIAL_BCAST(T)                             \
  void Communicator::Broadcast(std::span<T>, int root) const { \
    IMSRG_CHECK(root == 0, "broadcast: root out of range");    \
  }
IMSRG_COMM_SERIAL_BCAST(double)
IMSRG_COMM_SERIAL_BCAST(float)
IMSRG_COMM_SERIAL_BCAST(std::int32_t)
IMSRG_COMM_SERIAL_BCAST(std::int64_t)
IMSRG_COMM_SERIAL_BCAST(std::byte)
#undef IMSRG_COMM_SERIAL_BCAST

void Communicator::Broadcast(std::string&, int root) const {
  IMSRG_CHECK(root == 0, "broadcast: root out of range");
}

std::vector<double> Communicator::Allgather(double x) const { return {x}; }
std::vector<std::int64_t> Communicator::Allgather(std::int64_t x) const {
  return {x};
}

void Communicator::Alltoallv(std::span<const double> send,
                             std::span<const int> send_counts,
                             std::span<const int> send_displs,
                             std::span<double> recv,
                             std::span<const int> recv_counts,
                             std::span<const int> recv_displs) const {
  IMSRG_CHECK(send_counts.size() == 1 && send_displs.size() == 1 &&
                  recv_counts.size() == 1 && recv_displs.size() == 1,
              "alltoallv: count/displacement spans must have length size()");
  IMSRG_CHECK(send_counts[0] == recv_counts[0],
              "alltoallv: send and receive counts differ");
  IMSRG_CHECK(send_counts[0] >= 0 && send_displs[0] >= 0 && recv_displs[0] >= 0,
              "alltoallv: negative count or displacement");
  const auto n = static_cast<std::size_t>(send_counts[0]);
  IMSRG_CHECK(static_cast<std::size_t>(send_displs[0]) + n <= send.size() &&
                  static_cast<std::size_t>(recv_displs[0]) + n <= recv.size(),
              "alltoallv: buffer too small");
  if (n > 0) {
    std::memmove(recv.data() + recv_displs[0], send.data() + send_displs[0],
                 n * sizeof(double));
  }
}

Communicator Communicator::Split(int, int) const { return *this; }
Communicator Communicator::Duplicate() const { return *this; }

}  // namespace imsrg::comm
