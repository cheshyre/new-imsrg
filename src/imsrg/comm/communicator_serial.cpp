// Communicator in a build without MPI: only the one-rank communicator exists.
#include <algorithm>
#include <cstring>

#include "imsrg/base/assert.hpp"
#include "imsrg/comm/communicator.hpp"

namespace imsrg::comm {

struct Communicator::Impl {};

Communicator Communicator::serial() {
  static const auto impl = std::make_shared<const Impl>();
  return Communicator(impl);
}

Communicator Communicator::world_() { return serial(); }

int Communicator::rank() const noexcept { return 0; }
int Communicator::size() const noexcept { return 1; }
bool Communicator::is_serial() const noexcept { return true; }
void Communicator::barrier() const {}

#define IMSRG_COMM_SERIAL_REDUCE(T)                       \
  void Communicator::allreduce_sum(std::span<T>) const {} \
  void Communicator::allreduce_max(std::span<T>) const {} \
  void Communicator::allreduce_min(std::span<T>) const {}
IMSRG_COMM_SERIAL_REDUCE(double)
IMSRG_COMM_SERIAL_REDUCE(float)
IMSRG_COMM_SERIAL_REDUCE(std::int32_t)
IMSRG_COMM_SERIAL_REDUCE(std::int64_t)
#undef IMSRG_COMM_SERIAL_REDUCE

double Communicator::allreduce_sum(double x) const { return x; }
std::int64_t Communicator::allreduce_sum(std::int64_t x) const { return x; }
double Communicator::allreduce_max(double x) const { return x; }
std::int64_t Communicator::allreduce_max(std::int64_t x) const { return x; }
double Communicator::allreduce_min(double x) const { return x; }
std::int64_t Communicator::allreduce_min(std::int64_t x) const { return x; }

#define IMSRG_COMM_SERIAL_BCAST(T)                             \
  void Communicator::broadcast(std::span<T>, int root) const { \
    IMSRG_CHECK(root == 0, "broadcast: root out of range");    \
  }
IMSRG_COMM_SERIAL_BCAST(double)
IMSRG_COMM_SERIAL_BCAST(float)
IMSRG_COMM_SERIAL_BCAST(std::int32_t)
IMSRG_COMM_SERIAL_BCAST(std::int64_t)
IMSRG_COMM_SERIAL_BCAST(std::byte)
#undef IMSRG_COMM_SERIAL_BCAST

void Communicator::broadcast(std::string&, int root) const {
  IMSRG_CHECK(root == 0, "broadcast: root out of range");
}

std::vector<double> Communicator::allgather(double x) const { return {x}; }
std::vector<std::int64_t> Communicator::allgather(std::int64_t x) const { return {x}; }

void Communicator::alltoallv(std::span<const double> send, std::span<const int> send_counts,
                             std::span<const int> send_displs, std::span<double> recv,
                             std::span<const int> recv_counts,
                             std::span<const int> recv_displs) const {
  IMSRG_CHECK(send_counts.size() == 1 && send_displs.size() == 1 && recv_counts.size() == 1 &&
                  recv_displs.size() == 1,
              "alltoallv: count/displacement spans must have length size()");
  IMSRG_CHECK(send_counts[0] == recv_counts[0], "alltoallv: send and receive counts differ");
  IMSRG_CHECK(send_counts[0] >= 0 && send_displs[0] >= 0 && recv_displs[0] >= 0,
              "alltoallv: negative count or displacement");
  const auto n = static_cast<std::size_t>(send_counts[0]);
  IMSRG_CHECK(static_cast<std::size_t>(send_displs[0]) + n <= send.size() &&
                  static_cast<std::size_t>(recv_displs[0]) + n <= recv.size(),
              "alltoallv: buffer too small");
  if (n > 0) {
    std::memmove(recv.data() + recv_displs[0], send.data() + send_displs[0], n * sizeof(double));
  }
}

Communicator Communicator::split(int, int) const { return *this; }
Communicator Communicator::duplicate() const { return *this; }

}  // namespace imsrg::comm
