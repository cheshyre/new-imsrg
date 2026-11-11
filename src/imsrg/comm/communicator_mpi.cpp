// Communicator backed by MPI. The serial communicator (Communicator::serial()) makes no MPI calls.
#include <algorithm>
#include <cstring>
#include <string>

#include <mpi.h>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/comm/communicator.hpp"

namespace imsrg::comm {

struct Communicator::Impl {
  MPI_Comm comm = MPI_COMM_NULL;
  bool serial = false;
  bool owned = false;  // created by split/duplicate: freed with the last handle
  int rank = 0;
  int size = 1;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  ~Impl() {
    if (!owned || comm == MPI_COMM_NULL) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) MPI_Comm_free(&comm);
  }
};

namespace {

/// Throws RuntimeError with the MPI error string if `rc != MPI_SUCCESS`.
void check_mpi(int rc, const char* call) {
  if (rc == MPI_SUCCESS) return;
  char text[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  if (MPI_Error_string(rc, text, &len) != MPI_SUCCESS) len = 0;
  throw RuntimeError(std::string("comm: ") + call +
                     " failed: " + std::string(text, static_cast<std::size_t>(len)));
}

std::shared_ptr<const Communicator::Impl> wrap(MPI_Comm comm, bool owned) {
  auto impl = std::make_shared<Communicator::Impl>();
  impl->comm = comm;
  impl->owned = owned;
  check_mpi(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
  check_mpi(MPI_Comm_rank(comm, &impl->rank), "MPI_Comm_rank");
  check_mpi(MPI_Comm_size(comm, &impl->size), "MPI_Comm_size");
  return impl;
}

MPI_Datatype mpi_type(const double*) { return MPI_DOUBLE; }
MPI_Datatype mpi_type(const float*) { return MPI_FLOAT; }
MPI_Datatype mpi_type(const std::int32_t*) { return MPI_INT32_T; }
MPI_Datatype mpi_type(const std::int64_t*) { return MPI_INT64_T; }
MPI_Datatype mpi_type(const std::byte*) { return MPI_BYTE; }

int checked_count(std::size_t n) {
  IMSRG_CHECK(n <= static_cast<std::size_t>(INT32_MAX),
              "comm: message longer than 2^31-1 elements");
  return static_cast<int>(n);
}

template <class T>
void reduce_inplace(const Communicator::Impl& impl, std::span<T> data, MPI_Op op,
                    const char* name) {
  if (impl.serial || data.empty()) return;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, data.data(), checked_count(data.size()),
                          mpi_type(data.data()), op, impl.comm),
            name);
}

template <class T>
T reduce_scalar(const Communicator::Impl& impl, T x, MPI_Op op, const char* name) {
  reduce_inplace(impl, std::span<T>(&x, 1), op, name);
  return x;
}

template <class T>
void bcast(const Communicator::Impl& impl, std::span<T> data, int root) {
  IMSRG_CHECK(root >= 0 && root < impl.size, "broadcast: root out of range");
  if (impl.serial || data.empty()) return;
  check_mpi(
      MPI_Bcast(data.data(), checked_count(data.size()), mpi_type(data.data()), root, impl.comm),
      "MPI_Bcast");
}

template <class T>
std::vector<T> gather_all(const Communicator::Impl& impl, T x) {
  std::vector<T> out(static_cast<std::size_t>(impl.size), x);
  if (impl.serial) return out;
  check_mpi(MPI_Allgather(&x, 1, mpi_type(&x), out.data(), 1, mpi_type(&x), impl.comm),
            "MPI_Allgather");
  return out;
}

}  // namespace

Communicator Communicator::serial() {
  static const std::shared_ptr<const Impl> impl = [] {
    auto p = std::make_shared<Impl>();
    p->serial = true;
    return p;
  }();
  return Communicator(impl);
}

Communicator Communicator::world_() {
  static const std::shared_ptr<const Impl> impl = wrap(MPI_COMM_WORLD, false);
  return Communicator(impl);
}

int Communicator::rank() const noexcept { return impl_->rank; }
int Communicator::size() const noexcept { return impl_->size; }
bool Communicator::is_serial() const noexcept { return impl_->serial; }

void Communicator::barrier() const {
  if (impl_->serial) return;
  check_mpi(MPI_Barrier(impl_->comm), "MPI_Barrier");
}

#define IMSRG_COMM_REDUCE(T)                                  \
  void Communicator::allreduce_sum(std::span<T> d) const {    \
    reduce_inplace(*impl_, d, MPI_SUM, "MPI_Allreduce(sum)"); \
  }                                                           \
  void Communicator::allreduce_max(std::span<T> d) const {    \
    reduce_inplace(*impl_, d, MPI_MAX, "MPI_Allreduce(max)"); \
  }                                                           \
  void Communicator::allreduce_min(std::span<T> d) const {    \
    reduce_inplace(*impl_, d, MPI_MIN, "MPI_Allreduce(min)"); \
  }
IMSRG_COMM_REDUCE(double)
IMSRG_COMM_REDUCE(float)
IMSRG_COMM_REDUCE(std::int32_t)
IMSRG_COMM_REDUCE(std::int64_t)
#undef IMSRG_COMM_REDUCE

double Communicator::allreduce_sum(double x) const {
  return reduce_scalar(*impl_, x, MPI_SUM, "MPI_Allreduce(sum)");
}
std::int64_t Communicator::allreduce_sum(std::int64_t x) const {
  return reduce_scalar(*impl_, x, MPI_SUM, "MPI_Allreduce(sum)");
}
double Communicator::allreduce_max(double x) const {
  return reduce_scalar(*impl_, x, MPI_MAX, "MPI_Allreduce(max)");
}
std::int64_t Communicator::allreduce_max(std::int64_t x) const {
  return reduce_scalar(*impl_, x, MPI_MAX, "MPI_Allreduce(max)");
}
double Communicator::allreduce_min(double x) const {
  return reduce_scalar(*impl_, x, MPI_MIN, "MPI_Allreduce(min)");
}
std::int64_t Communicator::allreduce_min(std::int64_t x) const {
  return reduce_scalar(*impl_, x, MPI_MIN, "MPI_Allreduce(min)");
}

void Communicator::broadcast(std::span<double> d, int root) const { bcast(*impl_, d, root); }
void Communicator::broadcast(std::span<float> d, int root) const { bcast(*impl_, d, root); }
void Communicator::broadcast(std::span<std::int32_t> d, int root) const { bcast(*impl_, d, root); }
void Communicator::broadcast(std::span<std::int64_t> d, int root) const { bcast(*impl_, d, root); }
void Communicator::broadcast(std::span<std::byte> d, int root) const { bcast(*impl_, d, root); }

void Communicator::broadcast(std::string& s, int root) const {
  IMSRG_CHECK(root >= 0 && root < impl_->size, "broadcast: root out of range");
  if (impl_->serial) return;
  std::int64_t n = static_cast<std::int64_t>(s.size());
  bcast(*impl_, std::span<std::int64_t>(&n, 1), root);
  if (impl_->rank != root) s.resize(static_cast<std::size_t>(n));
  bcast(*impl_, std::as_writable_bytes(std::span<char>(s.data(), s.size())), root);
}

std::vector<double> Communicator::allgather(double x) const { return gather_all(*impl_, x); }
std::vector<std::int64_t> Communicator::allgather(std::int64_t x) const {
  return gather_all(*impl_, x);
}

void Communicator::alltoallv(std::span<const double> send, std::span<const int> send_counts,
                             std::span<const int> send_displs, std::span<double> recv,
                             std::span<const int> recv_counts,
                             std::span<const int> recv_displs) const {
  const auto p = static_cast<std::size_t>(impl_->size);
  IMSRG_CHECK(send_counts.size() == p && send_displs.size() == p && recv_counts.size() == p &&
                  recv_displs.size() == p,
              "alltoallv: count/displacement spans must have length size()");
  for (std::size_t r = 0; r < p; ++r) {
    IMSRG_CHECK(
        send_counts[r] >= 0 && send_displs[r] >= 0 && recv_counts[r] >= 0 && recv_displs[r] >= 0,
        "alltoallv: negative count or displacement");
    IMSRG_CHECK(
        static_cast<std::size_t>(send_displs[r]) + static_cast<std::size_t>(send_counts[r]) <=
                send.size() &&
            static_cast<std::size_t>(recv_displs[r]) + static_cast<std::size_t>(recv_counts[r]) <=
                recv.size(),
        "alltoallv: buffer too small");
  }
  if (impl_->serial) {
    if (send_counts[0] > 0) {
      std::memmove(recv.data() + recv_displs[0], send.data() + send_displs[0],
                   static_cast<std::size_t>(send_counts[0]) * sizeof(double));
    }
    return;
  }
  check_mpi(
      MPI_Alltoallv(send.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, recv.data(),
                    recv_counts.data(), recv_displs.data(), MPI_DOUBLE, impl_->comm),
      "MPI_Alltoallv");
}

Communicator Communicator::split(int color, int key) const {
  if (impl_->serial) return *this;
  IMSRG_CHECK(color >= 0, "split: color must be non-negative");
  MPI_Comm sub = MPI_COMM_NULL;
  check_mpi(MPI_Comm_split(impl_->comm, color, key, &sub), "MPI_Comm_split");
  return Communicator(wrap(sub, true));
}

Communicator Communicator::duplicate() const {
  if (impl_->serial) return *this;
  MPI_Comm dup = MPI_COMM_NULL;
  check_mpi(MPI_Comm_dup(impl_->comm, &dup), "MPI_Comm_dup");
  return Communicator(wrap(dup, true));
}

}  // namespace imsrg::comm
