// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Communicator backed by MPI. The serial communicator (Communicator::Serial())
// makes no MPI calls.
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/comm/detail/chunks.h"

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
void CheckMpi(int rc, const char* call) {
  if (rc == MPI_SUCCESS) return;
  char text[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  if (MPI_Error_string(rc, text, &len) != MPI_SUCCESS) len = 0;
  throw RuntimeError(std::string("comm: ") + call + " failed: " +
                     std::string(text, static_cast<std::size_t>(len)));
}

std::shared_ptr<const Communicator::Impl> Wrap(MPI_Comm comm, bool owned) {
  auto impl = std::make_shared<Communicator::Impl>();
  impl->comm = comm;
  impl->owned = owned;
  CheckMpi(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN),
           "MPI_Comm_set_errhandler");
  CheckMpi(MPI_Comm_rank(comm, &impl->rank), "MPI_Comm_rank");
  CheckMpi(MPI_Comm_size(comm, &impl->size), "MPI_Comm_size");
  return impl;
}

MPI_Datatype MpiType(const double*) { return MPI_DOUBLE; }
MPI_Datatype MpiType(const float*) { return MPI_FLOAT; }
MPI_Datatype MpiType(const std::int32_t*) { return MPI_INT32_T; }
MPI_Datatype MpiType(const std::int64_t*) { return MPI_INT64_T; }
MPI_Datatype MpiType(const std::byte*) { return MPI_BYTE; }

/// Calls `f(offset, count)` once per chunk of a message of `n` elements, in
/// order; `count` always fits in int (at most kMaxMessageElements).
template <class F>
void ForEachChunk(std::size_t n, F f) {
  for (const detail::Chunk& c : detail::ChunkRanges(
           static_cast<std::int64_t>(n), detail::kMaxMessageElements)) {
    f(static_cast<std::size_t>(c.offset), static_cast<int>(c.count));
  }
}

template <class T>
void ReduceInPlace(const Communicator::Impl& impl, std::span<T> data, MPI_Op op,
                   const char* name) {
  if (impl.serial || data.empty()) return;
  ForEachChunk(data.size(), [&](std::size_t offset, int count) {
    CheckMpi(MPI_Allreduce(MPI_IN_PLACE, data.data() + offset, count,
                           MpiType(data.data()), op, impl.comm),
             name);
  });
}

template <class T>
T ReduceScalar(const Communicator::Impl& impl, T x, MPI_Op op,
               const char* name) {
  ReduceInPlace(impl, std::span<T>(&x, 1), op, name);
  return x;
}

template <class T>
void Bcast(const Communicator::Impl& impl, std::span<T> data, int root) {
  IMSRG_CHECK(root >= 0 && root < impl.size, "broadcast: root out of range");
  if (impl.serial || data.empty()) return;
  ForEachChunk(data.size(), [&](std::size_t offset, int count) {
    CheckMpi(MPI_Bcast(data.data() + offset, count, MpiType(data.data()), root,
                       impl.comm),
             "MPI_Bcast");
  });
}

void ReduceToRoot(const Communicator::Impl& impl, std::span<double> data,
                  int root) {
  IMSRG_CHECK(root >= 0 && root < impl.size, "reduce: root out of range");
  if (impl.serial || data.empty()) return;
  const bool is_root = impl.rank == root;
  ForEachChunk(data.size(), [&](std::size_t offset, int count) {
    double* p = data.data() + offset;
    CheckMpi(MPI_Reduce(is_root ? MPI_IN_PLACE : p, is_root ? p : nullptr,
                        count, MPI_DOUBLE, MPI_SUM, root, impl.comm),
             "MPI_Reduce");
  });
}

template <class T>
std::vector<T> GatherAll(const Communicator::Impl& impl, T x) {
  std::vector<T> out(static_cast<std::size_t>(impl.size), x);
  if (impl.serial) return out;
  CheckMpi(
      MPI_Allgather(&x, 1, MpiType(&x), out.data(), 1, MpiType(&x), impl.comm),
      "MPI_Allgather");
  return out;
}

}  // namespace

Communicator Communicator::Serial() {
  static const auto* const kImpl = new std::shared_ptr<const Impl>([] {
    auto p = std::make_shared<Impl>();
    p->serial = true;
    return p;
  }());
  return Communicator(*kImpl);
}

Communicator Communicator::World() {
  static const auto* const kImpl =
      new std::shared_ptr<const Impl>(Wrap(MPI_COMM_WORLD, false));
  return Communicator(*kImpl);
}

int Communicator::rank() const noexcept { return impl_->rank; }
int Communicator::size() const noexcept { return impl_->size; }
bool Communicator::is_serial() const noexcept { return impl_->serial; }

void Communicator::Barrier() const {
  if (impl_->serial) return;
  CheckMpi(MPI_Barrier(impl_->comm), "MPI_Barrier");
}

#define IMSRG_COMM_REDUCE(T)                                 \
  void Communicator::AllreduceSum(std::span<T> d) const {    \
    ReduceInPlace(*impl_, d, MPI_SUM, "MPI_Allreduce(sum)"); \
  }                                                          \
  void Communicator::AllreduceMax(std::span<T> d) const {    \
    ReduceInPlace(*impl_, d, MPI_MAX, "MPI_Allreduce(max)"); \
  }                                                          \
  void Communicator::AllreduceMin(std::span<T> d) const {    \
    ReduceInPlace(*impl_, d, MPI_MIN, "MPI_Allreduce(min)"); \
  }
IMSRG_COMM_REDUCE(double)
IMSRG_COMM_REDUCE(float)
IMSRG_COMM_REDUCE(std::int32_t)
IMSRG_COMM_REDUCE(std::int64_t)
#undef IMSRG_COMM_REDUCE

double Communicator::AllreduceSum(double x) const {
  return ReduceScalar(*impl_, x, MPI_SUM, "MPI_Allreduce(sum)");
}
std::int64_t Communicator::AllreduceSum(std::int64_t x) const {
  return ReduceScalar(*impl_, x, MPI_SUM, "MPI_Allreduce(sum)");
}
double Communicator::AllreduceMax(double x) const {
  return ReduceScalar(*impl_, x, MPI_MAX, "MPI_Allreduce(max)");
}
std::int64_t Communicator::AllreduceMax(std::int64_t x) const {
  return ReduceScalar(*impl_, x, MPI_MAX, "MPI_Allreduce(max)");
}
double Communicator::AllreduceMin(double x) const {
  return ReduceScalar(*impl_, x, MPI_MIN, "MPI_Allreduce(min)");
}
std::int64_t Communicator::AllreduceMin(std::int64_t x) const {
  return ReduceScalar(*impl_, x, MPI_MIN, "MPI_Allreduce(min)");
}

void Communicator::Broadcast(std::span<double> d, int root) const {
  Bcast(*impl_, d, root);
}
void Communicator::Broadcast(std::span<float> d, int root) const {
  Bcast(*impl_, d, root);
}
void Communicator::Broadcast(std::span<std::int32_t> d, int root) const {
  Bcast(*impl_, d, root);
}
void Communicator::Broadcast(std::span<std::int64_t> d, int root) const {
  Bcast(*impl_, d, root);
}
void Communicator::Broadcast(std::span<std::byte> d, int root) const {
  Bcast(*impl_, d, root);
}

void Communicator::Broadcast(std::string& s, int root) const {
  IMSRG_CHECK(root >= 0 && root < impl_->size, "broadcast: root out of range");
  if (impl_->serial) return;
  std::int64_t n = static_cast<std::int64_t>(s.size());
  Bcast(*impl_, std::span<std::int64_t>(&n, 1), root);
  if (impl_->rank != root) s.resize(static_cast<std::size_t>(n));
  Bcast(*impl_, std::as_writable_bytes(std::span<char>(s.data(), s.size())),
        root);
}

void Communicator::ReduceSum(std::span<double> data, int root) const {
  ReduceToRoot(*impl_, data, root);
}

void Communicator::ReduceBcastSum(std::span<double> data) const {
  ReduceToRoot(*impl_, data, 0);
  Bcast(*impl_, data, 0);
}

double Communicator::ReduceBcastSum(double x) const {
  ReduceBcastSum(std::span<double>(&x, 1));
  return x;
}

std::vector<double> Communicator::Allgather(double x) const {
  return GatherAll(*impl_, x);
}
std::vector<std::int64_t> Communicator::Allgather(std::int64_t x) const {
  return GatherAll(*impl_, x);
}

void Communicator::Alltoallv(std::span<const double> send,
                             std::span<const int> send_counts,
                             std::span<const int> send_displs,
                             std::span<double> recv,
                             std::span<const int> recv_counts,
                             std::span<const int> recv_displs) const {
  const auto p = static_cast<std::size_t>(impl_->size);
  IMSRG_CHECK(send_counts.size() == p && send_displs.size() == p &&
                  recv_counts.size() == p && recv_displs.size() == p,
              "alltoallv: count/displacement spans must have length size()");
  for (std::size_t r = 0; r < p; ++r) {
    IMSRG_CHECK(send_counts[r] >= 0 && send_displs[r] >= 0 &&
                    recv_counts[r] >= 0 && recv_displs[r] >= 0,
                "alltoallv: negative count or displacement");
    IMSRG_CHECK(static_cast<std::size_t>(send_displs[r]) +
                            static_cast<std::size_t>(send_counts[r]) <=
                        send.size() &&
                    static_cast<std::size_t>(recv_displs[r]) +
                            static_cast<std::size_t>(recv_counts[r]) <=
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
  CheckMpi(MPI_Alltoallv(send.data(), send_counts.data(), send_displs.data(),
                         MPI_DOUBLE, recv.data(), recv_counts.data(),
                         recv_displs.data(), MPI_DOUBLE, impl_->comm),
           "MPI_Alltoallv");
}

Communicator Communicator::Split(int color, int key) const {
  if (impl_->serial) return *this;
  IMSRG_CHECK(color >= 0, "split: color must be non-negative");
  MPI_Comm sub = MPI_COMM_NULL;
  CheckMpi(MPI_Comm_split(impl_->comm, color, key, &sub), "MPI_Comm_split");
  return Communicator(Wrap(sub, true));
}

Communicator Communicator::Duplicate() const {
  if (impl_->serial) return *this;
  MPI_Comm dup = MPI_COMM_NULL;
  CheckMpi(MPI_Comm_dup(impl_->comm, &dup), "MPI_Comm_dup");
  return Communicator(Wrap(dup, true));
}

Communicator Communicator::SplitShared() const {
  if (impl_->serial) return *this;
  MPI_Comm sub = MPI_COMM_NULL;
  CheckMpi(MPI_Comm_split_type(impl_->comm, MPI_COMM_TYPE_SHARED, impl_->rank,
                               MPI_INFO_NULL, &sub),
           "MPI_Comm_split_type");
  return Communicator(Wrap(sub, true));
}

struct SharedMemory::Window {
  std::unique_ptr<std::byte[]> heap;  // serial communicator only
  MPI_Win win = MPI_WIN_NULL;

  Window() = default;
  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  ~Window() {
    if (win == MPI_WIN_NULL) return;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) MPI_Win_free(&win);
  }
};

SharedMemory Communicator::AllocateShared(std::size_t bytes) const {
  SharedMemory m;
  m.window_ = std::make_shared<SharedMemory::Window>();
  m.size_ = bytes;
  if (impl_->serial) {
    m.window_->heap =
        std::make_unique<std::byte[]>(std::max<std::size_t>(1, bytes));
    m.data_ = m.window_->heap.get();
    return m;
  }
  // All memory on rank 0 (at least one byte, so the query returns a valid
  // address), none on the others.
  const auto mine = static_cast<MPI_Aint>(
      impl_->rank == 0 ? std::max<std::size_t>(1, bytes) : 0);
  void* base = nullptr;
  CheckMpi(MPI_Win_allocate_shared(mine, 1, MPI_INFO_NULL, impl_->comm, &base,
                                   &m.window_->win),
           "MPI_Win_allocate_shared");
  MPI_Aint size0 = 0;
  int disp = 0;
  void* p0 = nullptr;
  CheckMpi(MPI_Win_shared_query(m.window_->win, 0, &size0, &disp, &p0),
           "MPI_Win_shared_query");
  m.data_ = static_cast<std::byte*>(p0);
  return m;
}

}  // namespace imsrg::comm
