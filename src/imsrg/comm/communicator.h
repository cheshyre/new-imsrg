// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMM_COMMUNICATOR_H_
#define IMSRG_COMM_COMMUNICATOR_H_
/// @file
/// `imsrg::comm::Communicator`: the MPI facade (design/10 sections 2 and 4). A
/// communicator is always passed explicitly; nothing in imsrg uses
/// MPI_COMM_WORLD implicitly. This header does not include <mpi.h>. In a build
/// without MPI (`IMSRG_ENABLE_MPI=OFF`) only the serial communicator exists and
/// every collective is a local copy.
///
/// Reductions use MPI's reduction order: results are identical on all ranks,
/// but they are invariant under the rank count only up to floating-point
/// reassociation (design/12 section 6). All collectives must be called by every
/// rank of the communicator in the same order. Span collectives split
/// messages longer than 2^30 elements into consecutive chunks (one MPI call
/// each, same order on all ranks); `Alltoallv` does not. MPI failures are
/// reported as `imsrg::RuntimeError` carrying the MPI error string.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::comm {

/// Cheap, copyable handle to a group of ranks. Copies share the underlying
/// communicator, which is freed (if it was created by `split`/`duplicate`) when
/// the last handle is destroyed, never after MPI_Finalize. Not thread-safe
/// beyond MPI_THREAD_FUNNELED: call from one thread.
class Communicator {
 public:
  /// The serial communicator (same as `Serial()`), so option structs can hold a
  /// communicator with a safe default.
  Communicator() : Communicator(Serial()) {}

  Communicator(const Communicator&) = default;
  Communicator& operator=(const Communicator&) = default;
  Communicator(Communicator&&) = default;
  Communicator& operator=(Communicator&&) = default;

  /// The one-rank communicator (rank 0, size 1). Makes no MPI calls, so it is
  /// also usable in MPI builds without an `Environment`.
  static Communicator Serial();

  /// Rank of the calling process in [0, size()).
  int rank() const noexcept;
  /// Number of ranks.
  int size() const noexcept;
  /// True on rank 0.
  bool is_root() const noexcept { return rank() == 0; }
  /// True for the serial communicator (size 1, no MPI involved), including
  /// serial split/duplicate.
  bool is_serial() const noexcept;

  /// Blocks until all ranks arrive.
  void Barrier() const;

  /// In-place element-wise sum over all ranks; every rank passes the same
  /// length.
  void AllreduceSum(std::span<double> data) const;
  /// @copydoc AllreduceSum(std::span<double>) const
  void AllreduceSum(std::span<float> data) const;
  /// @copydoc AllreduceSum(std::span<double>) const
  void AllreduceSum(std::span<std::int32_t> data) const;
  /// @copydoc AllreduceSum(std::span<double>) const
  void AllreduceSum(std::span<std::int64_t> data) const;
  /// In-place element-wise maximum over all ranks.
  void AllreduceMax(std::span<double> data) const;
  /// @copydoc AllreduceMax(std::span<double>) const
  void AllreduceMax(std::span<float> data) const;
  /// @copydoc AllreduceMax(std::span<double>) const
  void AllreduceMax(std::span<std::int32_t> data) const;
  /// @copydoc AllreduceMax(std::span<double>) const
  void AllreduceMax(std::span<std::int64_t> data) const;
  /// In-place element-wise minimum over all ranks.
  void AllreduceMin(std::span<double> data) const;
  /// @copydoc AllreduceMin(std::span<double>) const
  void AllreduceMin(std::span<float> data) const;
  /// @copydoc AllreduceMin(std::span<double>) const
  void AllreduceMin(std::span<std::int32_t> data) const;
  /// @copydoc AllreduceMin(std::span<double>) const
  void AllreduceMin(std::span<std::int64_t> data) const;

  /// Scalar sum over all ranks.
  double AllreduceSum(double x) const;
  /// Scalar sum over all ranks.
  std::int64_t AllreduceSum(std::int64_t x) const;
  /// Scalar maximum over all ranks.
  double AllreduceMax(double x) const;
  /// Scalar maximum over all ranks.
  std::int64_t AllreduceMax(std::int64_t x) const;
  /// Scalar minimum over all ranks.
  double AllreduceMin(double x) const;
  /// Scalar minimum over all ranks.
  std::int64_t AllreduceMin(std::int64_t x) const;

  /// Broadcasts `data` (same length on all ranks) from `root` (precondition: 0
  /// <= root < size()).
  void Broadcast(std::span<double> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void Broadcast(std::span<float> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void Broadcast(std::span<std::int32_t> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void Broadcast(std::span<std::int64_t> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void Broadcast(std::span<std::byte> data, int root) const;
  /// Broadcasts a string from `root`; other ranks' strings are resized and
  /// overwritten.
  void Broadcast(std::string& s, int root) const;

  /// Collective. Element-wise sum over all ranks into `data` of rank `root`
  /// (MPI_Reduce, in place on the root); on the other ranks `data` is left
  /// unchanged. Every rank passes the same length. Precondition: 0 <= root <
  /// size(). Messages longer than 2^30 elements are sent in chunks.
  void ReduceSum(std::span<double> data, int root) const;
  /// Collective. Sums `data` over all ranks to rank 0 and broadcasts the result
  /// from rank 0, so every rank ends with rank 0's bits. Unlike MPI_Allreduce,
  /// this guarantees bitwise-identical results on all ranks (design M12b-Q2).
  /// Every rank passes the same length.
  void ReduceBcastSum(std::span<double> data) const;
  /// Collective. Scalar version of `ReduceBcastSum(std::span<double>)`: the
  /// sum over all ranks, bitwise identical on every rank.
  double ReduceBcastSum(double x) const;

  /// Gathers one value per rank, ordered by rank, on every rank.
  std::vector<double> Allgather(double x) const;
  /// Gathers one value per rank, ordered by rank, on every rank.
  std::vector<std::int64_t> Allgather(std::int64_t x) const;

  /// Personalized all-to-all exchange (MPI_Alltoallv). `send_counts[r]` doubles
  /// starting at `send_displs[r]` go to rank r; `recv_counts[r]` starting at
  /// `recv_displs[r]` arrive from r. All four count/displacement spans have
  /// length size(); the data spans must cover the ranges (checked). Counts and
  /// displacements are in elements and are `int`: per-peer counts and
  /// displacements must fit in int (checked), as there is no chunking here.
  void Alltoallv(std::span<const double> send, std::span<const int> send_counts,
                 std::span<const int> send_displs, std::span<double> recv,
                 std::span<const int> recv_counts,
                 std::span<const int> recv_displs) const;

  /// MPI_Comm_split: ranks with equal `color` form a new communicator ordered
  /// by `key` (ties by old rank). Collective.
  Communicator Split(int color, int key) const;
  /// MPI_Comm_dup: a communicator over the same ranks with an independent
  /// message context.
  Communicator Duplicate() const;

  /// Implementation detail (defined in the communicator_*.cc file of the
  /// build).
  struct Impl;

 private:
  friend class Environment;
  explicit Communicator(std::shared_ptr<const Impl> impl) noexcept
      : impl_(std::move(impl)) {}
  /// Wraps MPI_COMM_WORLD (serial communicator in a build without MPI). Used by
  /// `Environment`.
  static Communicator World();

  std::shared_ptr<const Impl> impl_;
};

}  // namespace imsrg::comm

#endif  // IMSRG_COMM_COMMUNICATOR_H_
