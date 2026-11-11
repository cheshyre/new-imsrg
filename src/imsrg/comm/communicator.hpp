#pragma once
/// @file
/// `imsrg::comm::Communicator`: the MPI facade (design/10 sections 2 and 4). A communicator is
/// always passed explicitly; nothing in imsrg uses MPI_COMM_WORLD implicitly. This header does not
/// include <mpi.h>. In a build without MPI (`IMSRG_ENABLE_MPI=OFF`) only the serial communicator
/// exists and every collective is a local copy.
///
/// Reductions use MPI's reduction order: results are identical on all ranks, but they are
/// invariant under the rank count only up to floating-point reassociation (design/12 section 6).
/// All collectives must be called by every rank of the communicator in the same order. MPI
/// failures are reported as `imsrg::RuntimeError` carrying the MPI error string.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace imsrg::comm {

class Environment;

/// Cheap, copyable handle to a group of ranks. Copies share the underlying communicator, which
/// is freed (if it was created by `split`/`duplicate`) when the last handle is destroyed, never
/// after MPI_Finalize. Not thread-safe beyond MPI_THREAD_FUNNELED: call from one thread.
class Communicator {
 public:
  /// The serial communicator (same as `serial()`), so option structs can hold a communicator
  /// with a safe default.
  Communicator() : Communicator(serial()) {}

  /// The one-rank communicator (rank 0, size 1). Makes no MPI calls, so it is also usable in MPI
  /// builds without an `Environment`.
  static Communicator serial();

  /// Rank of the calling process in [0, size()).
  int rank() const noexcept;
  /// Number of ranks.
  int size() const noexcept;
  /// True on rank 0.
  bool is_root() const noexcept { return rank() == 0; }
  /// True for the serial communicator (size 1, no MPI involved), including serial split/duplicate.
  bool is_serial() const noexcept;

  /// Blocks until all ranks arrive.
  void barrier() const;

  /// In-place element-wise sum over all ranks; every rank passes the same length.
  void allreduce_sum(std::span<double> data) const;
  /// @copydoc allreduce_sum(std::span<double>) const
  void allreduce_sum(std::span<float> data) const;
  /// @copydoc allreduce_sum(std::span<double>) const
  void allreduce_sum(std::span<std::int32_t> data) const;
  /// @copydoc allreduce_sum(std::span<double>) const
  void allreduce_sum(std::span<std::int64_t> data) const;
  /// In-place element-wise maximum over all ranks.
  void allreduce_max(std::span<double> data) const;
  /// @copydoc allreduce_max(std::span<double>) const
  void allreduce_max(std::span<float> data) const;
  /// @copydoc allreduce_max(std::span<double>) const
  void allreduce_max(std::span<std::int32_t> data) const;
  /// @copydoc allreduce_max(std::span<double>) const
  void allreduce_max(std::span<std::int64_t> data) const;
  /// In-place element-wise minimum over all ranks.
  void allreduce_min(std::span<double> data) const;
  /// @copydoc allreduce_min(std::span<double>) const
  void allreduce_min(std::span<float> data) const;
  /// @copydoc allreduce_min(std::span<double>) const
  void allreduce_min(std::span<std::int32_t> data) const;
  /// @copydoc allreduce_min(std::span<double>) const
  void allreduce_min(std::span<std::int64_t> data) const;

  /// Scalar sum over all ranks.
  double allreduce_sum(double x) const;
  /// Scalar sum over all ranks.
  std::int64_t allreduce_sum(std::int64_t x) const;
  /// Scalar maximum over all ranks.
  double allreduce_max(double x) const;
  /// Scalar maximum over all ranks.
  std::int64_t allreduce_max(std::int64_t x) const;
  /// Scalar minimum over all ranks.
  double allreduce_min(double x) const;
  /// Scalar minimum over all ranks.
  std::int64_t allreduce_min(std::int64_t x) const;

  /// Broadcasts `data` (same length on all ranks) from `root` (precondition: 0 <= root < size()).
  void broadcast(std::span<double> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void broadcast(std::span<float> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void broadcast(std::span<std::int32_t> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void broadcast(std::span<std::int64_t> data, int root) const;
  /// @copydoc broadcast(std::span<double>, int) const
  void broadcast(std::span<std::byte> data, int root) const;
  /// Broadcasts a string from `root`; other ranks' strings are resized and overwritten.
  void broadcast(std::string& s, int root) const;

  /// Gathers one value per rank, ordered by rank, on every rank.
  std::vector<double> allgather(double x) const;
  /// Gathers one value per rank, ordered by rank, on every rank.
  std::vector<std::int64_t> allgather(std::int64_t x) const;

  /// Personalized all-to-all exchange (MPI_Alltoallv). `send_counts[r]` doubles starting at
  /// `send_displs[r]` go to rank r; `recv_counts[r]` starting at `recv_displs[r]` arrive from r.
  /// All four count/displacement spans have length size(); the data spans must cover the ranges
  /// (checked). Counts and displacements are in elements.
  void alltoallv(std::span<const double> send, std::span<const int> send_counts,
                 std::span<const int> send_displs, std::span<double> recv,
                 std::span<const int> recv_counts, std::span<const int> recv_displs) const;

  /// MPI_Comm_split: ranks with equal `color` form a new communicator ordered by `key`
  /// (ties by old rank). Collective.
  Communicator split(int color, int key) const;
  /// MPI_Comm_dup: a communicator over the same ranks with an independent message context.
  Communicator duplicate() const;

  /// Implementation detail (defined in the communicator_*.cpp file of the build).
  struct Impl;

 private:
  friend class Environment;
  explicit Communicator(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
  /// Wraps MPI_COMM_WORLD (serial communicator in a build without MPI). Used by `Environment`.
  static Communicator world_();

  std::shared_ptr<const Impl> impl_;
};

}  // namespace imsrg::comm
