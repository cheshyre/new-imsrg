#pragma once
/// @file
/// `imsrg::comm::Environment`: RAII owner of the MPI runtime (design/10 section 4).
///
/// Order of construction in `main`: create the `Environment` BEFORE `exec::Runtime` (Kokkos) and
/// destroy it AFTER it (declare it first, so normal scope rules give this order).
/// In a build without MPI everything here is a no-op and `world()` is the serial communicator.

#include "imsrg/comm/communicator.hpp"

namespace imsrg::comm {

/// Initializes MPI with MPI_THREAD_FUNNELED unless it is already initialized (then it does not
/// own the runtime and does not finalize it). The destructor calls MPI_Finalize only if this
/// object initialized MPI and MPI is not yet finalized. Non-copyable, non-movable.
class Environment {
 public:
  /// Initializes MPI without command-line arguments. Throws `RuntimeError` on failure.
  Environment();
  /// Initializes MPI, letting it consume/adjust the command line. Throws `RuntimeError` on failure.
  Environment(int& argc, char**& argv);
  ~Environment();
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(Environment&&) = delete;
  Environment& operator=(Environment&&) = delete;

  /// A communicator over all ranks (MPI_COMM_WORLD with MPI_ERRORS_RETURN); the serial
  /// communicator without MPI.
  Communicator world() const;

  /// True if the library was built with MPI.
  static bool mpi_enabled() noexcept;
  /// True if MPI is initialized and not finalized (always false without MPI).
  static bool is_initialized() noexcept;
  /// The provided thread level (MPI_THREAD_* value); 0 without MPI.
  int thread_level() const noexcept { return thread_level_; }

 private:
  void init(int* argc, char*** argv);
  [[maybe_unused]] bool owns_ = false;  // unused in builds without MPI
  int thread_level_ = 0;
};

}  // namespace imsrg::comm
