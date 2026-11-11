#pragma once
/// @file
/// Process-wide execution runtime: RAII guard around Kokkos initialization (design/10 section 4,
/// design/12 section 2). This header does not include Kokkos.
///
/// MPI is NOT initialized here: `comm::Environment` does that. Create the `comm::Environment`
/// BEFORE the exec `Runtime` and destroy it AFTER (declare it first in the enclosing scope).
/// The combined `imsrg::Runtime` of design/10 section 4 composes both in a higher module.

#include <string>

namespace imsrg::exec {

/// Options of the execution runtime. A value of -1 selects the Kokkos default (which honours
/// `OMP_NUM_THREADS`, `--kokkos-*` arguments, and the environment).
struct RuntimeOptions {
  int num_threads = -1;              ///< Host threads for the OpenMP/Threads backends.
  int device_id = -1;                ///< Accelerator index for CUDA/HIP backends.
  bool print_configuration = false;  ///< Print the Kokkos configuration to stdout on start.
};

/// RAII guard: initializes Kokkos on construction, fences and finalizes it on destruction.
/// Non-copyable and non-movable. Kokkos can be initialized once per process, so a second
/// `Runtime` (also after the first was destroyed) throws `RuntimeError`.
class Runtime {
 public:
  /// Initializes Kokkos with `options`.
  /// @throws RuntimeError if Kokkos is already initialized or has been finalized.
  explicit Runtime(const RuntimeOptions& options = {});

  /// Initializes Kokkos with `options` and the `--kokkos-*` command-line arguments, which are
  /// removed from `argv` (`argc` is updated); explicit arguments win over `options`.
  /// @throws RuntimeError as above.
  Runtime(int& argc, char** argv, const RuntimeOptions& options = {});

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  /// Fences all execution spaces and finalizes Kokkos.
  ~Runtime();

  /// True while Kokkos is initialized (between construction and destruction of the `Runtime`).
  static bool is_initialized() noexcept;

  /// Name of the default execution space, e.g. "Serial", "OpenMP", "Cuda", "HIP".
  static std::string execution_space_name();

  /// Maximum concurrency (threads) of the default execution space; at least 1.
  static int concurrency();
};

/// True if memory allocated in the default (device) memory space can be read by host code, so
/// that `DeviceBuffer` host mirrors alias the device data.
bool default_memory_is_host_accessible() noexcept;

/// Opens a named profiling region (Kokkos Tools). Matches the hook type `void (*)(const char*)`
/// of base's TimerRegistry. Requires an initialized Runtime.
void push_profiling_region(const char* name);

/// Closes the innermost profiling region. Matches `void (*)()`.
void pop_profiling_region();

}  // namespace imsrg::exec
