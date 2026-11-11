/// @file
/// imsrg.Runtime: process singleton for Kokkos (and optionally MPI), design/18 section 4.3.

#include <memory>
#include <mutex>

#include <nanobind/stl/string.h>

#include "bindings.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/comm/environment.hpp"
#include "imsrg/exec/runtime.hpp"

namespace imsrg::py {
namespace {

// Process singleton (design/18 section 4.3). Heap-allocated and never deleted: its members are
// torn down explicitly by finalize_locked(), never by static destruction order.
struct State {
  std::mutex mutex;
  std::unique_ptr<comm::Environment> env;
  std::unique_ptr<exec::Runtime> rt;
  bool finalized = false;
};

State& state() {
  static State* s = new State;
  return *s;
}

void init_locked(State& s, const exec::RuntimeOptions& opt, bool mpi) {
  if (s.finalized) throw imsrg::RuntimeError("imsrg Runtime was finalized; it cannot be restarted");
  if (s.rt) throw imsrg::RuntimeError("imsrg Runtime is already initialized");
  if (mpi) s.env = std::make_unique<comm::Environment>();
  s.rt = std::make_unique<exec::Runtime>(opt);
}

void finalize_locked(State& s) {
  if (!s.rt) return;
  if (live_operators().load() > 0)
    throw imsrg::RuntimeError("cannot finalize the imsrg Runtime while Operator objects are alive");
  s.rt.reset();
  s.env.reset();
  s.finalized = true;
}

struct RuntimeHandle {};

}  // namespace

void ensure_runtime() {
  auto& s = state();
  std::lock_guard lock(s.mutex);
  if (s.rt) return;
  init_locked(s, {}, false);
}

void bind_runtime(nb::module_& m) {
  nb::class_<RuntimeHandle>(
      m, "Runtime",
      "Process-wide Kokkos (and optional MPI) runtime. Constructing it initializes explicitly; "
      "otherwise it is initialized lazily by the first ModelSpace or Operator. Usable as a "
      "context manager (finalizes on exit).")
      .def(
          "__init__",
          [](RuntimeHandle* self, int num_threads, int device_id, bool mpi,
             bool print_configuration) {
            auto& s = state();
            {
              std::lock_guard lock(s.mutex);
              init_locked(s, {num_threads, device_id, print_configuration}, mpi);
            }
            new (self) RuntimeHandle();
          },
          nb::arg("num_threads") = -1, nb::arg("device_id") = -1, nb::arg("mpi") = false,
          nb::arg("print_configuration") = false)
      .def("__enter__", [](nb::object self) { return self; })
      .def("__exit__", [](RuntimeHandle&, nb::args) {
        auto& s = state();
        std::lock_guard lock(s.mutex);
        finalize_locked(s);
      })
      .def_static("ensure", &ensure_runtime, "Initializes with defaults if not yet initialized.")
      .def_static(
          "finalize",
          [] {
            auto& s = state();
            std::lock_guard lock(s.mutex);
            finalize_locked(s);
          },
          "Finalizes Kokkos (and MPI); raises if Operators are alive. No re-init after.")
      .def_static("is_initialized", [] { return exec::Runtime::is_initialized(); })
      .def_static("execution_space", [] { return exec::Runtime::execution_space_name(); })
      .def_static("concurrency", [] { return exec::Runtime::concurrency(); })
      .def_static("mpi_enabled", [] { return comm::Environment::mpi_enabled(); });

  // Called by imsrg's atexit handler after gc.collect(): finalize only if nothing is alive.
  m.def("_atexit", [] {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    if (s.rt && live_operators().load() == 0) finalize_locked(s);
  });
}

}  // namespace imsrg::py
