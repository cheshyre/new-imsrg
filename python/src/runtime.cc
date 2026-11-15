// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// imsrg.Runtime: process singleton for Kokkos (and optionally MPI), design/18
/// section 4.3.

#include "imsrg/exec/runtime.h"

#include <memory>
#include <mutex>

#include <nanobind/stl/string.h>

#include "imsrg/base/error.h"
#include "imsrg/comm/environment.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

// Process singleton (design/18 section 4.3). Heap-allocated and never deleted:
// its members are torn down explicitly by FinalizeLocked(), never by static
// destruction order.
struct State {
  std::mutex mutex;
  std::unique_ptr<comm::Environment> env;
  std::unique_ptr<exec::Runtime> rt;
  bool finalized = false;
};

State& GetState() {
  static State* s = new State;
  return *s;
}

void InitLocked(State& s, const exec::RuntimeOptions& opt, bool mpi) {
  if (s.finalized)
    throw imsrg::RuntimeError(
        "imsrg Runtime was finalized; it cannot be restarted");
  if (s.rt) throw imsrg::RuntimeError("imsrg Runtime is already initialized");
  if (mpi) s.env = std::make_unique<comm::Environment>();
  s.rt = std::make_unique<exec::Runtime>(opt);
}

void FinalizeLocked(State& s, bool force) {
  if (!s.rt) return;
  if (CallsInFlight() > 0)
    throw imsrg::RuntimeError(
        "cannot finalize the imsrg Runtime while a call is running (another "
        "thread, or this one from inside a callback)");
  ClearWorkspaceCache();
  if (LiveObjects() > 0) {
    if (!force)
      throw imsrg::RuntimeError(
          "cannot finalize the imsrg Runtime while Operator or solver objects "
          "are alive (Runtime.finalize(force=True) frees their device data)");
    const int blocked = ReleaseAllLive();
    if (blocked < 0)
      throw imsrg::RuntimeError("cannot force-finalize: a call is running");
    if (blocked > 0)
      throw imsrg::RuntimeError(
          "cannot force-finalize: " + std::to_string(blocked) +
          " object(s) have numpy views alive; delete the views first");
  }
  s.rt.reset();
  s.env.reset();
  s.finalized = true;
}

struct RuntimeHandle {};

}  // namespace

void EnsureRuntime() {
  auto& s = GetState();
  std::lock_guard lock(s.mutex);
  if (s.rt) return;
  InitLocked(s, {}, false);
}

void BindRuntime(nb::module_& m) {
  nb::class_<RuntimeHandle>(m, "Runtime",
                            "Process-wide Kokkos (and optional MPI) runtime. "
                            "Constructing it initializes explicitly; "
                            "otherwise it is initialized lazily by the first "
                            "ModelSpace or Operator. Usable as a "
                            "context manager (finalizes on exit).")
      .def(
          "__init__",
          [](RuntimeHandle* self, int num_threads, int device_id, bool mpi,
             bool print_configuration) {
            auto& s = GetState();
            {
              std::lock_guard lock(s.mutex);
              InitLocked(s, {num_threads, device_id, print_configuration}, mpi);
            }
            new (self) RuntimeHandle();
          },
          nb::arg("num_threads") = -1, nb::arg("device_id") = -1,
          nb::arg("mpi") = false, nb::arg("print_configuration") = false)
      .def("__enter__", [](nb::object self) { return self; })
      .def("__exit__",
           [](RuntimeHandle&, nb::args) {
             auto& s = GetState();
             std::lock_guard lock(s.mutex);
             FinalizeLocked(s, false);
           })
      .def_static("ensure", &EnsureRuntime,
                  "Initializes with defaults if not yet initialized.")
      .def_static(
          "finalize",
          [](bool force) {
            auto& s = GetState();
            std::lock_guard lock(s.mutex);
            FinalizeLocked(s, force);
          },
          nb::arg("force") = false,
          "Finalizes Kokkos (and MPI). Raises RuntimeError if Operators or "
          "solvers are alive, unless force=True: then their device data are "
          "freed first (later use of them raises StaleHostError; objects "
          "with numpy views alive block it). No re-init after.")
      .def_static(
          "is_initialized", [] { return exec::Runtime::IsInitialized(); },
          "True while Kokkos is initialized.")
      .def_static(
          "execution_space", [] { return exec::Runtime::ExecutionSpaceName(); },
          "Name of the default execution space (OpenMP, Serial, Cuda, ...).")
      .def_static(
          "concurrency", [] { return exec::Runtime::Concurrency(); },
          "Maximum concurrency (threads) of the default execution space.")
      .def_static(
          "mpi_enabled", [] { return comm::Environment::MpiEnabled(); },
          "True if the build has MPI.");

  // Called by imsrg's atexit handler after gc.collect(): frees the device data
  // of objects still alive (M11-Q1, D41) and finalizes, unless numpy views of
  // them are alive (then Kokkos is left to process exit).
  m.def("_atexit", [] {
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    ClearWorkspaceCache();
    if (s.rt && CallsInFlight() == 0 && ReleaseAllLive() == 0)
      FinalizeLocked(s, false);
  });
  m.def(
      "live_objects", [] { return LiveObjects(); },
      "Number of live objects (operators, solvers, workspaces) that own "
      "device data.");
}

}  // namespace imsrg::py
