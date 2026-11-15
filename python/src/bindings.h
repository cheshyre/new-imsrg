// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef PYTHON_SRC_BINDINGS_H_
#define PYTHON_SRC_BINDINGS_H_
/// @file
/// Shared declarations of the nanobind bindings (design/18). No Kokkos headers.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <nanobind/nanobind.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/operator.h"

namespace imsrg {

/// A file that does not exist (Python: `imsrg.InputFileNotFoundError`, also a
/// `FileNotFoundError`).
class MissingFileError : public InputError {
 public:
  using InputError::InputError;
};

/// A file that exists but has malformed or inconsistent content (Python:
/// `imsrg.FileFormatError`).
class FileFormatError : public InputError {
 public:
  using InputError::InputError;
};

/// Operands on different model spaces (Python: `imsrg.ModelSpaceError`).
class ModelSpaceError : public Error {
 public:
  using Error::Error;
};

/// Raised when a host view is requested of an operator whose host mirror is
/// stale, or when the device data of an operator were released by a forced
/// `Runtime.finalize` (design/18 section 2.2). Python: `imsrg.StaleHostError`.
class StaleHostError : public Error {
 public:
  using Error::Error;
};

}  // namespace imsrg

namespace imsrg::py {

/// Python-side ModelSpace: a handle to the immutable C++ model space (value
/// semantics, shared).
struct PyModelSpace {
  /// Wraps a model space.
  explicit PyModelSpace(std::shared_ptr<const basis::ModelSpace> p)
      : ms(std::move(p)) {}
  PyModelSpace(const PyModelSpace&) = default;
  PyModelSpace(PyModelSpace&&) = default;
  PyModelSpace& operator=(const PyModelSpace&) = default;
  PyModelSpace& operator=(PyModelSpace&&) = default;
  /// Drops the handle; then frees commutator workspaces nobody else uses, so
  /// that `del ms` frees the model space (design/18 section 4.3).
  ~PyModelSpace();
  std::shared_ptr<const basis::ModelSpace> ms;
};

/// Frees the cached commutator workspaces whose model space nothing else
/// refers to.
void PurgeWorkspaceCache();

/// Number of calls that run without the GIL at the moment. Thread-safe.
int CallsInFlight() noexcept;

/// Marks a call that runs without the GIL (or with Kokkos kernels) as in
/// flight, so that a forced finalize refuses while it runs.
/// @throws StaleHostError if a forced release is in progress.
class InFlight {
 public:
  InFlight();
  InFlight(const InFlight&) = delete;
  InFlight& operator=(const InFlight&) = delete;
  ~InFlight();
};

/// `InFlight` plus `nb::gil_scoped_release`: declare it where the binding
/// releases the GIL (the in-flight mark is taken first and dropped last).
class GilFree {
 public:
  GilFree() = default;
  GilFree(const GilFree&) = delete;
  GilFree& operator=(const GilFree&) = delete;

 private:
  InFlight in_flight_;
  nanobind::gil_scoped_release release_;
};

/// A per-object busy flag (design/18 section 4.1): the object may be used by
/// one thread at a time; re-entry from the same thread (a callback of the
/// running solve) is allowed.
class BusyFlag {
 public:
  BusyFlag() = default;
  BusyFlag(const BusyFlag&) = delete;
  BusyFlag& operator=(const BusyFlag&) = delete;

  /// Enters; @throws RuntimeError if another thread is inside.
  void Enter();
  /// Leaves.
  void Leave() noexcept;

 private:
  std::mutex mutex_;
  std::thread::id owner_;
  int depth_ = 0;
};

/// RAII `BusyFlag::Enter`/`Leave`.
class BusyScope {
 public:
  explicit BusyScope(BusyFlag& f) : flag_(f) { flag_.Enter(); }
  BusyScope(const BusyScope&) = delete;
  BusyScope& operator=(const BusyScope&) = delete;
  ~BusyScope() { flag_.Leave(); }

 private:
  BusyFlag& flag_;
};

/// Runs Python's signal handlers; if one raised (Ctrl-C), throws the Python
/// error. Call with the GIL held, at a safe point (a step boundary).
void ThrowIfSignalPending();

/// Which side of an operator holds the current data (design/18 section 2.2).
/// Each Python `Operator` carries one of these.
///
/// State machine (the transitions are the contract tested in
/// tests/python/test_residency.py):
///   * a zero operator made by the constructor: kBoth;
///   * every result of a library call (arithmetic, normal ordering, readers,
///     HF, commutators, flow, ...): kDeviceCurrent;
///   * `sync_to_host()`: kDeviceCurrent -> kBoth (otherwise unchanged);
///   * `sync_to_device()`: kHostCurrent -> kBoth (otherwise unchanged);
///   * a read-only host view needs host data: raises StaleHostError in
///     kDeviceCurrent (when residency is enforced), leaves the state alone;
///   * a writable host view needs host data as well and sets kHostCurrent;
///   * a library call that takes the operator as input first pushes it to the
///     device if it is kHostCurrent (-> kBoth); a call that modifies it in
///     place leaves it kDeviceCurrent;
///   * `set_zero()` -> kDeviceCurrent; `copy` of an operator inherits the state
///     (after a push if kHostCurrent).
enum class Residency : std::uint8_t {
  kHostCurrent,    ///< Host mirror has edits the device data lack.
  kDeviceCurrent,  ///< Device data are current, the host mirror may be stale.
  kBoth            ///< Both sides hold the same data.
};

/// Whether a stale host view raises (design/18 section 2.2).
enum class ResidencyPolicy : std::uint8_t {
  kAuto,     ///< Strict if the device memory is not host accessible (GPU),
             ///< implicit syncing if it is (Serial/OpenMP: a fence only).
  kStrict,   ///< Always raise StaleHostError on a stale view (portable
             ///< scripts; the setting of the residency tests).
  kImplicit  ///< Never raise: sync implicitly (opt-in; hides device-to-host
             ///< copies).
};

/// Sets the process-wide policy.
void SetResidencyPolicy(ResidencyPolicy p);
/// The process-wide policy (default kAuto; the environment variable
/// `IMSRG_RESIDENCY=strict|implicit|auto` sets the initial value).
ResidencyPolicy GetResidencyPolicy();
/// True if a stale host view currently raises.
bool ResidencyEnforced();

/// An object that owns device (Kokkos) data and registers itself in the live
/// registry, so that `Runtime.finalize(force=True)` and the exit hook can free
/// it (M11-Q1 answer of the owner, D41).
class Releasable {
 public:
  Releasable();
  Releasable(const Releasable&);
  Releasable& operator=(const Releasable&) = default;
  virtual ~Releasable();
  /// False while numpy views of the object's storage are alive: freeing it
  /// would leave them dangling.
  virtual bool CanRelease() const noexcept = 0;
  /// Frees the device data; later use raises StaleHostError. Precondition:
  /// CanRelease().
  virtual void Release() noexcept = 0;
  /// True once `Release()` ran (set by `ReleaseAllLive`).
  bool released = false;
};

/// Number of live objects that still own device data (not released).
/// Thread-safe.
int LiveObjects() noexcept;

/// Frees the device data of every live object (the force-finalize of
/// M11-Q1). Returns the number of objects that could not be freed because
/// numpy views of them are alive (nothing is freed in that case: all or
/// nothing). Thread-safe.
int ReleaseAllLive() noexcept;

/// An `op::Operator` that counts itself as a live object and carries its
/// residency state.
struct PyOperator : op::Operator, Releasable {
  /// Wraps (moves) a library result: the device data are current.
  explicit PyOperator(op::Operator&& o, Residency r = Residency::kDeviceCurrent)
      : op::Operator(std::move(o)), residency(r) {}
  /// Copies an operator (library result, device current).
  explicit PyOperator(const op::Operator& o,
                      Residency r = Residency::kDeviceCurrent)
      : op::Operator(o), residency(r) {}
  PyOperator(const PyOperator& o)
      : op::Operator(o), Releasable(o), residency(o.residency) {}
  PyOperator(PyOperator&& o) noexcept
      : op::Operator(std::move(o)), Releasable(o), residency(o.residency) {}
  PyOperator& operator=(const PyOperator& o) {
    op::Operator::operator=(o);
    residency = o.residency;
    return *this;
  }
  PyOperator& operator=(PyOperator&& o) noexcept {
    op::Operator::operator=(std::move(o));
    residency = o.residency;
    return *this;
  }
  ~PyOperator() override = default;

  bool CanRelease() const noexcept override { return views->load() == 0; }
  void Release() noexcept override { op::Operator::operator=(op::Operator()); }

  Residency residency = Residency::kBoth;  ///< See `Residency`.
  /// Number of live numpy views of this operator's host storage.
  std::shared_ptr<std::atomic<int>> views =
      std::make_shared<std::atomic<int>>(0);
  /// Number of those views that are writable: while one lives the operator
  /// stays HostCurrent (every compute call pushes first).
  std::shared_ptr<std::atomic<int>> writable_views =
      std::make_shared<std::atomic<int>>(0);
};

/// Checks that the device data were not released; returns `o`.
/// @throws StaleHostError if `Runtime.finalize(force=True)` freed them.
const PyOperator& Alive(const PyOperator& o);

/// Prepares `o` as the input of a library call: pushes host edits to the device
/// if the state is kHostCurrent (then kBoth) and returns the C++ operator.
/// @throws StaleHostError if released.
const op::Operator& Input(const PyOperator& o);

/// Prepares host access to `o` (a view of the host mirror). Raises
/// StaleHostError if the state is kDeviceCurrent and residency is enforced;
/// otherwise syncs implicitly. `writable` sets the state to kHostCurrent.
/// @throws StaleHostError as described, or if released.
void PrepareHostAccess(PyOperator& o, bool writable);

/// Makes a library result the Python object: state kDeviceCurrent.
inline PyOperator Result(op::Operator&& o) { return PyOperator(std::move(o)); }

/// A guard object for the ndarray `owner` of a numpy view of `self`: keeps the
/// Python operator alive and counts the view in `PyOperator::views` (and
/// `writable_views`).
nanobind::object MakeViewGuard(nanobind::handle self, bool writable = false);

/// Records that `o` was modified in place by a library call: DeviceCurrent, or
/// (live writable view) the host mirror is refreshed and the state stays
/// HostCurrent so the view's later edits are pushed.
void MarkModifiedInPlace(PyOperator& o);

/// @throws ModelSpaceError if `a` and `b` are not on the same model space.
void RequireSameModelSpace(const op::Operator& a, const op::Operator& b);

/// @throws MissingFileError if `path` does not exist.
void RequireFile(const std::filesystem::path& path);

/// Runs a reader on `path`: a missing file raises MissingFileError, any other
/// InputError of the reader (malformed or inconsistent content) is reported as
/// FileFormatError with the same message.
template <class F>
auto GuardRead(const std::filesystem::path& path, F&& f) -> decltype(f()) {
  RequireFile(path);
  try {
    return f();
  } catch (const MissingFileError&) {
    throw;
  } catch (const ConfigError&) {
    throw;
  } catch (const InputError& e) {
    throw FileFormatError(e.what());
  }
}

/// A commutator workspace (Kokkos scratch buffers and tables of one model
/// space), shared by the commutator and BCH bindings. Not thread safe: users
/// hold `mutex()` during a call.
class PyWorkspace : public Releasable {
 public:
  /// Builds the workspace (needs the Runtime).
  explicit PyWorkspace(std::shared_ptr<const basis::ModelSpace> ms)
      : ms_(ms), ws_(std::make_unique<commutator::Workspace>(std::move(ms))) {}
  PyWorkspace(const PyWorkspace&) = delete;
  PyWorkspace& operator=(const PyWorkspace&) = delete;
  bool CanRelease() const noexcept override { return true; }
  void Release() noexcept override { ws_.reset(); }
  /// The workspace. @throws StaleHostError if released.
  commutator::Workspace& ws() {
    if (!ws_) throw StaleHostError("the commutator workspace was released");
    return *ws_;
  }
  /// The model space the workspace was built for.
  const std::shared_ptr<const basis::ModelSpace>& model_space() const {
    return ms_;
  }
  /// Guards the scratch buffers.
  std::mutex& mutex() { return mutex_; }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::unique_ptr<commutator::Workspace> ws_;
  std::mutex mutex_;
};

/// The workspace for `ms` from a small process-wide cache (built on first use).
/// The cache is dropped by `ClearWorkspaceCache`.
std::shared_ptr<PyWorkspace> WorkspaceFor(
    const std::shared_ptr<const basis::ModelSpace>& ms);

/// Drops the cached commutator workspaces (they hold Kokkos buffers); called
/// before the Runtime is finalized. Workspaces a Python object still refers to
/// stay alive.
void ClearWorkspaceCache();

/// Initializes the process Runtime (Kokkos, optionally MPI) if it is not yet;
/// idempotent. Called by every constructor of a ModelSpace or Operator (lazy
/// init, design/18 section 4.3).
/// @throws RuntimeError if the Runtime was finalized.
void EnsureRuntime();

/// Binds imsrg.Runtime and the atexit hook.
void BindRuntime(nanobind::module_& m);
/// Binds Orbit, Channel, ModelSpace.
void BindBasis(nanobind::module_& m);
/// Binds Operator, OperatorSpec, enums, dump I/O, norms, random operators.
void BindOperator(nanobind::module_& m);
/// Binds RunConfig, load_config and the pipeline run.
void BindConfig(nanobind::module_& m);
/// Binds the readers and writers of the io module.
void BindIo(nanobind::module_& m);
/// Binds hf and mbpt.
void BindHf(nanobind::module_& m);
/// Binds the commutator.
void BindCommutator(nanobind::module_& m);
/// Binds the generators, the solvers and the BCH functions of the flow module.
void BindFlow(nanobind::module_& m);
/// Binds the operator library (kinetic energy, center of mass, radii).
void BindOpslib(nanobind::module_& m);

}  // namespace imsrg::py

#endif  // PYTHON_SRC_BINDINGS_H_
