#pragma once
/// @file
/// Shared declarations of the nanobind bindings (design/18). No Kokkos headers.

#include <atomic>
#include <memory>
#include <utility>

#include <nanobind/nanobind.h>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::py {

namespace nb = nanobind;

/// Python-side ModelSpace: a handle to the immutable C++ model space (value semantics, shared).
struct PyModelSpace {
  std::shared_ptr<const basis::ModelSpace> ms;
};

/// Number of live Python `Operator` objects (they own Kokkos buffers); the Runtime refuses to
/// finalize while this is nonzero. Thread-safe.
std::atomic<int>& live_operators() noexcept;

/// An `op::Operator` that counts itself in `live_operators()`.
struct PyOperator : op::Operator {
  /// Wraps (moves) an operator.
  explicit PyOperator(op::Operator&& o) : op::Operator(std::move(o)) { ++live_operators(); }
  /// Copies an operator.
  explicit PyOperator(const op::Operator& o) : op::Operator(o) { ++live_operators(); }
  PyOperator(const PyOperator& o) : op::Operator(o) { ++live_operators(); }
  PyOperator(PyOperator&& o) noexcept : op::Operator(std::move(o)) { ++live_operators(); }
  PyOperator& operator=(const PyOperator&) = default;
  PyOperator& operator=(PyOperator&&) = default;
  ~PyOperator() { --live_operators(); }
};

/// Initializes the process Runtime (Kokkos, optionally MPI) if it is not yet; idempotent.
/// Called by every constructor of a ModelSpace or Operator (lazy init, design/18 section 4.3).
/// @throws RuntimeError if the Runtime was finalized.
void ensure_runtime();

/// Binds imsrg.Runtime and the atexit hook.
void bind_runtime(nb::module_& m);
/// Binds Orbit, Channel, ModelSpace.
void bind_basis(nb::module_& m);
/// Binds Operator, OperatorSpec, enums, dump I/O, norms, random operators.
void bind_operator(nb::module_& m);
/// Binds RunConfig and load_config.
void bind_config(nb::module_& m);

}  // namespace imsrg::py
