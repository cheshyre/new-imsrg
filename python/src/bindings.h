// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef PYTHON_SRC_BINDINGS_H_
#define PYTHON_SRC_BINDINGS_H_
/// @file
/// Shared declarations of the nanobind bindings (design/18). No Kokkos headers.

#include <atomic>
#include <memory>
#include <utility>

#include <nanobind/nanobind.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::py {

namespace nb = nanobind;

/// Python-side ModelSpace: a handle to the immutable C++ model space (value
/// semantics, shared).
struct PyModelSpace {
  std::shared_ptr<const basis::ModelSpace> ms;
};

/// Number of live Python `Operator` objects (they own Kokkos buffers); the
/// Runtime refuses to finalize while this is nonzero. Thread-safe.
std::atomic<int>& LiveOperators() noexcept;

/// An `op::Operator` that counts itself in `LiveOperators()`.
struct PyOperator : op::Operator {
  /// Wraps (moves) an operator.
  explicit PyOperator(op::Operator&& o) : op::Operator(std::move(o)) {
    ++LiveOperators();
  }
  /// Copies an operator.
  explicit PyOperator(const op::Operator& o) : op::Operator(o) {
    ++LiveOperators();
  }
  PyOperator(const PyOperator& o) : op::Operator(o) { ++LiveOperators(); }
  PyOperator(PyOperator&& o) noexcept : op::Operator(std::move(o)) {
    ++LiveOperators();
  }
  PyOperator& operator=(const PyOperator&) = default;
  PyOperator& operator=(PyOperator&&) = default;
  ~PyOperator() { --LiveOperators(); }
};

/// Initializes the process Runtime (Kokkos, optionally MPI) if it is not yet;
/// idempotent. Called by every constructor of a ModelSpace or Operator (lazy
/// init, design/18 section 4.3).
/// @throws RuntimeError if the Runtime was finalized.
void EnsureRuntime();

/// Binds imsrg.Runtime and the atexit hook.
void BindRuntime(nb::module_& m);
/// Binds Orbit, Channel, ModelSpace.
void BindBasis(nb::module_& m);
/// Binds Operator, OperatorSpec, enums, dump I/O, norms, random operators.
void BindOperator(nb::module_& m);
/// Binds RunConfig and load_config.
void BindConfig(nb::module_& m);

}  // namespace imsrg::py

#endif  // PYTHON_SRC_BINDINGS_H_
