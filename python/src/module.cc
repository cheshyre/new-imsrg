// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// Module definition of imsrg._core: exception hierarchy and the binding
/// groups.

#include "imsrg/base/error.h"
#include "python/src/bindings.h"

NB_MODULE(_core, m) {
  namespace nb = nanobind;
  m.doc() =
      "imsrg C++ core (nanobind). Use the imsrg package, not this module.";
  // Translators are tried most recently registered first: base class first,
  // derived after.
  nb::exception<imsrg::Error> err(m, "ImsrgError", PyExc_RuntimeError);
  nb::exception<imsrg::InputError> input(m, "InputError", err.ptr());
  nb::exception<imsrg::ConfigError>(m, "ConfigError", input.ptr());
  nb::exception<imsrg::InvariantError>(m, "InvariantError", err.ptr());
  nb::exception<imsrg::NumericalError>(m, "NumericalError", err.ptr());
  nb::exception<imsrg::RuntimeError>(m, "RuntimeError", err.ptr());
  nb::exception<imsrg::NotImplementedError>(m, "NotImplementedError",
                                            err.ptr());
  nb::exception<imsrg::StaleHostError>(m, "StaleHostError", err.ptr());

  imsrg::py::BindRuntime(m);
  imsrg::py::BindBasis(m);
  imsrg::py::BindOperator(m);
  imsrg::py::BindConfig(m);
  imsrg::py::BindOpslib(m);
  imsrg::py::BindIo(m);
  imsrg::py::BindHf(m);
  imsrg::py::BindCommutator(m);
  imsrg::py::BindFlow(m);
}
