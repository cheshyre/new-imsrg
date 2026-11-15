// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// Module definition of imsrg._core: exception hierarchy and the binding
/// groups.

#include <exception>
#include <string>

#include "imsrg/base/error.h"
#include "python/src/bindings.h"

namespace {

namespace nb = nanobind;

/// Creates `imsrg._core.<name>` deriving from `bases` (a tuple) and adds it to
/// the module.
nb::object MakeException(nb::module_& m, const char* name, const char* doc,
                         nb::tuple bases) {
  const std::string qualified = std::string("imsrg._core.") + name;
  PyObject* type =
      PyErr_NewExceptionWithDoc(qualified.c_str(), doc, bases.ptr(), nullptr);
  if (type == nullptr) throw nb::python_error();
  nb::object obj = nb::steal(type);
  m.attr(name) = obj;
  return obj;
}

/// Registers a translator of the C++ exception `E` to the Python type `type`
/// (tried before the ones registered earlier, so register base classes first).
template <class E>
void Translate(const nb::object& type) {
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const E& e) {
          PyErr_SetString(static_cast<PyObject*>(payload), e.what());
        }
      },
      type.ptr());
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() =
      "imsrg C++ core (nanobind). Use the imsrg package, not this module.";
  // Translators are tried most recently registered first: base class first,
  // derived after.
  nb::object err =
      MakeException(m, "ImsrgError", "Base class of all imsrg errors.",
                    nb::make_tuple(nb::handle(PyExc_RuntimeError)));
  Translate<imsrg::Error>(err);
  nb::object input = MakeException(
      m, "InputError", "Invalid user input: parameters, files, options.",
      nb::make_tuple(err));
  Translate<imsrg::InputError>(input);
  Translate<imsrg::ConfigError>(MakeException(
      m, "ConfigError", "Invalid or inconsistent run configuration.",
      nb::make_tuple(input)));
  // A missing file is also a FileNotFoundError, so `except FileNotFoundError`
  // works.
  Translate<imsrg::MissingFileError>(MakeException(
      m, "InputFileNotFoundError", "An input file does not exist.",
      nb::make_tuple(input, nb::handle(PyExc_FileNotFoundError))));
  Translate<imsrg::FileFormatError>(MakeException(
      m, "FileFormatError",
      "An input file exists but its content is malformed or inconsistent "
      "with the stated truncation.",
      nb::make_tuple(input)));
  Translate<imsrg::InvariantError>(
      MakeException(m, "InvariantError", "A violated internal invariant.",
                    nb::make_tuple(err)));
  nb::object numerical = MakeException(m, "NumericalError",
                                       "A numerical failure (divergence, NaN).",
                                       nb::make_tuple(err));
  Translate<imsrg::NumericalError>(numerical);
  MakeException(m, "ConvergenceError",
                "An iteration or flow did not converge (raised by "
                "pipeline.run on divergence and by hf.solve with "
                "raise_on_failure=True).",
                nb::make_tuple(numerical));
  Translate<imsrg::RuntimeError>(MakeException(
      m, "RuntimeError", "Runtime/environment misuse (Kokkos, threads).",
      nb::make_tuple(err)));
  // `except NotImplementedError` (the builtin) catches these as well.
  Translate<imsrg::NotImplementedError>(MakeException(
      m, "NotImplementedError", "The requested feature is not implemented.",
      nb::make_tuple(err, nb::handle(PyExc_NotImplementedError))));
  Translate<imsrg::ModelSpaceError>(MakeException(
      m, "ModelSpaceError", "Operands live on different model spaces.",
      nb::make_tuple(err)));
  MakeException(m, "SymmetryError",
                "Operands of incompatible symmetry (rank, parity, Tz); "
                "reserved for tensor operators, which the core does not "
                "support yet.",
                nb::make_tuple(err));
  Translate<imsrg::StaleHostError>(MakeException(
      m, "StaleHostError",
      "A host view of an operator whose host mirror is stale, or use of an "
      "operator whose device data were released.",
      nb::make_tuple(err)));

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
