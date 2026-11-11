/// @file
/// Module definition of imsrg._core: exception hierarchy and the binding groups.

#include "bindings.hpp"

#include "imsrg/base/error.hpp"

namespace imsrg::py {

std::atomic<int>& live_operators() noexcept {
  static std::atomic<int> n{0};
  return n;
}

}  // namespace imsrg::py

NB_MODULE(_core, m) {
  namespace nb = nanobind;
  m.doc() = "imsrg C++ core (nanobind). Use the imsrg package, not this module.";
  // Translators are tried most recently registered first: base class first, derived after.
  nb::exception<imsrg::Error> err(m, "ImsrgError", PyExc_RuntimeError);
  nb::exception<imsrg::InputError> input(m, "InputError", err.ptr());
  nb::exception<imsrg::ConfigError>(m, "ConfigError", input.ptr());
  nb::exception<imsrg::InvariantError>(m, "InvariantError", err.ptr());
  nb::exception<imsrg::NumericalError>(m, "NumericalError", err.ptr());
  nb::exception<imsrg::RuntimeError>(m, "RuntimeError", err.ptr());
  nb::exception<imsrg::NotImplementedError>(m, "NotImplementedError", err.ptr());

  imsrg::py::bind_runtime(m);
  imsrg::py::bind_basis(m);
  imsrg::py::bind_operator(m);
  imsrg::py::bind_config(m);
}
