#pragma once
/// @file
/// Typed exceptions of the imsrg library (design/00 D16).
///
/// Library code never calls `exit()` or `abort()`. User and input errors throw `InputError`
/// (or a subclass); violated invariants throw `InvariantError` (see `IMSRG_CHECK` in
/// `imsrg/base/assert.hpp`). Everything derives from `imsrg::Error`, so callers (the executable,
/// the Python bindings) can catch one type. Exceptions are host-only: device code never throws.

#include <stdexcept>
#include <string>

namespace imsrg {

/// Base class of all exceptions thrown by imsrg.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Invalid user input: bad parameters, malformed or inconsistent input files, unsupported options.
/// Messages name the offending parameter or file (and position) so the user can fix the input.
class InputError : public Error {
 public:
  using Error::Error;
};

/// A configuration value that is missing, has the wrong type, or is out of range.
class ConfigError : public InputError {
 public:
  using InputError::InputError;
};

/// An internal invariant does not hold (a bug, or an API precondition violated by the caller).
/// Thrown by `IMSRG_CHECK` and by `IMSRG_ASSERT` in builds with assertions.
class InvariantError : public Error {
 public:
  using Error::Error;
};

/// A numerical procedure failed (e.g. an eigensolver did not converge, a matrix is singular).
class NumericalError : public Error {
 public:
  using Error::Error;
};

/// The runtime environment could not be set up or was used incorrectly (Kokkos or MPI
/// initialization, double initialization, use after finalization).
class RuntimeError : public Error {
 public:
  using Error::Error;
};

/// A feature that is designed but not implemented yet.
class NotImplementedError : public Error {
 public:
  using Error::Error;
};

}  // namespace imsrg
