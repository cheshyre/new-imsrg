#pragma once
/// @file
/// A minimal `Expected<T, E>` (value or error), standing in for C++23 `std::expected`, which the
/// C++20 toolchains of this project do not provide. Only the operations the driver needs exist.
/// If `base` grows the same facility later, this header becomes an alias to it.

#include <type_traits>
#include <utility>
#include <variant>

#include "imsrg/base/error.hpp"

namespace imsrg::driver {

/// The error alternative of an `Expected`, as in `return Unexpected{issues};`.
template <class E>
struct Unexpected {
  E error;  ///< The error value.
};

template <class E>
Unexpected(E) -> Unexpected<E>;

/// Either a value of type `T` or an error of type `E` (never both, never neither). `T` and `E`
/// may be the same type. Accessing the wrong alternative throws `InvariantError`.
template <class T, class E>
class [[nodiscard]] Expected {
 public:
  /// Holds a value.
  Expected(const T& value) : data_(std::in_place_index<0>, value) {}
  /// Holds a value.
  Expected(T&& value) : data_(std::in_place_index<0>, std::move(value)) {}
  /// Holds an error.
  Expected(Unexpected<E> error) : data_(std::in_place_index<1>, std::move(error.error)) {}

  /// True if a value is held.
  bool has_value() const noexcept { return data_.index() == 0; }
  /// True if a value is held.
  explicit operator bool() const noexcept { return has_value(); }

  /// The value. @throws InvariantError if an error is held.
  T& value() & {
    require_value();
    return std::get<0>(data_);
  }
  /// The value. @throws InvariantError if an error is held.
  const T& value() const& {
    require_value();
    return std::get<0>(data_);
  }
  /// The value (moved out). @throws InvariantError if an error is held.
  T&& value() && {
    require_value();
    return std::get<0>(std::move(data_));
  }
  /// The error. @throws InvariantError if a value is held.
  E& error() & {
    require_error();
    return std::get<1>(data_);
  }
  /// The error. @throws InvariantError if a value is held.
  const E& error() const& {
    require_error();
    return std::get<1>(data_);
  }

  /// Member access to the value. @throws InvariantError if an error is held.
  T* operator->() { return &value(); }
  /// Member access to the value. @throws InvariantError if an error is held.
  const T* operator->() const { return &value(); }
  /// The value. @throws InvariantError if an error is held.
  T& operator*() & { return value(); }
  /// The value. @throws InvariantError if an error is held.
  const T& operator*() const& { return value(); }

 private:
  void require_value() const {
    if (!has_value()) throw InvariantError("Expected: value() called on an error");
  }
  void require_error() const {
    if (has_value()) throw InvariantError("Expected: error() called on a value");
  }

  std::variant<T, E> data_;
};

}  // namespace imsrg::driver
