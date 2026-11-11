#pragma once
/// @file
/// Invariant checks (design/00 D16, AGENTS.md "Code conventions").
///
/// * `IMSRG_CHECK(cond)` / `IMSRG_CHECK(cond, message)`: always evaluated; throws
///   `imsrg::InvariantError` with the condition text, file, line and the optional message.
/// * `IMSRG_ASSERT(cond)` / `IMSRG_ASSERT(cond, message)`: the same, but only when assertions
///   are enabled (no `NDEBUG`, or `IMSRG_ENABLE_ASSERTS` defined); otherwise the condition is not
///   evaluated (it must not have side effects).
///
/// The message is any expression convertible to `std::string_view` (e.g. a literal or the
/// result of `fmt::format(...)`); it is evaluated only when the check fails.
/// Both macros are host-only: they throw, so they must not be used inside kernels
/// (design/12-compute-backend.md section 5). Kernel code uses `KOKKOS_ASSERT` via
/// `imsrg/exec/kokkos_backend.hpp`.

#include <string_view>

namespace imsrg::detail {

/// Throws `InvariantError` describing a failed check. Not for direct use; see `IMSRG_CHECK`.
[[noreturn]] void check_failed(const char* condition, const char* file, int line,
                               std::string_view message = {});

}  // namespace imsrg::detail

#define IMSRG_CHECK(cond, ...)                                                             \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      ::imsrg::detail::check_failed(#cond, __FILE__, __LINE__ __VA_OPT__(, ) __VA_ARGS__); \
    }                                                                                      \
  } while (false)

#if !defined(NDEBUG) || defined(IMSRG_ENABLE_ASSERTS)
#define IMSRG_ASSERTS_ENABLED 1
#define IMSRG_ASSERT(cond, ...) IMSRG_CHECK(cond __VA_OPT__(, ) __VA_ARGS__)
#else
#define IMSRG_ASSERTS_ENABLED 0
#define IMSRG_ASSERT(cond, ...) \
  do {                          \
    (void)sizeof(!(cond));      \
  } while (false)
#endif
