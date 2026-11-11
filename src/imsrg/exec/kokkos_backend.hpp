#pragma once
/// @file
/// The restricted Kokkos vocabulary for kernels (design/12 section 2). Includes Kokkos: only for
/// `.cpp` files in `src/imsrg/exec/` and `src/imsrg/*/kernels/` (design/20 section 4); never
/// include it from a header that other modules include.
///
/// Kernel rules (design/12 section 5): lambdas are `IMSRG_LAMBDA` and capture by value only
/// (views and PODs, never `this`, containers or strings); no exceptions, allocation or virtual
/// calls in kernels; every kernel has a name; kernels live in free functions, never in a `TEST`
/// body or a private/protected member function.
///
/// `parallel_for` and `team_for` launch asynchronously on async backends; `parallel_reduce*`
/// return the result in a host variable and therefore block.

#include <cstddef>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "imsrg/exec/detail/device_buffer_impl.hpp"
#include "imsrg/exec/device_buffer.hpp"

/// Lambda annotation for device lambdas (= KOKKOS_LAMBDA); value captures only.
#ifndef IMSRG_LAMBDA
#define IMSRG_LAMBDA KOKKOS_LAMBDA
#endif

namespace imsrg::exec {

/// Member type handed to team kernels.
using TeamMember = Kokkos::TeamPolicy<DefaultExecSpace>::member_type;

/// 1-D range policy with 64-bit indices on the default execution space.
using RangePolicy = Kokkos::RangePolicy<DefaultExecSpace, Kokkos::IndexType<Index>>;

/// The device view of `buf` (empty view for an empty buffer). Valid until `buf` is resized,
/// moved from or destroyed.
template <class T>
DeviceView<T> view(DeviceBuffer<T>& buf) {
  return detail::DeviceBufferAccess<T>::view(buf);
}

/// Read-only device view of `buf`.
template <class T>
DeviceView<const T> view(const DeviceBuffer<T>& buf) {
  return detail::DeviceBufferAccess<T>::view(buf);
}

/// Runs `f(i)` for i in [0, n) on the default execution space. `f` is an `IMSRG_LAMBDA` with
/// value captures. Asynchronous on async backends.
template <class F>
void parallel_for(const char* name, Index n, const F& f) {
  Kokkos::parallel_for(name, RangePolicy(0, n), f);
}

/// Runs `f(i, j)` for i in [0, n0), j in [0, n1) (MDRangePolicy, rank 2).
template <class F>
void parallel_for_md(const char* name, Index n0, Index n1, const F& f) {
  using Policy = Kokkos::MDRangePolicy<DefaultExecSpace, Kokkos::Rank<2>, Kokkos::IndexType<Index>>;
  Kokkos::parallel_for(name, Policy({0, 0}, {n0, n1}), f);
}

/// `result = sum over i in [0, n) of the contributions `f(i, partial)` add to `partial`
/// (`partial += ...`). `result` is overwritten. Blocking. Floating-point sums depend on the
/// reduction order.
template <class F, class T>
void parallel_reduce(const char* name, Index n, const F& f, T& result) {
  Kokkos::parallel_reduce(name, RangePolicy(0, n), f, result);
}

/// `result = max over i in [0, n) of the values `f(i, m)` fold into `m`
/// (`if (x > m) m = x`). Blocking. For `n == 0` the result is the lowest value of `T`.
template <class F, class T>
void parallel_reduce_max(const char* name, Index n, const F& f, T& result) {
  Kokkos::parallel_reduce(name, RangePolicy(0, n), f, Kokkos::Max<T>(result));
}

/// Runs `f(member)` once per team: `league_size` teams of `team_size` threads (<= 0: chosen by
/// Kokkos, AUTO), each with `scratch_bytes_per_team` bytes of level-0 team scratch, available in
/// the kernel through `member.team_scratch(0)`. Asynchronous on async backends.
template <class F>
void team_for(const char* name, int league_size, int team_size, std::size_t scratch_bytes_per_team,
              const F& f) {
  Kokkos::TeamPolicy<DefaultExecSpace> policy =
      team_size > 0 ? Kokkos::TeamPolicy<DefaultExecSpace>(league_size, team_size)
                    : Kokkos::TeamPolicy<DefaultExecSpace>(league_size, Kokkos::AUTO);
  if (scratch_bytes_per_team > 0) {
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes_per_team));
  }
  Kokkos::parallel_for(name, policy, f);
}

/// Waits for all work on the default execution space.
inline void fence(const char* name) { DefaultExecSpace().fence(name); }

}  // namespace imsrg::exec
