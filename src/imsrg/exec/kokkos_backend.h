// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_EXEC_KOKKOS_BACKEND_H_
#define IMSRG_EXEC_KOKKOS_BACKEND_H_
/// @file
/// The restricted Kokkos vocabulary for kernels (design/12 section 2). Includes
/// Kokkos: only for `.cc` files in `src/imsrg/exec/` and `src/imsrg/*/kernels/`
/// (design/20 section 4); never include it from a header that other modules
/// include.
///
/// Kernel rules (design/12 section 5): lambdas are `IMSRG_LAMBDA` and capture
/// by value only (views and PODs, never `this`, containers or strings); no
/// exceptions, allocation or virtual calls in kernels; every kernel has a name;
/// kernels live in free functions, never in a `TEST` body or a
/// private/protected member function.
///
/// `ParallelFor` and `TeamFor` launch asynchronously on async backends;
/// `parallel_reduce*` return the result in a host variable and therefore block.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "imsrg/base/assert.h"
#include "imsrg/exec/detail/device_buffer_impl.h"
#include "imsrg/exec/device_buffer.h"

/// Lambda annotation for device lambdas (= KOKKOS_LAMBDA); value captures only.
#ifndef IMSRG_LAMBDA
#define IMSRG_LAMBDA KOKKOS_LAMBDA
#endif

namespace imsrg::exec {

/// True if kernels run on an accelerator (CUDA/HIP), not on host threads.
/// Kernels with a host-tuned shape (one thread per row looping over a long
/// ragged range) select a device shape with more, shorter items by it (M14);
/// both shapes must do the same per-element arithmetic, so results agree
/// bitwise across the choice.
inline constexpr bool kDeviceKernels =
    !std::is_same_v<DefaultExecSpace, HostExecSpace>;

/// Member type handed to team kernels.
using TeamMember = Kokkos::TeamPolicy<DefaultExecSpace>::member_type;

/// 1-D range policy with 64-bit indices on the default execution space.
using RangePolicy =
    Kokkos::RangePolicy<DefaultExecSpace, Kokkos::IndexType<Index>>;

/// The device view of `buf` (empty view for an empty buffer). Valid until `buf`
/// is resized, moved from or destroyed.
template <class T>
DeviceView<T> View(DeviceBuffer<T>& buf) {
  return detail::DeviceBufferAccess<T>::view(buf);
}

/// Read-only device view of `buf`.
template <class T>
DeviceView<const T> View(const DeviceBuffer<T>& buf) {
  return detail::DeviceBufferAccess<T>::view(buf);
}

/// Runs `f(i)` for i in [0, n) on the default execution space. `f` is an
/// `IMSRG_LAMBDA` with value captures. Asynchronous on async backends.
template <class F>
void ParallelFor(const char* name, Index n, const F& f) {
  IMSRG_CHECK(n >= 0, name);
  Kokkos::parallel_for(name, RangePolicy(0, n), f);
}

/// `ParallelFor` with dynamic scheduling (`Kokkos::Schedule<Dynamic>`): for
/// ragged work items whose costs differ a lot (e.g. one item per column of
/// the two-body blocks); host backends then hand out items on demand instead
/// of in equal contiguous chunks. Device backends ignore the hint.
template <class F>
void ParallelForDynamic(const char* name, Index n, const F& f) {
  using Policy = Kokkos::RangePolicy<DefaultExecSpace, Kokkos::IndexType<Index>,
                                     Kokkos::Schedule<Kokkos::Dynamic>>;
  IMSRG_CHECK(n >= 0, name);
  Kokkos::parallel_for(name, Policy(0, n), f);
}

/// Runs `f(i, j)` for i in [0, n0), j in [0, n1) (MDRangePolicy, rank 2).
template <class F>
void ParallelForMd(const char* name, Index n0, Index n1, const F& f) {
  using Policy = Kokkos::MDRangePolicy<DefaultExecSpace, Kokkos::Rank<2>,
                                       Kokkos::IndexType<Index>>;
  IMSRG_CHECK(n0 >= 0 && n1 >= 0, name);
  Kokkos::parallel_for(name, Policy({0, 0}, {n0, n1}), f);
}

/// `result = sum over i in [0, n) of the contributions `f(i, partial)` add to
/// `partial` (`partial += ...`). `result` is overwritten. Blocking.
/// Floating-point sums depend on the reduction order.
template <class F, class T>
void ParallelReduce(const char* name, Index n, const F& f, T& result) {
  IMSRG_CHECK(n >= 0, name);
  Kokkos::parallel_reduce(name, RangePolicy(0, n), f, result);
}

/// `result = max over i in [0, n) of the values `f(i, m)` fold into `m` (`if (x
/// > m) m = x`). Blocking. For `n == 0` the result is the lowest value of `T`.
template <class F, class T>
void ParallelReduceMax(const char* name, Index n, const F& f, T& result) {
  IMSRG_CHECK(n >= 0, name);
  Kokkos::parallel_reduce(name, RangePolicy(0, n), f, Kokkos::Max<T>(result));
}

/// Work distribution of a team launch. `kStatic`: the league is split in
/// fixed chunks (deterministic assignment). `kDynamic`: teams take league
/// indices on demand, chunk 1 (for ragged work such as the triples tiles).
enum class Schedule { kStatic, kDynamic };

/// Team scratch level request: `kAuto` picks level 0 if `scratch_bytes` fit
/// there, else level 1; `kLevel0`/`kLevel1` force the level.
enum class ScratchLevel { kAuto, kLevel0, kLevel1 };

/// Launch options of `TeamFor`/`TeamReduce`. Trivially copyable.
struct TeamOptions {
  Schedule schedule = Schedule::kStatic;
  int team_size = 0;      ///< Threads per team; <= 0: AUTO (Serial uses 1).
  int vector_length = 0;  ///< Vector length; <= 0: AUTO.
  ScratchLevel level = ScratchLevel::kAuto;
  std::size_t scratch_bytes = 0;  ///< Team scratch per team, in bytes.
};
static_assert(std::is_trivially_copyable_v<TeamOptions>);

namespace detail {
using StaticTeamPolicy = Kokkos::TeamPolicy<DefaultExecSpace>;
using DynamicTeamPolicy =
    Kokkos::TeamPolicy<DefaultExecSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
static_assert(std::is_same_v<StaticTeamPolicy::member_type,
                             DynamicTeamPolicy::member_type>,
              "team member types of both schedules must agree");
}  // namespace detail

/// Largest team scratch (bytes per team) the default execution space offers
/// at `level` (0 or 1).
inline std::size_t TeamScratchMax(int level) {
  IMSRG_CHECK(level == 0 || level == 1, "TeamScratchMax: level must be 0 or 1");
  return static_cast<std::size_t>(
      detail::StaticTeamPolicy::scratch_size_max(level));
}

/// The scratch level (0 or 1) `opts` uses: the forced one, or for `kAuto`
/// level 0 if `scratch_bytes <= TeamScratchMax(0)`, else level 1.
/// IMSRG_CHECK (throws) that `scratch_bytes` fits the chosen level.
inline int ResolveScratchLevel(const TeamOptions& opts) {
  int level = 0;
  if (opts.level == ScratchLevel::kLevel1) {
    level = 1;
  } else if (opts.level == ScratchLevel::kAuto &&
             opts.scratch_bytes > TeamScratchMax(0)) {
    level = 1;
  }
  IMSRG_CHECK(opts.scratch_bytes <= TeamScratchMax(level),
              "ResolveScratchLevel: the team scratch does not fit");
  return level;
}

namespace detail {
template <class Policy>
Policy MakeTeamPolicy(int league_size, const TeamOptions& opts) {
  const int level = ResolveScratchLevel(opts);
  Policy policy =
      opts.team_size > 0
          ? (opts.vector_length > 0
                 ? Policy(league_size, opts.team_size, opts.vector_length)
                 : Policy(league_size, opts.team_size, Kokkos::AUTO))
          : (opts.vector_length > 0
                 ? Policy(league_size, Kokkos::AUTO, opts.vector_length)
                 : Policy(league_size, Kokkos::AUTO, Kokkos::AUTO));
  if (opts.scratch_bytes > 0) {
    policy.set_scratch_size(level, Kokkos::PerTeam(opts.scratch_bytes));
  }
  return policy;
}
}  // namespace detail

/// Runs `f(member)` on EVERY thread of `league_size` teams (shape and scratch
/// from `opts`; team scratch via `member.team_scratch(level)` with the level
/// of `ResolveScratchLevel(opts)`). Kernels must therefore split work with
/// `Kokkos::TeamThreadRange` and guard per-team writes with
/// `Kokkos::single(PerTeam)` and `member.team_barrier()`; CPU tests with team
/// size 1 do not exercise this, so review it. `Schedule::kDynamic` uses chunk
/// size 1. Asynchronous on async backends.
template <class F>
void TeamFor(const char* name, int league_size, const TeamOptions& opts,
             const F& f) {
  IMSRG_CHECK(league_size >= 0, name);
  if (opts.schedule == Schedule::kDynamic) {
    auto policy =
        detail::MakeTeamPolicy<detail::DynamicTeamPolicy>(league_size, opts);
    policy.set_chunk_size(1);
    Kokkos::parallel_for(name, policy, f);
  } else {
    Kokkos::parallel_for(
        name,
        detail::MakeTeamPolicy<detail::StaticTeamPolicy>(league_size, opts), f);
  }
}

/// `sum = sum over teams of the contributions `f(member, partial)` add to
/// `partial` (`partial += ...`, once per team; guard with
/// `Kokkos::single(PerTeam)` as in `TeamFor`). `sum` is overwritten. Blocking.
/// Determinism: with `Schedule::kDynamic` (and on any threaded backend) the
/// reduction order, hence the floating-point result, may vary from run to run.
template <class F>
void TeamReduce(const char* name, int league_size, const TeamOptions& opts,
                const F& f, double& sum) {
  IMSRG_CHECK(league_size >= 0, name);
  if (opts.schedule == Schedule::kDynamic) {
    auto policy =
        detail::MakeTeamPolicy<detail::DynamicTeamPolicy>(league_size, opts);
    policy.set_chunk_size(1);
    Kokkos::parallel_reduce(name, policy, f, Kokkos::Sum<double>(sum));
  } else {
    Kokkos::parallel_reduce(
        name,
        detail::MakeTeamPolicy<detail::StaticTeamPolicy>(league_size, opts), f,
        Kokkos::Sum<double>(sum));
  }
}

/// Convenience overload (the launch before `TeamOptions`): static schedule,
/// team size `team_size` (<= 0: AUTO), vector length 1, and
/// `scratch_bytes_per_team` bytes of LEVEL-0 scratch, which its callers read
/// as `member.team_scratch(0)` (IMSRG_CHECK that it fits level 0).
template <class F>
void TeamFor(const char* name, int league_size, int team_size,
             std::size_t scratch_bytes_per_team, const F& f) {
  TeamOptions opts;
  opts.team_size = team_size;
  opts.vector_length = 1;
  opts.level = ScratchLevel::kLevel0;
  opts.scratch_bytes = scratch_bytes_per_team;
  TeamFor(name, league_size, opts, f);
}

/// Waits for all work on the default execution space.
inline void Fence(const char* name) { DefaultExecSpace().fence(name); }

}  // namespace imsrg::exec

#endif  // IMSRG_EXEC_KOKKOS_BACKEND_H_
