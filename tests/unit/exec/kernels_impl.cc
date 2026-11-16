// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Kernels written against the exec facade. Free functions only (design/12
// section 5).
#include "tests/unit/exec/kernels_impl.h"

#include <cstdint>
#include <limits>

#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::exec {
template class DeviceBuffer<test::Pod>;  // explicit instantiation via the impl
                                         // header
}

namespace imsrg::exec::test {

void FillTwiceIndex(DeviceBuffer<double>& a) {
  auto v = View(a);
  ParallelFor(
      "test::fill_twice_index", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i) { v(i) = 2.0 * static_cast<double>(i); });
  Fence("test::fill_twice_index");
}

void FillOddDynamic(DeviceBuffer<std::int64_t>& a) {
  auto v = View(a);
  ParallelForDynamic(
      "test::fill_odd_dynamic", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i) { v(i) = 2 * i + 1; });
  Fence("test::fill_odd_dynamic");
}

std::int64_t Sum(const DeviceBuffer<std::int64_t>& a) {
  auto v = View(a);
  std::int64_t result = 0;
  ParallelReduce(
      "test::sum", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i, std::int64_t& partial) { partial += v(i); },
      result);
  return result;
}

std::int64_t MaxValue(const DeviceBuffer<std::int64_t>& a) {
  auto v = View(a);
  std::int64_t result = std::numeric_limits<std::int64_t>::lowest();
  ParallelReduceMax(
      "test::max_value", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i, std::int64_t& m) {
        if (v(i) > m) {
          m = v(i);
        }
      },
      result);
  return result;
}

void FillMatrix(DeviceBuffer<std::int64_t>& m, std::int64_t n0,
                std::int64_t n1) {
  auto v = View(m);
  ParallelForMd(
      "test::fill_matrix", n0, n1, IMSRG_LAMBDA(const Index i, const Index j) {
        v(i * n1 + j) = 100 * i + j;
      });
  Fence("test::fill_matrix");
}

void RowSumsTeam(const DeviceBuffer<std::int64_t>& m, std::int64_t n0,
                 std::int64_t n1, DeviceBuffer<std::int64_t>& sums) {
  auto in = View(m);
  auto out = View(sums);
  const std::size_t scratch =
      static_cast<std::size_t>(n1) * sizeof(std::int64_t);
  TeamFor(
      "test::row_sums_team", static_cast<int>(n0), 0, scratch,
      IMSRG_LAMBDA(const TeamMember& team) {
        const Index row = team.league_rank();
        Kokkos::View<std::int64_t*, DefaultExecSpace::scratch_memory_space,
                     Kokkos::MemoryUnmanaged>
            buf(team.team_scratch(0), static_cast<std::size_t>(n1));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n1),
                             [&](const Index j) { buf(j) = in(row * n1 + j); });
        team.team_barrier();
        std::int64_t total = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n1),
            [&](const Index j, std::int64_t& s) { s += buf(j); }, total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { out(row) = total; });
      });
  Fence("test::row_sums_team");
}

namespace {
TeamOptions MakeOptions(bool dynamic, int level_kind, std::size_t bytes) {
  TeamOptions o;
  o.schedule = dynamic ? Schedule::kDynamic : Schedule::kStatic;
  o.level = level_kind == 1   ? ScratchLevel::kLevel0
            : level_kind == 2 ? ScratchLevel::kLevel1
                              : ScratchLevel::kAuto;
  o.scratch_bytes = bytes;
  return o;
}
}  // namespace

void VisitLeague(DeviceBuffer<std::int64_t>& counts, bool dynamic) {
  auto v = View(counts);
  TeamFor(
      "test::visit_league", static_cast<int>(v.extent(0)),
      MakeOptions(dynamic, 0, 0), IMSRG_LAMBDA(const TeamMember& team) {
        const Index i = team.league_rank();
        Kokkos::single(Kokkos::PerTeam(team), [&]() { v(i) += 1; });
      });
  Fence("test::visit_league");
}

void RowSumsTeamLevel(const DeviceBuffer<std::int64_t>& m, std::int64_t n0,
                      std::int64_t n1, DeviceBuffer<std::int64_t>& sums,
                      int level_kind) {
  auto in = View(m);
  auto out = View(sums);
  const TeamOptions opts = MakeOptions(
      true, level_kind, static_cast<std::size_t>(n1) * sizeof(std::int64_t));
  const int level = ResolveScratchLevel(opts);
  TeamFor(
      "test::row_sums_team_level", static_cast<int>(n0), opts,
      IMSRG_LAMBDA(const TeamMember& team) {
        const Index row = team.league_rank();
        Kokkos::View<std::int64_t*, DefaultExecSpace::scratch_memory_space,
                     Kokkos::MemoryUnmanaged>
            buf(team.team_scratch(level), static_cast<std::size_t>(n1));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n1),
                             [&](const Index j) { buf(j) = in(row * n1 + j); });
        team.team_barrier();
        std::int64_t total = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n1),
            [&](const Index j, std::int64_t& s) { s += buf(j); }, total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { out(row) = total; });
      });
  Fence("test::row_sums_team_level");
}

double TeamReduceHalfIntegers(int n, bool dynamic) {
  double sum = -1.0;
  TeamReduce(
      "test::team_reduce", n, MakeOptions(dynamic, 0, 0),
      IMSRG_LAMBDA(const TeamMember& team, double& partial) {
        const double x = static_cast<double>(team.league_rank()) + 0.5;
        Kokkos::single(Kokkos::PerTeam(team), [&]() { partial += x; });
      },
      sum);
  return sum;
}

int ResolveLevel(int level_kind, std::size_t bytes) {
  return ResolveScratchLevel(MakeOptions(false, level_kind, bytes));
}

std::size_t ScratchMax(int level) { return TeamScratchMax(level); }

void FillPods(DeviceBuffer<Pod>& p) {
  auto v = View(p);
  ParallelFor(
      "test::fill_pods", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i) {
        v(i).a = static_cast<double>(i);
        v(i).b = static_cast<std::int32_t>(i);
        v(i).c = static_cast<std::int32_t>(-i);
      });
  Fence("test::fill_pods");
}

}  // namespace imsrg::exec::test
