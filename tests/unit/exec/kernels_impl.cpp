// Kernels written against the exec facade. Free functions only (design/12 section 5).
#include "kernels_impl.hpp"

#include <cstdint>
#include <limits>

#include "imsrg/exec/kokkos_backend.hpp"

namespace imsrg::exec {
template class DeviceBuffer<test::Pod>;  // explicit instantiation via the impl header
}

namespace imsrg::exec::test {

void fill_twice_index(DeviceBuffer<double>& a) {
  auto v = view(a);
  parallel_for(
      "test::fill_twice_index", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i) { v(i) = 2.0 * static_cast<double>(i); });
  fence("test::fill_twice_index");
}

std::int64_t sum(const DeviceBuffer<std::int64_t>& a) {
  auto v = view(a);
  std::int64_t result = 0;
  parallel_reduce(
      "test::sum", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i, std::int64_t& partial) { partial += v(i); }, result);
  return result;
}

std::int64_t max_value(const DeviceBuffer<std::int64_t>& a) {
  auto v = view(a);
  std::int64_t result = std::numeric_limits<std::int64_t>::lowest();
  parallel_reduce_max(
      "test::max_value", static_cast<Index>(v.extent(0)),
      IMSRG_LAMBDA(const Index i, std::int64_t& m) {
        if (v(i) > m) {
          m = v(i);
        }
      },
      result);
  return result;
}

void fill_matrix(DeviceBuffer<std::int64_t>& m, std::int64_t n0, std::int64_t n1) {
  auto v = view(m);
  parallel_for_md(
      "test::fill_matrix", n0, n1,
      IMSRG_LAMBDA(const Index i, const Index j) { v(i * n1 + j) = 100 * i + j; });
  fence("test::fill_matrix");
}

void row_sums_team(const DeviceBuffer<std::int64_t>& m, std::int64_t n0, std::int64_t n1,
                   DeviceBuffer<std::int64_t>& sums) {
  auto in = view(m);
  auto out = view(sums);
  const std::size_t scratch = static_cast<std::size_t>(n1) * sizeof(std::int64_t);
  team_for(
      "test::row_sums_team", static_cast<int>(n0), 0, scratch,
      IMSRG_LAMBDA(const TeamMember& team) {
        const Index row = team.league_rank();
        Kokkos::View<std::int64_t*, DefaultExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>
            buf(team.team_scratch(0), static_cast<std::size_t>(n1));
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n1),
                             [&](const Index j) { buf(j) = in(row * n1 + j); });
        team.team_barrier();
        std::int64_t total = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n1), [&](const Index j, std::int64_t& s) { s += buf(j); },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { out(row) = total; });
      });
  fence("test::row_sums_team");
}

void fill_pods(DeviceBuffer<Pod>& p) {
  auto v = view(p);
  parallel_for(
      "test::fill_pods", static_cast<Index>(v.extent(0)), IMSRG_LAMBDA(const Index i) {
        v(i).a = static_cast<double>(i);
        v(i).b = static_cast<std::int32_t>(i);
        v(i).c = static_cast<std::int32_t>(-i);
      });
  fence("test::fill_pods");
}

}  // namespace imsrg::exec::test
