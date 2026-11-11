#pragma once
// Test-local kernel entry points (Kokkos-free interface); the kernels are in kernels_impl.cpp.
#include <cstdint>

#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::exec::test {

/// Plain-old-data element used to test explicit DeviceBuffer instantiation in a kernel TU.
struct Pod {
  double a;
  std::int32_t b;
  std::int32_t c;
};

/// a[i] = 2 i.
void fill_twice_index(DeviceBuffer<double>& a);
/// Exact sum of a[i] (integer valued).
std::int64_t sum(const DeviceBuffer<std::int64_t>& a);
/// Maximum of a[i]; the lowest int64 for an empty buffer.
std::int64_t max_value(const DeviceBuffer<std::int64_t>& a);
/// m(i, j) = 100 i + j for an n0 x n1 row-major matrix stored in `m`.
void fill_matrix(DeviceBuffer<std::int64_t>& m, std::int64_t n0, std::int64_t n1);
/// sums[t] = sum_j m(t, j) for each row t, via team_for with team scratch.
void row_sums_team(const DeviceBuffer<std::int64_t>& m, std::int64_t n0, std::int64_t n1,
                   DeviceBuffer<std::int64_t>& sums);
/// p[i] = {i, i, -i}.
void fill_pods(DeviceBuffer<Pod>& p);

}  // namespace imsrg::exec::test
