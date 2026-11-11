#pragma once
// Test-local kernel entry points (Kokkos-free interface); the kernels are in tables_kernels.cpp.
#include <cstdint>

#include "imsrg/angmom/tables.hpp"
#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::angmom::test {

/// Sum over every (ia, ib, J, ic, id, J') of the table's nj^4 nJ^2 index space of the device view
/// value; `abs_sum` receives the sum of the absolute values.
double sum_sixj_device(const SixJTable& t, double& abs_sum);
/// Sum of all stored 9j LS-jj coefficients read through the device view (over the index space).
double sum_ninej_device(const NineJLSjjTable& t);
/// out[q] = M(query row q) through the device view; each query row holds
/// N L n l n1 l1 n2 l2 lambda (9 int32).
void eval_moshinsky_device(const MoshinskyTable& t, const exec::DeviceBuffer<std::int32_t>& queries,
                           exec::DeviceBuffer<double>& out);

}  // namespace imsrg::angmom::test
