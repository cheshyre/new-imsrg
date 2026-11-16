// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_ANGMOM_SIXJ3_KERNELS_H_
#define TESTS_UNIT_ANGMOM_SIXJ3_KERNELS_H_
// Test-local kernel entry points (Kokkos-free interface); the kernels are in
// sixj3_kernels.cc.
#include <cstdint>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/angmom/tables.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::angmom::test {

/// out[q] = device-view value of `t` for the query row q, which holds the six
/// doubled arguments (2 j1, 2 j2, 2 J12, 2 j3, 2 J, 2 J23) as int32.
void EvalSixJ3Device(const SixJ3Table& t,
                     const exec::DeviceBuffer<std::int32_t>& queries,
                     exec::DeviceBuffer<double>& out);

/// out[q] = device-view value of the Pandya table `t` for the query row q,
/// which holds the six doubled arguments (2 ja, 2 jb, 2 J, 2 jc, 2 jd, 2 J').
void EvalSixJDevice(const SixJTable& t,
                    const exec::DeviceBuffer<std::int32_t>& queries,
                    exec::DeviceBuffer<double>& out);

}  // namespace imsrg::angmom::test

#endif  // TESTS_UNIT_ANGMOM_SIXJ3_KERNELS_H_
