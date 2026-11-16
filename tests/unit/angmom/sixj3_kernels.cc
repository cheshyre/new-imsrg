// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Kernels reading the SixJ3Table / SixJTable device views. Free functions only
// (design/12 section 5).
#include "tests/unit/angmom/sixj3_kernels.h"

#include <cstdint>

#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::angmom::test {

void EvalSixJ3Device(const SixJ3Table& t,
                     const exec::DeviceBuffer<std::int32_t>& queries,
                     exec::DeviceBuffer<double>& out) {
  const SixJ3View v = t.device_view();
  const auto q = exec::View(queries);
  auto o = exec::View(out);
  exec::ParallelFor(
      "test::eval_sixj3", static_cast<exec::Index>(o.extent(0)),
      IMSRG_LAMBDA(const exec::Index i) {
        const exec::Index b = 6 * i;
        o(i) = v(TwoJ{q(b)}, TwoJ{q(b + 1)}, TwoJ{q(b + 2)}, TwoJ{q(b + 3)},
                 TwoJ{q(b + 4)}, TwoJ{q(b + 5)});
      });
  exec::Fence("test::eval_sixj3");
}

void EvalSixJDevice(const SixJTable& t,
                    const exec::DeviceBuffer<std::int32_t>& queries,
                    exec::DeviceBuffer<double>& out) {
  const SixJView v = t.device_view();
  const auto q = exec::View(queries);
  auto o = exec::View(out);
  exec::ParallelFor(
      "test::eval_sixj", static_cast<exec::Index>(o.extent(0)),
      IMSRG_LAMBDA(const exec::Index i) {
        const exec::Index b = 6 * i;
        o(i) = v(TwoJ{q(b)}, TwoJ{q(b + 1)}, TwoJ{q(b + 2)}, TwoJ{q(b + 3)},
                 TwoJ{q(b + 4)}, TwoJ{q(b + 5)});
      });
  exec::Fence("test::eval_sixj");
}

}  // namespace imsrg::angmom::test
