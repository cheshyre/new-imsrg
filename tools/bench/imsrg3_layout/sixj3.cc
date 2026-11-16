// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/sixj3.h"

#include <cmath>
#include <cstddef>
#include <span>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {

SixJ3Table::SixJ3Table(int emax)
    : emax_(emax),
      values_(
          [&] {
            IMSRG_CHECK(emax >= 0, "SixJ3Table: emax must be >= 0");
            return static_cast<std::size_t>(ValueCount(emax));
          }(),
          "imsrg3_layout::SixJ3Table"),
      hat_(static_cast<std::size_t>(2 * emax + 2),
           "imsrg3_layout::SixJ3Table::hat") {
  const int nj = emax + 1;
  const int nJ = 2 * emax + 2;
  const int nJ3 = 3 * emax + 2;
  std::span<double> out = values_.host_span();
  std::size_t i = 0;
  for (int i1 = 0; i1 < nj; ++i1) {
    const int j1 = 2 * i1 + 1;
    for (int i2 = 0; i2 < nj; ++i2) {
      const int j2 = 2 * i2 + 1;
      for (int J12 = 0; J12 < nJ; ++J12) {
        for (int i3 = 0; i3 < nj; ++i3) {
          const int j3 = 2 * i3 + 1;
          for (int iJ = 0; iJ < nJ3; ++iJ) {
            const int J = 2 * iJ + 1;
            for (int J23 = 0; J23 < nJ; ++J23, ++i) {
              // The four triads of {j1 j2 J12; j3 J J23}.
              const bool ok =
                  Triangle2(j1, j2, 2 * J12) && Triangle2(j1, J, 2 * J23) &&
                  Triangle2(j3, j2, 2 * J23) && Triangle2(j3, J, 2 * J12);
              out[i] = ok ? angmom::Wigner6j(TwoJ{j1}, TwoJ{j2}, TwoJ{2 * J12},
                                             TwoJ{j3}, TwoJ{J}, TwoJ{2 * J23})
                          : 0.0;
            }
          }
        }
      }
    }
  }
  values_.SyncToDevice();
  std::span<double> hat = hat_.host_span();
  for (int J = 0; J < 2 * emax + 2; ++J) {
    hat[static_cast<std::size_t>(J)] = std::sqrt(2.0 * J + 1.0);
  }
  hat_.SyncToDevice();
}

std::int64_t SixJ3Table::size_bytes() const noexcept {
  return static_cast<std::int64_t>((values_.size() + hat_.size()) *
                                   sizeof(double));
}

SixJ3View SixJ3Table::host_view() const noexcept {
  return SixJ3View{values_.host_span().data(), hat_.host_span().data(), emax_};
}

SixJ3View SixJ3Table::device_view() const noexcept {
  return SixJ3View{values_.device_data(), hat_.device_data(), emax_};
}

std::int64_t SixJ3Table::ValueCount(int emax) noexcept {
  const std::int64_t nj = emax + 1;
  const std::int64_t nJ = 2 * emax + 2;
  const std::int64_t nJ3 = 3 * emax + 2;
  return nj * nj * nj * nJ * nJ * nJ3;
}

}  // namespace imsrg::layout3
