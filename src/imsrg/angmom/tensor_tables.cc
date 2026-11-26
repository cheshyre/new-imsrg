// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Recoupling tables of rank-lambda tensor operators: host construction,
// DeviceBuffer storage.
#include "imsrg/angmom/tensor_tables.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"

namespace imsrg::angmom {

namespace {

int TriangleCount(int a2, int b2) {
  return (a2 + b2) / 2 - std::abs(a2 - b2) / 2 + 1;
}

}  // namespace

LambdaSixJTable::LambdaSixJTable(int emax, int lambda)
    : emax_(emax), lambda_(lambda), values_([&] {
        IMSRG_CHECK(emax >= 0 && lambda >= 0,
                    "LambdaSixJTable: emax and lambda must be >= 0");
        const auto nj = static_cast<std::size_t>(emax + 1);
        const auto nJ = static_cast<std::size_t>(2 * emax + 2);
        return exec::DeviceBuffer<double>(nJ * nJ * nj * nj * nj,
                                          "imsrg::LambdaSixJTable");
      }()) {
  const int nj = emax + 1;
  const int nJ = 2 * emax + 2;
  std::span<double> out = values_.host_span();
  std::size_t i = 0;
  for (int J1 = 0; J1 < nJ; ++J1)
    for (int J2 = 0; J2 < nJ; ++J2)
      for (int ia = 0; ia < nj; ++ia)
        for (int ib = 0; ib < nj; ++ib)
          for (int ic = 0; ic < nj; ++ic, ++i)
            out[i] =
                Wigner6j(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{2 * lambda},
                         TwoJ{2 * ia + 1}, TwoJ{2 * ib + 1}, TwoJ{2 * ic + 1});
  values_.SyncToDevice();
}

LambdaSixJView LambdaSixJTable::host_view() const noexcept {
  return LambdaSixJView{values_.host_span().data(), emax_};
}
LambdaSixJView LambdaSixJTable::device_view() const noexcept {
  return LambdaSixJView{values_.device_data(), emax_};
}

NineJTable::NineJTable(int emax, int lambda, double skip_threshold)
    : emax_(emax), lambda_(lambda), threshold_(skip_threshold) {
  IMSRG_CHECK(emax >= 0 && lambda >= 0 && skip_threshold >= 0.0,
              "NineJTable: emax, lambda and the threshold must be >= 0");
  const int nj = emax + 1;
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(nj) * nj * nj *
                                    nj);
  std::int64_t total = 0;
  std::size_t q = 0;
  for (int A = 1; A <= 2 * emax + 1; A += 2)
    for (int B = 1; B <= 2 * emax + 1; B += 2)
      for (int D = 1; D <= 2 * emax + 1; D += 2)
        for (int E = 1; E <= 2 * emax + 1; E += 2, ++q) {
          offsets[q] = total;
          total += static_cast<std::int64_t>(TriangleCount(A, B)) *
                   TriangleCount(D, E) * TriangleCount(A, D) *
                   TriangleCount(B, E);
        }
  offsets_ = exec::DeviceBuffer<std::int64_t>(offsets.size(),
                                              "imsrg::NineJTable::offsets");
  offsets_.CopyFromHost(offsets);
  values_ = exec::DeviceBuffer<double>(static_cast<std::size_t>(total),
                                       "imsrg::NineJTable");
  std::span<double> out = values_.host_span();
  const TwoJ L{2 * lambda};
  std::size_t i = 0;
  for (int A = 1; A <= 2 * emax + 1; A += 2)
    for (int B = 1; B <= 2 * emax + 1; B += 2)
      for (int D = 1; D <= 2 * emax + 1; D += 2)
        for (int E = 1; E <= 2 * emax + 1; E += 2)
          for (int C = std::abs(A - B); C <= A + B; C += 2)
            for (int F = std::abs(D - E); F <= D + E; F += 2)
              for (int G = std::abs(A - D); G <= A + D; G += 2)
                for (int H = std::abs(B - E); H <= B + E; H += 2, ++i) {
                  double v = 0.0;
                  if (Triangle(TwoJ{C}, TwoJ{F}, L) &&
                      Triangle(TwoJ{G}, TwoJ{H}, L)) {
                    v = Wigner9j(TwoJ{A}, TwoJ{B}, TwoJ{C}, TwoJ{D}, TwoJ{E},
                                 TwoJ{F}, TwoJ{G}, TwoJ{H}, L);
                    if (std::abs(v) < skip_threshold) v = 0.0;
                  }
                  out[i] = v;
                }
  IMSRG_CHECK(static_cast<std::int64_t>(i) == total);
  values_.SyncToDevice();
}

NineJView NineJTable::host_view() const noexcept {
  return NineJView{offsets_.host_span().data(), values_.host_span().data(),
                   emax_};
}
NineJView NineJTable::device_view() const noexcept {
  return NineJView{offsets_.device_data(), values_.device_data(), emax_};
}

}  // namespace imsrg::angmom
