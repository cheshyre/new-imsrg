// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/elementwise.h"

#include <cmath>

#include "imsrg/base/assert.h"
#include "imsrg/operator/detail/part_kernels.h"

namespace imsrg::op {

void ElementwiseAbs(Operator& x) {
  IMSRG_CHECK(!x.empty(), "ElementwiseAbs: empty operator");
  x.zero_body() = std::abs(x.zero_body());
  detail::Abs(x.one_body().buffer());
  if (x.two_body().allocated()) detail::Abs(x.two_body().matrix().buffer());
  x.set_hermiticity(Hermiticity::kNonHermitian);
}

void ElementwiseDivide(Operator& num, const Operator& denom) {
  IMSRG_CHECK(!num.empty() && !denom.empty(),
              "ElementwiseDivide: empty operator");
  IMSRG_CHECK(SameModelSpace(num, denom),
              "ElementwiseDivide: different model spaces");
  IMSRG_CHECK(num.spec().SameSymmetry(denom.spec()) &&
                  num.spec().particle_rank == denom.spec().particle_rank,
              "ElementwiseDivide: operators differ in symmetry or rank");
  num.zero_body() /= denom.zero_body();
  detail::Divide(num.one_body().buffer(), denom.one_body().buffer());
  if (num.two_body().allocated()) {
    IMSRG_CHECK(denom.two_body().allocated(),
                "ElementwiseDivide: denominator has no two-body part");
    detail::Divide(num.two_body().matrix().buffer(),
                   denom.two_body().matrix().buffer());
  }
  num.set_hermiticity(Hermiticity::kNonHermitian);
}

void AddToEveryElement(Operator& x, double c) {
  IMSRG_CHECK(!x.empty(), "AddToEveryElement: empty operator");
  x.zero_body() += c;
  detail::AddScalar(c, x.one_body().buffer());
  if (x.two_body().allocated()) {
    detail::AddScalar(c, x.two_body().matrix().buffer());
  }
  x.set_hermiticity(Hermiticity::kNonHermitian);
}

}  // namespace imsrg::op
