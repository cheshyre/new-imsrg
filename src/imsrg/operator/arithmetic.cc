// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/arithmetic.h"

#include <utility>

#include "imsrg/base/assert.h"
#include "imsrg/operator/detail/part_kernels.h"

namespace imsrg::op {

Operator& Axpy(double a, const Operator& x, Operator& y) {
  IMSRG_CHECK(!x.empty() && !y.empty(), "axpy: empty operator");
  IMSRG_CHECK(SameModelSpace(x, y),
              "axpy: operators live on different model spaces");
  IMSRG_CHECK(x.spec().SameSymmetry(y.spec()),
              "axpy: operators differ in symmetry");
  IMSRG_CHECK(!x.three_body() || !y.three_body() ||
                  (x.three_body()->size() == y.three_body()->size() &&
                   x.three_body()->space().NumRows() ==
                       y.three_body()->space().NumRows()),
              "axpy: operators differ in three-body layout");
  IMSRG_CHECK(x.spec().particle_rank <= y.spec().particle_rank,
              "axpy: y must hold every part of x (particle_rank of x exceeds "
              "that of y)");
  if (a == 0.0) {
    return y;
  }
  y.zero_body() += a * x.zero_body();
  detail::Axpy(a, x.one_body().buffer(), y.one_body().buffer());
  if (x.two_body().allocated()) {
    IMSRG_CHECK(y.two_body().allocated(), "axpy: y has no two-body part");
    detail::Axpy(a, x.two_body().matrix().buffer(),
                 y.two_body().matrix().buffer());
  }
  if (x.three_body()) {
    IMSRG_CHECK(y.three_body().has_value(), "axpy: y has no three-body part");
    ThreeBody& y3 = *y.three_body();
    const ThreeBody& x3 = *x.three_body();
    if (x3.hermiticity() == y3.hermiticity()) {
      detail::Axpy(a, x3.buffer(), y3.buffer());
    } else {
      // Mixed symmetry: the combination is non-Hermitian, so both parts go to
      // the dense layout first (the packed layouts differ from it).
      y3.set_hermiticity(Hermiticity::kNonHermitian);
      ThreeBody x3_dense = x3;
      x3_dense.set_hermiticity(Hermiticity::kNonHermitian);
      detail::Axpy(a, x3_dense.buffer(), y3.buffer());
    }
  }
  y.set_hermiticity(CombinedHermiticity(y.hermiticity(), x.hermiticity()));
  return y;
}

Operator& Scale(Operator& x, double a) {
  x.zero_body() *= a;
  detail::Scale(a, x.one_body().buffer());
  if (x.two_body().allocated()) {
    detail::Scale(a, x.two_body().matrix().buffer());
  }
  if (x.three_body()) detail::Scale(a, x.three_body()->buffer());
  return x;
}

Operator& operator+=(Operator& y, const Operator& x) { return Axpy(1.0, x, y); }
Operator& operator-=(Operator& y, const Operator& x) {
  return Axpy(-1.0, x, y);
}
Operator& operator*=(Operator& x, double a) { return Scale(x, a); }
Operator& operator/=(Operator& x, double a) {
  IMSRG_CHECK(a != 0.0, "operator/=: division by zero");
  return Scale(x, 1.0 / a);
}

Operator operator+(const Operator& x, const Operator& y) {
  IMSRG_CHECK(!x.empty() && !y.empty(), "operator+: empty operator");
  if (x.spec().particle_rank >= y.spec().particle_rank) {
    Operator r = x;
    Axpy(1.0, y, r);
    return r;
  }
  Operator r = y;
  Axpy(1.0, x, r);
  return r;
}

Operator operator-(const Operator& x, const Operator& y) {
  IMSRG_CHECK(!x.empty() && !y.empty(), "operator-: empty operator");
  if (x.spec().particle_rank >= y.spec().particle_rank) {
    Operator r = x;
    Axpy(-1.0, y, r);
    return r;
  }
  Operator r = y;
  Scale(r, -1.0);
  Axpy(1.0, x, r);
  return r;
}

Operator operator-(const Operator& x) {
  Operator r = x;
  Scale(r, -1.0);
  return r;
}

Operator operator*(double a, const Operator& x) {
  Operator r = x;
  Scale(r, a);
  return r;
}

Operator operator*(const Operator& x, double a) { return a * x; }

Operator operator/(const Operator& x, double a) {
  Operator r = x;
  r /= a;
  return r;
}

// A nonzero constant is Hermitian, so adding it to an anti-Hermitian operator
// gives a non-Hermitian one.
Operator& operator+=(Operator& x, double c) {
  x.zero_body() += c;
  if (c != 0.0 && x.hermiticity() == Hermiticity::kAntiHermitian) {
    x.set_hermiticity(Hermiticity::kNonHermitian);
  }
  return x;
}

Operator& operator-=(Operator& x, double c) { return x += -c; }

}  // namespace imsrg::op
