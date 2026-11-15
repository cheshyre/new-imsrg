// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/compare.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "imsrg/base/error.h"

namespace imsrg::op {

namespace {

double MaxAbsOf(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::abs(x));
  return m;
}

/// max |a - b| over two vectors; a missing (empty) one counts as zero.
double MaxAbsDiffOf(const std::vector<double>& a,
                    const std::vector<double>& b) {
  if (a.empty()) return MaxAbsOf(b);
  if (b.empty()) return MaxAbsOf(a);
  IMSRG_CHECK(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

std::vector<double> TwoBodyValues(const TwoBody& t) {
  if (!t.allocated()) return {};
  return t.matrix().buffer().ToVector();
}

}  // namespace

double PartMaxima::max() const noexcept {
  return std::max({zero_body, one_body, two_body, three_body});
}

PartMaxima MaxAbs(const Operator& op) {
  PartMaxima m;
  m.zero_body = std::abs(op.zero_body());
  m.one_body = MaxAbsOf(op.one_body().buffer().ToVector());
  m.two_body = MaxAbsOf(TwoBodyValues(op.two_body()));
  if (op.three_body()) m.three_body = MaxAbsOf(op.three_body()->buffer().ToVector());
  return m;
}

PartMaxima MaxAbsDifference(const Operator& a, const Operator& b) {
  IMSRG_CHECK(SameModelSpace(a, b));
  IMSRG_CHECK(a.spec().SameSymmetry(b.spec()));
  PartMaxima m;
  m.zero_body = std::abs(a.zero_body() - b.zero_body());
  m.one_body = MaxAbsDiffOf(a.one_body().buffer().ToVector(),
                            b.one_body().buffer().ToVector());
  m.two_body =
      MaxAbsDiffOf(TwoBodyValues(a.two_body()), TwoBodyValues(b.two_body()));
  if (a.three_body() || b.three_body()) {
    ThreeBody a3 = a.three_body() ? *a.three_body() : ThreeBody();
    ThreeBody b3 = b.three_body() ? *b.three_body() : ThreeBody();
    if (!a3.empty() && !b3.empty() && a3.hermiticity() != b3.hermiticity()) {
      a3.set_hermiticity(Hermiticity::kNonHermitian);
      b3.set_hermiticity(Hermiticity::kNonHermitian);
    }
    m.three_body = MaxAbsDiffOf(a3.empty() ? std::vector<double>() : a3.buffer().ToVector(),
                                b3.empty() ? std::vector<double>() : b3.buffer().ToVector());
  }
  return m;
}

}  // namespace imsrg::op
