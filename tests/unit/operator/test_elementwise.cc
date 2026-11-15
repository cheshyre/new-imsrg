// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Element-wise abs, division and scalar offset of all stored elements (flow
// review item 6): compared with host loops over the part buffers.
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/elementwise.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> Space() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "O16", "O16");
}

Operator Random(std::uint64_t seed, Hermiticity h) {
  RandomOperatorOptions ro;
  ro.seed = seed;
  ro.zero_body_scale = 1.0;
  return RandomOperator(Space(), OperatorSpec::Scalar(2, h), ro);
}

std::vector<double> Flat(const Operator& o) {
  std::vector<double> v;
  v.push_back(o.zero_body());
  for (double x : o.one_body().buffer().ToVector()) v.push_back(x);
  for (double x : o.two_body().matrix().buffer().ToVector()) v.push_back(x);
  return v;
}

TEST(Elementwise, AbsMatchesHostLoopAndMakesNonHermitian) {
  Operator a = Random(11, Hermiticity::kAntiHermitian);
  const std::vector<double> before = Flat(a);
  ElementwiseAbs(a);
  const std::vector<double> after = Flat(a);
  ASSERT_EQ(before.size(), after.size());
  bool any_negative = false;
  for (std::size_t i = 0; i < before.size(); ++i) {
    any_negative = any_negative || before[i] < 0.0;
    EXPECT_EQ(after[i], std::abs(before[i]));
  }
  EXPECT_TRUE(any_negative);
  EXPECT_EQ(a.hermiticity(), Hermiticity::kNonHermitian);
}

TEST(Elementwise, DivideMatchesHostLoop) {
  Operator n = Random(21, Hermiticity::kHermitian);
  Operator d = Random(22, Hermiticity::kHermitian);
  AddToEveryElement(d, 3.0);  // keeps every denominator away from zero
  const std::vector<double> nn = Flat(n), dd = Flat(d);
  ElementwiseDivide(n, d);
  const std::vector<double> q = Flat(n);
  for (std::size_t i = 0; i < q.size(); ++i) EXPECT_EQ(q[i], nn[i] / dd[i]);
}

TEST(Elementwise, AddToEveryElementTouchesAllPartsNotOnlyE0) {
  Operator x = Random(31, Hermiticity::kHermitian);
  const std::vector<double> before = Flat(x);
  AddToEveryElement(x, 0.25);
  const std::vector<double> after = Flat(x);
  for (std::size_t i = 0; i < before.size(); ++i)
    EXPECT_EQ(after[i], before[i] + 0.25);
  // The operator+= of arithmetic.h only changes the zero-body part.
  Operator y = Random(31, Hermiticity::kHermitian);
  y += 0.25;
  EXPECT_EQ(Flat(y)[0], before[0] + 0.25);
  EXPECT_EQ(Flat(y)[1], before[1]);
}

TEST(Elementwise, DivisionByZeroGivesNonFinite) {
  Operator n = Random(41, Hermiticity::kHermitian);
  Operator z = n;
  Scale(z, 0.0);
  ElementwiseDivide(n, z);
  EXPECT_FALSE(std::isfinite(Norm(n)));
}

}  // namespace
}  // namespace imsrg::op
