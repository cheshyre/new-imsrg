// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tests/support/three_body_test_support.h"

#include <memory>
#include <utility>

#include "imsrg/base/assert.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::testing {

std::shared_ptr<const basis::ModelSpace> MakeThreeBodySpace(
    int emax, std::string_view reference, const basis::ThreeBodyCuts& cuts) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0,
                                      reference, reference)
      ->WithThreeBody(cuts);
}

std::shared_ptr<const basis::ModelSpace> AddThreeBodySpace(
    const std::shared_ptr<const basis::ModelSpace>& ms,
    const basis::ThreeBodyCuts& cuts) {
  IMSRG_CHECK(ms != nullptr, "AddThreeBodySpace: null model space");
  return ms->WithThreeBody(cuts);
}

namespace {

op::Operator RandomTwoBody(std::shared_ptr<const basis::ModelSpace> ms,
                           Hermiticity h, std::uint64_t seed) {
  op::RandomOperatorOptions o;
  o.seed = seed;
  o.one_body_scale = 1.0;
  o.two_body_scale = 1.0;
  return op::RandomOperator(std::move(ms), op::OperatorSpec::Scalar(2, h), o);
}

}  // namespace

op::Operator RandomThreeBodyOperator(const commutator::ThreeBodyContext& ctx,
                                     Hermiticity h, std::uint64_t seed,
                                     double three_body_scale) {
  const auto& ms = ctx.model_space_ptr();
  IMSRG_CHECK(ms != nullptr, "RandomThreeBodyOperator: moved-from context");
  op::Operator x(ms, op::OperatorSpec::Scalar(3, h));
  const op::Operator x2 = RandomTwoBody(ms, h, seed);
  x.one_body() = x2.one_body();
  x.two_body() = x2.two_body();

  Hermiticity ha = Hermiticity::kHermitian;
  if (h == Hermiticity::kHermitian) ha = Hermiticity::kAntiHermitian;
  if (h == Hermiticity::kNonHermitian) ha = Hermiticity::kNonHermitian;
  const op::Operator a = RandomTwoBody(ms, ha, 1000 + 2 * seed);
  const op::Operator b =
      RandomTwoBody(ms, Hermiticity::kHermitian, 1001 + 2 * seed);
  op::ThreeBody z3 = commutator::Comm223(a, b, ctx);
  IMSRG_CHECK(z3.hermiticity() == h,
              "RandomThreeBodyOperator: unexpected Comm223 hermiticity");
  *x.three_body() = std::move(z3);
  if (three_body_scale != 1.0) {
    op::Operator only3(ms, op::OperatorSpec::Scalar(3, h));
    *only3.three_body() = *x.three_body();
    x.three_body()->set_zero();
    op::Axpy(three_body_scale, only3, x);
  }
  return x;
}

}  // namespace imsrg::testing
