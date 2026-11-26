// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of op::Truncate (emax -> emax_imsrg restriction, design/50 C5).
#include "imsrg/operator/truncate.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> Ms(int emax, const char* ref,
                                            const char* val) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, ref, val);
}

// Every element of the truncated operator equals the source element of the
// same orbits (matched by quantum numbers); spec and hermiticity are kept.
void CheckSubmatrix(Hermiticity h) {
  const auto big = Ms(4, "Al27", "sd-shell");
  const auto small = Ms(2, "Al27", "sd-shell");
  Operator A = RandomOperator(big, OperatorSpec::Scalar(2, h),
                              {.seed = 7, .zero_body_scale = 1.0});
  const Operator T = Truncate(A, small);
  A.SyncToHost();
  T.SyncToHost();
  EXPECT_EQ(&T.model_space(), small.get());
  EXPECT_EQ(T.hermiticity(), h);
  EXPECT_EQ(T.zero_body(), A.zero_body());
  const auto& sb = small->basis();
  const auto& bb = big->basis();
  auto m = [&](std::int32_t a) {
    return bb.index_of(sb.orbit(OrbitIndex(a)));
  };
  const std::int32_t n = sb.num_orbits();
  for (std::int32_t a = 0; a < n; ++a)
    for (std::int32_t b = 0; b < n; ++b)
      EXPECT_EQ(T.one_body().get(OrbitIndex(a), OrbitIndex(b)),
                A.one_body().get(m(a), m(b)));
  std::int64_t count = 0;
  for (std::int32_t a = 0; a < n; ++a)
    for (std::int32_t b = a; b < n; ++b)
      for (std::int32_t c = 0; c < n; ++c)
        for (std::int32_t d = c; d < n; ++d)
          for (int J2 = 0; J2 <= 2 * sb.max_j2().twice; J2 += 2) {
            const double t = T.two_body().get_normalized(
                OrbitIndex(a), OrbitIndex(b), OrbitIndex(c), OrbitIndex(d),
                TwoJ{J2});
            EXPECT_EQ(t, A.two_body().get_normalized(m(a), m(b), m(c), m(d),
                                                     TwoJ{J2}));
            count += t != 0.0;
          }
  EXPECT_GT(count, 1000);
}

TEST(OperatorTruncate, HermitianSubmatrix) {
  CheckSubmatrix(Hermiticity::kHermitian);
}

TEST(OperatorTruncate, AntiHermitianSubmatrix) {
  CheckSubmatrix(Hermiticity::kAntiHermitian);
}

TEST(OperatorTruncate, SameSpaceIsIdentity) {
  const auto ms = Ms(3, "O16", "O16");
  Operator A = RandomOperator(ms, OperatorSpec::Scalar(2), {.seed = 3});
  const Operator T = Truncate(A, Ms(3, "O16", "O16"));
  A.SyncToHost();
  T.SyncToHost();
  const auto& x = A.two_body().matrix().buffer();
  const auto& y = T.two_body().matrix().buffer();
  ASSERT_EQ(x.size(), y.size());
  const auto xs = x.ToVector();
  const auto ys = y.ToVector();
  EXPECT_EQ(xs, ys);
}

TEST(OperatorTruncate, MissingOrbitThrows) {
  const Operator A =
      RandomOperator(Ms(2, "O16", "O16"), OperatorSpec::Scalar(2));
  EXPECT_THROW(Truncate(A, Ms(3, "O16", "O16")), InputError);
}

TEST(OperatorTruncate, OneBodyOperator) {
  const Operator A = RandomOperator(Ms(3, "O16", "O16"),
                                    OperatorSpec::Scalar(1), {.seed = 9});
  const auto small = Ms(1, "O16", "O16");
  const Operator T = Truncate(A, small);
  EXPECT_EQ(T.spec().particle_rank, 1);
  A.SyncToHost();
  T.SyncToHost();
  EXPECT_EQ(T.one_body().get(OrbitIndex(0), OrbitIndex(0)),
            A.one_body().get(OrbitIndex(0), OrbitIndex(0)));
}

}  // namespace
}  // namespace imsrg::op
