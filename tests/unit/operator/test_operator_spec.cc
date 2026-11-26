// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of OperatorSpec helpers: hermiticity codes and validate().
#include <cstdint>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::op {
namespace {

TEST(OperatorSpec, HermiticityCodesRoundTrip) {
  EXPECT_EQ(HermiticityCode(Hermiticity::kHermitian), 'H');
  EXPECT_EQ(HermiticityCode(Hermiticity::kAntiHermitian), 'A');
  EXPECT_EQ(HermiticityCode(Hermiticity::kNonHermitian), 'N');
  for (const char c : {'H', 'A', 'N'}) {
    EXPECT_EQ(HermiticityCode(HermiticityFromCode(c)), c);
  }
}

TEST(OperatorSpec, BadHermiticityCodeThrowsInputError) {
  EXPECT_THROW(HermiticityFromCode('X'), InputError);
  EXPECT_THROW(HermiticityFromCode('h'), InputError);
  EXPECT_THROW(HermiticityFromCode('\0'), InputError);
}

TEST(OperatorSpec, HermiticitySigns) {
  EXPECT_EQ(HermiticitySign(Hermiticity::kHermitian), 1);
  EXPECT_EQ(HermiticitySign(Hermiticity::kAntiHermitian), -1);
  EXPECT_EQ(HermiticitySign(Hermiticity::kNonHermitian), 0);
}

TEST(OperatorSpec, ScalarRanksOneAndTwoAreValid) {
  for (const auto h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
                       Hermiticity::kNonHermitian}) {
    EXPECT_NO_THROW(Validate(OperatorSpec::Scalar(1, h)));
    EXPECT_NO_THROW(Validate(OperatorSpec::Scalar(2, h)));
  }
}

TEST(OperatorSpec, OutOfRangeFieldsAreInputErrors) {
  OperatorSpec s = OperatorSpec::Scalar(2);
  s.particle_rank = 0;
  EXPECT_THROW(Validate(s), InputError);
  s.particle_rank = 4;
  EXPECT_THROW(Validate(s), InputError);
  s = OperatorSpec::Scalar(2);
  s.rank_J = -1;
  EXPECT_THROW(Validate(s), InputError);
  s = OperatorSpec::Scalar(2);
  s.delta_Tz = -1;
  EXPECT_THROW(Validate(s), InputError);
  s = OperatorSpec::Scalar(2);
  s.delta_particle_number = 2;
  EXPECT_THROW(Validate(s), InputError);
  s.delta_particle_number = -2;
  EXPECT_THROW(Validate(s), InputError);
}

TEST(OperatorSpec, TensorSpecsAreAcceptedAndReduced) {
  OperatorSpec s = OperatorSpec::Scalar(2);
  EXPECT_FALSE(s.is_reduced());
  for (const auto h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    for (std::int32_t J = 0; J <= 5; ++J) {
      for (const auto p : {Parity::kEven, Parity::kOdd}) {
        for (std::int32_t dtz = 0; dtz <= 2; ++dtz) {
          s = OperatorSpec::Scalar(2, h);
          s.rank_J = J;
          s.parity = p;
          s.delta_Tz = dtz;
          EXPECT_NO_THROW(Validate(s));
          EXPECT_EQ(s.is_reduced(), !s.is_scalar());
        }
      }
    }
  }
  s = OperatorSpec::Scalar(2);
  s.delta_Tz = 3;
  EXPECT_THROW(Validate(s), InputError);
}

TEST(OperatorSpec, UnimplementedFeaturesThrowNotImplemented) {
  OperatorSpec s = OperatorSpec::Scalar(3);
  s.rank_J = 1;
  EXPECT_THROW(Validate(s), NotImplementedError);
  s = OperatorSpec::Scalar(2, Hermiticity::kNonHermitian);
  s.parity = Parity::kOdd;
  EXPECT_THROW(Validate(s), NotImplementedError);
  s = OperatorSpec::Scalar(2);
  s.delta_particle_number = 1;
  EXPECT_THROW(Validate(s), NotImplementedError);
  s.delta_particle_number = -1;
  EXPECT_THROW(Validate(s), NotImplementedError);
  s = OperatorSpec::Scalar(3);
  EXPECT_NO_THROW(Validate(s));
}

}  // namespace
}  // namespace imsrg::op
