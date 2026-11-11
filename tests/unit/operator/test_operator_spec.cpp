// Tests of OperatorSpec helpers: hermiticity codes and validate().
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

TEST(OperatorSpec, HermiticityCodesRoundTrip) {
  EXPECT_EQ(hermiticity_code(Hermiticity::Hermitian), 'H');
  EXPECT_EQ(hermiticity_code(Hermiticity::AntiHermitian), 'A');
  EXPECT_EQ(hermiticity_code(Hermiticity::NonHermitian), 'N');
  for (const char c : {'H', 'A', 'N'}) {
    EXPECT_EQ(hermiticity_code(hermiticity_from_code(c)), c);
  }
}

TEST(OperatorSpec, BadHermiticityCodeThrowsInputError) {
  EXPECT_THROW(hermiticity_from_code('X'), InputError);
  EXPECT_THROW(hermiticity_from_code('h'), InputError);
  EXPECT_THROW(hermiticity_from_code('\0'), InputError);
}

TEST(OperatorSpec, HermiticitySigns) {
  EXPECT_EQ(hermiticity_sign(Hermiticity::Hermitian), 1);
  EXPECT_EQ(hermiticity_sign(Hermiticity::AntiHermitian), -1);
  EXPECT_EQ(hermiticity_sign(Hermiticity::NonHermitian), 0);
}

TEST(OperatorSpec, ScalarRanksOneAndTwoAreValid) {
  for (const auto h :
       {Hermiticity::Hermitian, Hermiticity::AntiHermitian, Hermiticity::NonHermitian}) {
    EXPECT_NO_THROW(validate(OperatorSpec::scalar(1, h)));
    EXPECT_NO_THROW(validate(OperatorSpec::scalar(2, h)));
  }
}

TEST(OperatorSpec, OutOfRangeFieldsAreInputErrors) {
  OperatorSpec s = OperatorSpec::scalar(2);
  s.particle_rank = 0;
  EXPECT_THROW(validate(s), InputError);
  s.particle_rank = 4;
  EXPECT_THROW(validate(s), InputError);
  s = OperatorSpec::scalar(2);
  s.rank_J = -1;
  EXPECT_THROW(validate(s), InputError);
  s = OperatorSpec::scalar(2);
  s.delta_Tz = -1;
  EXPECT_THROW(validate(s), InputError);
  s = OperatorSpec::scalar(2);
  s.delta_particle_number = 2;
  EXPECT_THROW(validate(s), InputError);
  s.delta_particle_number = -2;
  EXPECT_THROW(validate(s), InputError);
}

TEST(OperatorSpec, UnimplementedFeaturesThrowNotImplemented) {
  OperatorSpec s = OperatorSpec::scalar(2);
  s.rank_J = 1;
  EXPECT_THROW(validate(s), NotImplementedError);
  s = OperatorSpec::scalar(2);
  s.parity = Parity::Odd;
  EXPECT_THROW(validate(s), NotImplementedError);
  s = OperatorSpec::scalar(2);
  s.delta_Tz = 1;
  EXPECT_THROW(validate(s), NotImplementedError);
  s = OperatorSpec::scalar(2);
  s.delta_particle_number = 1;
  EXPECT_THROW(validate(s), NotImplementedError);
  s.delta_particle_number = -1;
  EXPECT_THROW(validate(s), NotImplementedError);
  s = OperatorSpec::scalar(3);
  EXPECT_THROW(validate(s), NotImplementedError);
}

}  // namespace
