// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of Operator: construction, sizing, hermiticity propagation,
// same_model_space, copies.
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> MakeMs(int emax,
                                                const char* nucleus = "O16") {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, nucleus, nucleus);
}

TEST(Operator, DefaultIsEmpty) {
  const Operator o;
  EXPECT_TRUE(o.empty());
  EXPECT_EQ(o.model_space_ptr(), nullptr);
  EXPECT_THROW((void)o.model_space(), InvariantError);
}

TEST(Operator, SizesFollowTheModelSpace) {
  const auto ms = MakeMs(1);
  const Operator o(ms, OperatorSpec::Scalar(2));
  EXPECT_FALSE(o.empty());
  EXPECT_EQ(&o.model_space(), ms.get());
  EXPECT_EQ(o.one_body().num_orbits(), ms->num_orbits());
  EXPECT_TRUE(o.two_body().allocated());
  EXPECT_EQ(o.two_body().matrix().num_blocks(),
            static_cast<std::size_t>(ms->two_body().num_channels()));
  EXPECT_EQ(o.zero_body(), 0.0);
  EXPECT_FALSE(o.three_body().has_value());
  EXPECT_EQ(o.hermiticity(), Hermiticity::kHermitian);

  const Operator o1(ms, OperatorSpec::Scalar(1));
  EXPECT_FALSE(o1.two_body().allocated());
  EXPECT_EQ(o1.one_body().num_orbits(), ms->num_orbits());
}

TEST(Operator, ConstructorValidates) {
  const auto ms = MakeMs(1);
  OperatorSpec bad = OperatorSpec::Scalar(2);
  bad.particle_rank = 0;
  EXPECT_THROW(Operator(ms, bad), InputError);
  EXPECT_THROW(Operator(ms, OperatorSpec::Scalar(3)), NotImplementedError);
  EXPECT_THROW(Operator(nullptr, OperatorSpec::Scalar(2)), InvariantError);
}

TEST(Operator, SetHermiticityUpdatesSpecAndParts) {
  Operator o(MakeMs(1), OperatorSpec::Scalar(2));
  o.set_hermiticity(Hermiticity::kAntiHermitian);
  EXPECT_EQ(o.hermiticity(), Hermiticity::kAntiHermitian);
  EXPECT_EQ(o.spec().hermiticity, Hermiticity::kAntiHermitian);
  EXPECT_EQ(o.one_body().hermiticity(), Hermiticity::kAntiHermitian);
  EXPECT_EQ(o.two_body().hermiticity(), Hermiticity::kAntiHermitian);
  o.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 1.0);
  EXPECT_EQ(o.one_body().get(OrbitIndex{1}, OrbitIndex{0}), -1.0);
}

TEST(Operator, SetZeroClearsAllParts) {
  Operator o(MakeMs(1), OperatorSpec::Scalar(2));
  o.zero_body() = 3.0;
  o.one_body().set(OrbitIndex{0}, OrbitIndex{0}, 1.0);
  o.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                   LocalKetIndex{0}, 2.0);
  o.set_zero();
  EXPECT_EQ(o.zero_body(), 0.0);
  EXPECT_EQ(o.one_body().get(OrbitIndex{0}, OrbitIndex{0}), 0.0);
  EXPECT_EQ(o.two_body().get(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                             LocalKetIndex{0}),
            0.0);
}

TEST(Operator, SyncRoundTripKeepsHostData) {
  Operator o(MakeMs(1), OperatorSpec::Scalar(2));
  o.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 4.0);
  o.SyncToDevice();
  o.SyncToHost();
  EXPECT_EQ(o.one_body().get(OrbitIndex{1}, OrbitIndex{0}), 4.0);
}

TEST(Operator, SameModelSpace) {
  const auto ms = MakeMs(1);
  const Operator a(ms, OperatorSpec::Scalar(2));
  const Operator b(ms, OperatorSpec::Scalar(1));
  EXPECT_TRUE(SameModelSpace(a, b));
  const Operator rebuilt(MakeMs(1), OperatorSpec::Scalar(2));
  EXPECT_NE(rebuilt.model_space_ptr(), a.model_space_ptr());
  EXPECT_TRUE(SameModelSpace(a, rebuilt));
  const Operator he4(MakeMs(1, "He4"), OperatorSpec::Scalar(2));
  EXPECT_FALSE(SameModelSpace(a, he4));
  const Operator bigger(MakeMs(2), OperatorSpec::Scalar(2));
  EXPECT_FALSE(SameModelSpace(a, bigger));
  const Operator empty;
  EXPECT_FALSE(SameModelSpace(a, empty));
  EXPECT_FALSE(SameModelSpace(empty, a));
  EXPECT_FALSE(SameModelSpace(empty, empty));
}

TEST(Operator, CopyIsDeepAndShareslModelSpace) {
  Operator a(MakeMs(1), OperatorSpec::Scalar(2));
  a.zero_body() = 1.0;
  a.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 2.0);
  a.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                   LocalKetIndex{0}, 3.0);
  Operator c = a;
  EXPECT_EQ(c.model_space_ptr(), a.model_space_ptr());
  c.zero_body() = 10.0;
  c.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 20.0);
  c.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                   LocalKetIndex{0}, 30.0);
  EXPECT_EQ(a.zero_body(), 1.0);
  EXPECT_EQ(a.one_body().get(OrbitIndex{0}, OrbitIndex{1}), 2.0);
  EXPECT_EQ(a.two_body().get(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                             LocalKetIndex{0}),
            3.0);
}

TEST(Operator, MovedFromIsEmpty) {
  Operator a(MakeMs(1), OperatorSpec::Scalar(2));
  a.zero_body() = 5.0;
  const auto* data = a.one_body().host_view().data;
  Operator b = std::move(a);
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move): documented
                           // moved-from state
  EXPECT_FALSE(b.empty());
  EXPECT_EQ(b.zero_body(), 5.0);
  EXPECT_EQ(b.one_body().host_view().data, data);
}

}  // namespace
}  // namespace imsrg::op
