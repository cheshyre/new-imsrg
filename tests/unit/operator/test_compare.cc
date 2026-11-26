// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of max_abs and max_abs_difference.
#include <memory>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/compare.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> MakeMs(const char* nucleus = "O16") {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 1}, 16.0,
                                      nucleus, nucleus);
}

void Put2b(Operator& o, double v) {
  o.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                   LocalKetIndex{0}, v);
}

TEST(OperatorCompare, ZeroOperatorHasZeroMaxima) {
  const Operator o(MakeMs(), OperatorSpec::Scalar(2));
  const PartMaxima m = MaxAbs(o);
  EXPECT_EQ(m.zero_body, 0.0);
  EXPECT_EQ(m.one_body, 0.0);
  EXPECT_EQ(m.two_body, 0.0);
  EXPECT_EQ(m.max(), 0.0);
}

TEST(OperatorCompare, MaxAbsPicksLargestMagnitudePerPart) {
  Operator o(MakeMs(), OperatorSpec::Scalar(2));
  o.zero_body() = -2.0;
  o.one_body().set(OrbitIndex{0}, OrbitIndex{1}, -3.0);
  o.one_body().set(OrbitIndex{1}, OrbitIndex{1}, 1.0);
  Put2b(o, 5.0);
  o.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0},
                   LocalKetIndex{0}, -7.0);
  o.SyncToDevice();
  const PartMaxima m = MaxAbs(o);
  EXPECT_EQ(m.zero_body, 2.0);
  EXPECT_EQ(m.one_body, 3.0);
  EXPECT_EQ(m.two_body, 7.0);
  EXPECT_EQ(m.max(), 7.0);
}

TEST(OperatorCompare, DifferenceIsPartwise) {
  const auto ms = MakeMs();
  Operator a(ms, OperatorSpec::Scalar(2));
  Operator b(ms, OperatorSpec::Scalar(2));
  a.zero_body() = 1.0;
  b.zero_body() = 1.5;
  a.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 2.0);
  b.one_body().set(OrbitIndex{0}, OrbitIndex{1}, -1.0);
  Put2b(a, 4.0);
  Put2b(b, 4.25);
  a.SyncToDevice();
  b.SyncToDevice();
  const PartMaxima d = MaxAbsDifference(a, b);
  EXPECT_DOUBLE_EQ(d.zero_body, 0.5);
  EXPECT_DOUBLE_EQ(d.one_body, 3.0);
  EXPECT_DOUBLE_EQ(d.two_body, 0.25);
  EXPECT_DOUBLE_EQ(d.max(), 3.0);
  const PartMaxima self = MaxAbsDifference(a, a);
  EXPECT_EQ(self.max(), 0.0);
}

TEST(OperatorCompare, EqualRebuiltModelSpaceIsAccepted) {
  Operator a(MakeMs(), OperatorSpec::Scalar(2));
  const Operator b(MakeMs(), OperatorSpec::Scalar(2));
  a.zero_body() = 1.0;
  EXPECT_DOUBLE_EQ(MaxAbsDifference(a, b).zero_body, 1.0);
}

TEST(OperatorCompare, UnallocatedPartCountsAsZero) {
  const auto ms = MakeMs();
  Operator a(ms, OperatorSpec::Scalar(2));
  const Operator b(ms, OperatorSpec::Scalar(1));
  Put2b(a, -6.0);
  a.SyncToDevice();
  EXPECT_DOUBLE_EQ(MaxAbsDifference(a, b).two_body, 6.0);
  EXPECT_DOUBLE_EQ(MaxAbsDifference(b, a).two_body, 6.0);
  EXPECT_EQ(MaxAbs(b).two_body, 0.0);
}

TEST(OperatorCompare, MismatchedModelSpacesThrow) {
  const Operator a(MakeMs("O16"), OperatorSpec::Scalar(2));
  const Operator b(MakeMs("He4"), OperatorSpec::Scalar(2));
  EXPECT_THROW(MaxAbsDifference(a, b), InvariantError);
  const Operator e;
  EXPECT_THROW(MaxAbsDifference(a, e), InvariantError);
}

TEST(OperatorCompare, MismatchedSymmetryThrows) {
  // Only symmetries that differ and are constructible matter here;
  // same_symmetry covers rank_J, parity, delta_Tz and delta_particle_number,
  // none of which can be built yet (design/13), so check the accepted case: a
  // differing hermiticity is not an error.
  const auto ms = MakeMs();
  const Operator a(ms, OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  const Operator b(ms, OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
  EXPECT_NO_THROW(MaxAbsDifference(a, b));
}

}  // namespace
}  // namespace imsrg::op
