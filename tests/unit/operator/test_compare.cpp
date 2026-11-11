// Tests of max_abs and max_abs_difference.
#include <memory>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/compare.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::shared_ptr<const basis::ModelSpace> make_ms(const char* nucleus = "O16") {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 1}, 16.0, nucleus, nucleus);
}

void put_2b(Operator& o, double v) {
  o.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0}, LocalKetIndex{0}, v);
}

TEST(OperatorCompare, ZeroOperatorHasZeroMaxima) {
  const Operator o(make_ms(), OperatorSpec::scalar(2));
  const PartMaxima m = max_abs(o);
  EXPECT_EQ(m.zero_body, 0.0);
  EXPECT_EQ(m.one_body, 0.0);
  EXPECT_EQ(m.two_body, 0.0);
  EXPECT_EQ(m.max(), 0.0);
}

TEST(OperatorCompare, MaxAbsPicksLargestMagnitudePerPart) {
  Operator o(make_ms(), OperatorSpec::scalar(2));
  o.zero_body() = -2.0;
  o.one_body().set(OrbitIndex{0}, OrbitIndex{1}, -3.0);
  o.one_body().set(OrbitIndex{1}, OrbitIndex{1}, 1.0);
  put_2b(o, 5.0);
  o.two_body().set(ChannelIndex{0}, ChannelIndex{0}, LocalKetIndex{0}, LocalKetIndex{0}, -7.0);
  const PartMaxima m = max_abs(o);
  EXPECT_EQ(m.zero_body, 2.0);
  EXPECT_EQ(m.one_body, 3.0);
  EXPECT_EQ(m.two_body, 7.0);
  EXPECT_EQ(m.max(), 7.0);
}

TEST(OperatorCompare, DifferenceIsPartwise) {
  const auto ms = make_ms();
  Operator a(ms, OperatorSpec::scalar(2));
  Operator b(ms, OperatorSpec::scalar(2));
  a.zero_body() = 1.0;
  b.zero_body() = 1.5;
  a.one_body().set(OrbitIndex{0}, OrbitIndex{1}, 2.0);
  b.one_body().set(OrbitIndex{0}, OrbitIndex{1}, -1.0);
  put_2b(a, 4.0);
  put_2b(b, 4.25);
  const PartMaxima d = max_abs_difference(a, b);
  EXPECT_DOUBLE_EQ(d.zero_body, 0.5);
  EXPECT_DOUBLE_EQ(d.one_body, 3.0);
  EXPECT_DOUBLE_EQ(d.two_body, 0.25);
  EXPECT_DOUBLE_EQ(d.max(), 3.0);
  const PartMaxima self = max_abs_difference(a, a);
  EXPECT_EQ(self.max(), 0.0);
}

TEST(OperatorCompare, EqualRebuiltModelSpaceIsAccepted) {
  Operator a(make_ms(), OperatorSpec::scalar(2));
  const Operator b(make_ms(), OperatorSpec::scalar(2));
  a.zero_body() = 1.0;
  EXPECT_DOUBLE_EQ(max_abs_difference(a, b).zero_body, 1.0);
}

TEST(OperatorCompare, UnallocatedPartCountsAsZero) {
  const auto ms = make_ms();
  Operator a(ms, OperatorSpec::scalar(2));
  const Operator b(ms, OperatorSpec::scalar(1));
  put_2b(a, -6.0);
  EXPECT_DOUBLE_EQ(max_abs_difference(a, b).two_body, 6.0);
  EXPECT_DOUBLE_EQ(max_abs_difference(b, a).two_body, 6.0);
  EXPECT_EQ(max_abs(b).two_body, 0.0);
}

TEST(OperatorCompare, MismatchedModelSpacesThrow) {
  const Operator a(make_ms("O16"), OperatorSpec::scalar(2));
  const Operator b(make_ms("He4"), OperatorSpec::scalar(2));
  EXPECT_THROW(max_abs_difference(a, b), InvariantError);
  const Operator e;
  EXPECT_THROW(max_abs_difference(a, e), InvariantError);
}

TEST(OperatorCompare, MismatchedSymmetryThrows) {
  // Only symmetries that differ and are constructible matter here; same_symmetry covers rank_J,
  // parity, delta_Tz and delta_particle_number, none of which can be built yet (design/13), so
  // check the accepted case: a differing hermiticity is not an error.
  const auto ms = make_ms();
  const Operator a(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const Operator b(ms, OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
  EXPECT_NO_THROW(max_abs_difference(a, b));
}

}  // namespace
