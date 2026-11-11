// Tests of OneBody: partner semantics, storage layout, views, value semantics.
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/operator/one_body.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

OrbitIndex oi(int i) { return OrbitIndex{i}; }

TEST(OneBody, DefaultIsEmptyAndSizedIsZero) {
  const OneBody e;
  EXPECT_EQ(e.num_orbits(), 0);
  const OneBody b(5, Hermiticity::Hermitian);
  EXPECT_EQ(b.num_orbits(), 5);
  for (int a = 0; a < 5; ++a) {
    for (int c = 0; c < 5; ++c) EXPECT_EQ(b.get(oi(a), oi(c)), 0.0);
  }
}

TEST(OneBody, HermitianSetWritesPartner) {
  OneBody b(4, Hermiticity::Hermitian);
  b.set(oi(1), oi(3), 2.5);
  EXPECT_EQ(b.get(oi(1), oi(3)), 2.5);
  EXPECT_EQ(b.get(oi(3), oi(1)), 2.5);
}

TEST(OneBody, AntiHermitianSetNegatesPartner) {
  OneBody b(4, Hermiticity::AntiHermitian);
  b.set(oi(1), oi(3), 2.5);
  EXPECT_EQ(b.get(oi(3), oi(1)), -2.5);
}

TEST(OneBody, NonHermitianSetHasNoPartner) {
  OneBody b(4, Hermiticity::NonHermitian);
  b.set(oi(1), oi(3), 2.5);
  EXPECT_EQ(b.get(oi(1), oi(3)), 2.5);
  EXPECT_EQ(b.get(oi(3), oi(1)), 0.0);
}

TEST(OneBody, DiagonalIsWrittenOnce) {
  OneBody b(3, Hermiticity::Hermitian);
  b.add(oi(2), oi(2), 1.5);
  b.add(oi(2), oi(2), 1.5);
  EXPECT_EQ(b.get(oi(2), oi(2)), 3.0);
  b.set(oi(0), oi(0), 4.0);
  EXPECT_EQ(b.get(oi(0), oi(0)), 4.0);
}

TEST(OneBody, AddAccumulatesWithPartner) {
  OneBody b(3, Hermiticity::AntiHermitian);
  b.add(oi(0), oi(2), 1.0);
  b.add(oi(0), oi(2), 2.0);
  EXPECT_EQ(b.get(oi(0), oi(2)), 3.0);
  EXPECT_EQ(b.get(oi(2), oi(0)), -3.0);
  b.add(oi(2), oi(0), 1.0);  // the partner entry gets +1, the (0,2) entry gets -1
  EXPECT_EQ(b.get(oi(0), oi(2)), 2.0);
  EXPECT_EQ(b.get(oi(2), oi(0)), -2.0);
}

TEST(OneBody, SetElementHasNoPartner) {
  OneBody b(3, Hermiticity::Hermitian);
  b.set_element(oi(0), oi(1), 7.0);
  EXPECT_EQ(b.get(oi(0), oi(1)), 7.0);
  EXPECT_EQ(b.get(oi(1), oi(0)), 0.0);
}

TEST(OneBody, SetHermiticityChangesLaterWritesOnly) {
  OneBody b(3, Hermiticity::Hermitian);
  b.set(oi(0), oi(1), 1.0);
  b.set_hermiticity(Hermiticity::AntiHermitian);
  EXPECT_EQ(b.hermiticity(), Hermiticity::AntiHermitian);
  EXPECT_EQ(b.get(oi(1), oi(0)), 1.0);
  b.set(oi(0), oi(2), 1.0);
  EXPECT_EQ(b.get(oi(2), oi(0)), -1.0);
}

TEST(OneBody, SetZero) {
  OneBody b(3, Hermiticity::Hermitian);
  b.set(oi(0), oi(1), 1.0);
  b.set(oi(2), oi(2), 3.0);
  b.set_zero();
  for (int a = 0; a < 3; ++a) {
    for (int c = 0; c < 3; ++c) EXPECT_EQ(b.get(oi(a), oi(c)), 0.0);
  }
}

TEST(OneBody, ViewsAreColumnMajor) {
  OneBody b(4, Hermiticity::NonHermitian);
  b.set(oi(1), oi(3), 9.0);
  const auto v = b.host_view();
  EXPECT_EQ(v.rows, 4);
  EXPECT_EQ(v.cols, 4);
  EXPECT_EQ(v.ld, 4);
  EXPECT_EQ(v(1, 3), 9.0);
  EXPECT_EQ(v.data[1 + 3 * 4], 9.0);
  const OneBody& cb = b;
  EXPECT_EQ(cb.host_view()(1, 3), 9.0);
  EXPECT_EQ(cb.host_view().data, v.data);
  EXPECT_EQ(b.device_view().rows, 4);
  EXPECT_EQ(b.device_view().ld, 4);
  EXPECT_EQ(cb.device_view().cols, 4);
  EXPECT_NE(b.device_view().data, nullptr);
  b.host_view()(0, 0) = 5.0;
  EXPECT_EQ(b.get(oi(0), oi(0)), 5.0);
  EXPECT_EQ(b.buffer().host_span().size(), 16u);
}

TEST(OneBody, CopyIsDeepMoveIsCheap) {
  OneBody a(3, Hermiticity::Hermitian);
  a.set(oi(0), oi(1), 1.0);
  OneBody c = a;
  c.set(oi(0), oi(1), 2.0);
  EXPECT_EQ(a.get(oi(0), oi(1)), 1.0);
  EXPECT_EQ(c.get(oi(1), oi(0)), 2.0);
  const double* p = a.host_view().data;
  OneBody m = std::move(a);
  EXPECT_EQ(m.host_view().data, p);
  EXPECT_EQ(m.get(oi(1), oi(0)), 1.0);
}

}  // namespace
