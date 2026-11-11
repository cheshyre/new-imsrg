// Tests of TwoBody: block layout, orbit-based accessors with exchange phases, partner rules.
#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/two_body.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::shared_ptr<const basis::ModelSpace> ms2() {
  static const auto ms =
      basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 2}, 16.0, "O16", "O16");
  return ms;
}

/// Orbit |n l 2j tz2> of the shared model space.
OrbitIndex orb(int n, int l, int j2, int tz2) {
  return ms2()->basis().index_of(basis::Orbit{n, l, TwoJ{j2}, TwoTz{tz2}});
}

int jtwice(OrbitIndex a) { return ms2()->basis().orbit(a).j2.twice; }

/// Exchange phase of |ba; J> relative to |ab; J>.
int ph(OrbitIndex a, OrbitIndex b, TwoJ J) {
  return basis::exchange_phase(TwoJ{jtwice(a)}, TwoJ{jtwice(b)}, J);
}

TwoBody make(Hermiticity h, int rank = 2) { return TwoBody(ms2(), OperatorSpec::scalar(rank, h)); }

struct Case {
  OrbitIndex a, b, c, d;
  TwoJ J;
};

std::vector<Case> cases() {
  const auto p0s = orb(0, 0, 1, -1);
  const auto n0s = orb(0, 0, 1, +1);
  const auto p1s = orb(1, 0, 1, -1);
  const auto p0p32 = orb(0, 1, 3, -1);
  const auto p0p12 = orb(0, 1, 1, -1);
  const auto p0d52 = orb(0, 2, 5, -1);
  const auto p0d32 = orb(0, 2, 3, -1);
  const auto n0d52 = orb(0, 2, 5, +1);
  return {
      {p0p32, p0p12, p0p32, p0p12, TwoJ{2}},  // bra == ket (diagonal), J = 1
      {p0p32, p0p12, p0d52, p0d32, TwoJ{4}},  // pp, J = 2, different j on each side
      {p0s, p0s, p1s, p1s, TwoJ{0}},          // a == b and c == d
      {p0s, n0s, n0d52, p0d52, TwoJ{2}},      // pn, reversed ket order
      {n0s, p0s, p0d52, n0d52, TwoJ{2}},      // pn, reversed both
      {p0d52, p0d52, p0p32, p0p12, TwoJ{4}},  // a == b only
  };
}

TEST(TwoBody, BlockLayoutCoversAllChannels) {
  const TwoBody t = make(Hermiticity::Hermitian);
  const auto& tb = t.channels();
  ASSERT_TRUE(t.allocated());
  ASSERT_EQ(t.matrix().num_blocks(), static_cast<std::size_t>(tb.num_channels()));
  std::int64_t total = 0;
  for (int ch = 0; ch < tb.num_channels(); ++ch) {
    const ChannelIndex c{ch};
    const auto k = t.block_index(c, c);
    ASSERT_TRUE(k.has_value());
    EXPECT_EQ(*k, static_cast<std::size_t>(ch));
    const auto& info = t.matrix().block_info(*k);
    const std::int64_t nk = tb.channel(c).num_kets;
    EXPECT_EQ(info.rows, nk);
    EXPECT_EQ(info.cols, nk);
    EXPECT_EQ(info.row_block.value(), ch);
    EXPECT_EQ(info.col_block.value(), ch);
    total += nk * nk;
    EXPECT_EQ(t.host_block(c).rows, nk);
    EXPECT_EQ(t.host_block(c).cols, nk);
  }
  EXPECT_EQ(t.matrix().total_size(), total);
  EXPECT_FALSE(t.block_index(ChannelIndex{0}, ChannelIndex{1}).has_value());
  EXPECT_FALSE(
      t.block_index(ChannelIndex{tb.num_channels()}, ChannelIndex{tb.num_channels()}).has_value());
  EXPECT_FALSE(t.block_index(ChannelIndex{-1}, ChannelIndex{-1}).has_value());
}

TEST(TwoBody, RankOneIsNotAllocated) {
  TwoBody t = make(Hermiticity::Hermitian, 1);
  EXPECT_FALSE(t.allocated());
  EXPECT_FALSE(t.block_index(ChannelIndex{0}, ChannelIndex{0}).has_value());
  const Case c = cases()[0];
  EXPECT_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), 0.0);
  EXPECT_THROW(t.set_normalized(c.a, c.b, c.c, c.d, c.J, 1.0), InvariantError);
  EXPECT_NO_THROW(t.set_zero());
}

TEST(TwoBody, DefaultConstructedIsEmpty) {
  TwoBody t;
  EXPECT_FALSE(t.allocated());
  EXPECT_THROW((void)t.channels(), InvariantError);
}

TEST(TwoBody, NullModelSpaceThrows) {
  EXPECT_THROW(TwoBody(nullptr, OperatorSpec::scalar(2)), InvariantError);
}

TEST(TwoBody, FourOrderingsCarryExchangePhases) {
  for (const auto h : {Hermiticity::Hermitian, Hermiticity::AntiHermitian}) {
    for (const Case& c : cases()) {
      TwoBody t = make(h);
      const double v = 1.25;
      t.set_normalized(c.a, c.b, c.c, c.d, c.J, v);
      const double pb = ph(c.a, c.b, c.J);
      const double pk = ph(c.c, c.d, c.J);
      // An anti-Hermitian part on its own diagonal is zero by construction, skip those.
      const bool diag = (c.a == c.c && c.b == c.d);
      if (diag && h == Hermiticity::AntiHermitian) continue;
      EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.b, c.a, c.c, c.d, c.J), pb * v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.d, c.c, c.J), pk * v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.b, c.a, c.d, c.c, c.J), pb * pk * v);
    }
  }
}

TEST(TwoBody, PartnerFollowsHermiticity) {
  const double v = 0.75;
  for (const Case& c : cases()) {
    if (c.a == c.c && c.b == c.d) continue;
    {
      TwoBody t = make(Hermiticity::Hermitian);
      t.set_normalized(c.a, c.b, c.c, c.d, c.J, v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), v);
    }
    {
      TwoBody t = make(Hermiticity::AntiHermitian);
      t.set_normalized(c.a, c.b, c.c, c.d, c.J, v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), -v);
    }
    {
      TwoBody t = make(Hermiticity::NonHermitian);
      t.set_normalized(c.a, c.b, c.c, c.d, c.J, v);
      EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), v);
      EXPECT_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), 0.0);
    }
  }
}

TEST(TwoBody, DiagonalElementIsWrittenOnce) {
  TwoBody t = make(Hermiticity::Hermitian);
  const Case c = cases()[0];
  t.add_normalized(c.a, c.b, c.c, c.d, c.J, 1.5);
  t.add_normalized(c.a, c.b, c.c, c.d, c.J, 1.5);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), 3.0);
}

TEST(TwoBody, AddNormalizedAppliesSignToPartnerForAntiHermitian) {
  // imsrg++ AddToTBME omits the sign on the partner; here it is applied (design/30 section 6).
  const Case c = cases()[1];
  TwoBody t = make(Hermiticity::AntiHermitian);
  t.add_normalized(c.a, c.b, c.c, c.d, c.J, 2.0);
  t.add_normalized(c.a, c.b, c.c, c.d, c.J, 1.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), 3.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), -3.0);
}

TEST(TwoBody, AddNormalizedHermitianAndNonHermitian) {
  const Case c = cases()[1];
  TwoBody h = make(Hermiticity::Hermitian);
  h.add_normalized(c.a, c.b, c.c, c.d, c.J, 2.0);
  EXPECT_DOUBLE_EQ(h.get_normalized(c.c, c.d, c.a, c.b, c.J), 2.0);
  TwoBody n = make(Hermiticity::NonHermitian);
  n.add_normalized(c.a, c.b, c.c, c.d, c.J, 2.0);
  EXPECT_EQ(n.get_normalized(c.c, c.d, c.a, c.b, c.J), 0.0);
}

TEST(TwoBody, AddNormalizedWithSwappedOrderUsesPhase) {
  const Case c = cases()[1];
  TwoBody t = make(Hermiticity::Hermitian);
  t.add_normalized(c.b, c.a, c.c, c.d, c.J, 1.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.b, c.a, c.c, c.d, c.J), 1.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), ph(c.a, c.b, c.J));
}

TEST(TwoBody, MissingKetsReadZeroAndThrowOnWrite) {
  TwoBody t = make(Hermiticity::Hermitian);
  const auto p0s = orb(0, 0, 1, -1);
  const auto p1s = orb(1, 0, 1, -1);
  const auto p0p32 = orb(0, 1, 3, -1);
  const auto p0d52 = orb(0, 2, 5, -1);
  const auto n0s = orb(0, 0, 1, +1);
  // Pauli: |aa; J odd> does not exist.
  EXPECT_EQ(t.get_normalized(p0p32, p0p32, p1s, p1s, TwoJ{2}), 0.0);
  EXPECT_THROW(t.set_normalized(p0p32, p0p32, p0p32, p0p32, TwoJ{2}, 1.0), InvariantError);
  // Triangle violated: 0s1/2 x 0s1/2 cannot couple to J = 2.
  EXPECT_EQ(t.get_normalized(p0s, p1s, p0s, p1s, TwoJ{4}), 0.0);
  EXPECT_THROW(t.set_normalized(p0s, p1s, p0s, p1s, TwoJ{4}, 1.0), InvariantError);
  // Different parity (s-s vs s-p) and different Tz (pp vs pn) are not connected.
  EXPECT_EQ(t.get_normalized(p0s, p1s, p0s, orb(0, 1, 1, -1), TwoJ{2}), 0.0);
  EXPECT_THROW(t.set_normalized(p0s, p1s, p0s, orb(0, 1, 1, -1), TwoJ{2}, 1.0), InvariantError);
  EXPECT_EQ(t.get_normalized(p0s, p1s, p0s, n0s, TwoJ{2}), 0.0);
  EXPECT_THROW(t.set_normalized(p0s, p1s, p0s, n0s, TwoJ{2}, 1.0), InvariantError);
  EXPECT_THROW(t.add_normalized(p0s, p1s, p0s, n0s, TwoJ{2}, 1.0), InvariantError);
  (void)p0d52;
}

TEST(TwoBody, GetUnnormalizedScalesByDiagonalKets) {
  const auto cs = cases();
  const double v = 1.5;
  struct Expect {
    int idx;
    double factor;
  };
  for (const Expect e : {Expect{0, 1.0}, Expect{2, 2.0}, Expect{5, std::sqrt(2.0)}}) {
    const Case& c = cs[static_cast<std::size_t>(e.idx)];
    TwoBody t = make(Hermiticity::Hermitian);
    t.set_normalized(c.a, c.b, c.c, c.d, c.J, v);
    EXPECT_DOUBLE_EQ(t.get_unnormalized(c.a, c.b, c.c, c.d, c.J), e.factor * v);
  }
}

TEST(TwoBody, BlockViewsAgreeWithAccessors) {
  TwoBody t = make(Hermiticity::NonHermitian);
  const auto& tb = t.channels();
  for (int ch = 0; ch < tb.num_channels(); ++ch) {
    auto blk = t.host_block(ChannelIndex{ch});
    for (std::int64_t j = 0; j < blk.cols; ++j) {
      for (std::int64_t i = 0; i < blk.rows; ++i)
        blk(i, j) = ch + 0.01 * static_cast<double>(i) + 0.0001 * static_cast<double>(j);
    }
  }
  const TwoBody& ct = t;
  for (int ch = 0; ch < tb.num_channels(); ++ch) {
    const auto blk = ct.host_block(ChannelIndex{ch});
    EXPECT_EQ(blk.data, ct.matrix().host_block(static_cast<std::size_t>(ch)).data);
    for (std::int64_t j = 0; j < blk.cols; j += 3) {
      for (std::int64_t i = 0; i < blk.rows; i += 2) {
        EXPECT_EQ(t.get(ChannelIndex{ch}, ChannelIndex{ch}, LocalKetIndex{static_cast<int>(i)},
                        LocalKetIndex{static_cast<int>(j)}),
                  blk(i, j));
      }
    }
    EXPECT_EQ(ct.device_block(ChannelIndex{ch}).rows, blk.rows);
    EXPECT_EQ(t.device_block(ChannelIndex{ch}).cols, blk.cols);
  }
  // Orbit-based access agrees with the block element.
  const Case c = cases()[1];
  const auto ch = *tb.find_channel(c.J, Parity::Even, TwoTz{-2});
  const auto bra = *tb.local_index(ch, tb.ket_index(c.a, c.b));
  const auto ket = *tb.local_index(ch, tb.ket_index(c.c, c.d));
  EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), t.get(ch, ch, bra, ket));
}

TEST(TwoBody, SetStoredElementAndOutOfBlockAccess) {
  TwoBody t = make(Hermiticity::Hermitian);
  const ChannelIndex ch{0};
  ASSERT_GT(t.host_block(ch).rows, 0);
  t.set(ch, ch, LocalKetIndex{0}, LocalKetIndex{0}, 3.0);
  EXPECT_EQ(t.get(ch, ch, LocalKetIndex{0}, LocalKetIndex{0}), 3.0);
  EXPECT_EQ(t.get(ch, ChannelIndex{1}, LocalKetIndex{0}, LocalKetIndex{0}), 0.0);
  EXPECT_THROW(t.set(ch, ChannelIndex{1}, LocalKetIndex{0}, LocalKetIndex{0}, 1.0), InvariantError);
  EXPECT_THROW((void)t.host_block(ChannelIndex{100000}), InvariantError);
}

TEST(TwoBody, SetZeroAndDeepCopy) {
  TwoBody t = make(Hermiticity::Hermitian);
  const Case c = cases()[1];
  t.set_normalized(c.a, c.b, c.c, c.d, c.J, 2.0);
  TwoBody copy = t;
  copy.set_normalized(c.a, c.b, c.c, c.d, c.J, 5.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), 2.0);
  EXPECT_DOUBLE_EQ(copy.get_normalized(c.a, c.b, c.c, c.d, c.J), 5.0);
  EXPECT_NE(t.matrix().buffer().host_span().data(), copy.matrix().buffer().host_span().data());
  t.set_zero();
  EXPECT_EQ(t.get_normalized(c.a, c.b, c.c, c.d, c.J), 0.0);
  EXPECT_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), 0.0);
  EXPECT_DOUBLE_EQ(copy.get_normalized(c.a, c.b, c.c, c.d, c.J), 5.0);
}

TEST(TwoBody, SetHermiticityAffectsLaterWrites) {
  TwoBody t = make(Hermiticity::Hermitian);
  EXPECT_EQ(t.hermiticity(), Hermiticity::Hermitian);
  t.set_hermiticity(Hermiticity::AntiHermitian);
  EXPECT_EQ(t.hermiticity(), Hermiticity::AntiHermitian);
  const Case c = cases()[1];
  t.set_normalized(c.a, c.b, c.c, c.d, c.J, 1.0);
  EXPECT_DOUBLE_EQ(t.get_normalized(c.c, c.d, c.a, c.b, c.J), -1.0);
}

}  // namespace
