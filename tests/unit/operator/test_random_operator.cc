// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of random_operator: determinism, symmetries, selection rules, scales
// and the generator's fixed output.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> MakeMs() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "O16", "O16");
}

}  // namespace

TEST(OperatorRandom, SameSeedSameOperatorDifferentSeedDiffers) {
  const auto ms = MakeMs();
  const RandomOperatorOptions o{.seed = 42, .zero_body_scale = 3.0};
  const Operator a = RandomOperator(ms, OperatorSpec::Scalar(2), o);
  const Operator b = RandomOperator(ms, OperatorSpec::Scalar(2), o);
  EXPECT_EQ(MaxAbsDifference(a, b).max(), 0.0);
  const Operator c = RandomOperator(ms, OperatorSpec::Scalar(2),
                                    {.seed = 43, .zero_body_scale = 3.0});
  EXPECT_GT(MaxAbsDifference(a, c).one_body, 0.0);
  EXPECT_GT(MaxAbsDifference(a, c).two_body, 0.0);
}

TEST(OperatorRandom, ZeroBodyFollowsSplitmix64) {
  // splitmix64 with seed 0 starts with 0xE220A8397B1DCDAF.
  const auto ms = MakeMs();
  const Operator a = RandomOperator(ms, OperatorSpec::Scalar(2),
                                    {.seed = 0, .zero_body_scale = 2.0});
  const double u = static_cast<double>(0xE220A8397B1DCDAFULL >> 11) * 0x1.0p-53;
  EXPECT_DOUBLE_EQ(a.zero_body(), 2.0 * (2.0 * u - 1.0));
  const Operator z = RandomOperator(ms, OperatorSpec::Scalar(2));
  EXPECT_EQ(z.zero_body(), 0.0);
}

TEST(OperatorRandom, HermitianSymmetriesAndSelectionRule) {
  const auto ms = MakeMs();
  const Operator o =
      RandomOperator(ms, OperatorSpec::Scalar(2),
                     {.seed = 7, .one_body_scale = 0.5, .two_body_scale = 2.0});
  const std::int32_t n = ms->num_orbits();
  double max1 = 0.0;
  bool any_nonzero = false;
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < n; ++b) {
      const double v = o.one_body().get(OrbitIndex{a}, OrbitIndex{b});
      EXPECT_EQ(v, o.one_body().get(OrbitIndex{b}, OrbitIndex{a}));
      const auto& oa = ms->basis().orbit(OrbitIndex{a});
      const auto& ob = ms->basis().orbit(OrbitIndex{b});
      if (oa.l != ob.l || oa.j2.twice != ob.j2.twice ||
          oa.tz2.twice != ob.tz2.twice) {
        EXPECT_EQ(v, 0.0);
      } else {
        any_nonzero = any_nonzero || v != 0.0;
      }
      max1 = std::max(max1, std::abs(v));
    }
  }
  EXPECT_TRUE(any_nonzero);
  EXPECT_LE(max1, 0.5);
  const auto& chans = o.two_body().channels();
  double max2 = 0.0;
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const std::int32_t nk = chans.channel(ChannelIndex{c}).num_kets;
    for (std::int32_t i = 0; i < nk; ++i) {
      for (std::int32_t j = 0; j < nk; ++j) {
        const double v = o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                          LocalKetIndex{i}, LocalKetIndex{j});
        EXPECT_EQ(v, o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                      LocalKetIndex{j}, LocalKetIndex{i}));
        max2 = std::max(max2, std::abs(v));
      }
    }
  }
  EXPECT_LE(max2, 2.0);
  EXPECT_GT(max2, 1.0);  // the whole range is used
}

TEST(OperatorRandom, AntiHermitianHasZeroDiagonalAndNoZeroBody) {
  const auto ms = MakeMs();
  const Operator o =
      RandomOperator(ms, OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian),
                     {.seed = 8, .zero_body_scale = 5.0});
  EXPECT_EQ(o.zero_body(), 0.0);
  const std::int32_t n = ms->num_orbits();
  for (std::int32_t a = 0; a < n; ++a) {
    EXPECT_EQ(o.one_body().get(OrbitIndex{a}, OrbitIndex{a}), 0.0);
    for (std::int32_t b = 0; b < n; ++b) {
      EXPECT_EQ(o.one_body().get(OrbitIndex{a}, OrbitIndex{b}),
                -o.one_body().get(OrbitIndex{b}, OrbitIndex{a}));
    }
  }
  const auto& chans = o.two_body().channels();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const std::int32_t nk = chans.channel(ChannelIndex{c}).num_kets;
    for (std::int32_t i = 0; i < nk; ++i) {
      EXPECT_EQ(o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                 LocalKetIndex{i}, LocalKetIndex{i}),
                0.0);
      for (std::int32_t j = 0; j < nk; ++j) {
        EXPECT_EQ(o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                   LocalKetIndex{i}, LocalKetIndex{j}),
                  -o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                    LocalKetIndex{j}, LocalKetIndex{i}));
      }
    }
  }
}

TEST(OperatorRandom, NonHermitianHasIndependentElements) {
  const auto ms = MakeMs();
  const Operator o = RandomOperator(
      ms, OperatorSpec::Scalar(2, Hermiticity::kNonHermitian), {.seed = 9});
  const auto& chans = o.two_body().channels();
  bool asymmetric = false;
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const std::int32_t nk = chans.channel(ChannelIndex{c}).num_kets;
    for (std::int32_t i = 0; i < nk; ++i) {
      for (std::int32_t j = i + 1; j < nk; ++j) {
        asymmetric = asymmetric ||
                     o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                      LocalKetIndex{i}, LocalKetIndex{j}) !=
                         o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                          LocalKetIndex{j}, LocalKetIndex{i});
      }
    }
  }
  EXPECT_TRUE(asymmetric);
}

TEST(OperatorRandom, OneBodyOnlySpecHasNoTwoBody) {
  const auto ms = MakeMs();
  const Operator o = RandomOperator(ms, OperatorSpec::Scalar(1), {.seed = 2});
  EXPECT_FALSE(o.two_body().allocated());
  EXPECT_GT(MaxAbs(o).one_body, 0.0);
}

}  // namespace imsrg::op
