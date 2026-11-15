// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of the operator norms against brute-force sums.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/detail/part_kernels.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> MakeMs() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "O16", "O16");
}

class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : s_(seed) {}
  double Next() {
    s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(static_cast<std::int64_t>(s_ >> 11)) /
               4503599627370496.0 -
           1.0;
  }

 private:
  std::uint64_t s_;
};

/// Random Hermitian operator; 1B entries between all orbit pairs (also
/// different l, j, tz).
Operator MakeOp(int rank, std::uint64_t seed) {
  Operator o(MakeMs(), OperatorSpec::Scalar(rank));
  Lcg rng(seed);
  o.zero_body() = 100.0 * rng.Next();
  const int n = o.model_space().num_orbits();
  for (int p = 0; p < n; ++p) {
    for (int q = p; q < n; ++q) {
      o.one_body().set(OrbitIndex{p}, OrbitIndex{q}, rng.Next());
    }
  }
  if (o.two_body().allocated()) {
    const auto& chans = o.two_body().channels();
    for (int c = 0; c < chans.num_channels(); ++c) {
      const int nk = chans.channels()[static_cast<std::size_t>(c)].num_kets;
      for (int i = 0; i < nk; ++i) {
        for (int j = i; j < nk; ++j) {
          const double v = rng.Next();
          o.two_body().set(ChannelIndex{c}, ChannelIndex{c}, LocalKetIndex{i},
                           LocalKetIndex{j}, v);
          if (i != j) {
            o.two_body().set(ChannelIndex{c}, ChannelIndex{c}, LocalKetIndex{j},
                             LocalKetIndex{i}, v);
          }
        }
      }
    }
  }
  return o;
}

double BruteOneBody(const Operator& o) {
  const auto& b = o.model_space().basis();
  const int n = b.num_orbits();
  double s = 0.0;
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      const auto& op = b.orbit(OrbitIndex{p});
      const auto& oq = b.orbit(OrbitIndex{q});
      if (op.l == oq.l && op.j2 == oq.j2 && op.tz2 == oq.tz2) {
        const double v = o.one_body().get(OrbitIndex{p}, OrbitIndex{q});
        s += v * v * (op.j2.twice + 1);
      }
    }
  }
  return std::sqrt(s);
}

double BruteTwoBody(const Operator& o) {
  if (!o.two_body().allocated()) return 0.0;
  const auto& chans = o.two_body().channels();
  double s = 0.0;
  for (int c = 0; c < chans.num_channels(); ++c) {
    const auto& info = chans.channels()[static_cast<std::size_t>(c)];
    const double w = 2.0 * info.J() + 1.0;
    for (int i = 0; i < info.num_kets; ++i) {
      for (int j = 0; j < info.num_kets; ++j) {
        const double v = o.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                          LocalKetIndex{i}, LocalKetIndex{j});
        s += w * w * v * v;
      }
    }
  }
  return std::sqrt(s);
}

void ExpectRel(double a, double b) {
  EXPECT_NEAR(a, b, 1.0e-12 * std::max(1.0, std::abs(b)));
}

TEST(OperatorNorms, ZeroOperatorHasZeroNorms) {
  const Operator o(MakeMs(), OperatorSpec::Scalar(2));
  EXPECT_EQ(OneBodyNorm(o), 0.0);
  EXPECT_EQ(TwoBodyNorm(o), 0.0);
  EXPECT_EQ(ThreeBodyNorm(o), 0.0);
  EXPECT_EQ(Norm(o), 0.0);
}

TEST(OperatorNorms, MatchBruteForce) {
  const Operator o = MakeOp(2, 1);
  const double n1 = BruteOneBody(o);
  const double n2 = BruteTwoBody(o);
  ASSERT_GT(n1, 1.0);
  ASSERT_GT(n2, 1.0);
  ExpectRel(OneBodyNorm(o), n1);
  ExpectRel(TwoBodyNorm(o), n2);
  EXPECT_EQ(ThreeBodyNorm(o), 0.0);
  ExpectRel(Norm(o), std::sqrt(n1 * n1 + n2 * n2));
}

TEST(OperatorNorms, OneBodyIgnoresDifferentLjtzEntries) {
  Operator o(MakeMs(), OperatorSpec::Scalar(1));
  const auto& b = o.model_space().basis();
  // find two orbits that differ in (l, 2j, tz2), and one orbit pair with equal
  // labels
  OrbitIndex p{0};
  OrbitIndex q{-1};
  for (int i = 1; i < b.num_orbits(); ++i) {
    const auto& a = b.orbit(OrbitIndex{0});
    const auto& c = b.orbit(OrbitIndex{i});
    if (a.l != c.l || a.j2 != c.j2 || a.tz2 != c.tz2) {
      q = OrbitIndex{i};
      break;
    }
  }
  ASSERT_GE(q.value(), 0);
  o.one_body().set(p, q, 5.0);
  EXPECT_EQ(OneBodyNorm(o), 0.0);
  const double w = b.orbit(p).j2.twice + 1;
  o.one_body().set(p, p, 2.0);
  EXPECT_DOUBLE_EQ(OneBodyNorm(o), std::sqrt(4.0 * w));
  // an off-diagonal pair with equal (l, j, tz) but different n counts
  for (int i = 1; i < b.num_orbits(); ++i) {
    const auto& a = b.orbit(p);
    const auto& c = b.orbit(OrbitIndex{i});
    if (a.l == c.l && a.j2 == c.j2 && a.tz2 == c.tz2) {
      o.one_body().set(p, OrbitIndex{i}, 1.0);
      EXPECT_DOUBLE_EQ(OneBodyNorm(o), std::sqrt(4.0 * w + 2.0 * w));
      break;
    }
  }
}

TEST(OperatorNorms, ZeroBodyDoesNotEnterNorm) {
  Operator o = MakeOp(2, 2);
  const double before = Norm(o);
  o.zero_body() += 1.0e3;
  EXPECT_EQ(Norm(o), before);
}

TEST(OperatorNorms, RankOneHasNoTwoBodyNorm) {
  const Operator o = MakeOp(1, 3);
  EXPECT_EQ(TwoBodyNorm(o), 0.0);
  EXPECT_EQ(Norm(o), OneBodyNorm(o));
  ExpectRel(Norm(o), BruteOneBody(o));
}

TEST(OperatorNorms, HomogeneousUnderScaling) {
  const Operator o = MakeOp(2, 4);
  for (const double a : {2.5, -3.0, 0.0}) {
    const Operator s = a * o;
    ExpectRel(Norm(s), std::abs(a) * Norm(o));
    ExpectRel(OneBodyNorm(s), std::abs(a) * OneBodyNorm(o));
    ExpectRel(TwoBodyNorm(s), std::abs(a) * TwoBodyNorm(o));
  }
}

double HostWeightedSum(const std::vector<double>& x,
                       const std::vector<std::int64_t>& off,
                       const std::vector<double>& w) {
  double sum = 0.0;
  for (std::size_t b = 0; b + 1 < off.size(); ++b) {
    double s = 0.0;
    for (std::int64_t i = off[b]; i < off[b + 1]; ++i) {
      s += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    }
    sum += w[b] * s;
  }
  return sum;
}

void ExpectWeightedSum(const std::vector<std::int64_t>& off,
                       const std::vector<double>& w, std::size_t n) {
  Lcg rng(7);
  std::vector<double> x(n);
  for (double& v : x) v = rng.Next();
  exec::DeviceBuffer<double> buf(n);
  buf.CopyFromHost(std::span<const double>(x));
  const double ref = HostWeightedSum(x, off, w);
  const double got = detail::WeightedBlockSumOfSquares(buf, off, w);
  EXPECT_NEAR(got, ref, 1.0e-14 * std::abs(ref));
}

TEST(WeightedBlockSumOfSquares, EmptyBlocksBigBlocksAndChunkBoundaries) {
  // chunk size is 4096: block 2 is larger than a chunk, block 4 ends exactly on
  // a chunk boundary, empty blocks sit at the start, middle and end.
  const std::vector<std::int64_t> off = {
      0, 0, 5, 5, 4000, 12000, 12288, 12288, 12290, 20000, 20000};
  std::vector<double> w(off.size() - 1);
  for (std::size_t b = 0; b < w.size(); ++b)
    w[b] = 1.0 + 0.5 * static_cast<double>(b);
  ExpectWeightedSum(off, w, 20500);
}

TEST(WeightedBlockSumOfSquares, SingleBlockAndTinyBlocks) {
  ExpectWeightedSum({0, 10000}, {3.0}, 10000);
  ExpectWeightedSum({0, 1}, {2.0}, 1);
  ExpectWeightedSum({0, 1, 2, 3, 4097, 4098}, {1.0, 2.0, 3.0, 4.0, 5.0}, 4100);
}

TEST(WeightedBlockSumOfSquares, NoBlocksIsZero) {
  const exec::DeviceBuffer<double> buf(4);
  EXPECT_EQ(detail::WeightedBlockSumOfSquares(buf, {}, {}), 0.0);
}

TEST(OperatorNorms, TwoBodyNormMatchesPerBlockFormulaOnRandomOperators) {
  for (const int emax : {2, 4}) {
    const auto ms = basis::ModelSpace::FromNames(
        basis::BasisTruncation{.emax = emax}, 16.0, "O16", "O16");
    const Operator o = RandomOperator(ms, OperatorSpec::Scalar(2),
                                      RandomOperatorOptions{.seed = 11});
    const std::vector<double> data = o.two_body().matrix().buffer().ToVector();
    const auto channels = o.two_body().channels().channels();
    double sum = 0.0;
    for (const auto& info : o.two_body().matrix().blocks()) {
      double s = 0.0;
      for (std::int64_t i = 0; i < info.rows * info.cols; ++i) {
        const double v = data[static_cast<std::size_t>(info.offset + i)];
        s += v * v;
      }
      const double w =
          2.0 * channels[static_cast<std::size_t>(info.row_block.value())].J() +
          1.0;
      sum += w * w * s;
    }
    const double ref = std::sqrt(sum);
    ASSERT_GT(ref, 1.0);
    EXPECT_NEAR(TwoBodyNorm(o), ref, 1.0e-14 * ref);
  }
}

}  // namespace
}  // namespace imsrg::op
