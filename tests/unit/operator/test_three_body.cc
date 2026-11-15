// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of the three-body operator part: packed/dense storage and element
// access, hermiticity conversions, the imsrg++ norm, Operator integration,
// arithmetic with three-body parts and the guarded functions.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/elementwise.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::op {
namespace {

using basis::KeptRow3Index;

std::shared_ptr<const basis::ModelSpace> He4Space() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "He4", "He4")
      ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 4});
}

OperatorSpec Spec3(Hermiticity h) {
  OperatorSpec s = OperatorSpec::Scalar(3);
  s.hermiticity = h;
  return s;
}

/// A channel with at least `min_dim` rows.
Channel3Index BigChannel(const ThreeBody& t, std::int32_t min_dim) {
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    if (t.dim(Channel3Index{c}) >= min_dim) return Channel3Index{c};
  }
  ADD_FAILURE() << "no channel with dimension " << min_dim;
  return Channel3Index{0};
}

/// Deterministic nonzero-ish value of (seed, ch, i, j); every 7th is zero.
double Val(int seed, std::int32_t ch, std::int64_t i, std::int64_t j) {
  const std::int64_t k = (seed * 131 + ch * 37 + i * 17 + j * 5) % 97;
  if (k % 7 == 3) return 0.0;
  return 0.01 * static_cast<double>(k - 48) + 1.0 / 3.0;
}

/// Fills the whole part through `set` on the lower triangle and the diagonal
/// (all elements for N), consistent with the hermiticity.
void Fill(ThreeBody& t, int seed) {
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    const std::int64_t n = t.dim(Channel3Index{c});
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = t.packed() ? j : 0; i < n; ++i) {
        double v = Val(seed, c, i, j);
        if (i == j && t.hermiticity() == Hermiticity::kAntiHermitian) v = 0.0;
        t.set(Channel3Index{c}, KeptRow3Index{static_cast<std::int32_t>(i)},
              KeptRow3Index{static_cast<std::int32_t>(j)}, v);
      }
    }
  }
  t.SyncToDevice();
}

/// All elements of the full matrices through `get`.
std::vector<double> FullMatrices(const ThreeBody& t) {
  std::vector<double> out;
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    const std::int32_t n = t.dim(Channel3Index{c});
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        out.push_back(
            t.get(Channel3Index{c}, KeptRow3Index{i}, KeptRow3Index{j}));
      }
    }
  }
  return out;
}

TEST(ThreeBody, PackedIndexIsColumnMajorLowerTriangle) {
  for (std::int64_t n = 1; n <= 6; ++n) {
    std::int64_t expected = 0;
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = j; i < n; ++i) {
        EXPECT_EQ(ThreeBody::PackedIndex(n, i, j), expected);
        ++expected;
      }
    }
    EXPECT_EQ(expected, ThreeBody::BlockSize(n, true));
    EXPECT_EQ(ThreeBody::BlockSize(n, false), n * n);
  }
}

TEST(ThreeBody, ConstructionAndLayout) {
  EXPECT_TRUE(ThreeBody().empty());
  EXPECT_EQ(ThreeBody().num_channels(), 0);
  EXPECT_EQ(ImsrgppNorm(ThreeBody()), 0.0);
  EXPECT_THROW(ThreeBody(nullptr, Hermiticity::kHermitian), InputError);
  const auto no3 = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 1}, 16.0, "O16", "O16");
  EXPECT_THROW(ThreeBody(no3, Hermiticity::kHermitian), InputError);

  const auto ms = He4Space();
  const basis::ThreeBodySpace& sp = *ms->three_body_space();
  for (const auto h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
                       Hermiticity::kNonHermitian}) {
    const ThreeBody t(ms, h);
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.packed(), h != Hermiticity::kNonHermitian);
    ASSERT_EQ(t.num_channels(), sp.num_channels());
    const auto off = t.block_offsets();
    ASSERT_EQ(off.size(), static_cast<std::size_t>(sp.num_channels()) + 1);
    std::int64_t total = 0;
    for (std::int32_t c = 0; c < sp.num_channels(); ++c) {
      const std::int64_t n = sp.dim(Channel3Index{c});
      EXPECT_EQ(off[static_cast<std::size_t>(c)], total);
      total += ThreeBody::BlockSize(n, t.packed());
    }
    EXPECT_EQ(off.back(), total);
    EXPECT_EQ(t.size(), static_cast<std::size_t>(total));
    for (const double v : t.host_span()) EXPECT_EQ(v, 0.0);
  }
}

TEST(ThreeBody, ElementAccessHermitian) {
  const auto ms = He4Space();
  ThreeBody t(ms, Hermiticity::kHermitian);
  const Channel3Index ch = BigChannel(t, 3);
  const std::int64_t n = t.dim(ch);
  const std::int64_t off = t.block_offsets()[ch.ToSize()];
  t.set(ch, KeptRow3Index{2}, KeptRow3Index{1}, 1.5);
  EXPECT_EQ(t.host_span()[static_cast<std::size_t>(
                off + ThreeBody::PackedIndex(n, 2, 1))],
            1.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{1}), 1.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{2}), 1.5);
  // Writing the upper element stores it in the lower triangle.
  t.set(ch, KeptRow3Index{0}, KeptRow3Index{2}, -2.0);
  EXPECT_EQ(t.host_span()[static_cast<std::size_t>(
                off + ThreeBody::PackedIndex(n, 2, 0))],
            -2.0);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{0}), -2.0);
  t.Add(ch, KeptRow3Index{0}, KeptRow3Index{2}, 0.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{0}), -1.5);
  t.set(ch, KeptRow3Index{1}, KeptRow3Index{1}, 4.0);
  t.Add(ch, KeptRow3Index{1}, KeptRow3Index{1}, 1.0);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{1}), 5.0);
  t.set_zero();
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{1}), 0.0);
}

TEST(ThreeBody, ElementAccessAntiHermitian) {
  const auto ms = He4Space();
  ThreeBody t(ms, Hermiticity::kAntiHermitian);
  const Channel3Index ch = BigChannel(t, 3);
  const std::int64_t n = t.dim(ch);
  const std::int64_t off = t.block_offsets()[ch.ToSize()];
  t.set(ch, KeptRow3Index{2}, KeptRow3Index{1}, 1.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{1}), 1.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{2}), -1.5);
  t.set(ch, KeptRow3Index{0}, KeptRow3Index{2}, 2.0);  // stores -2 at (2, 0)
  EXPECT_EQ(t.host_span()[static_cast<std::size_t>(
                off + ThreeBody::PackedIndex(n, 2, 0))],
            -2.0);
  EXPECT_EQ(t.get(ch, KeptRow3Index{0}, KeptRow3Index{2}), 2.0);
  t.Add(ch, KeptRow3Index{0}, KeptRow3Index{2}, 1.0);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{0}), -3.0);
  // The diagonal is zero: zero writes are fine, nonzero ones fail.
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{1}), 0.0);
  EXPECT_NO_THROW(t.set(ch, KeptRow3Index{1}, KeptRow3Index{1}, 0.0));
  EXPECT_THROW(t.set(ch, KeptRow3Index{1}, KeptRow3Index{1}, 1.0),
               InvariantError);
  EXPECT_THROW(t.Add(ch, KeptRow3Index{1}, KeptRow3Index{1}, 1.0),
               InvariantError);
}

TEST(ThreeBody, ElementAccessNonHermitianIsDense) {
  const auto ms = He4Space();
  ThreeBody t(ms, Hermiticity::kNonHermitian);
  const Channel3Index ch = BigChannel(t, 3);
  const std::int64_t n = t.dim(ch);
  const std::int64_t off = t.block_offsets()[ch.ToSize()];
  t.set(ch, KeptRow3Index{2}, KeptRow3Index{1}, 1.5);
  EXPECT_EQ(t.host_span()[static_cast<std::size_t>(off + 2 + 1 * n)], 1.5);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{2}), 0.0);
  t.Add(ch, KeptRow3Index{1}, KeptRow3Index{2}, 0.25);
  EXPECT_EQ(t.host_span()[static_cast<std::size_t>(off + 1 + 2 * n)], 0.25);
  t.set(ch, KeptRow3Index{1}, KeptRow3Index{1}, 3.0);  // any diagonal is fine
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{1}), 3.0);
}

TEST(ThreeBody, SetHermiticityConversions) {
  const auto ms = He4Space();
  // H -> N keeps the represented matrix; N -> H keeps the lower triangle.
  ThreeBody t(ms, Hermiticity::kHermitian);
  Fill(t, 1);
  const std::vector<double> full_h = FullMatrices(t);
  const std::size_t packed_size = t.size();
  t.set_hermiticity(Hermiticity::kNonHermitian);
  EXPECT_FALSE(t.packed());
  EXPECT_GT(t.size(), packed_size);
  EXPECT_EQ(FullMatrices(t), full_h);
  t.set_hermiticity(Hermiticity::kHermitian);
  EXPECT_EQ(t.size(), packed_size);
  EXPECT_EQ(FullMatrices(t), full_h);

  // H -> A flips the flag only: lower values are kept, the upper triangle
  // changes sign, the diagonal becomes zero.
  const Channel3Index ch = BigChannel(t, 3);
  const double lower = t.get(ch, KeptRow3Index{2}, KeptRow3Index{1});
  t.set_hermiticity(Hermiticity::kAntiHermitian);
  EXPECT_EQ(t.get(ch, KeptRow3Index{2}, KeptRow3Index{1}), lower);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{2}), -lower);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{1}), 0.0);
  t.set_hermiticity(Hermiticity::kHermitian);
  EXPECT_EQ(t.get(ch, KeptRow3Index{1}, KeptRow3Index{2}), lower);

  // A -> N uses the old sign for the upper triangle.
  ThreeBody a(ms, Hermiticity::kAntiHermitian);
  Fill(a, 2);
  const std::vector<double> full_a = FullMatrices(a);
  a.set_hermiticity(Hermiticity::kNonHermitian);
  EXPECT_EQ(FullMatrices(a), full_a);
  // N -> A drops the upper triangle and zeroes the diagonal.
  ThreeBody n(ms, Hermiticity::kNonHermitian);
  Fill(n, 3);
  const double diag = n.get(ch, KeptRow3Index{0}, KeptRow3Index{0});
  const double low = n.get(ch, KeptRow3Index{2}, KeptRow3Index{0});
  n.set_hermiticity(Hermiticity::kAntiHermitian);
  EXPECT_EQ(n.get(ch, KeptRow3Index{0}, KeptRow3Index{0}), 0.0);
  EXPECT_EQ(n.get(ch, KeptRow3Index{2}, KeptRow3Index{0}), low);
  EXPECT_EQ(n.get(ch, KeptRow3Index{0}, KeptRow3Index{2}), -low);
  (void)diag;
}

TEST(ThreeBody, NormIsImsrgppPackedSum) {
  const auto ms = He4Space();
  for (const auto h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    ThreeBody t(ms, h);
    Fill(t, 4);
    double sum = 0.0;
    for (const double v : t.host_span()) sum += v * v;  // packed sum
    EXPECT_NEAR(ImsrgppNorm(t), std::sqrt(sum), 1.0e-13 * std::sqrt(sum));
    EXPECT_GT(sum, 0.0);
    // sqrt((||Z||_F^2 + sum_i z_ii^2) / 2) with Z the full matrices.
    double frob2 = 0.0;
    double diag2 = 0.0;
    for (std::int32_t c = 0; c < t.num_channels(); ++c) {
      const std::int32_t n = t.dim(Channel3Index{c});
      for (std::int32_t j = 0; j < n; ++j) {
        for (std::int32_t i = 0; i < n; ++i) {
          const double v =
              t.get(Channel3Index{c}, KeptRow3Index{i}, KeptRow3Index{j});
          frob2 += v * v;
          if (i == j) diag2 += v * v;
        }
      }
    }
    EXPECT_NEAR(ImsrgppNorm(t), std::sqrt((frob2 + diag2) / 2.0),
                1.0e-13 * std::sqrt(sum));
  }
}

TEST(ThreeBody, NormOfKnownValues) {
  const auto ms = He4Space();
  ThreeBody t(ms, Hermiticity::kHermitian);
  const Channel3Index ch = BigChannel(t, 2);
  t.set(ch, KeptRow3Index{0}, KeptRow3Index{0}, 3.0);
  t.set(ch, KeptRow3Index{1}, KeptRow3Index{0}, 4.0);  // counted once
  t.SyncToDevice();
  EXPECT_NEAR(ImsrgppNorm(t), 5.0, 1.0e-15);
  // Dense (non-Hermitian): the lower triangle only.
  ThreeBody d(ms, Hermiticity::kNonHermitian);
  d.set(ch, KeptRow3Index{0}, KeptRow3Index{0}, 3.0);
  d.set(ch, KeptRow3Index{1}, KeptRow3Index{0}, 4.0);
  d.set(ch, KeptRow3Index{0}, KeptRow3Index{1}, 100.0);  // upper: ignored
  d.SyncToDevice();
  EXPECT_NEAR(ImsrgppNorm(d), 5.0, 1.0e-15);
}

TEST(ThreeBodyOperator, ConstructionAndNorms) {
  const auto ms = He4Space();
  Operator o2(ms, OperatorSpec::Scalar(2));
  EXPECT_FALSE(o2.three_body().has_value());
  EXPECT_EQ(ThreeBodyNorm(o2), 0.0);

  Operator o3(ms, Spec3(Hermiticity::kHermitian));
  ASSERT_TRUE(o3.three_body().has_value());
  EXPECT_TRUE(o3.two_body().allocated());
  EXPECT_EQ(ThreeBodyNorm(o3), 0.0);
  Fill(*o3.three_body(), 5);
  o3.SyncToDevice();
  const double n3 = ImsrgppNorm(*o3.three_body());
  EXPECT_GT(n3, 0.0);
  EXPECT_EQ(ThreeBodyNorm(o3), n3);
  EXPECT_NEAR(Norm(o3), n3, 1.0e-15);  // the other parts are zero

  // Deep copy: zeroing the copy leaves the original.
  Operator c2 = o3;
  c2.set_zero();
  EXPECT_EQ(ThreeBodyNorm(c2), 0.0);
  EXPECT_EQ(ThreeBodyNorm(o3), n3);

  // Hermiticity: H <-> A keeps the data; N converts.
  o3.set_hermiticity(Hermiticity::kNonHermitian);
  EXPECT_FALSE(o3.three_body()->packed());
  EXPECT_EQ(o3.hermiticity(), Hermiticity::kNonHermitian);

  // Model space without three-body space.
  const auto no3 = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 2}, 16.0, "He4", "He4");
  EXPECT_THROW(Operator(no3, Spec3(Hermiticity::kHermitian)), InputError);
}

TEST(ThreeBodyOperator, AxpyScaleAndRankPromotion) {
  const auto ms = He4Space();
  Operator x(ms, Spec3(Hermiticity::kHermitian));
  Operator y(ms, Spec3(Hermiticity::kHermitian));
  Fill(*x.three_body(), 6);
  Fill(*y.three_body(), 7);
  x.zero_body() = 1.0;

  Operator r = y;
  Axpy(2.0, x, r);
  r.SyncToHost();
  const auto xs = std::as_const(x).three_body()->host_span();
  const auto ys = std::as_const(y).three_body()->host_span();
  const auto rs = std::as_const(r).three_body()->host_span();
  ASSERT_EQ(rs.size(), xs.size());
  for (std::size_t k = 0; k < rs.size(); ++k) {
    EXPECT_DOUBLE_EQ(rs[k], ys[k] + 2.0 * xs[k]);
  }
  EXPECT_EQ(r.hermiticity(), Hermiticity::kHermitian);

  Operator s = x + y;
  s.SyncToHost();
  for (std::size_t k = 0; k < rs.size(); ++k) {
    EXPECT_DOUBLE_EQ(s.three_body()->host_span()[k], xs[k] + ys[k]);
  }
  Operator d = x - y;
  d.SyncToHost();
  for (std::size_t k = 0; k < rs.size(); ++k) {
    EXPECT_DOUBLE_EQ(d.three_body()->host_span()[k], xs[k] - ys[k]);
  }
  Operator m = 3.0 * x;
  m.SyncToHost();
  Operator neg = -x;
  neg.SyncToHost();
  Operator half = x / 2.0;
  half.SyncToHost();
  for (std::size_t k = 0; k < rs.size(); ++k) {
    EXPECT_DOUBLE_EQ(m.three_body()->host_span()[k], 3.0 * xs[k]);
    EXPECT_DOUBLE_EQ(neg.three_body()->host_span()[k], -xs[k]);
    EXPECT_DOUBLE_EQ(half.three_body()->host_span()[k], 0.5 * xs[k]);
  }
  EXPECT_NEAR(ThreeBodyNorm(m), 3.0 * ThreeBodyNorm(x),
              1.0e-12 * ThreeBodyNorm(x));

  // Rank promotion: a rank-2 operand counts as a zero three-body part.
  Operator two(ms, OperatorSpec::Scalar(2));
  two.zero_body() = 2.0;
  Operator p1 = two + x;
  Operator p2 = x + two;
  Operator p3 = two - x;
  EXPECT_EQ(p1.spec().particle_rank, 3);
  EXPECT_EQ(p2.spec().particle_rank, 3);
  EXPECT_EQ(p3.spec().particle_rank, 3);
  EXPECT_DOUBLE_EQ(p1.zero_body(), 3.0);
  p3.SyncToHost();
  for (std::size_t k = 0; k < rs.size(); ++k) {
    EXPECT_DOUBLE_EQ(p1.three_body()->host_span()[k], xs[k]);
    EXPECT_DOUBLE_EQ(p3.three_body()->host_span()[k], -xs[k]);
  }
  // A rank-3 operand cannot be added to a rank-2 one in place.
  Operator t2(ms, OperatorSpec::Scalar(2));
  EXPECT_THROW(Axpy(1.0, x, t2), InvariantError);
}

TEST(ThreeBodyOperator, MixedHermiticityGivesDenseNonHermitian) {
  const auto ms = He4Space();
  Operator h(ms, Spec3(Hermiticity::kHermitian));
  Operator a(ms, Spec3(Hermiticity::kAntiHermitian));
  Fill(*h.three_body(), 8);
  Fill(*a.three_body(), 9);
  const std::vector<double> fh = FullMatrices(*h.three_body());
  const std::vector<double> fa = FullMatrices(*a.three_body());
  for (const bool h_first : {true, false}) {
    Operator r = h_first ? h + a : a + h;
    EXPECT_EQ(r.hermiticity(), Hermiticity::kNonHermitian);
    ASSERT_FALSE(r.three_body()->packed());
    r.SyncToHost();
    const std::vector<double> fr = FullMatrices(*r.three_body());
    ASSERT_EQ(fr.size(), fh.size());
    for (std::size_t k = 0; k < fr.size(); ++k) {
      EXPECT_DOUBLE_EQ(fr[k], fh[k] + fa[k]);
    }
  }
  // With a rank-2 anti-Hermitian operand the flag turns non-Hermitian too.
  Operator two(ms, OperatorSpec::Scalar(2));
  two.set_hermiticity(Hermiticity::kAntiHermitian);
  Operator r = h + two;
  r.SyncToHost();
  EXPECT_FALSE(r.three_body()->packed());
  EXPECT_EQ(FullMatrices(*r.three_body()), fh);
  // MaxAbsDifference compares through the full matrices.
  Operator hn = h;
  hn.set_hermiticity(Hermiticity::kNonHermitian);
  EXPECT_EQ(MaxAbsDifference(h, hn).three_body, 0.0);
  EXPECT_GT(MaxAbs(h).three_body, 0.0);
}

TEST(ThreeBodyOperator, UnsupportedFunctionsThrowNotImplemented) {
  const auto ms = He4Space();
  Operator x(ms, Spec3(Hermiticity::kHermitian));
  EXPECT_THROW((void)NormalOrder(x), NotImplementedError);
  EXPECT_THROW((void)UndoNormalOrder(x), NotImplementedError);
  EXPECT_THROW((void)RandomOperator(ms, Spec3(Hermiticity::kHermitian)),
               NotImplementedError);
  EXPECT_THROW((void)WithModelSpace(x, ms), NotImplementedError);
  EXPECT_THROW(ElementwiseAbs(x), NotImplementedError);
  EXPECT_THROW(AddToEveryElement(x, 1.0), NotImplementedError);
  Operator y = x;
  EXPECT_THROW(ElementwiseDivide(x, y), NotImplementedError);
  const std::int64_t norb = ms->num_orbits();
  std::vector<double> c(static_cast<std::size_t>(norb * norb), 0.0);
  for (std::int64_t i = 0; i < norb; ++i) {
    c[static_cast<std::size_t>(i + i * norb)] = 1.0;
  }
  EXPECT_THROW(
      (void)TransformBasis(
          x, linalg::MatrixView<const double>{c.data(), norb, norb, norb}),
      NotImplementedError);
}

}  // namespace
}  // namespace imsrg::op
