// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::linalg {

namespace {

template <class T>
Matrix<T> RandomMatrix(std::mt19937_64& rng, std::int64_t r, std::int64_t c) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  Matrix<T> m(r, c);
  for (std::int64_t j = 0; j < c; ++j)
    for (std::int64_t i = 0; i < r; ++i) m(i, j) = static_cast<T>(d(rng));
  return m;
}

/// Naive reference: result = alpha op(A) op(B) + beta C0 (C0 ignored if beta ==
/// 0).
template <class T>
Matrix<double> Naive(T alpha, MatrixView<const T> A, Op oa,
                     MatrixView<const T> B, Op ob, T beta,
                     MatrixView<const T> C0) {
  const bool ta = oa == Op::kTrans, tb = ob == Op::kTrans;
  const std::int64_t m = ta ? A.cols : A.rows, k = ta ? A.rows : A.cols,
                     n = tb ? B.rows : B.cols;
  Matrix<double> r(m, n);
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < m; ++i) {
      double s = 0;
      for (std::int64_t p = 0; p < k; ++p) {
        const double a = ta ? A(p, i) : A(i, p);
        const double b = tb ? B(j, p) : B(p, j);
        s += a * b;
      }
      r(i, j) = alpha * s +
                (beta == T{0} ? 0.0 : static_cast<double>(beta) * C0(i, j));
    }
  return r;
}

template <class T>
void RunCase(std::mt19937_64& rng, std::int64_t m, std::int64_t n,
             std::int64_t k, Op oa, Op ob, T alpha, T beta, bool nan_c,
             double tol_unit) {
  Matrix<T> A = oa == Op::kTrans ? RandomMatrix<T>(rng, k, m)
                                 : RandomMatrix<T>(rng, m, k);
  Matrix<T> B = ob == Op::kTrans ? RandomMatrix<T>(rng, n, k)
                                 : RandomMatrix<T>(rng, k, n);
  Matrix<T> C = RandomMatrix<T>(rng, m, n);
  if (nan_c) {
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = 0; i < m; ++i) {
        C(i, j) = std::numeric_limits<T>::quiet_NaN();
      }
    }
  }
  Matrix<double> ref = Naive<T>(alpha, A.const_view(), oa, B.const_view(), ob,
                                beta, C.const_view());
  Gemm(alpha, A.const_view(), oa, B.const_view(), ob, beta, C.view());
  double scale = 1.0;
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < m; ++i)
      scale = std::max(scale, std::abs(ref(i, j)));
  const double tol =
      tol_unit * scale * std::max<double>(1.0, static_cast<double>(k));
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < m; ++i) {
      ASSERT_FALSE(std::isnan(C(i, j))) << "NaN at " << i << "," << j;
      ASSERT_NEAR(C(i, j), ref(i, j), tol) << m << "x" << n << "x" << k;
    }
}

}  // namespace

TEST(LinalgGemm, AllOpCombinationsAndShapes) {
  std::mt19937_64 rng(12345);
  const std::int64_t shapes[][3] = {{5, 4, 3}, {1, 7, 5},   {6, 1, 5},
                                    {1, 1, 1}, {13, 9, 17}, {3, 3, 1}};
  for (auto& s : shapes)
    for (Op oa : {Op::kNoTrans, Op::kTrans})
      for (Op ob : {Op::kNoTrans, Op::kTrans})
        RunCase<double>(rng, s[0], s[1], s[2], oa, ob, 1.0, 0.0, false,
                        1.0e-12);
}

TEST(LinalgGemm, AlphaBeta) {
  std::mt19937_64 rng(2);
  const double ab[][2] = {
      {1.0, 1.0}, {2.5, -0.5}, {0.0, 1.0}, {0.0, 0.0}, {-1.0, 2.0}};
  for (auto& p : ab)
    for (Op oa : {Op::kNoTrans, Op::kTrans})
      for (Op ob : {Op::kNoTrans, Op::kTrans})
        RunCase<double>(rng, 7, 5, 6, oa, ob, p[0], p[1], false, 1.0e-12);
}

TEST(LinalgGemm, BetaZeroIgnoresNaN) {
  std::mt19937_64 rng(3);
  RunCase<double>(rng, 6, 5, 4, Op::kNoTrans, Op::kNoTrans, 1.5, 0.0, true,
                  1.0e-12);
  RunCase<double>(rng, 6, 5, 4, Op::kTrans, Op::kTrans, 1.0, 0.0, true,
                  1.0e-12);
  // Also the k == 0 path.
  RunCase<double>(rng, 3, 3, 0, Op::kNoTrans, Op::kNoTrans, 1.0, 0.0, true,
                  1.0e-12);
}

TEST(LinalgGemm, ZeroSized) {
  std::mt19937_64 rng(4);
  RunCase<double>(rng, 0, 4, 3, Op::kNoTrans, Op::kNoTrans, 1.0, 0.0, false,
                  1.0e-12);
  RunCase<double>(rng, 4, 0, 3, Op::kNoTrans, Op::kNoTrans, 1.0, 0.0, false,
                  1.0e-12);
  RunCase<double>(rng, 4, 3, 0, Op::kNoTrans, Op::kNoTrans, 1.0, 2.0, false,
                  1.0e-12);
  RunCase<double>(rng, 4, 3, 0, Op::kTrans, Op::kTrans, 1.0, 0.0, false,
                  1.0e-12);
}

TEST(LinalgGemm, SubViewsWithLargerLd) {
  std::mt19937_64 rng(5);
  Matrix<double> bigA = RandomMatrix<double>(rng, 10, 9);
  Matrix<double> bigB = RandomMatrix<double>(rng, 11, 8);
  Matrix<double> bigC = RandomMatrix<double>(rng, 12, 10);
  // A: 4x5 at (2,1) of bigA; B: 5x3 at (3,2) of bigB; C: 4x3 at (5,4) of bigC.
  MatrixView<const double> A{&bigA(2, 1), 4, 5, 10};
  MatrixView<const double> B{&bigB(3, 2), 5, 3, 11};
  MatrixView<double> C{&bigC(5, 4), 4, 3, 12};
  Matrix<double> before = bigC;
  Matrix<double> ref =
      Naive<double>(1.3, A, Op::kNoTrans, B, Op::kNoTrans, 0.7, C);
  Gemm(1.3, A, Op::kNoTrans, B, Op::kNoTrans, 0.7, C);
  for (std::int64_t j = 0; j < 10; ++j)
    for (std::int64_t i = 0; i < 12; ++i) {
      const bool inside = i >= 5 && i < 9 && j >= 4 && j < 7;
      if (inside)
        EXPECT_NEAR(bigC(i, j), ref(i - 5, j - 4), 1.0e-12);
      else
        EXPECT_EQ(bigC(i, j), before(i, j));
    }
}

TEST(LinalgGemm, Float) {
  std::mt19937_64 rng(6);
  for (Op oa : {Op::kNoTrans, Op::kTrans})
    for (Op ob : {Op::kNoTrans, Op::kTrans}) {
      RunCase<float>(rng, 9, 7, 5, oa, ob, 1.0f, 0.0f, false, 1.0e-5);
      RunCase<float>(rng, 9, 7, 5, oa, ob, 2.0f, 0.5f, false, 1.0e-5);
      RunCase<float>(rng, 5, 1, 4, oa, ob, 1.0f, 0.0f, true, 1.0e-5);
    }
}

TEST(LinalgGemm, ShapeErrorsThrow) {
  Matrix<double> A(3, 4), B(5, 2), C(3, 2);
  EXPECT_THROW(Gemm(1.0, A.const_view(), Op::kNoTrans, B.const_view(),
                    Op::kNoTrans, 0.0, C.view()),
               imsrg::InvariantError);
  Matrix<double> B2(4, 2), Cbad(2, 2);
  EXPECT_THROW(Gemm(1.0, A.const_view(), Op::kNoTrans, B2.const_view(),
                    Op::kNoTrans, 0.0, Cbad.view()),
               imsrg::InvariantError);
  MatrixView<const double> badld{A.view().data, 3, 4, 2};
  EXPECT_THROW(Gemm(1.0, badld, Op::kNoTrans, B2.const_view(), Op::kNoTrans,
                    0.0, C.view()),
               imsrg::InvariantError);
}

TEST(LinalgGroupedGemm, Mix) {
  std::mt19937_64 rng(7);
  Matrix<double> A0 = RandomMatrix<double>(rng, 4, 3),
                 B0 = RandomMatrix<double>(rng, 3, 5);
  Matrix<double> A1 = RandomMatrix<double>(rng, 6, 2),
                 B1 = RandomMatrix<double>(rng, 4, 6);
  Matrix<double> A2 = RandomMatrix<double>(rng, 0, 3),
                 B2 = RandomMatrix<double>(rng, 3, 2);
  Matrix<double> C0(4, 5), C1 = RandomMatrix<double>(rng, 2, 4), C2(0, 2);
  Matrix<double> C1_0 = C1;
  std::vector<GemmProblem> ps = {
      {1.0, A0.const_view(), Op::kNoTrans, B0.const_view(), Op::kNoTrans, 0.0,
       C0.view()},
      {2.0, A1.const_view(), Op::kTrans, B1.const_view(), Op::kTrans, 0.5,
       C1.view()},
      {1.0, A2.const_view(), Op::kNoTrans, B2.const_view(), Op::kNoTrans, 0.0,
       C2.view()}};
  GroupedGemm(ps);
  auto r0 = Naive<double>(1.0, A0.const_view(), Op::kNoTrans, B0.const_view(),
                          Op::kNoTrans, 0.0, C0.const_view());
  auto r1 = Naive<double>(2.0, A1.const_view(), Op::kTrans, B1.const_view(),
                          Op::kTrans, 0.5, C1_0.const_view());
  for (int j = 0; j < 5; ++j)
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(C0(i, j), r0(i, j), 1.0e-12);
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 2; ++i) EXPECT_NEAR(C1(i, j), r1(i, j), 1.0e-12);
  GroupedGemm({});
}

/// A group of problems of mixed shapes (dimensions 32..96, all transpose
/// combinations, alpha and beta not in {0, 1}); owns the storage.
struct Group {
  std::vector<Matrix<double>> A, B, C, C0;
  std::vector<GemmProblem> ps;
  std::vector<double> alpha, beta;
  std::vector<Op> oa, ob;
};

Group MakeGroup(std::mt19937_64& rng, int count) {
  Group g;
  std::uniform_int_distribution<int> dim(32, 96);
  for (int p = 0; p < count; ++p) {
    const int m = dim(rng), n = dim(rng), k = dim(rng);
    const Op oa = (p & 1) ? Op::kTrans : Op::kNoTrans;
    const Op ob = (p & 2) ? Op::kTrans : Op::kNoTrans;
    g.A.push_back(oa == Op::kTrans ? RandomMatrix<double>(rng, k, m)
                                   : RandomMatrix<double>(rng, m, k));
    g.B.push_back(ob == Op::kTrans ? RandomMatrix<double>(rng, n, k)
                                   : RandomMatrix<double>(rng, k, n));
    g.C.push_back(RandomMatrix<double>(rng, m, n));
    g.alpha.push_back(0.5 + 0.25 * p);
    g.beta.push_back(p % 3 == 0 ? 0.0 : -0.75 + 0.1 * p);
    g.oa.push_back(oa);
    g.ob.push_back(ob);
  }
  g.C0 = g.C;
  for (std::size_t p = 0; p < g.C.size(); ++p) {
    g.ps.push_back({g.alpha[p], g.A[p].const_view(), g.oa[p],
                    g.B[p].const_view(), g.ob[p], g.beta[p], g.C[p].view()});
  }
  return g;
}

void ExpectMatchesNaive(const Group& g, std::size_t p) {
  const Matrix<double> ref = Naive<double>(
      g.alpha[p], g.A[p].const_view(), g.oa[p], g.B[p].const_view(), g.ob[p],
      g.beta[p], g.C0[p].const_view());
  for (std::int64_t j = 0; j < ref.cols(); ++j) {
    for (std::int64_t i = 0; i < ref.rows(); ++i) {
      ASSERT_NEAR(g.C[p](i, j), ref(i, j),
                  1.0e-13 * std::max(1.0, std::abs(ref(i, j))))
          << "problem " << p << " (" << i << ", " << j << ")";
    }
  }
}

TEST(LinalgGroupedGemm, LargeMixedGroupMatchesNaive) {
  // 12 problems of about 2 * 64^3 flops each: several Mflop in total, above
  // kConcurrentGemmMinFlops, so with Accelerate and >= 2 host threads the
  // problems run concurrently.
  std::mt19937_64 rng(11);
  Group g = MakeGroup(rng, 12);
  GroupedGemm(g.ps);
  for (std::size_t p = 0; p < g.ps.size(); ++p) {
    ExpectMatchesNaive(g, p);
  }
}

TEST(LinalgGroupedGemm, InvalidLastProblemThrowsAndLeavesAllCUntouched) {
  std::mt19937_64 rng(12);
  Group g = MakeGroup(rng, 9);
  Matrix<double> Ab = RandomMatrix<double>(rng, 40, 50);
  Matrix<double> Bb = RandomMatrix<double>(rng, 49, 30);  // inner 50 != 49
  Matrix<double> Cb = RandomMatrix<double>(rng, 40, 30);
  const Matrix<double> Cb0 = Cb;
  g.ps.push_back({1.0, Ab.const_view(), Op::kNoTrans, Bb.const_view(),
                  Op::kNoTrans, 0.0, Cb.view()});
  EXPECT_THROW(GroupedGemm(g.ps), imsrg::InvariantError);
  auto same = [](const Matrix<double>& x, const Matrix<double>& y) {
    if (x.rows() != y.rows() || x.cols() != y.cols()) return false;
    for (std::int64_t j = 0; j < x.cols(); ++j)
      for (std::int64_t i = 0; i < x.rows(); ++i)
        if (std::memcmp(&x(i, j), &y(i, j), sizeof(double)) != 0) return false;
    return true;
  };
  for (std::size_t p = 0; p < g.C.size(); ++p) {
    EXPECT_TRUE(same(g.C[p], g.C0[p])) << "problem " << p;
  }
  EXPECT_TRUE(same(Cb, Cb0));
}

TEST(LinalgGroupedGemm, EmptyAndSingleProblemGroups) {
  EXPECT_NO_THROW(GroupedGemm(std::span<const GemmProblem>{}));
  std::mt19937_64 rng(13);
  Group g = MakeGroup(rng, 1);
  GroupedGemm(g.ps);
  ExpectMatchesNaive(g, 0);
}

}  // namespace imsrg::linalg
