#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/linalg/blas.hpp"

using namespace imsrg::linalg;

namespace {

template <class T>
Matrix<T> random_matrix(std::mt19937_64& rng, std::int64_t r, std::int64_t c) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  Matrix<T> m(r, c);
  for (std::int64_t j = 0; j < c; ++j)
    for (std::int64_t i = 0; i < r; ++i) m(i, j) = static_cast<T>(d(rng));
  return m;
}

/// Naive reference: result = alpha op(A) op(B) + beta C0 (C0 ignored if beta == 0).
template <class T>
Matrix<double> naive(T alpha, MatrixView<const T> A, Op oa, MatrixView<const T> B, Op ob, T beta,
                     MatrixView<const T> C0) {
  const bool ta = oa == Op::Trans, tb = ob == Op::Trans;
  const std::int64_t m = ta ? A.cols : A.rows, k = ta ? A.rows : A.cols, n = tb ? B.rows : B.cols;
  Matrix<double> r(m, n);
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < m; ++i) {
      double s = 0;
      for (std::int64_t p = 0; p < k; ++p) {
        const double a = ta ? A(p, i) : A(i, p);
        const double b = tb ? B(j, p) : B(p, j);
        s += a * b;
      }
      r(i, j) = alpha * s + (beta == T{0} ? 0.0 : static_cast<double>(beta) * C0(i, j));
    }
  return r;
}

template <class T>
void run_case(std::mt19937_64& rng, std::int64_t m, std::int64_t n, std::int64_t k, Op oa, Op ob,
              T alpha, T beta, bool nan_c, double tol_unit) {
  Matrix<T> A = oa == Op::Trans ? random_matrix<T>(rng, k, m) : random_matrix<T>(rng, m, k);
  Matrix<T> B = ob == Op::Trans ? random_matrix<T>(rng, n, k) : random_matrix<T>(rng, k, n);
  Matrix<T> C = random_matrix<T>(rng, m, n);
  if (nan_c)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < m; ++i) C(i, j) = std::numeric_limits<T>::quiet_NaN();
  Matrix<double> ref =
      naive<T>(alpha, A.const_view(), oa, B.const_view(), ob, beta, C.const_view());
  gemm(alpha, A.const_view(), oa, B.const_view(), ob, beta, C.view());
  double scale = 1.0;
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < m; ++i) scale = std::max(scale, std::abs(ref(i, j)));
  const double tol = tol_unit * scale * std::max<double>(1.0, static_cast<double>(k));
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
    for (Op oa : {Op::NoTrans, Op::Trans})
      for (Op ob : {Op::NoTrans, Op::Trans})
        run_case<double>(rng, s[0], s[1], s[2], oa, ob, 1.0, 0.0, false, 1e-12);
}

TEST(LinalgGemm, AlphaBeta) {
  std::mt19937_64 rng(2);
  const double ab[][2] = {{1.0, 1.0}, {2.5, -0.5}, {0.0, 1.0}, {0.0, 0.0}, {-1.0, 2.0}};
  for (auto& p : ab)
    for (Op oa : {Op::NoTrans, Op::Trans})
      for (Op ob : {Op::NoTrans, Op::Trans})
        run_case<double>(rng, 7, 5, 6, oa, ob, p[0], p[1], false, 1e-12);
}

TEST(LinalgGemm, BetaZeroIgnoresNaN) {
  std::mt19937_64 rng(3);
  run_case<double>(rng, 6, 5, 4, Op::NoTrans, Op::NoTrans, 1.5, 0.0, true, 1e-12);
  run_case<double>(rng, 6, 5, 4, Op::Trans, Op::Trans, 1.0, 0.0, true, 1e-12);
  // Also the k == 0 path.
  run_case<double>(rng, 3, 3, 0, Op::NoTrans, Op::NoTrans, 1.0, 0.0, true, 1e-12);
}

TEST(LinalgGemm, ZeroSized) {
  std::mt19937_64 rng(4);
  run_case<double>(rng, 0, 4, 3, Op::NoTrans, Op::NoTrans, 1.0, 0.0, false, 1e-12);
  run_case<double>(rng, 4, 0, 3, Op::NoTrans, Op::NoTrans, 1.0, 0.0, false, 1e-12);
  run_case<double>(rng, 4, 3, 0, Op::NoTrans, Op::NoTrans, 1.0, 2.0, false, 1e-12);
  run_case<double>(rng, 4, 3, 0, Op::Trans, Op::Trans, 1.0, 0.0, false, 1e-12);
}

TEST(LinalgGemm, SubViewsWithLargerLd) {
  std::mt19937_64 rng(5);
  Matrix<double> bigA = random_matrix<double>(rng, 10, 9);
  Matrix<double> bigB = random_matrix<double>(rng, 11, 8);
  Matrix<double> bigC = random_matrix<double>(rng, 12, 10);
  // A: 4x5 at (2,1) of bigA; B: 5x3 at (3,2) of bigB; C: 4x3 at (5,4) of bigC.
  MatrixView<const double> A{&bigA(2, 1), 4, 5, 10};
  MatrixView<const double> B{&bigB(3, 2), 5, 3, 11};
  MatrixView<double> C{&bigC(5, 4), 4, 3, 12};
  Matrix<double> before = bigC;
  Matrix<double> ref = naive<double>(1.3, A, Op::NoTrans, B, Op::NoTrans, 0.7, C);
  gemm(1.3, A, Op::NoTrans, B, Op::NoTrans, 0.7, C);
  for (std::int64_t j = 0; j < 10; ++j)
    for (std::int64_t i = 0; i < 12; ++i) {
      const bool inside = i >= 5 && i < 9 && j >= 4 && j < 7;
      if (inside)
        EXPECT_NEAR(bigC(i, j), ref(i - 5, j - 4), 1e-12);
      else
        EXPECT_EQ(bigC(i, j), before(i, j));
    }
}

TEST(LinalgGemm, Float) {
  std::mt19937_64 rng(6);
  for (Op oa : {Op::NoTrans, Op::Trans})
    for (Op ob : {Op::NoTrans, Op::Trans}) {
      run_case<float>(rng, 9, 7, 5, oa, ob, 1.0f, 0.0f, false, 1e-5);
      run_case<float>(rng, 9, 7, 5, oa, ob, 2.0f, 0.5f, false, 1e-5);
      run_case<float>(rng, 5, 1, 4, oa, ob, 1.0f, 0.0f, true, 1e-5);
    }
}

TEST(LinalgGemm, ShapeErrorsThrow) {
  Matrix<double> A(3, 4), B(5, 2), C(3, 2);
  EXPECT_THROW(gemm(1.0, A.const_view(), Op::NoTrans, B.const_view(), Op::NoTrans, 0.0, C.view()),
               imsrg::InvariantError);
  Matrix<double> B2(4, 2), Cbad(2, 2);
  EXPECT_THROW(
      gemm(1.0, A.const_view(), Op::NoTrans, B2.const_view(), Op::NoTrans, 0.0, Cbad.view()),
      imsrg::InvariantError);
  MatrixView<const double> badld{A.view().data, 3, 4, 2};
  EXPECT_THROW(gemm(1.0, badld, Op::NoTrans, B2.const_view(), Op::NoTrans, 0.0, C.view()),
               imsrg::InvariantError);
}

TEST(LinalgGroupedGemm, Mix) {
  std::mt19937_64 rng(7);
  Matrix<double> A0 = random_matrix<double>(rng, 4, 3), B0 = random_matrix<double>(rng, 3, 5);
  Matrix<double> A1 = random_matrix<double>(rng, 6, 2), B1 = random_matrix<double>(rng, 4, 6);
  Matrix<double> A2 = random_matrix<double>(rng, 0, 3), B2 = random_matrix<double>(rng, 3, 2);
  Matrix<double> C0(4, 5), C1 = random_matrix<double>(rng, 2, 4), C2(0, 2);
  Matrix<double> C1_0 = C1;
  std::vector<GemmProblem> ps = {
      {1.0, A0.const_view(), Op::NoTrans, B0.const_view(), Op::NoTrans, 0.0, C0.view()},
      {2.0, A1.const_view(), Op::Trans, B1.const_view(), Op::Trans, 0.5, C1.view()},
      {1.0, A2.const_view(), Op::NoTrans, B2.const_view(), Op::NoTrans, 0.0, C2.view()}};
  grouped_gemm(ps);
  auto r0 = naive<double>(1.0, A0.const_view(), Op::NoTrans, B0.const_view(), Op::NoTrans, 0.0,
                          C0.const_view());
  auto r1 = naive<double>(2.0, A1.const_view(), Op::Trans, B1.const_view(), Op::Trans, 0.5,
                          C1_0.const_view());
  for (int j = 0; j < 5; ++j)
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(C0(i, j), r0(i, j), 1e-12);
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 2; ++i) EXPECT_NEAR(C1(i, j), r1(i, j), 1e-12);
  grouped_gemm({});
}
