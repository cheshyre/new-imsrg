#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/linalg/blas.hpp"

using namespace imsrg::linalg;

TEST(LinalgEigh, RandomSymmetricSizes) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  for (int n = 1; n <= 60; ++n) {
    Matrix<double> A(n, n);
    for (int j = 0; j < n; ++j)
      for (int i = j; i < n; ++i) A(i, j) = A(j, i) = d(rng);
    EighResult r = eigh(A.const_view());
    ASSERT_EQ(static_cast<int>(r.eigenvalues.size()), n);
    EXPECT_TRUE(std::is_sorted(r.eigenvalues.begin(), r.eigenvalues.end()));
    const Matrix<double>& V = r.eigenvectors;
    double anorm = 0;
    for (double w : r.eigenvalues) anorm = std::max(anorm, std::abs(w));
    const double tol = 1e-12 * std::max(1.0, anorm) * n;
    for (int c = 0; c < n; ++c) {
      // Residual ||A v - w v||_inf.
      for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int p = 0; p < n; ++p) s += A(i, p) * V(p, c);
        ASSERT_NEAR(s, r.eigenvalues[static_cast<std::size_t>(c)] * V(i, c), tol) << "n=" << n;
      }
      for (int c2 = 0; c2 <= c; ++c2) {
        double s = 0;
        for (int i = 0; i < n; ++i) s += V(i, c) * V(i, c2);
        ASSERT_NEAR(s, c == c2 ? 1.0 : 0.0, 1e-12 * n) << "n=" << n;
      }
    }
  }
}

TEST(LinalgEigh, TwoByTwoAnalytic) {
  Matrix<double> A(2, 2);
  A(0, 0) = 2;
  A(1, 0) = A(0, 1) = 1;
  A(1, 1) = 2;
  EighResult r = eigh(A.const_view());
  EXPECT_NEAR(r.eigenvalues[0], 1.0, 1e-14);
  EXPECT_NEAR(r.eigenvalues[1], 3.0, 1e-14);
  const double s = std::sqrt(0.5);
  EXPECT_NEAR(std::abs(r.eigenvectors(0, 0)), s, 1e-14);
  EXPECT_NEAR(std::abs(r.eigenvectors(1, 0)), s, 1e-14);
  EXPECT_NEAR(r.eigenvectors(0, 0) * r.eigenvectors(1, 0), -0.5, 1e-14);
  EXPECT_NEAR(r.eigenvectors(0, 1) * r.eigenvectors(1, 1), 0.5, 1e-14);
}

TEST(LinalgEigh, DiagonalInput) {
  Matrix<double> A(4, 4);
  const double diag[4] = {3.0, -1.0, 2.0, 0.5};
  for (int i = 0; i < 4; ++i) A(i, i) = diag[i];
  EighResult r = eigh(A.const_view());
  const double expect[4] = {-1.0, 0.5, 2.0, 3.0};
  const int src[4] = {1, 3, 2, 0};
  for (int c = 0; c < 4; ++c) {
    EXPECT_NEAR(r.eigenvalues[static_cast<std::size_t>(c)], expect[c], 1e-14);
    EXPECT_NEAR(std::abs(r.eigenvectors(src[c], c)), 1.0, 1e-14);
  }
}

TEST(LinalgEigh, ReadsOnlyLowerTriangleAndEmpty) {
  Matrix<double> A(2, 2);
  A(0, 0) = 1;
  A(1, 0) = 0.5;
  A(0, 1) = 1e6;  // must be ignored
  A(1, 1) = 3;
  EighResult r = eigh(A.const_view());
  const double mid = 2.0, h = std::sqrt(1.0 + 0.25);
  EXPECT_NEAR(r.eigenvalues[0], mid - h, 1e-13);
  EXPECT_NEAR(r.eigenvalues[1], mid + h, 1e-13);
  Matrix<double> E(0, 0);
  EXPECT_EQ(eigh(E.const_view()).eigenvalues.size(), 0u);
  Matrix<double> R(2, 3), V(2, 3);
  std::vector<double> w(2);
  EXPECT_THROW(eigh(R.const_view(), w, V.view()), imsrg::InvariantError);
}
