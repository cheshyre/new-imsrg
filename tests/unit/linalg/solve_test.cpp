#include <algorithm>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/linalg/blas.hpp"

using namespace imsrg::linalg;

namespace {
double frob(const Matrix<double>& m) {
  double s = 0;
  for (std::int64_t j = 0; j < m.cols(); ++j)
    for (std::int64_t i = 0; i < m.rows(); ++i) s += m(i, j) * m(i, j);
  return std::sqrt(s);
}
}  // namespace

TEST(LinalgSolve, RandomSystems) {
  std::mt19937_64 rng(99);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  for (int n : {1, 2, 5, 17, 40}) {
    for (int nrhs : {1, 3}) {
      Matrix<double> A(n, n), B(n, nrhs);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) A(i, j) = d(rng) + (i == j ? 3.0 : 0.0);
      for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) B(i, j) = d(rng);
      Matrix<double> X = B;
      Matrix<double> Acopy = A;
      solve(A.const_view(), X.view());
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) ASSERT_EQ(A(i, j), Acopy(i, j));  // A untouched
      Matrix<double> R(n, nrhs);
      gemm(1.0, A.const_view(), Op::NoTrans, X.const_view(), Op::NoTrans, 0.0, R.view());
      double res = 0;
      for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) res = std::max(res, std::abs(R(i, j) - B(i, j)));
      EXPECT_LE(res, 1e-12 * frob(A) * frob(X)) << "n=" << n;
    }
  }
}

TEST(LinalgSolve, SingularThrows) {
  Matrix<double> A(3, 3), B(3, 1);
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) A(i, j) = (i + 1) * (j + 1);  // rank 1
  B(0, 0) = 1;
  EXPECT_THROW(solve(A.const_view(), B.view()), imsrg::NumericalError);
  Matrix<double> Z(2, 2), b(2, 1);
  EXPECT_THROW(solve(Z.const_view(), b.view()), imsrg::NumericalError);
}

TEST(LinalgSolve, ShapeErrorsAndEmpty) {
  Matrix<double> A(3, 2), B(3, 1);
  EXPECT_THROW(solve(A.const_view(), B.view()), imsrg::InvariantError);
  Matrix<double> A2(3, 3), B2(2, 1);
  EXPECT_THROW(solve(A2.const_view(), B2.view()), imsrg::InvariantError);
  Matrix<double> E(0, 0), EB(0, 2);
  EXPECT_NO_THROW(solve(E.const_view(), EB.view()));
}
