#include "imsrg/linalg/matrix_view.hpp"

#include <type_traits>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"

using imsrg::linalg::Matrix;
using imsrg::linalg::MatrixView;

TEST(LinalgMatrixView, ColumnMajorIndexing) {
  double buf[12];
  for (int i = 0; i < 12; ++i) buf[i] = i;
  MatrixView<double> v{buf, 3, 3, 4};  // ld > rows
  EXPECT_EQ(v(0, 0), 0.0);
  EXPECT_EQ(v(2, 0), 2.0);
  EXPECT_EQ(v(0, 1), 4.0);
  EXPECT_EQ(v(1, 2), 9.0);
  v(1, 1) = -1.0;
  EXPECT_EQ(buf[5], -1.0);
}

TEST(LinalgMatrixView, ConstConversion) {
  double buf[4] = {1, 2, 3, 4};
  MatrixView<double> v{buf, 2, 2, 2};
  MatrixView<const double> c = v;
  EXPECT_EQ(c.data, buf);
  EXPECT_EQ(c(1, 1), 4.0);
  static_assert(std::is_convertible_v<MatrixView<double>, MatrixView<const double>>);
  static_assert(!std::is_convertible_v<MatrixView<const double>, MatrixView<double>>);
}

TEST(LinalgMatrix, ZeroInitAndViews) {
  Matrix<double> m(3, 2);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 2);
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 3; ++i) EXPECT_EQ(m(i, j), 0.0);
  m(2, 1) = 5.0;
  EXPECT_EQ(m.view()(2, 1), 5.0);
  EXPECT_EQ(m.const_view().ld, 3);
  Matrix<double> copy = m;
  copy(0, 0) = 1.0;
  EXPECT_EQ(m(0, 0), 0.0);
}

TEST(LinalgMatrix, NegativeDimensionThrows) {
  EXPECT_THROW(Matrix<double>(-1, 2), imsrg::InvariantError);
}
