#pragma once
/// @file
/// Column-major matrix views and an owning host matrix. Column-major (element (i, j) at
/// `data[i + j * ld]`) matches BLAS/LAPACK and imsrg++ (Armadillo), so blocks can be handed to
/// BLAS without transposition.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/device.hpp"

namespace imsrg::linalg {

/// Whether a gemm operand is used as is or transposed.
enum class Op : std::uint8_t { NoTrans, Trans };

/// Non-owning column-major view of a `rows x cols` matrix with leading dimension `ld >= rows`.
/// Trivially copyable, so it can be captured by value in kernels (design/12 section 5). The view
/// does not know which memory space `data` points to. Indices are not bounds-checked here.
template <class T>
struct MatrixView {
  T* data = nullptr;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t ld = 0;

  /// Element (i, j); precondition `0 <= i < rows`, `0 <= j < cols`. O(1).
  IMSRG_INLINE_FUNCTION T& operator()(std::int64_t i, std::int64_t j) const {
    return data[i + j * ld];
  }

  /// Implicit conversion to a view of const elements.
  IMSRG_INLINE_FUNCTION operator MatrixView<const T>() const
    requires(!std::is_const_v<T>)
  {
    return MatrixView<const T>{data, rows, cols, ld};
  }
};

static_assert(std::is_trivially_copyable_v<MatrixView<double>>);
static_assert(std::is_trivially_copyable_v<MatrixView<const float>>);

/// Owning dense column-major host matrix (`ld == rows`), zero-initialized. Value semantics.
/// Instantiated for double and float.
template <class T>
class Matrix {
 public:
  /// Empty 0 x 0 matrix.
  Matrix() = default;
  /// `rows x cols` zero matrix. Precondition: rows, cols >= 0 (IMSRG_CHECK).
  Matrix(std::int64_t rows, std::int64_t cols);

  /// Number of rows.
  std::int64_t rows() const noexcept { return rows_; }
  /// Number of columns.
  std::int64_t cols() const noexcept { return cols_; }

  /// Element (i, j), bounds-checked with IMSRG_ASSERT in debug builds.
  T& operator()(std::int64_t i, std::int64_t j) {
    IMSRG_ASSERT(i >= 0 && i < rows_ && j >= 0 && j < cols_);
    return data_[static_cast<std::size_t>(i + j * rows_)];
  }
  /// Read-only element access, see the non-const overload.
  const T& operator()(std::int64_t i, std::int64_t j) const {
    IMSRG_ASSERT(i >= 0 && i < rows_ && j >= 0 && j < cols_);
    return data_[static_cast<std::size_t>(i + j * rows_)];
  }

  /// Mutable view (ld = rows).
  MatrixView<T> view() noexcept { return {data_.data(), rows_, cols_, rows_}; }
  /// Read-only view (ld = rows).
  MatrixView<const T> const_view() const noexcept { return {data_.data(), rows_, cols_, rows_}; }

 private:
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  std::vector<T> data_;
};

extern template class Matrix<double>;
extern template class Matrix<float>;

}  // namespace imsrg::linalg
