#include "imsrg/linalg/matrix_view.hpp"

namespace imsrg::linalg {

template <class T>
Matrix<T>::Matrix(std::int64_t rows, std::int64_t cols) : rows_(rows), cols_(cols) {
  IMSRG_CHECK(rows >= 0 && cols >= 0, "Matrix dimensions must be non-negative");
  data_.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), T{});
}

template class Matrix<double>;
template class Matrix<float>;

}  // namespace imsrg::linalg
