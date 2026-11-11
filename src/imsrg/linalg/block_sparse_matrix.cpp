#include "imsrg/linalg/block_sparse_matrix.hpp"

#include <algorithm>
#include <utility>

#include "imsrg/base/assert.hpp"

namespace imsrg::linalg {

template <class T>
BlockSparseMatrix<T>::BlockSparseMatrix(const std::vector<BlockShape>& shapes, std::string label)
    : label_(std::move(label)) {
  blocks_.reserve(shapes.size());
  std::int64_t offset = 0;
  for (const BlockShape& s : shapes) {
    IMSRG_CHECK(s.rows >= 0 && s.cols >= 0, "BlockSparseMatrix: negative block size");
    blocks_.push_back(BlockInfo{s.row_block, s.col_block, s.rows, s.cols, offset});
    offset += s.rows * s.cols;
  }
  total_size_ = offset;
  sorted_.resize(blocks_.size());
  for (std::size_t i = 0; i < sorted_.size(); ++i) sorted_[i] = i;
  std::sort(sorted_.begin(), sorted_.end(), [this](std::size_t a, std::size_t b) {
    const BlockInfo& x = blocks_[a];
    const BlockInfo& y = blocks_[b];
    if (x.row_block != y.row_block) return x.row_block < y.row_block;
    return x.col_block < y.col_block;
  });
  for (std::size_t i = 1; i < sorted_.size(); ++i) {
    const BlockInfo& x = blocks_[sorted_[i - 1]];
    const BlockInfo& y = blocks_[sorted_[i]];
    IMSRG_CHECK(!(x.row_block == y.row_block && x.col_block == y.col_block),
                "BlockSparseMatrix: duplicate (row, col) block");
  }
  buffer_ = exec::DeviceBuffer<T>(static_cast<std::size_t>(total_size_), label_);
}

template <class T>
const BlockInfo& BlockSparseMatrix<T>::block_info(std::size_t k) const {
  IMSRG_CHECK(k < blocks_.size(), "BlockSparseMatrix: block index out of range");
  return blocks_[k];
}

template <class T>
std::optional<std::size_t> BlockSparseMatrix<T>::find_block(BlockIndex row, BlockIndex col) const {
  auto it = std::lower_bound(sorted_.begin(), sorted_.end(), std::pair{row, col},
                             [this](std::size_t k, const std::pair<BlockIndex, BlockIndex>& key) {
                               const BlockInfo& b = blocks_[k];
                               if (b.row_block != key.first) return b.row_block < key.first;
                               return b.col_block < key.second;
                             });
  if (it == sorted_.end()) return std::nullopt;
  const BlockInfo& b = blocks_[*it];
  if (b.row_block != row || b.col_block != col) return std::nullopt;
  return *it;
}

template <class T>
MatrixView<T> BlockSparseMatrix<T>::host_block(std::size_t k) {
  const BlockInfo& b = block_info(k);
  return {buffer_.host_span().data() + b.offset, b.rows, b.cols, b.rows};
}

template <class T>
MatrixView<const T> BlockSparseMatrix<T>::host_block(std::size_t k) const {
  const BlockInfo& b = block_info(k);
  return {buffer_.host_span().data() + b.offset, b.rows, b.cols, b.rows};
}

template <class T>
MatrixView<T> BlockSparseMatrix<T>::device_block(std::size_t k) {
  const BlockInfo& b = block_info(k);
  return {buffer_.device_data() + b.offset, b.rows, b.cols, b.rows};
}

template <class T>
MatrixView<const T> BlockSparseMatrix<T>::device_block(std::size_t k) const {
  const BlockInfo& b = block_info(k);
  return {buffer_.device_data() + b.offset, b.rows, b.cols, b.rows};
}

template class BlockSparseMatrix<double>;
template class BlockSparseMatrix<float>;

}  // namespace imsrg::linalg
