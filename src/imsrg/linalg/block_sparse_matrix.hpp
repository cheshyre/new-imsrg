#pragma once
/// @file
/// `BlockSparseMatrix<T>`: a set of dense column-major blocks stored back to back in ONE
/// `exec::DeviceBuffer<T>` (design/10 section 3.3). Each block is labelled by a (row block, column
/// block) pair (e.g. 2B channel numbers) and has leading dimension equal to its row count.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/exec/device_buffer.hpp"
#include "imsrg/linalg/matrix_view.hpp"

namespace imsrg::linalg {

/// Label of a row or column block (for 2B operators: the channel number).
using BlockIndex = StrongIndex<struct BlockTag>;

/// Requested block: its labels and size.
struct BlockShape {
  BlockIndex row_block;
  BlockIndex col_block;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
};

/// Layout of one stored block: `rows x cols` elements starting at element `offset` of the buffer.
struct BlockInfo {
  BlockIndex row_block;
  BlockIndex col_block;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t offset = 0;
};

static_assert(std::is_trivially_copyable_v<BlockInfo>);
static_assert(std::is_trivially_copyable_v<BlockShape>);

/// Block-sparse matrix with value semantics (copy = deep copy of the data, move is cheap).
/// Instantiated for double and float. Needs an initialized `exec::Runtime`.
template <class T>
class BlockSparseMatrix {
 public:
  /// Empty matrix without blocks.
  BlockSparseMatrix() = default;

  /// Allocates the blocks contiguously in the given order (offsets are prefix sums of
  /// rows * cols), zero-initialized, with `label` as buffer label. Throws InvariantError on
  /// negative sizes or duplicate (row_block, col_block) pairs. Complexity O(n log n) in the block
  /// count plus the allocation.
  explicit BlockSparseMatrix(const std::vector<BlockShape>& shapes,
                             std::string label = "imsrg::BlockSparseMatrix");

  /// Number of blocks.
  std::size_t num_blocks() const noexcept { return blocks_.size(); }
  /// Block layouts in storage order.
  std::span<const BlockInfo> blocks() const noexcept { return blocks_; }
  /// Layout of block `k`. Precondition: `k < num_blocks()` (IMSRG_CHECK).
  const BlockInfo& block_info(std::size_t k) const;

  /// Index (storage position) of the block labelled (row, col), or nullopt. O(log n).
  std::optional<std::size_t> find_block(BlockIndex row, BlockIndex col) const;

  /// Total number of stored elements.
  std::int64_t total_size() const noexcept { return total_size_; }
  /// The label given at construction.
  std::string_view label() const noexcept { return label_; }

  /// The underlying buffer (all blocks back to back).
  exec::DeviceBuffer<T>& buffer() noexcept { return buffer_; }
  /// The underlying buffer, read-only.
  const exec::DeviceBuffer<T>& buffer() const noexcept { return buffer_; }

  /// View of block `k` in the HOST MIRROR (creating it on first use). On backends with separate
  /// device memory the caller must `buffer().sync_to_host()` before reading and
  /// `sync_to_device()` after writing. Precondition: `k < num_blocks()`.
  MatrixView<T> host_block(std::size_t k);
  /// Read-only host view, see the non-const overload.
  MatrixView<const T> host_block(std::size_t k) const;

  /// View of block `k` in DEVICE memory: for kernels, and for BLAS calls on host-accessible
  /// backends (Serial/OpenMP). Do not dereference on the host on GPU backends.
  MatrixView<T> device_block(std::size_t k);
  /// Read-only device view, see the non-const overload.
  MatrixView<const T> device_block(std::size_t k) const;

 private:
  std::vector<BlockInfo> blocks_;
  std::vector<std::size_t> sorted_;  // storage indices sorted by (row_block, col_block)
  std::int64_t total_size_ = 0;
  std::string label_;
  exec::DeviceBuffer<T> buffer_;
};

extern template class BlockSparseMatrix<double>;
extern template class BlockSparseMatrix<float>;

}  // namespace imsrg::linalg
