#include "imsrg/linalg/block_sparse_matrix.hpp"

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"

using namespace imsrg::linalg;

namespace {
BlockIndex B(int v) { return BlockIndex{v}; }

std::vector<BlockShape> shapes() {
  return {{B(2), B(2), 3, 3}, {B(0), B(0), 2, 4}, {B(5), B(1), 0, 2}, {B(1), B(7), 1, 1}};
}
}  // namespace

TEST(LinalgBlockSparse, ContiguousOffsets) {
  BlockSparseMatrix<double> m(shapes());
  ASSERT_EQ(m.num_blocks(), 4u);
  EXPECT_EQ(m.block_info(0).offset, 0);
  EXPECT_EQ(m.block_info(1).offset, 9);
  EXPECT_EQ(m.block_info(2).offset, 17);
  EXPECT_EQ(m.block_info(3).offset, 17);
  EXPECT_EQ(m.total_size(), 18);
  EXPECT_EQ(m.buffer().size(), 18u);
  EXPECT_THROW(m.block_info(4), imsrg::InvariantError);
}

TEST(LinalgBlockSparse, FindBlock) {
  BlockSparseMatrix<double> m(shapes());
  EXPECT_EQ(m.find_block(B(2), B(2)), 0u);
  EXPECT_EQ(m.find_block(B(0), B(0)), 1u);
  EXPECT_EQ(m.find_block(B(1), B(7)), 3u);
  EXPECT_FALSE(m.find_block(B(2), B(0)).has_value());
  EXPECT_FALSE(m.find_block(B(9), B(9)).has_value());
  EXPECT_FALSE(m.find_block(B(-1), B(0)).has_value());
}

TEST(LinalgBlockSparse, ZeroInitAndHostBlockAliasing) {
  BlockSparseMatrix<double> m(shapes());
  for (double x : m.buffer().host_span()) EXPECT_EQ(x, 0.0);
  auto b = m.host_block(1);
  EXPECT_EQ(b.rows, 2);
  EXPECT_EQ(b.cols, 4);
  EXPECT_EQ(b.ld, 2);
  b(1, 3) = 7.0;
  EXPECT_EQ(m.buffer().host_span()[9 + 1 + 3 * 2], 7.0);
  const auto& cm = m;
  EXPECT_EQ(cm.host_block(1)(1, 3), 7.0);
}

TEST(LinalgBlockSparse, DeepCopy) {
  BlockSparseMatrix<double> m(shapes());
  m.host_block(0)(1, 1) = 3.0;
  BlockSparseMatrix<double> c = m;
  EXPECT_EQ(c.host_block(0)(1, 1), 3.0);
  c.host_block(0)(1, 1) = 4.0;
  EXPECT_EQ(m.host_block(0)(1, 1), 3.0);
  EXPECT_EQ(c.find_block(B(2), B(2)), 0u);
}

TEST(LinalgBlockSparse, DuplicateRejected) {
  std::vector<BlockShape> s = {{B(1), B(1), 2, 2}, {B(0), B(0), 1, 1}, {B(1), B(1), 3, 3}};
  EXPECT_THROW(BlockSparseMatrix<double>{s}, imsrg::InvariantError);
}

TEST(LinalgBlockSparse, NegativeSizeRejectedAndFloat) {
  EXPECT_THROW(BlockSparseMatrix<double>(std::vector<BlockShape>{{B(0), B(0), -1, 2}}),
               imsrg::InvariantError);
  BlockSparseMatrix<float> f(shapes());
  f.host_block(0)(0, 0) = 1.5f;
  EXPECT_EQ(f.host_block(0)(0, 0), 1.5f);
}
