// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Per-row addressing of the three-body parts of the IMSRG(3)-N7 kernels
// (detail/n7_views.h): the row-offset tables of the context against the
// channel-block layout of `op::ThreeBody` and `basis::RowSet`, and the halo
// contract of `ThreeBodyPartView` (rows living in a second buffer).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/three_body_test_support.h"
#include "tests/unit/commutator/n7_test_support.h"

namespace imsrg::commutator {
namespace {

using testing::N7Inputs;
using testing::RandomN7Inputs;

/// Checks the row table of `in` for storage mode `packed` against the block
/// offsets of the `op::ThreeBody` of `in.x` and against `RowSet::Whole`.
void CheckRowTables(const N7Inputs& in, bool packed, const std::string& tag) {
  SCOPED_TRACE(tag + (packed ? " packed" : " dense"));
  const ThreeBodyContext::Impl& impl = in.ctx->impl();
  const op::ThreeBody& t = *in.x.three_body();
  ASSERT_EQ(t.packed(), packed);
  const std::vector<std::int64_t> tab_v =
      packed ? impl.row_off_packed.ToVector() : impl.row_off_dense.ToVector();
  const basis::ThreeBodySpace& space = *in.ms3->three_body_space();
  ASSERT_EQ(static_cast<std::int64_t>(tab_v.size()), space.NumRows());
  const std::int64_t* tp = tab_v.data();
  const basis::RowSet whole = basis::RowSet::Whole(space);
  const std::vector<std::int64_t> ch_row = impl.ch_row_offset.ToVector();
  const auto offsets = t.block_offsets();
  for (std::size_t c = 0; c < space.channels().size(); ++c) {
    const std::int64_t n = space.channels()[c].dim;
    const std::int64_t f = ch_row[c];
    for (std::int64_t j = 0; j < n; ++j) {
      const std::int64_t first = packed ? j : 0;
      const std::int64_t rs = whole.LocalRowOffset(
          Channel3Index{static_cast<std::int32_t>(c)},
          basis::KeptRow3Index{static_cast<std::int32_t>(j)}, packed);
      ASSERT_EQ(tp[f + j] + first, rs);
      for (std::int64_t i = first; i < n; ++i) {
        const std::int64_t expect =
            offsets[c] +
            (packed ? op::ThreeBody::PackedIndex(n, i, j) : i + j * n);
        ASSERT_EQ(tp[f + j] + i, expect);
      }
    }
  }
}

/// Compares `ThreeBodyLocal` of the views of `in.x` (host reads of the device
/// data, valid where the device data are host accessible) with the host
/// getter of the operator.
void CheckLocalValues(N7Inputs& in, const std::string& tag) {
  SCOPED_TRACE(tag);
  op::ThreeBody& t = *in.x.three_body();
  if (!t.buffer().host_is_device()) return;
  t.SyncToHost();
  const detail::N7Views v = in.ctx->impl().N7(&in.x, &in.y);
  const basis::ThreeBodySpace& space = *in.ms3->three_body_space();
  for (std::size_t c = 0; c < space.channels().size(); ++c) {
    const int n = space.channels()[c].dim;
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const double got =
            detail::ThreeBodyLocal(v.x3, v.sp, static_cast<int>(c), i, j);
        const double want =
            t.get(Channel3Index{static_cast<int>(c)}, basis::KeptRow3Index{i},
                  basis::KeptRow3Index{j});
        ASSERT_EQ(got, want) << "ch " << c << " (" << i << ", " << j << ")";
      }
    }
  }
}

/// Both storage modes on `ms3`.
void CheckSpace(const std::shared_ptr<const basis::ModelSpace>& ms3,
                const std::string& tag, std::uint64_t seed) {
  N7Inputs h = RandomN7Inputs(ms3, Hermiticity::kHermitian,
                              Hermiticity::kAntiHermitian, seed);
  CheckRowTables(h, true, tag);
  CheckLocalValues(h, tag + " packed");
  N7Inputs d = RandomN7Inputs(ms3, Hermiticity::kNonHermitian,
                              Hermiticity::kNonHermitian, seed + 7);
  CheckRowTables(d, false, tag);
  CheckLocalValues(d, tag + " dense");
}

TEST(N7Views, RowTablesMatchBlockLayout) {
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  CheckSpace(testing::MakeThreeBodySpace(2, "He4", e3), "He4 emax 2", 5);
  CheckSpace(testing::MakeThreeBodySpace(2, "O16", e3), "O16 emax 2", 6);
  basis::ThreeBodyCuts de3;
  de3.e3max = 6;
  de3.de3max = 2.0;
  CheckSpace(testing::MakeThreeBodySpace(2, "O16", de3), "O16 dE3max 2", 7);
}

/// ThreeBodyLocal on `full` (all rows local) and on a view in which the rows
/// `halo_rows` of channel `ch` (dimension `n`) live in a halo buffer (row_off
/// pointing into the halo with a bias, the local copy poisoned with NaN) must
/// agree for every element of the channel. `full` addresses `data`.
void CheckHalo(const detail::Space3View& sp,
               const detail::ThreeBodyPartView& full, std::int64_t size, int ch,
               int n, const std::vector<int>& halo_rows) {
  const std::int64_t f0 = sp.ch_row_offset[ch];
  const std::int64_t nrows =
      sp.ch_row_offset[sp.num_channels - 1] + sp.ch_dim[sp.num_channels - 1];
  const bool packed = full.packed != 0;
  std::vector<std::int64_t> off(full.row_off, full.row_off + nrows);
  std::vector<std::int8_t> buf(static_cast<std::size_t>(nrows), 0);
  std::vector<double> local(full.local, full.local + size);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> halo(7, nan);  // bias: rows start after a 7-entry pad
  for (const int k : halo_rows) {
    const std::int64_t first = packed ? k : 0;
    const std::int64_t start = static_cast<std::int64_t>(halo.size());
    for (std::int64_t m = first; m < n; ++m) {
      halo.push_back(full.local[full.row_off[f0 + k] + m]);
      local[static_cast<std::size_t>(full.row_off[f0 + k] + m)] = nan;
    }
    off[static_cast<std::size_t>(f0 + k)] = start - first;
    buf[static_cast<std::size_t>(f0 + k)] = 1;
  }
  detail::ThreeBodyPartView part = full;
  part.local = local.data();
  part.halo = halo.data();
  part.row_off = off.data();
  part.row_buf = buf.data();
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double want = detail::ThreeBodyLocal(full, sp, ch, i, j);
      const double got = detail::ThreeBodyLocal(part, sp, ch, i, j);
      ASSERT_EQ(got, want) << "(" << i << ", " << j << ") sign " << full.sign;
    }
  }
}

TEST(N7Views, HaloRowsReadLikeLocalRows) {
  basis::ThreeBodyCuts e3;
  e3.e3max = 6;
  const auto ms3 = testing::MakeThreeBodySpace(2, "He4", e3);
  const basis::ThreeBodySpace& space = *ms3->three_body_space();
  std::size_t ch = 0;
  for (std::size_t c = 0; c < space.channels().size(); ++c) {
    if (space.channels()[c].dim > space.channels()[ch].dim) {
      ch = c;
    }
  }
  const int n = space.channels()[ch].dim;
  ASSERT_GE(n, 5);
  const std::vector<int> halo_rows = {1, 2, n - 1};
  // x Hermitian (sign +1) or anti-Hermitian (sign -1), packed; y dense.
  for (const Hermiticity h :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    N7Inputs p = RandomN7Inputs(ms3, h, Hermiticity::kNonHermitian, 21);
    op::ThreeBody& tp = *p.x.three_body();
    op::ThreeBody& td = *p.y.three_body();
    if (!tp.buffer().host_is_device()) return;
    tp.SyncToHost();
    td.SyncToHost();
    const detail::N7Views v = p.ctx->impl().N7(&p.x, &p.y);
    ASSERT_EQ(v.x3.packed, 1);
    ASSERT_EQ(v.y3.packed, 0);
    EXPECT_EQ(v.x3.sign, h == Hermiticity::kHermitian ? 1 : -1);
    CheckHalo(v.sp, v.x3, static_cast<std::int64_t>(tp.size()),
              static_cast<int>(ch), n, halo_rows);
    CheckHalo(v.sp, v.y3, static_cast<std::int64_t>(td.size()),
              static_cast<int>(ch), n, halo_rows);
  }
}

}  // namespace
}  // namespace imsrg::commutator
