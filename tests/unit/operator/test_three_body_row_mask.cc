// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of the masking helpers MaskOutsideLines and OnlyRowsNonzero.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/row_mask.h"

namespace imsrg::testing {
namespace {

using basis::KeptRow3Index;
using basis::RowRange;
using basis::RowSet;
using basis::Triple3Index;

std::shared_ptr<const basis::ModelSpace> He4Space() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "He4", "He4")
      ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 4});
}

/// A set holding the first triple of every channel with at least 2 triples
/// and the last triple of channels with at least 3.
RowSet SomeRows(const basis::ThreeBodySpace& s) {
  std::vector<RowRange> r;
  auto add = [&](Channel3Index ch, std::int32_t t) {
    const basis::OrbitTriple& tr = s.triple(Triple3Index{t});
    r.push_back(RowRange{ch, Triple3Index{t}, Triple3Index{t + 1}, tr.row_begin,
                         tr.row_end});
  };
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const basis::KeptChannel3& kc = s.channel(Channel3Index{c});
    const std::int32_t nt = kc.triple_end.value() - kc.triple_begin.value();
    if (nt >= 2) add(Channel3Index{c}, kc.triple_begin.value());
    if (nt >= 3) add(Channel3Index{c}, kc.triple_end.value() - 1);
  }
  return RowSet::FromRanges(s, std::move(r));
}

/// Independent count of the elements off the lines of `rows`.
std::int64_t CountOffLines(const basis::ThreeBodySpace& s, const RowSet& rows,
                           bool packed) {
  std::int64_t count = 0;
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const Channel3Index ch{c};
    for (std::int32_t j = 0; j < s.dim(ch); ++j) {
      for (std::int32_t i = packed ? j : 0; i < s.dim(ch); ++i) {
        if (!rows.Contains(ch, KeptRow3Index{i}) &&
            !rows.Contains(ch, KeptRow3Index{j})) {
          ++count;
        }
      }
    }
  }
  return count;
}

void CheckMask(Hermiticity h) {
  const auto ms = He4Space();
  op::ThreeBody b(ms, h);
  const basis::ThreeBodySpace& s = b.space();
  for (double& x : b.host_span()) x = 1.5;
  b.SyncToDevice();
  const RowSet rows = SomeRows(s);
  ASSERT_GT(rows.NumRows(), 0);
  MaskOutsideLines(b, rows);
  b.SyncToHost();
  std::int64_t nans = 0;
  for (const double x : b.host_span()) {
    if (std::isnan(x)) {
      ++nans;
    } else {
      EXPECT_EQ(x, 1.5);
    }
  }
  EXPECT_EQ(nans, CountOffLines(s, rows, b.packed()));
  EXPECT_GT(nans, 0);
  // Element-wise: NaN exactly off the lines.
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::int64_t n = s.dim(ch);
    const std::int64_t off = b.block_offsets()[static_cast<std::size_t>(c)];
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = b.packed() ? j : 0; i < n; ++i) {
        const std::int64_t idx =
            off +
            (b.packed() ? op::ThreeBody::PackedIndex(n, i, j) : i + j * n);
        const bool on =
            rows.Contains(ch, KeptRow3Index{static_cast<std::int32_t>(i)}) ||
            rows.Contains(ch, KeptRow3Index{static_cast<std::int32_t>(j)});
        ASSERT_EQ(std::isnan(b.host_span()[static_cast<std::size_t>(idx)]),
                  !on);
      }
    }
  }
}

TEST(RowMaskTest, MasksOffLinePackedElements) {
  CheckMask(Hermiticity::kHermitian);
}

TEST(RowMaskTest, MasksOffLineDenseElements) {
  CheckMask(Hermiticity::kNonHermitian);
}

TEST(RowMaskTest, OnlyRowsNonzeroDistinguishesRows) {
  for (const Hermiticity h :
       {Hermiticity::kHermitian, Hermiticity::kNonHermitian}) {
    const auto ms = He4Space();
    op::ThreeBody z(ms, h);
    const basis::ThreeBodySpace& s = z.space();
    const RowSet rows = SomeRows(s);
    const RowSet whole = RowSet::Whole(s);
    EXPECT_TRUE(OnlyRowsNonzero(z, rows));  // all zero
    // Write into a row of the set: still true.
    const RowRange r0 = rows.ranges().front();
    const std::int64_t off =
        z.block_offsets()[static_cast<std::size_t>(r0.channel.value())];
    const std::int64_t n = z.dim(r0.channel);
    const std::int64_t k = r0.row_begin.value();
    const std::int64_t at =
        off + (z.packed() ? op::ThreeBody::PackedIndex(n, k, k) : k + k * n);
    z.host_span()[static_cast<std::size_t>(at)] = 2.0;
    z.SyncToDevice();
    EXPECT_TRUE(OnlyRowsNonzero(z, rows));
    EXPECT_TRUE(OnlyRowsNonzero(z, whole));
    // Write into the first row of channel 0..: a row outside the set.
    std::int32_t other = -1;
    for (std::int32_t kk = 0; kk < n; ++kk) {
      if (!rows.Contains(r0.channel, KeptRow3Index{kk})) {
        other = kk;
        break;
      }
    }
    ASSERT_GE(other, 0);
    const std::int64_t at2 =
        off + (z.packed() ? op::ThreeBody::PackedIndex(n, other, other)
                          : other + other * n);
    z.host_span()[static_cast<std::size_t>(at2)] = 1.0e-300;
    z.SyncToDevice();
    EXPECT_FALSE(OnlyRowsNonzero(z, rows));
    EXPECT_TRUE(OnlyRowsNonzero(z, whole));
  }
}

}  // namespace
}  // namespace imsrg::testing
