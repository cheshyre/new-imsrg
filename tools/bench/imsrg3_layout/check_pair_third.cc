// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark (pair_third); see
// imsrg3_layout_check.cc. Counts of the pair-third states, R = E^T for
// triples of distinct orbits, the symmetry factors for coinciding orbits.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/constants.h"
#include "imsrg/basis/model_space.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

constexpr double kNoCut = 1.0e10;
constexpr double kTol = 1.0e-12;

CanonicalSpace MakeSpace(int emax, int e3max, double de3max) {
  auto ms = basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0,
                                         "O16", "O16")
                ->WithThreeBody(basis::ThreeBodyCuts{e3max});
  return CanonicalSpace(ms, de3max, -1.0);
}

/// Entry (row, col) of a host CSR matrix (0 if absent). O(row length).
double At(const CsrView& m, std::int64_t row, std::int32_t col) {
  for (std::int64_t k = m.ptr[row]; k < m.ptr[row + 1]; ++k) {
    if (m.col[k] == col) return m.val[k];
  }
  return 0.0;
}

/// Index data of one triple: flat rows and states of its channel-local ranges.
struct TripleIndex {
  std::int32_t t = 0;
  std::int32_t channel = 0;
  std::int32_t row_begin = 0;  ///< Channel-local.
  std::int32_t row_end = 0;
  std::int32_t pt_begin = 0;  ///< Channel-local.
  std::int32_t pt_end = 0;
  std::int64_t flat_row0 = 0;  ///< Flat index of channel-local row 0.
  std::int64_t flat_pt0 = 0;   ///< Flat index of channel-local state 0.
};

TripleIndex IndexOf(const CanonicalView& cv, const PairThirdView& pv,
                    std::int32_t t) {
  TripleIndex x;
  x.t = t;
  x.channel = cv.tr_channel[t];
  x.row_begin = cv.tr_row_begin[t];
  x.row_end = cv.tr_row_end[t];
  x.pt_begin = pv.tr_pt_begin[t];
  x.pt_end = pv.tr_pt_end[t];
  x.flat_row0 = cv.ch_row_offset[x.channel];
  x.flat_pt0 = pv.ch_pt_offset[x.channel];
  return x;
}

/// Max |R(row r, state s) - scale * E(state s, row r)| over the triple's rows
/// and the states s in [s0, s1) (channel-local), and the same for the states
/// outside, which must keep R = scale' E^T only when asked; here only [s0,
/// s1).
double MaxRMinusScaledET(const CsrView& rm, const CsrView& em,
                         const TripleIndex& x, std::int32_t s0, std::int32_t s1,
                         double scale) {
  double dev = 0.0;
  for (std::int32_t r = x.row_begin; r < x.row_end; ++r) {
    for (std::int32_t s = s0; s < s1; ++s) {
      const double rv = At(rm, x.flat_row0 + r, s);
      const double ev = At(em, x.flat_pt0 + s, r);
      dev = std::max(dev, std::abs(rv - scale * ev));
    }
  }
  return dev;
}

void ExpectCounts(int emax, int e3max, double de3max, std::int64_t sum_n,
                  std::int64_t sum_n2, int max_n) {
  const CanonicalSpace space = MakeSpace(emax, e3max, de3max);
  const SixJ3Table sixj(emax);
  const PairThirdSpace pt(space, sixj);
  const SpaceCounts c = pt.Counts();
  const SpaceCounts o = PairThirdSpace::CountOnly(space);
  EXPECT_EQ(c.sum_n, sum_n);
  EXPECT_EQ(c.sum_n2, sum_n2);
  EXPECT_EQ(c.max_n, max_n);
  EXPECT_EQ(pt.NumStates(), sum_n);
  EXPECT_EQ(c.num_channels, space.Counts().num_channels);
  EXPECT_EQ(c.num_triples, space.Counts().num_triples);
  EXPECT_EQ(o.sum_n, c.sum_n);
  EXPECT_EQ(o.sum_n2, c.sum_n2);
  EXPECT_EQ(o.sum_packed, c.sum_packed);
  EXPECT_EQ(o.max_n, c.max_n);
  EXPECT_EQ(o.num_channels, c.num_channels);
  EXPECT_EQ(o.num_triples, c.num_triples);
}

TEST(PairThirdCounts, O16Emax2E3max6) {
  ExpectCounts(2, 6, kNoCut, 5754, 1135982, 320);
}
TEST(PairThirdCounts, O16Emax3E3max9) {
  ExpectCounts(3, 9, kNoCut, 37072, 36423808, 1586);
}
TEST(PairThirdCounts, O16Emax3E3max7) {
  ExpectCounts(3, 7, kNoCut, 25696, 19272500, 1314);
}
TEST(PairThirdCounts, O16Emax3E3max9De3max3) {
  ExpectCounts(3, 9, 3.0, 17096, 8510804, 799);
}

/// Checks R = E^T and orthogonality for all triples of distinct orbits.
void CheckDistinctTriples(int emax, int e3max) {
  const CanonicalSpace space = MakeSpace(emax, e3max, kNoCut);
  const SixJ3Table sixj(emax);
  const PairThirdSpace pt(space, sixj);
  const CanonicalView cv = space.host_view();
  const PairThirdView pv = pt.host_view();
  const CsrView rm = pt.RMap().host_view();
  const CsrView em = pt.EMap().host_view();
  int num_checked = 0;
  double max_dev = 0.0;
  double max_orth = 0.0;
  for (std::int32_t t = 0; t < cv.num_triples; ++t) {
    if (cv.tr_a[t] == cv.tr_b[t] || cv.tr_b[t] == cv.tr_c[t]) continue;
    ++num_checked;
    const TripleIndex x = IndexOf(cv, pv, t);
    max_dev = std::max(max_dev,
                       MaxRMinusScaledET(rm, em, x, x.pt_begin, x.pt_end, 1.0));
    // Splittings are runs of equal (p, q, r); each E block is square and
    // orthogonal.
    std::int32_t s0 = x.pt_begin;
    while (s0 < x.pt_end) {
      std::int32_t s1 = s0;
      const std::int64_t f0 = x.flat_pt0 + s0;
      while (s1 < x.pt_end && pv.pt_p[x.flat_pt0 + s1] == pv.pt_p[f0] &&
             pv.pt_q[x.flat_pt0 + s1] == pv.pt_q[f0] &&
             pv.pt_r[x.flat_pt0 + s1] == pv.pt_r[f0]) {
        ++s1;
      }
      EXPECT_EQ(s1 - s0, x.row_end - x.row_begin) << "triple " << t;
      for (std::int32_t i = s0; i < s1; ++i) {
        for (std::int32_t j = s0; j < s1; ++j) {
          double dot = 0.0;
          for (std::int32_t r = x.row_begin; r < x.row_end; ++r) {
            dot += At(em, x.flat_pt0 + i, r) * At(em, x.flat_pt0 + j, r);
          }
          max_orth = std::max(max_orth, std::abs(dot - (i == j ? 1.0 : 0.0)));
        }
      }
      s0 = s1;
    }
  }
  EXPECT_GT(num_checked, 0);
  EXPECT_LT(max_dev, kTol);
  EXPECT_LT(max_orth, kTol);
}

TEST(PairThirdDistinct, Emax2) { CheckDistinctTriples(2, 6); }
TEST(PairThirdDistinct, Emax3E3max7) { CheckDistinctTriples(3, 7); }

/// Coinciding orbits: R = sqrt(2) E^T on the (x x) y states, 2 E^T on the
/// (a c) a states of a == b < c.
void CheckCoincident(int emax, int e3max) {
  const CanonicalSpace space = MakeSpace(emax, e3max, kNoCut);
  const SixJ3Table sixj(emax);
  const PairThirdSpace pt(space, sixj);
  const CanonicalView cv = space.host_view();
  const PairThirdView pv = pt.host_view();
  const CsrView rm = pt.RMap().host_view();
  const CsrView em = pt.EMap().host_view();
  int num_pair_states = 0;
  int num_other_states = 0;
  double max_dev = 0.0;
  for (std::int32_t t = 0; t < cv.num_triples; ++t) {
    const int a = cv.tr_a[t];
    const int b = cv.tr_b[t];
    const int c = cv.tr_c[t];
    const bool aab = a == b && b < c;
    const bool abb = a < b && b == c;
    if (!aab && !abb) continue;
    const TripleIndex x = IndexOf(cv, pv, t);
    for (std::int32_t s = x.pt_begin; s < x.pt_end; ++s) {
      const std::int64_t f = x.flat_pt0 + s;
      const int p = pv.pt_p[f];
      const int q = pv.pt_q[f];
      double scale = 0.0;
      if (p == q) {
        scale = constants::kSqrt2;  // (x x) y with x the repeated orbit.
        ++num_pair_states;
      } else if (aab) {
        scale = 2.0;  // (a c) a: two permutations reach the state.
        ++num_other_states;
      } else {
        continue;  // (a b) b of a < b == b: not constrained here.
      }
      max_dev =
          std::max(max_dev, MaxRMinusScaledET(rm, em, x, s, s + 1, scale));
    }
  }
  EXPECT_GT(num_pair_states, 0);
  EXPECT_GT(num_other_states, 0);
  EXPECT_LT(max_dev, kTol);
}

TEST(PairThirdCoincident, Emax2) { CheckCoincident(2, 6); }
TEST(PairThirdCoincident, Emax3E3max7) { CheckCoincident(3, 7); }

TEST(PairThirdMaps, EveryRowHasAnREntryAndSizesAreReported) {
  const CanonicalSpace space = MakeSpace(3, 9, kNoCut);
  const SixJ3Table sixj(3);
  const PairThirdSpace pt(space, sixj);
  const CsrView rm = pt.RMap().host_view();
  for (std::int64_t r = 0; r < space.NumRows(); ++r) {
    ASSERT_LT(rm.ptr[r], rm.ptr[r + 1]) << "row " << r;
  }
  EXPECT_GT(pt.EMap().Nonzeros(), 0);
  EXPECT_GT(pt.StateBytes(), 0);
  RecordProperty("nnz_R_emax3_e3max9", std::to_string(pt.RMap().Nonzeros()));
  RecordProperty("nnz_E_emax3_e3max9", std::to_string(pt.EMap().Nonzeros()));
  std::cout << "PairThird emax 3 E3max 9: nnz(R) = " << pt.RMap().Nonzeros()
            << ", nnz(E) = " << pt.EMap().Nonzeros() << std::endl;
}

}  // namespace
}  // namespace imsrg::layout3
