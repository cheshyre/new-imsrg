// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark (layouts); see
// imsrg3_layout_check.cc. Counts of layouts A, B and C, block conversions
// (round trips, norms) and the outer maps L against dense products.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/layouts.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

constexpr double kNoCut = 1.0e10;
constexpr double kTol = 1.0e-12;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

CanonicalSpace MakeSpace(int emax, int e3max, double de3max) {
  auto ms = basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0,
                                         "O16", "O16")
                ->WithThreeBody(basis::ThreeBodyCuts{e3max});
  return CanonicalSpace(ms, de3max, -1.0);
}

/// Everything the layout checks need for one space.
struct Fixture {
  Fixture(int emax, int e3max, double de3max)
      : space(MakeSpace(emax, e3max, de3max)),
        sixj(emax),
        pt(space, sixj),
        a(space, pt),
        b(space, pt, sixj),
        c(space, pt) {}
  CanonicalSpace space;
  SixJ3Table sixj;
  PairThirdSpace pt;
  LayoutA a;
  LayoutB b;
  LayoutC c;
};

std::vector<double> RandomBlock(std::int64_t n2, std::mt19937* rng) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(U(n2));
  for (double& v : x) v = dist(*rng);
  return x;
}

/// max |x - y| / max(1, max |y|).
double RelDiff(std::span<const double> x, std::span<const double> y) {
  double diff = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    diff = std::max(diff, std::abs(x[i] - y[i]));
    scale = std::max(scale, std::abs(y[i]));
  }
  return diff / scale;
}

void ExpectSameCounts(const SpaceCounts& x, std::int64_t sum_n,
                      std::int64_t sum_n2, int max_n) {
  EXPECT_EQ(x.sum_n, sum_n);
  EXPECT_EQ(x.sum_n2, sum_n2);
  EXPECT_EQ(x.max_n, max_n);
}

template <class L>
void ExpectStructure(const L& l) {
  const CanonicalSpace& s = l.Space();
  ASSERT_EQ(static_cast<std::int32_t>(l.Dims().size()), s.num_channels());
  ASSERT_EQ(l.BlockOffsets().size(), l.Dims().size() + 1);
  EXPECT_EQ(l.BlockOffsets()[0], 0);
  const std::vector<linalg::BlockShape> shapes = l.Shapes();
  ASSERT_EQ(shapes.size(), l.Dims().size());
  std::int64_t rows = 0;
  for (std::size_t c = 0; c < l.Dims().size(); ++c) {
    const std::int64_t n = l.Dims()[c];
    EXPECT_EQ(l.BlockOffsets()[c + 1] - l.BlockOffsets()[c], n * n);
    EXPECT_EQ(shapes[c].row_block.value(), static_cast<std::int32_t>(c));
    EXPECT_EQ(shapes[c].col_block.value(), static_cast<std::int32_t>(c));
    EXPECT_EQ(shapes[c].rows, n);
    EXPECT_EQ(shapes[c].cols, n);
    rows += n;
  }
  EXPECT_EQ(l.total_size(), l.BlockOffsets().back());
  const LayoutView v = l.device_view();
  const CanonicalView cv = s.host_view();
  // Triples tile each channel contiguously; the outer map has one row per
  // layout row and its columns ascend.
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    std::int32_t expect_begin = 0;
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      EXPECT_EQ(v.tr_row_begin[t], expect_begin);
      EXPECT_GE(v.tr_row_end[t], v.tr_row_begin[t]);
      expect_begin = v.tr_row_end[t];
    }
    EXPECT_EQ(expect_begin, v.ch_dim[ch]);
  }
  EXPECT_EQ(v.outer.ptr[0], 0);
  for (std::int64_t r = 0; r < rows; ++r) {
    ASSERT_LE(v.outer.ptr[r], v.outer.ptr[r + 1]);
    for (std::int64_t k = v.outer.ptr[r] + 1; k < v.outer.ptr[r + 1]; ++k) {
      ASSERT_LT(v.outer.col[k - 1], v.outer.col[k]);
    }
  }
  EXPECT_GT(l.MetadataBytes(), 0);
}

void ExpectLayoutCounts(int emax, int e3max, double de3max,
                        const std::int64_t (&b)[2], int b_max,
                        const std::int64_t (&c)[2], int c_max) {
  const Fixture f(emax, e3max, de3max);
  const SpaceCounts sc = f.space.Counts();
  const SpaceCounts a = f.a.Counts();
  EXPECT_EQ(a.sum_n, sc.sum_n);
  EXPECT_EQ(a.sum_n2, sc.sum_n2);
  EXPECT_EQ(a.max_n, sc.max_n);
  EXPECT_EQ(f.a.total_size(), f.space.total_size());
  ExpectSameCounts(f.b.Counts(), b[0], b[1], b_max);
  ExpectSameCounts(f.c.Counts(), c[0], c[1], c_max);
  const SpaceCounts only = LayoutB::CountOnly(f.space);
  const SpaceCounts built = f.b.Counts();
  EXPECT_EQ(only.sum_n, built.sum_n);
  EXPECT_EQ(only.sum_n2, built.sum_n2);
  EXPECT_EQ(only.sum_packed, built.sum_packed);
  EXPECT_EQ(only.max_n, built.max_n);
  EXPECT_EQ(only.num_channels, built.num_channels);
  EXPECT_EQ(only.num_triples, built.num_triples);
  const SpaceCounts pc = f.pt.Counts();
  EXPECT_EQ(f.c.Counts().sum_n, pc.sum_n);
  std::cout << "[layout counts] emax " << emax << " E3max " << e3max
            << " dE3max " << de3max << ": A " << sc.sum_n << "/" << sc.sum_n2
            << "/" << sc.max_n << ", B " << built.sum_n << "/" << built.sum_n2
            << "/" << built.max_n << ", C " << f.c.Counts().sum_n << "/"
            << f.c.Counts().sum_n2 << "/" << f.c.Counts().max_n << "\n";
  ExpectStructure(f.a);
  ExpectStructure(f.b);
  ExpectStructure(f.c);
}

// Layout B counts equal the planning pass's prediction (1,842/119,790/103;
// 12,112/3,941,688/521; 8,400/2,083,528/432; 5,576/918,040/262): the
// canonical kets that vanish under antisymmetrization are dropped by the
// projection. Layout C counts are those of check_pair_third.cc.
TEST(LayoutCounts, O16Emax2E3max6) {
  ExpectLayoutCounts(2, 6, kNoCut, {1842, 119790}, 103, {5574, 1090366}, 311);
}
TEST(LayoutCounts, O16Emax3E3max9) {
  ExpectLayoutCounts(3, 9, kNoCut, {12112, 3941688}, 521, {36564, 35782440},
                     1574);
}
TEST(LayoutCounts, O16Emax3E3max7) {
  ExpectLayoutCounts(3, 7, kNoCut, {8400, 2083528}, 432, {25324, 18895336},
                     1304);
}
TEST(LayoutCounts, O16Emax3E3max9De3max3) {
  ExpectLayoutCounts(3, 9, 3.0, {5576, 918040}, 262, {16840, 8346800}, 793);
}

/// Random z_B per channel -> canonical -> layout B again.
TEST(LayoutRoundTrip, B) {
  std::mt19937 rng(7);
  for (const auto& [emax, e3max] : {std::pair{2, 6}, std::pair{3, 7}}) {
    const Fixture f(emax, e3max, kNoCut);
    const CanonicalView cv = f.space.host_view();
    double worst = 0.0;
    for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
      const std::int64_t nb = f.b.Dims()[U(ch)];
      const std::int64_t na = cv.ch_dim[ch];
      const std::vector<double> zb = RandomBlock(nb * nb, &rng);
      std::vector<double> za(U(na * na));
      std::vector<double> back(U(nb * nb));
      f.b.ToCanonical(ch, zb, za);
      f.b.FromCanonical(ch, za, back);
      worst = std::max(worst, RelDiff(back, zb));
    }
    EXPECT_LE(worst, kTol) << "emax " << emax;
  }
}

/// z_A = ToCanonical_B(random z_B), then C -> canonical returns z_A.
TEST(LayoutRoundTrip, C) {
  std::mt19937 rng(11);
  for (const auto& [emax, e3max] : {std::pair{2, 6}, std::pair{3, 7}}) {
    const Fixture f(emax, e3max, kNoCut);
    const CanonicalView cv = f.space.host_view();
    double worst = 0.0;
    for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
      const std::int64_t nb = f.b.Dims()[U(ch)];
      const std::int64_t na = cv.ch_dim[ch];
      const std::int64_t nc = f.c.Dims()[U(ch)];
      const std::vector<double> zb = RandomBlock(nb * nb, &rng);
      std::vector<double> za(U(na * na));
      f.b.ToCanonical(ch, zb, za);
      std::vector<double> zc(U(nc * nc));
      std::vector<double> back(U(na * na));
      f.c.FromCanonical(ch, za, zc);
      f.c.ToCanonical(ch, zc, back);
      worst = std::max(worst, RelDiff(back, za));
    }
    EXPECT_LE(worst, kTol) << "emax " << emax;
  }
}

TEST(LayoutRoundTrip, AIsTheIdentity) {
  const Fixture f(2, 6, kNoCut);
  std::mt19937 rng(3);
  for (std::int32_t ch = 0; ch < f.space.num_channels(); ++ch) {
    const std::int64_t n = f.a.Dims()[U(ch)];
    const std::vector<double> z = RandomBlock(n * n, &rng);
    std::vector<double> out(U(n * n));
    std::vector<double> back(U(n * n));
    f.a.ToCanonical(ch, z, out);
    f.a.FromCanonical(ch, out, back);
    EXPECT_EQ(out, z);
    EXPECT_EQ(back, z);
  }
}

/// sum z_B^2 = sum_ij w_i w_j z_A,ij^2 (w = sym / 6 of the row's triple) and
/// the imsrg++ norm of the layout equals that of its canonical image.
TEST(LayoutNorm, BMatchesWeightedCanonicalNorm) {
  std::mt19937 rng(5);
  const Fixture f(2, 6, kNoCut);
  const CanonicalView cv = f.space.host_view();
  std::vector<double> z_b(U(f.b.total_size()));
  std::vector<double> z_a(U(f.space.total_size()));
  double lhs = 0.0;
  double rhs = 0.0;
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t nb = f.b.Dims()[U(ch)];
    const std::int64_t na = cv.ch_dim[ch];
    const std::vector<double> zb = RandomBlock(nb * nb, &rng);
    std::copy(zb.begin(), zb.end(), z_b.begin() + f.b.BlockOffsets()[U(ch)]);
    std::vector<double> za(U(na * na));
    f.b.ToCanonical(ch, zb, za);
    std::copy(za.begin(), za.end(), z_a.begin() + cv.ch_block_offset[ch]);
    for (const double x : zb) lhs += x * x;
    const std::int32_t* row_triple = cv.row_triple + cv.ch_row_offset[ch];
    for (std::int64_t j = 0; j < na; ++j) {
      for (std::int64_t i = 0; i < na; ++i) {
        const double wi = cv.tr_sym[row_triple[i]] / 6.0;
        const double wj = cv.tr_sym[row_triple[j]] / 6.0;
        rhs += wi * wj * za[U(i + j * na)] * za[U(i + j * na)];
      }
    }
  }
  EXPECT_LE(std::abs(lhs - rhs), kTol * std::max(1.0, rhs));
  const double nb = LayoutImsrgppNorm(f.b, z_b);
  const double nc = ImsrgppNorm(f.space, z_a);
  EXPECT_LE(std::abs(nb - nc), kTol * std::max(1.0, nc));
}

/// Dense row `r` (flat layout row) of an outer map as a vector of length `n`.
std::vector<double> DenseRow(const CsrView& m, std::int64_t r, std::int64_t n) {
  std::vector<double> row(U(n), 0.0);
  for (std::int64_t k = m.ptr[r]; k < m.ptr[r + 1]; ++k) {
    row[U(m.col[k])] = m.val[k];
  }
  return row;
}

void AddScaled(std::vector<double>* acc, const std::vector<double>& x,
               double coef) {
  for (std::size_t i = 0; i < x.size(); ++i) (*acc)[i] += coef * x[i];
}

/// max over the rows of the layout of |L_csr - expected| where `expected(ch,
/// local_row)` returns the dense row of channel-local pair-third states.
template <class Expected>
double OuterMapDeviation(const LayoutView& v, const CanonicalView& cv,
                         const PairThirdView& pv, Expected expected) {
  double worst = 0.0;
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      for (std::int32_t r = v.tr_row_begin[t]; r < v.tr_row_end[t]; ++r) {
        const std::int64_t flat = v.ch_row_offset[ch] + r;
        const std::vector<double> got =
            DenseRow(v.outer, flat, pv.ch_num_pt[ch]);
        const std::vector<double> want = expected(ch, r);
        for (std::int64_t k = v.outer.ptr[flat]; k < v.outer.ptr[flat + 1];
             ++k) {
          EXPECT_GE(v.outer.col[k], pv.tr_pt_begin[t]);
          EXPECT_LT(v.outer.col[k], pv.tr_pt_end[t]);
        }
        for (std::size_t i = 0; i < got.size(); ++i) {
          worst = std::max(worst, std::abs(got[i] - want[i]));
        }
      }
    }
  }
  return worst;
}

TEST(LayoutOuterMap, AIsR) {
  const Fixture f(2, 6, kNoCut);
  const CanonicalView cv = f.space.host_view();
  const PairThirdView pv = f.pt.host_view();
  const CsrView rm = f.pt.RMap().host_view();
  const LayoutView v = f.a.device_view();
  const double dev =
      OuterMapDeviation(v, cv, pv, [&](std::int32_t ch, std::int32_t r) {
        return DenseRow(rm, cv.ch_row_offset[ch] + r, pv.ch_num_pt[ch]);
      });
  EXPECT_LE(dev, kTol);
  const std::int64_t rows = cv.num_rows;
  EXPECT_EQ(v.outer.ptr[rows], rm.ptr[rows]);
  for (std::int32_t t = 0; t < cv.num_triples; ++t) {
    const std::int32_t ch = cv.tr_channel[t];
    const std::int64_t row = cv.ch_row_offset[ch] + cv.tr_row_begin[t];
    if (cv.tr_row_end[t] > cv.tr_row_begin[t]) {
      EXPECT_DOUBLE_EQ(v.energy_weight[row], std::sqrt(cv.tr_sym[t] / 6.0));
    }
  }
}

TEST(LayoutOuterMap, CIsEr) {
  for (const auto& [emax, e3max] : {std::pair{2, 6}, std::pair{3, 7}}) {
    const Fixture f(emax, e3max, kNoCut);
    const CanonicalView cv = f.space.host_view();
    const PairThirdView pv = f.pt.host_view();
    const CsrView rm = f.pt.RMap().host_view();
    const CsrView em = f.pt.EMap().host_view();
    const LayoutView v = f.c.device_view();
    const double dev =
        OuterMapDeviation(v, cv, pv, [&](std::int32_t ch, std::int32_t s) {
          std::vector<double> acc(U(pv.ch_num_pt[ch]), 0.0);
          const std::int64_t flat = pv.ch_pt_offset[ch] + s;
          for (std::int64_t k = em.ptr[flat]; k < em.ptr[flat + 1]; ++k) {
            AddScaled(&acc,
                      DenseRow(rm, cv.ch_row_offset[ch] + em.col[k],
                               pv.ch_num_pt[ch]),
                      em.val[k]);
          }
          return acc;
        });
    EXPECT_LE(dev, kTol) << "emax " << emax;
    // Weights: s_T on the ABC states of the canonical rows, 0 elsewhere.
    double wsum = 0.0;
    for (std::int64_t i = 0; i < pv.num_states; ++i) wsum += v.energy_weight[i];
    double expect = 0.0;
    for (std::int64_t r = 0; r < cv.num_rows; ++r) {
      expect += std::sqrt(cv.tr_sym[cv.row_triple[r]] / 6.0);
    }
    EXPECT_NEAR(wsum, expect, 1.0e-9 * expect);
  }
}

/// S = (dB x nA) conversion matrix of layout B in channel `ch`, recovered from
/// `FromCanonical` of rank-1 canonical blocks (up to one global sign).
std::vector<double> RecoverS(const LayoutB& b, std::int32_t ch) {
  const std::int64_t na = b.Space().host_view().ch_dim[ch];
  const std::int64_t nb = b.Dims()[U(ch)];
  const auto convert = [&](std::int64_t i, std::int64_t j) {
    std::vector<double> za(U(na * na), 0.0);
    za[U(i + j * na)] = 1.0;
    std::vector<double> zb(U(nb * nb));
    b.FromCanonical(ch, za, zb);
    return zb;
  };
  std::int64_t js = 0;
  double best = -1.0;
  for (std::int64_t j = 0; j < na; ++j) {
    const std::vector<double> zb = convert(j, j);
    double tr = 0.0;
    for (std::int64_t k = 0; k < nb; ++k) tr += zb[U(k + k * nb)];
    if (tr > best) {
      best = tr;
      js = j;
    }
  }
  std::vector<double> u(U(nb), 0.0);
  std::int64_t kmax = 0;
  {
    const std::vector<double> zb = convert(js, js);
    for (std::int64_t k = 0; k < nb; ++k) {
      if (zb[U(k + k * nb)] > zb[U(kmax + kmax * nb)]) kmax = k;
    }
    const double root = std::sqrt(zb[U(kmax + kmax * nb)]);
    for (std::int64_t k = 0; k < nb; ++k) u[U(k)] = zb[U(k + kmax * nb)] / root;
  }
  std::vector<double> s(U(nb * na), 0.0);
  for (std::int64_t j = 0; j < na; ++j) {
    const std::vector<double> zb = convert(j, js);
    for (std::int64_t k = 0; k < nb; ++k) {
      s[U(k + j * nb)] = zb[U(k + kmax * nb)] / u[U(kmax)];
    }
  }
  return s;
}

TEST(LayoutOuterMap, BIsSR) {
  const Fixture f(2, 6, kNoCut);
  const CanonicalView cv = f.space.host_view();
  const PairThirdView pv = f.pt.host_view();
  const CsrView rm = f.pt.RMap().host_view();
  const LayoutView v = f.b.device_view();
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t na = cv.ch_dim[ch];
    const std::int64_t nb = f.b.Dims()[U(ch)];
    const std::vector<double> s = RecoverS(f.b, ch);
    // Global sign: the sign of S that matches the first nonzero of L.
    std::vector<std::vector<double>> want(U(nb));
    for (std::int64_t k = 0; k < nb; ++k) {
      std::vector<double> acc(U(pv.ch_num_pt[ch]), 0.0);
      for (std::int64_t j = 0; j < na; ++j) {
        AddScaled(&acc,
                  DenseRow(rm, cv.ch_row_offset[ch] + j, pv.ch_num_pt[ch]),
                  s[U(k + j * nb)]);
      }
      want[U(k)] = std::move(acc);
    }
    double sign = 1.0;
    for (std::int64_t k = 0; k < nb && sign == 1.0; ++k) {
      const std::vector<double> got =
          DenseRow(v.outer, v.ch_row_offset[ch] + k, pv.ch_num_pt[ch]);
      for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::abs(got[i]) > 1.0e-6) {
          sign = got[i] * want[U(k)][i] < 0.0 ? -1.0 : 1.0;
          k = nb;
          break;
        }
      }
    }
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      for (std::int32_t k = v.tr_row_begin[t]; k < v.tr_row_end[t]; ++k) {
        const std::int64_t flat = v.ch_row_offset[ch] + k;
        const std::vector<double> got =
            DenseRow(v.outer, flat, pv.ch_num_pt[ch]);
        for (std::size_t i = 0; i < got.size(); ++i) {
          EXPECT_NEAR(got[i], sign * want[U(k)][i], kTol) << "ch " << ch;
        }
        for (std::int64_t e = v.outer.ptr[flat]; e < v.outer.ptr[flat + 1];
             ++e) {
          EXPECT_GE(v.outer.col[e], pv.tr_pt_begin[t]);
          EXPECT_LT(v.outer.col[e], pv.tr_pt_end[t]);
        }
        EXPECT_EQ(v.energy_weight[flat], 1.0);
      }
    }
  }
}

/// Metadata bytes and build times at emax 3 (printed, not asserted).
TEST(LayoutMetadata, BytesAndBuildTimeEmax3) {
  const CanonicalSpace space = MakeSpace(3, 9, kNoCut);
  const SixJ3Table sixj(3);
  const PairThirdSpace pt(space, sixj);
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  auto t0 = Clock::now();
  const LayoutA a(space, pt);
  const double ta = ms(t0);
  t0 = Clock::now();
  const LayoutB b(space, pt, sixj);
  const double tb = ms(t0);
  t0 = Clock::now();
  const LayoutC c(space, pt);
  const double tc = ms(t0);
  std::cout << "[layout metadata] emax 3 E3max 9: A " << a.MetadataBytes()
            << " B, " << ta << " ms; B " << b.MetadataBytes() << " B, " << tb
            << " ms; C " << c.MetadataBytes() << " B, " << tc << " ms\n";
  EXPECT_GT(a.MetadataBytes(), 0);
  EXPECT_GT(b.MetadataBytes(), 0);
  EXPECT_GT(c.MetadataBytes(), 0);
}

}  // namespace
}  // namespace imsrg::layout3
