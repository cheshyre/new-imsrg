// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of SixJ3Table, the three-body recoupling helpers and the guards of
// SixJTable / SixJView against three-body J in a single-particle slot (the "6j
// trap", design/benchmarks/imsrg3-layout.md section 7).
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/angmom/tables.h"
#include "imsrg/angmom/three_body_recoupling.h"
#include "imsrg/angmom/wigner.h"
#include "imsrg/base/error.h"
#include "tests/unit/angmom/sixj3_kernels.h"

namespace imsrg::angmom {
namespace {

constexpr std::array<ThreeBodyPerm, 6> kAllPerms = {
    ThreeBodyPerm::kABC, ThreeBodyPerm::kBCA, ThreeBodyPerm::kCAB,
    ThreeBodyPerm::kACB, ThreeBodyPerm::kBAC, ThreeBodyPerm::kCBA};

// 6j family evaluated directly (no table), same interface as SixJ3View.
struct DirectSixJ {
  double operator()(TwoJ j1, TwoJ j2, TwoJ J12, TwoJ j3, TwoJ J,
                    TwoJ J23) const noexcept {
    return Wigner6j(j1, j2, J12, j3, J, J23);
  }
  double Hat(std::int32_t J) const noexcept { return std::sqrt(2.0 * J + 1.0); }
};

// Independent transcription of the documented formula (three_body_recoupling.h)
// straight from Wigner6j.
double ReferenceRecoupling(ThreeBodyPerm p, int a2, int b2, int c2, int Jin,
                           int Jab, int J2) {
  if (!(Triangle2(a2, b2, 2 * Jab) && Triangle2(c2, J2, 2 * Jab))) return 0.0;
  const double hh = std::sqrt((2.0 * Jin + 1.0) * (2.0 * Jab + 1.0));
  const auto sj = [&](int x, int y, int z) {
    return Wigner6j(TwoJ{x}, TwoJ{y}, TwoJ{2 * Jab}, TwoJ{z}, TwoJ{J2},
                    TwoJ{2 * Jin});
  };
  const auto sgn = [](int n) { return (n % 2 == 0) ? 1.0 : -1.0; };
  switch (p) {
    case ThreeBodyPerm::kABC:
      return Jab == Jin ? 1.0 : 0.0;
    case ThreeBodyPerm::kBAC:
      return Jab == Jin ? sgn((a2 + b2) / 2 - Jab) : 0.0;
    case ThreeBodyPerm::kBCA:
      return sgn((b2 + c2) / 2 + Jin + 1) * hh * sj(a2, b2, c2);
    case ThreeBodyPerm::kCAB:
      return sgn((a2 + b2) / 2 - Jab + 1) * hh * sj(b2, a2, c2);
    case ThreeBodyPerm::kACB:
      return sgn((b2 + c2) / 2 + Jin - Jab) * hh * sj(b2, a2, c2);
    case ThreeBodyPerm::kCBA:
      return -hh * sj(a2, b2, c2);
  }
  return 0.0;
}

TEST(SixJ3Table, EqualsWignerForEveryArgument) {
  const int emax = 3;
  const SixJ3Table t(emax);
  EXPECT_EQ(t.emax(), emax);
  EXPECT_EQ(SixJ3Table::ValueCount(emax), 4 * 4 * 4 * 8 * 8 * 11);
  const SixJ3View v = t.host_view();
  const int tmax = 2 * emax + 1;
  std::int64_t nonzero_triangles = 0;
  std::int64_t total = 0;
  double max_diff = 0.0;
  for (int j1 = 1; j1 <= tmax; j1 += 2) {
    for (int j2 = 1; j2 <= tmax; j2 += 2) {
      for (int J12 = 0; J12 <= 2 * tmax; J12 += 2) {
        for (int j3 = 1; j3 <= tmax; j3 += 2) {
          for (int J = 1; J <= 6 * emax + 3; J += 2) {
            for (int J23 = 0; J23 <= 2 * tmax; J23 += 2) {
              const double ref = Wigner6j(TwoJ{j1}, TwoJ{j2}, TwoJ{J12},
                                          TwoJ{j3}, TwoJ{J}, TwoJ{J23});
              const double got = v(TwoJ{j1}, TwoJ{j2}, TwoJ{J12}, TwoJ{j3},
                                   TwoJ{J}, TwoJ{J23});
              const bool ok = Triangle2(j1, j2, J12) && Triangle2(j1, J, J23) &&
                              Triangle2(j3, j2, J23) && Triangle2(j3, J, J12);
              ++total;
              if (ok) ++nonzero_triangles;
              if (!ok) {
                ASSERT_EQ(got, 0.0);
              }
              max_diff = std::max(max_diff, std::abs(got - ref));
              ASSERT_EQ(t(TwoJ{j1}, TwoJ{j2}, TwoJ{J12}, TwoJ{j3}, TwoJ{J},
                          TwoJ{J23}),
                        got);
            }
          }
        }
      }
    }
  }
  EXPECT_LT(max_diff, 1.0e-14);
  EXPECT_EQ(total, 45056);
  EXPECT_GT(nonzero_triangles, 1000);
  for (int J = 0; J <= 2 * emax + 1; ++J) {
    EXPECT_NEAR(v.Hat(J), std::sqrt(2.0 * J + 1.0), 1.0e-15);
  }
  EXPECT_GT(t.size_bytes(), 0);
}

TEST(SixJ3Table, ZeroOutsideRangesAndTriangles) {
  const SixJ3Table t(3);
  // Triangle (j1 j2 J12) fails: 1/2 + 1/2 < 2.
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{4}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  // 2j beyond 2 emax + 1, 2J beyond 6 emax + 3, J12 beyond 2 (2 emax + 1).
  EXPECT_EQ(t(TwoJ{9}, TwoJ{1}, TwoJ{4}, TwoJ{1}, TwoJ{5}, TwoJ{4}), 0.0);
  EXPECT_EQ(t(TwoJ{7}, TwoJ{7}, TwoJ{4}, TwoJ{7}, TwoJ{21}, TwoJ{4}), 0.0);
  EXPECT_EQ(t(TwoJ{7}, TwoJ{7}, TwoJ{16}, TwoJ{7}, TwoJ{7}, TwoJ{4}), 0.0);
  // Wrong parities and negative arguments.
  EXPECT_EQ(t(TwoJ{2}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{2}, TwoJ{0}), 0.0);
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(t(TwoJ{-1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(t(TwoJ{1}, TwoJ{1}, TwoJ{-2}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  // A valid nonzero entry for contrast: {1/2 1/2 0; 1/2 1/2 0} = -1/2.
  EXPECT_NEAR(t(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), -0.5,
              1.0e-15);
}

TEST(SixJ3Table, DeviceViewSpotCheck) {
  const int emax = 3;
  const SixJ3Table t(emax);
  std::vector<std::int32_t> q;
  std::vector<double> expect;
  const auto add = [&](int j1, int j2, int J12, int j3, int J, int J23) {
    q.insert(q.end(), {j1, j2, J12, j3, J, J23});
    expect.push_back(
        t(TwoJ{j1}, TwoJ{j2}, TwoJ{J12}, TwoJ{j3}, TwoJ{J}, TwoJ{J23}));
  };
  for (int j1 = 1; j1 <= 7; j1 += 4) {
    for (int j2 = 3; j2 <= 7; j2 += 2) {
      for (int J12 = 0; J12 <= 14; J12 += 2) {
        for (int J = 1; J <= 21; J += 4) {
          add(j1, j2, J12, 5, J, 6);
          add(j1, j2, J12, 7, J, 10);
        }
      }
    }
  }
  add(9, 1, 4, 1, 5, 4);  // out of range -> 0.0
  ASSERT_FALSE(expect.empty());
  exec::DeviceBuffer<std::int32_t> qb(q.size());
  qb.CopyFromHost(q);
  exec::DeviceBuffer<double> out(expect.size());
  test::EvalSixJ3Device(t, qb, out);
  const auto got = out.ToVector();
  bool any_nonzero = false;
  for (std::size_t i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(got[i], expect[i]) << i;
    any_nonzero = any_nonzero || expect[i] != 0.0;
  }
  EXPECT_TRUE(any_nonzero);
}

TEST(ThreeBodyRecoupling, MatchesDirectFormula) {
  const SixJ3Table t(2);
  const SixJ3View v = t.host_view();
  const DirectSixJ direct;
  std::int64_t nonzero = 0;
  for (const ThreeBodyPerm p : kAllPerms) {
    for (int a2 = 1; a2 <= 5; a2 += 2) {
      for (int b2 = 1; b2 <= 5; b2 += 2) {
        for (int c2 = 1; c2 <= 5; c2 += 2) {
          for (int J2 = 1; J2 <= 15; J2 += 2) {
            for (int Jin = 0; Jin <= 5; ++Jin) {
              for (int Jab = 0; Jab <= 5; ++Jab) {
                const double ref =
                    ReferenceRecoupling(p, a2, b2, c2, Jin, Jab, J2);
                const double via_table = RecouplingCoefficient(
                    p, TwoJ{a2}, TwoJ{b2}, TwoJ{c2}, Jin, Jab, TwoJ{J2}, v);
                const double via_direct =
                    RecouplingCoefficient(p, TwoJ{a2}, TwoJ{b2}, TwoJ{c2}, Jin,
                                          Jab, TwoJ{J2}, direct);
                ASSERT_NEAR(via_table, ref, 1.0e-14);
                ASSERT_NEAR(via_direct, ref, 1.0e-14);
                if (ref != 0.0) ++nonzero;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(nonzero, 0);
}

TEST(ThreeBodyRecoupling, PermuteAndPhase) {
  int o1 = 0;
  int o2 = 0;
  int o3 = 0;
  Permute(ThreeBodyPerm::kCBA, 1, 2, 3, o1, o2, o3);
  EXPECT_EQ(o1 * 100 + o2 * 10 + o3, 321);
  Permute(ThreeBodyPerm::kBCA, 1, 2, 3, o1, o2, o3);
  EXPECT_EQ(o1 * 100 + o2 * 10 + o3, 231);
  Permute(ThreeBodyPerm::kCAB, 1, 2, 3, o1, o2, o3);
  EXPECT_EQ(o1 * 100 + o2 * 10 + o3, 312);
  Permute(ThreeBodyPerm::kACB, 1, 2, 3, o1, o2, o3);
  EXPECT_EQ(o1 * 100 + o2 * 10 + o3, 132);
  Permute(ThreeBodyPerm::kBAC, 1, 2, 3, o1, o2, o3);
  EXPECT_EQ(o1 * 100 + o2 * 10 + o3, 213);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kABC), 1);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kBCA), 1);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kCAB), 1);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kACB), -1);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kBAC), -1);
  EXPECT_EQ(PermutationPhase(ThreeBodyPerm::kCBA), -1);
  EXPECT_EQ(Phase(0), 1);
  EXPECT_EQ(Phase(3), -1);
  EXPECT_EQ(Phase(-3), -1);
  EXPECT_EQ(Phase(-4), 1);
  EXPECT_TRUE(Triangle2(1, 1, 2));
  EXPECT_FALSE(Triangle2(1, 1, 4));
  EXPECT_FALSE(Triangle2(1, 1, 1));  // odd sum
  EXPECT_FALSE(Triangle2(5, 1, 2));
}

TEST(ThreeBodyRecoupling, SortOrbitsInvertsPermute) {
  for (int a_in = 0; a_in < 3; ++a_in) {
    for (int b_in = 0; b_in < 3; ++b_in) {
      for (int c_in = 0; c_in < 3; ++c_in) {
        int a = 0;
        int b = 0;
        int c = 0;
        const ThreeBodyPerm p = SortOrbits(a_in, b_in, c_in, a, b, c);
        EXPECT_LE(a, b);
        EXPECT_LE(b, c);
        int o1 = 0;
        int o2 = 0;
        int o3 = 0;
        Permute(p, a, b, c, o1, o2, o3);
        EXPECT_EQ(o1, a_in);
        EXPECT_EQ(o2, b_in);
        EXPECT_EQ(o3, c_in);
      }
    }
  }
}

TEST(ThreeBodyRecoupling, RecouplingMatricesAreOrthogonal) {
  const SixJ3Table t(2);
  const SixJ3View v = t.host_view();
  const std::array<int, 3> js = {1, 3, 5};
  int checked = 0;
  for (const int a2 : js) {
    for (const int b2 : js) {
      for (const int c2 : js) {
        for (int J2 = 1; J2 <= 15; J2 += 2) {
          // Columns: sorted ket (ab) Jab, c; rows: the permuted ket
          // (a_in b_in) Jab_in, c_in with (a_in, b_in, c_in) = Permute(p, ...).
          std::vector<int> cols;
          for (int Jab = 0; Jab <= 5; ++Jab) {
            if (Triangle2(a2, b2, 2 * Jab) && Triangle2(c2, J2, 2 * Jab)) {
              cols.push_back(Jab);
            }
          }
          for (const ThreeBodyPerm p : kAllPerms) {
            int ai = 0;
            int bi = 0;
            int ci = 0;
            Permute(p, a2, b2, c2, ai, bi, ci);
            std::vector<int> rows;
            for (int Jin = 0; Jin <= 5; ++Jin) {
              if (Triangle2(ai, bi, 2 * Jin) && Triangle2(ci, J2, 2 * Jin)) {
                rows.push_back(Jin);
              }
            }
            ASSERT_EQ(rows.size(), cols.size());
            const std::size_t n = rows.size();
            std::vector<double> m(n * n);
            for (std::size_t r = 0; r < n; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                m[r * n + c] =
                    PermutationPhase(p) *
                    RecouplingCoefficient(p, TwoJ{a2}, TwoJ{b2}, TwoJ{c2},
                                          rows[r], cols[c], TwoJ{J2}, v);
              }
            }
            for (std::size_t r = 0; r < n; ++r) {
              for (std::size_t s = 0; s < n; ++s) {
                double dot = 0.0;
                for (std::size_t c = 0; c < n; ++c) {
                  dot += m[r * n + c] * m[s * n + c];
                }
                ASSERT_NEAR(dot, r == s ? 1.0 : 0.0, 1.0e-13)
                    << "p=" << static_cast<int>(p) << " j=" << a2 << b2 << c2
                    << " 2J=" << J2;
              }
            }
            ++checked;
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// --- The 6j trap ------------------------------------------------------------

TEST(SixJTrap, HostLookupThrowsBeyondCoverage) {
  const int emax = 3;
  const SixJTable t(emax);
  const int too_big = 2 * emax + 3;  // 2j = 9, a valid j but beyond the table
  EXPECT_THROW(t(TwoJ{too_big}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}),
               InvariantError);
  EXPECT_THROW(t(TwoJ{1}, TwoJ{too_big}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}),
               InvariantError);
  EXPECT_THROW(t(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{too_big}, TwoJ{1}, TwoJ{0}),
               InvariantError);
  EXPECT_THROW(t(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{too_big}, TwoJ{0}),
               InvariantError);
  // Integer J = 2 emax + 2 (TwoJ 4 emax + 4) in either J slot.
  const int J_big = 2 * (2 * emax + 2);
  EXPECT_THROW(t(TwoJ{1}, TwoJ{1}, TwoJ{J_big}, TwoJ{1}, TwoJ{1}, TwoJ{0}),
               InvariantError);
  EXPECT_THROW(t(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{J_big}),
               InvariantError);
  // Three-body total J (half-integer 15/2) in a j slot.
  EXPECT_THROW(t(TwoJ{15}, TwoJ{1}, TwoJ{4}, TwoJ{1}, TwoJ{1}, TwoJ{4}),
               InvariantError);
  // In range: unchanged. Wrong parity or negative: still 0.0, no throw.
  EXPECT_EQ(t(TwoJ{7}, TwoJ{7}, TwoJ{14}, TwoJ{7}, TwoJ{7}, TwoJ{14}),
            Wigner6j(TwoJ{7}, TwoJ{7}, TwoJ{14}, TwoJ{7}, TwoJ{7}, TwoJ{14}));
  EXPECT_NO_THROW(
      t(TwoJ{10}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}));  // even 2j
  EXPECT_NO_THROW(
      t(TwoJ{1}, TwoJ{1}, TwoJ{21}, TwoJ{1}, TwoJ{1}, TwoJ{0}));  // odd 2J
  EXPECT_NO_THROW(
      t(TwoJ{-1}, TwoJ{1}, TwoJ{-2}, TwoJ{1}, TwoJ{1}, TwoJ{0}));  // negative
}

TEST(SixJTrap, ViewReturnsNanBeyondCoverage) {
  const int emax = 3;
  const SixJTable t(emax);
  const SixJView v = t.host_view();
  const int too_big = 2 * emax + 3;
  const int J_big = 2 * (2 * emax + 2);
  EXPECT_TRUE(std::isnan(
      v(TwoJ{too_big}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0})));
  EXPECT_TRUE(std::isnan(
      v(TwoJ{1}, TwoJ{too_big}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0})));
  EXPECT_TRUE(std::isnan(
      v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{too_big}, TwoJ{1}, TwoJ{0})));
  EXPECT_TRUE(std::isnan(
      v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{too_big}, TwoJ{0})));
  EXPECT_TRUE(
      std::isnan(v(TwoJ{1}, TwoJ{1}, TwoJ{J_big}, TwoJ{1}, TwoJ{1}, TwoJ{0})));
  EXPECT_TRUE(
      std::isnan(v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{J_big})));
  EXPECT_TRUE(
      std::isnan(v(TwoJ{15}, TwoJ{1}, TwoJ{4}, TwoJ{1}, TwoJ{1}, TwoJ{4})));
  // Wrong parity, negative arguments, failed triangles: 0.0.
  EXPECT_EQ(v(TwoJ{10}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(v(TwoJ{1}, TwoJ{1}, TwoJ{21}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(v(TwoJ{-1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  EXPECT_EQ(v(TwoJ{1}, TwoJ{1}, TwoJ{4}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  // In range: unchanged.
  EXPECT_EQ(v(TwoJ{7}, TwoJ{7}, TwoJ{14}, TwoJ{7}, TwoJ{7}, TwoJ{14}),
            Wigner6j(TwoJ{7}, TwoJ{7}, TwoJ{14}, TwoJ{7}, TwoJ{7}, TwoJ{14}));
  EXPECT_NEAR(v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), -0.5,
              1.0e-15);
}

TEST(SixJTrap, DeviceViewReturnsNanBeyondCoverage) {
  const int emax = 3;
  const SixJTable t(emax);
  const std::vector<std::int32_t> q = {
      9,  1, 0,  1, 1, 0,   // 2j too large
      1,  1, 16, 1, 1, 0,   // J = 8 > 2 emax + 1 = 7
      15, 1, 4,  1, 1, 4,   // three-body J in a j slot
      7,  7, 14, 7, 7, 14,  // in range
      2,  1, 0,  1, 1, 0,   // even 2j
  };
  exec::DeviceBuffer<std::int32_t> qb(q.size());
  qb.CopyFromHost(q);
  exec::DeviceBuffer<double> out(5);
  test::EvalSixJDevice(t, qb, out);
  const auto got = out.ToVector();
  EXPECT_TRUE(std::isnan(got[0]));
  EXPECT_TRUE(std::isnan(got[1]));
  EXPECT_TRUE(std::isnan(got[2]));
  EXPECT_EQ(got[3],
            Wigner6j(TwoJ{7}, TwoJ{7}, TwoJ{14}, TwoJ{7}, TwoJ{7}, TwoJ{14}));
  EXPECT_EQ(got[4], 0.0);
}

}  // namespace
}  // namespace imsrg::angmom
