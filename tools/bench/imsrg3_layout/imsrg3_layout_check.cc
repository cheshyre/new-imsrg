// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark prototypes (design/14 section 2):
// the 3B 6j table, the recoupling ports and the canonical space. The other
// check_*.cc files hold the pair-third, layout and kernel checks. Label `slow`
// (not run by tools/ci.sh).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.h"
#include "imsrg/basis/model_space.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/recoupling.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

std::shared_ptr<const basis::ModelSpace> Space3(int emax, int e3max) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, "O16",
                                      "O16")
      ->WithThreeBody(basis::ThreeBodyCuts{e3max});
}

TEST(Recoupling, SortOrbitsMatchesPermute) {
  // Every ordering of three distinct and of coinciding orbits sorts back.
  const int sets[][3] = {{1, 2, 3}, {3, 1, 2}, {2, 2, 5}, {5, 2, 2}, {4, 4, 4}};
  for (const auto& s : sets) {
    int a = 0;
    int b = 0;
    int c = 0;
    const Perm p = SortOrbits(s[0], s[1], s[2], a, b, c);
    EXPECT_LE(a, b);
    EXPECT_LE(b, c);
    int o1 = 0;
    int o2 = 0;
    int o3 = 0;
    Permute(p, a, b, c, o1, o2, o3);
    EXPECT_EQ(o1, s[0]);
    EXPECT_EQ(o2, s[1]);
    EXPECT_EQ(o3, s[2]);
  }
}

// The table equals angmom::Wigner6j on every argument it covers (emax 3), and
// reads 0 outside its range.
TEST(SixJ3, MatchesWigner6jEverywhere) {
  const int emax = 3;
  const SixJ3Table table(emax);
  const SixJ3View v = table.host_view();
  std::int64_t nonzero = 0;
  for (int j1 = 1; j1 <= 2 * emax + 1; j1 += 2) {
    for (int j2 = 1; j2 <= 2 * emax + 1; j2 += 2) {
      for (int J12 = 0; J12 <= 2 * emax + 1; ++J12) {
        for (int j3 = 1; j3 <= 2 * emax + 1; j3 += 2) {
          for (int J = 1; J <= 6 * emax + 3; J += 2) {
            for (int J23 = 0; J23 <= 2 * emax + 1; ++J23) {
              const double ref =
                  angmom::Wigner6j(TwoJ{j1}, TwoJ{j2}, TwoJ{2 * J12}, TwoJ{j3},
                                   TwoJ{J}, TwoJ{2 * J23});
              const double got = v(TwoJ{j1}, TwoJ{j2}, TwoJ{2 * J12}, TwoJ{j3},
                                   TwoJ{J}, TwoJ{2 * J23});
              ASSERT_EQ(got, ref) << j1 << " " << j2 << " " << J12 << " " << j3
                                  << " " << J << " " << J23;
              if (got != 0.0) ++nonzero;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(nonzero, 1000);
  // Outside the range: J beyond 6 emax + 3, even 2J, even 2j.
  EXPECT_EQ(v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{6 * emax + 5}, TwoJ{0}),
            0.0);
  EXPECT_EQ(v(TwoJ{1}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{2}, TwoJ{0}), 0.0);
  EXPECT_EQ(v(TwoJ{2}, TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{0}), 0.0);
  for (int J = 0; J <= 2 * emax + 1; ++J) {
    EXPECT_DOUBLE_EQ(v.Hat(J), std::sqrt(2.0 * J + 1.0));
  }
}

// Integer J range of a pair (doubled j1, j2) that also couples with doubled
// j3 to doubled J3.
void PairRange(int j1, int j2, int j3, int J3, int& lo, int& hi) {
  lo = std::max(std::abs(j1 - j2), std::abs(j3 - J3)) / 2;
  hi = std::min(j1 + j2, j3 + J3) / 2;
}

// M_p[J_in, J] = PermutationPhase(p) RecouplingCoefficient(p, ...) over the
// given J lists (row: J_in, column: J), row-major.
std::vector<double> RecouplingMatrix(Perm p, int ja, int jb, int jc, int J3,
                                     const std::vector<int>& j_in,
                                     const std::vector<int>& j_out,
                                     const SixJ3View& v) {
  std::vector<double> m(j_in.size() * j_out.size());
  for (std::size_t r = 0; r < j_in.size(); ++r) {
    for (std::size_t c = 0; c < j_out.size(); ++c) {
      m[r * j_out.size() + c] =
          PermutationPhase(p) * RecouplingCoefficient(p, TwoJ{ja}, TwoJ{jb},
                                                      TwoJ{jc}, j_in[r],
                                                      j_out[c], TwoJ{J3}, v);
    }
  }
  return m;
}

double MaxDevFromIdentity(const std::vector<double>& a,
                          const std::vector<double>& b, std::size_t n) {
  // max |(A B^T)_ij - delta_ij| for n x n row-major A, B.
  double dev = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += a[i * n + k] * b[j * n + k];
      dev = std::max(dev, std::abs(s - (i == j ? 1.0 : 0.0)));
    }
  }
  return dev;
}

// Distinct orbits: every recoupling matrix is orthogonal. b == c: M_ACB is a
// symmetric involution. a == b == c: Q = (1 + M_CBA + M_ACB) / 3 over even
// Jab is a symmetric projector. All j up to 7/2 and every total J.
TEST(Recoupling, OrthogonalityAndProjectors) {
  const int emax = 3;
  const SixJ3Table table(emax);
  const SixJ3View v = table.host_view();
  const int tmax = 2 * emax + 1;
  int checked = 0;
  for (int ja = 1; ja <= tmax; ja += 2) {
    for (int jb = 1; jb <= tmax; jb += 2) {
      for (int jc = 1; jc <= tmax; jc += 2) {
        for (int J3 = 1; J3 <= ja + jb + jc; J3 += 2) {
          int lo = 0;
          int hi = 0;
          PairRange(ja, jb, jc, J3, lo, hi);
          if (lo > hi) continue;
          std::vector<int> jout;
          for (int J = lo; J <= hi; ++J) jout.push_back(J);
          for (const Perm p :
               {Perm::kBCA, Perm::kCAB, Perm::kACB, Perm::kCBA}) {
            const int j[3] = {ja, jb, jc};
            int o1 = 0;
            int o2 = 0;
            int o3 = 0;
            Permute(p, 0, 1, 2, o1, o2, o3);
            int ilo = 0;
            int ihi = 0;
            PairRange(j[o1], j[o2], j[o3], J3, ilo, ihi);
            std::vector<int> jin;
            for (int J = ilo; J <= ihi; ++J) jin.push_back(J);
            ASSERT_EQ(jin.size(), jout.size());
            const auto m = RecouplingMatrix(p, ja, jb, jc, J3, jin, jout, v);
            EXPECT_LT(MaxDevFromIdentity(m, m, jout.size()), 1.0e-12)
                << "p=" << static_cast<int>(p) << " j=" << ja << "," << jb
                << "," << jc << " J=" << J3;
            ++checked;
          }
          if (jb == jc) {
            const auto m =
                RecouplingMatrix(Perm::kACB, ja, jb, jc, J3, jout, jout, v);
            const std::size_t n = jout.size();
            for (std::size_t r = 0; r < n; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                EXPECT_NEAR(m[r * n + c], m[c * n + r], 1.0e-12);
              }
            }
            EXPECT_LT(MaxDevFromIdentity(m, m, n), 1.0e-12);
          }
          if (ja == jb && jb == jc) {
            std::vector<int> even;
            for (int J : jout) {
              if (J % 2 == 0) even.push_back(J);
            }
            const std::size_t n = even.size();
            if (n == 0) continue;
            const auto m1 =
                RecouplingMatrix(Perm::kCBA, ja, jb, jc, J3, even, even, v);
            const auto m2 =
                RecouplingMatrix(Perm::kACB, ja, jb, jc, J3, even, even, v);
            std::vector<double> q(n * n);
            for (std::size_t i = 0; i < n * n; ++i) {
              q[i] = (m1[i] + m2[i]) / 3.0;
            }
            for (std::size_t i = 0; i < n; ++i) q[i * n + i] += 1.0 / 3.0;
            for (std::size_t r = 0; r < n; ++r) {
              for (std::size_t c = 0; c < n; ++c) {
                double qq = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                  qq += q[r * n + k] * q[k * n + c];
                }
                EXPECT_NEAR(qq, q[r * n + c], 1.0e-12);
                EXPECT_NEAR(q[r * n + c], q[c * n + r], 1.0e-12);
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 500);
}

struct CountCase {
  int emax;
  int e3max;
  double de3max;
  std::int64_t sum_n;
  std::int64_t sum_n2;
  std::int32_t max_n;
};

// Layout-A counts; the values agree with the planning pass, which reproduced
// imsrg++'s pn storage sizes.
TEST(CanonicalSpace, CountsMatchThePlan) {
  const CountCase cases[] = {{2, 6, 1.0e10, 2099, 146895, 113},
                             {3, 9, 1.0e10, 13223, 4509669, 554},
                             {3, 7, 1.0e10, 8909, 2285903, 449},
                             {3, 9, 3.0, 5832, 988640, 270}};
  for (const CountCase& k : cases) {
    const auto ms = Space3(k.emax, k.e3max);
    const CanonicalSpace space(ms, k.de3max, -1.0);
    const SpaceCounts s = space.Counts();
    EXPECT_EQ(s.sum_n, k.sum_n) << k.emax << " " << k.e3max << " " << k.de3max;
    EXPECT_EQ(s.sum_n2, k.sum_n2);
    EXPECT_EQ(s.max_n, k.max_n);
    EXPECT_EQ(space.NumRows(), s.sum_n);
    EXPECT_EQ(space.total_size(), s.sum_n2);
    if (k.de3max > 100.0) {
      EXPECT_EQ(s.sum_n, ms->three_body()->SumChannelKets());
    }
  }
}

// Structural invariants: rows of a triple are contiguous with the triple's
// orbits, channels are ordered, ImsrgppNorm of an all-ones buffer is
// sqrt(sum N (N + 1) / 2).
TEST(CanonicalSpace, TablesAreConsistent) {
  const auto ms = Space3(2, 6);
  const CanonicalSpace space(ms, 1.0e10, -1.0);
  const CanonicalView v = space.host_view();
  const basis::ThreeBodyChannels& tbc = *ms->three_body();
  std::int64_t rows = 0;
  for (std::int32_t c = 0; c < v.num_channels; ++c) {
    EXPECT_EQ(v.ch_row_offset[c], rows);
    rows += v.ch_dim[c];
    std::int32_t next_row = 0;
    for (std::int32_t t = v.ch_triple_begin[c]; t < v.ch_triple_end[c]; ++t) {
      EXPECT_EQ(v.tr_channel[t], c);
      EXPECT_EQ(v.tr_row_begin[t], next_row);
      EXPECT_LT(v.tr_row_begin[t], v.tr_row_end[t]);
      next_row = v.tr_row_end[t];
      EXPECT_LE(v.tr_a[t], v.tr_b[t]);
      EXPECT_LE(v.tr_b[t], v.tr_c[t]);
      for (std::int32_t r = v.tr_row_begin[t]; r < v.tr_row_end[t]; ++r) {
        const std::int64_t flat = v.ch_row_offset[c] + r;
        EXPECT_EQ(v.row_triple[flat], t);
        EXPECT_TRUE(tbc.find_ket(OrbitIndex{v.tr_a[t]}, OrbitIndex{v.tr_b[t]},
                                 OrbitIndex{v.tr_c[t]},
                                 TwoJ{v.row_two_jab[flat]})
                        .has_value());
      }
    }
    EXPECT_EQ(next_row, v.ch_dim[c]);
  }
  EXPECT_EQ(rows, space.NumRows());
  std::vector<double> ones(static_cast<std::size_t>(space.total_size()), 1.0);
  EXPECT_NEAR(ImsrgppNorm(space, ones),
              std::sqrt(static_cast<double>(space.Counts().sum_packed)),
              1.0e-9);
}

}  // namespace
}  // namespace imsrg::layout3
