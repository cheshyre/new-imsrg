// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of PairThirdSpace (pair-third states and the maps R and E), ported
// from the benchmark prototype checks (tools/bench/imsrg3_layout/
// check_pair_third.cc): state counts, index conventions, R = E^T for triples
// of distinct orbits, orthogonality of the E blocks, the symmetry factors for
// coinciding orbits, the recoupling projector identities, a spot check against
// an independent Wigner6j recomputation, and the device round trip.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/angmom/three_body_recoupling.h"
#include "imsrg/angmom/wigner.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/operator/pair_third.h"
#include "tests/unit/operator/pair_third_kernels.h"

namespace imsrg::op {
namespace {

constexpr double kNoCut = 1.0e30;
constexpr double kTol = 1.0e-12;

/// A model space with its pair-third space.
struct PtFixture {
  std::shared_ptr<const basis::ModelSpace> ms;
  std::unique_ptr<angmom::SixJ3Table> sixj;
  std::unique_ptr<PairThirdSpace> pt;

  PtFixture(int emax, int e3max, double de3max) {
    basis::ThreeBodyCuts cuts;
    cuts.e3max = e3max;
    cuts.de3max = de3max;
    ms = basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, "O16",
                                      "O16")
             ->WithThreeBody(cuts);
    sixj = std::make_unique<angmom::SixJ3Table>(emax);
    pt = std::make_unique<PairThirdSpace>(*ms->three_body_space(), ms->basis(),
                                          *sixj);
  }
  const basis::ThreeBodySpace& space() const { return *ms->three_body_space(); }
};

/// Entry (row, col) of a host CSR matrix (0 if absent). O(row length).
double At(const CsrView& m, std::int64_t row, std::int32_t col) {
  for (std::int64_t k = m.ptr[row]; k < m.ptr[row + 1]; ++k) {
    if (m.col[k] == col) return m.val[k];
  }
  return 0.0;
}

/// Index data of one triple: channel-local ranges and the flat offsets.
struct TripleIndex {
  std::int32_t t = 0;
  std::int32_t row_begin = 0;  ///< Channel-local.
  std::int32_t row_end = 0;
  std::int32_t st_begin = 0;  ///< Channel-local.
  std::int32_t st_end = 0;
  std::int64_t flat_row0 = 0;  ///< Flat index of channel-local row 0.
  std::int64_t flat_st0 = 0;   ///< Flat index of channel-local state 0.
};

TripleIndex IndexOf(const PtFixture& s, std::int32_t t) {
  const PairThirdView pv = s.pt->host_view();
  const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
  TripleIndex x;
  x.t = t;
  x.row_begin = tr.row_begin.value();
  x.row_end = tr.row_end.value();
  x.st_begin = pv.tr_state_begin[t];
  x.st_end = pv.tr_state_end[t];
  x.flat_row0 = s.space().channel(tr.channel).row_offset;
  x.flat_st0 = pv.ch_state_offset[tr.channel.value()];
  return x;
}

/// Max |R(row r, state s) - scale * E(state s, row r)| over the triple's rows
/// and the channel-local states [s0, s1).
double MaxRMinusScaledET(const PtFixture& setup, const TripleIndex& x,
                         std::int32_t s0, std::int32_t s1, double scale) {
  const CsrView rm = setup.pt->r_map().host_view();
  const CsrView em = setup.pt->e_map().host_view();
  double dev = 0.0;
  for (std::int32_t r = x.row_begin; r < x.row_end; ++r) {
    for (std::int32_t s = s0; s < s1; ++s) {
      const double rv = At(rm, x.flat_row0 + r, s);
      const double ev = At(em, x.flat_st0 + s, r);
      dev = std::max(dev, std::abs(rv - scale * ev));
    }
  }
  return dev;
}

// ---------------------------------------------------------------------------
// Counts and index conventions.
// ---------------------------------------------------------------------------

struct Case {
  int emax;
  int e3max;
  double de3max;
};
constexpr Case kCases[] = {
    {2, 6, kNoCut},
    {3, 9, kNoCut},
    {3, 9, 3.0},
};

/// Expected state counts of the benchmark prototype (same enumeration).
void ExpectCounts(const Case& c, std::int64_t sum_n, std::int64_t sum_n2,
                  int max_n) {
  const PtFixture s(c.emax, c.e3max, c.de3max);
  EXPECT_EQ(s.pt->num_states(), sum_n);
  EXPECT_EQ(s.pt->sum_states2(), sum_n2);
  EXPECT_EQ(s.pt->max_channel_states(), max_n);
  EXPECT_EQ(s.pt->num_channels(), s.space().num_channels());
  EXPECT_EQ(s.pt->num_triples(), s.space().num_triples());
}

TEST(PairThirdCounts, O16Emax2E3max6) {
  ExpectCounts({2, 6, kNoCut}, 5574, 1090366, 311);
}
TEST(PairThirdCounts, O16Emax3E3max9) {
  ExpectCounts({3, 9, kNoCut}, 36564, 35782440, 1574);
}
TEST(PairThirdCounts, O16Emax3E3max7) {
  ExpectCounts({3, 7, kNoCut}, 25324, 18895336, 1304);
}
TEST(PairThirdCounts, O16Emax3E3max9De3max3) {
  ExpectCounts({3, 9, 3.0}, 16840, 8346800, 793);
}

TEST(PairThirdIndex, ChannelAndTripleRangesAreContiguous) {
  for (const Case& c : kCases) {
    const PtFixture s(c.emax, c.e3max, c.de3max);
    const basis::ThreeBodySpace& sp = s.space();
    const PairThirdSpace& pt = *s.pt;
    const PairThirdView pv = pt.host_view();
    std::int64_t flat = 0;
    for (std::int32_t ch = 0; ch < sp.num_channels(); ++ch) {
      const Channel3Index chi{ch};
      EXPECT_EQ(pt.state_offset(chi), flat);
      EXPECT_EQ(pv.ch_state_offset[ch], flat);
      EXPECT_EQ(pv.ch_num_states[ch], pt.num_states(chi));
      std::int32_t local = 0;
      const basis::KeptChannel3& kc = sp.channel(chi);
      for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
           ++t) {
        const basis::Triple3Index ti{t};
        EXPECT_EQ(pt.state_begin(ti), local);
        EXPECT_GT(pt.state_end(ti), pt.state_begin(ti)) << "triple " << t;
        local = pt.state_end(ti);
      }
      EXPECT_EQ(local, pt.num_states(chi));
      if (kc.dim == 0) {
        EXPECT_EQ(pt.num_states(chi), 0);
      }
      flat += local;
    }
    EXPECT_EQ(flat, pt.num_states());
    EXPECT_EQ(pt.r_map().num_rows(), sp.NumRows());
    EXPECT_EQ(pt.e_map().num_rows(), pt.num_states());
    EXPECT_GT(pt.StateBytes(), 0);
    EXPECT_EQ(pt.Bytes(),
              pt.StateBytes() + pt.r_map().Bytes() + pt.e_map().Bytes());
  }
}

TEST(PairThirdIndex, StatesAreWellFormed) {
  const PtFixture s(3, 9, 3.0);
  const PairThirdView pv = s.pt->host_view();
  const auto orbits = s.ms->basis().orbits();
  for (std::int32_t ch = 0; ch < pv.num_channels; ++ch) {
    const Channel3Index chi{ch};
    const int two_j = s.space().channel(chi).two_j.twice;
    for (std::int32_t k = 0; k < pv.ch_num_states[ch]; ++k) {
      const std::int64_t f = pv.ch_state_offset[ch] + k;
      const int p = pv.st_p[f];
      const int q = pv.st_q[f];
      const int r = pv.st_r[f];
      ASSERT_LE(p, q);
      const int jp = orbits[static_cast<std::size_t>(p)].j2.twice;
      const int jq = orbits[static_cast<std::size_t>(q)].j2.twice;
      const int jr = orbits[static_cast<std::size_t>(r)].j2.twice;
      const int tjpq = pv.st_two_jpq[f];
      EXPECT_EQ(tjpq % 2, 0);
      EXPECT_TRUE(tjpq >= std::abs(jp - jq) && tjpq <= jp + jq);
      EXPECT_TRUE(tjpq >= std::abs(jr - two_j) && tjpq <= jr + two_j);
      if (p == q) {
        EXPECT_EQ((tjpq / 2) % 2, 0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// R = E^T for distinct orbits, orthogonality.
// ---------------------------------------------------------------------------

void CheckDistinctTriples(const Case& c) {
  const PtFixture s(c.emax, c.e3max, c.de3max);
  const PairThirdView pv = s.pt->host_view();
  const CsrView em = s.pt->e_map().host_view();
  int num_checked = 0;
  double max_dev = 0.0;
  double max_orth = 0.0;
  for (std::int32_t t = 0; t < s.space().num_triples(); ++t) {
    const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
    if (tr.a == tr.b || tr.b == tr.c) continue;
    ++num_checked;
    const TripleIndex x = IndexOf(s, t);
    max_dev =
        std::max(max_dev, MaxRMinusScaledET(s, x, x.st_begin, x.st_end, 1.0));
    // Splittings are runs of equal (p, q, r); each E block is square and
    // orthogonal.
    std::int32_t s0 = x.st_begin;
    while (s0 < x.st_end) {
      std::int32_t s1 = s0;
      const std::int64_t f0 = x.flat_st0 + s0;
      while (s1 < x.st_end && pv.st_p[x.flat_st0 + s1] == pv.st_p[f0] &&
             pv.st_q[x.flat_st0 + s1] == pv.st_q[f0] &&
             pv.st_r[x.flat_st0 + s1] == pv.st_r[f0]) {
        ++s1;
      }
      EXPECT_EQ(s1 - s0, x.row_end - x.row_begin) << "triple " << t;
      for (std::int32_t i = s0; i < s1; ++i) {
        for (std::int32_t j = s0; j < s1; ++j) {
          double dot = 0.0;
          for (std::int32_t r = x.row_begin; r < x.row_end; ++r) {
            dot += At(em, x.flat_st0 + i, r) * At(em, x.flat_st0 + j, r);
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

TEST(PairThirdDistinct, Emax2) { CheckDistinctTriples({2, 6, kNoCut}); }
TEST(PairThirdDistinct, Emax3) { CheckDistinctTriples({3, 9, kNoCut}); }
TEST(PairThirdDistinct, Emax3De3max3) { CheckDistinctTriples({3, 9, 3.0}); }

/// Coinciding orbits: R = sqrt(2) E^T on the (x x) y states, 2 E^T on the
/// (a c) a states of a == b < c.
void CheckCoincident(const Case& c) {
  const PtFixture s(c.emax, c.e3max, c.de3max);
  const PairThirdView pv = s.pt->host_view();
  int num_pair_states = 0;
  int num_other_states = 0;
  double max_dev = 0.0;
  for (std::int32_t t = 0; t < s.space().num_triples(); ++t) {
    const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
    const bool aab = tr.a == tr.b && tr.b < tr.c;
    const bool abb = tr.a < tr.b && tr.b == tr.c;
    if (!aab && !abb) continue;
    const TripleIndex x = IndexOf(s, t);
    for (std::int32_t k = x.st_begin; k < x.st_end; ++k) {
      const std::int64_t f = x.flat_st0 + k;
      double scale = 0.0;
      if (pv.st_p[f] == pv.st_q[f]) {
        scale = constants::kSqrt2;  // (x x) y with x the repeated orbit.
        ++num_pair_states;
      } else if (aab) {
        scale = 2.0;  // (a c) a: two permutations reach the state.
        ++num_other_states;
      } else {
        continue;  // (a b) b of a < b == b: not constrained here.
      }
      max_dev = std::max(max_dev, MaxRMinusScaledET(s, x, k, k + 1, scale));
    }
  }
  EXPECT_GT(num_pair_states, 0);
  EXPECT_GT(num_other_states, 0);
  EXPECT_LT(max_dev, kTol);
}

TEST(PairThirdCoincident, Emax2) { CheckCoincident({2, 6, kNoCut}); }
TEST(PairThirdCoincident, Emax3) { CheckCoincident({3, 9, kNoCut}); }
TEST(PairThirdCoincident, Emax3De3max3) { CheckCoincident({3, 9, 3.0}); }

// A canonical row without R entries is a canonical ket that vanishes under
// antisymmetrization; it can only occur for triples with b == c.
TEST(PairThirdMaps, EmptyRRowsOnlyForNullKets) {
  for (const Case& c : kCases) {
    const PtFixture s(c.emax, c.e3max, c.de3max);
    const CsrView rm = s.pt->r_map().host_view();
    for (std::int64_t r = 0; r < s.space().NumRows(); ++r) {
      if (rm.ptr[r] < rm.ptr[r + 1]) continue;
      const basis::OrbitTriple& tr = s.space().triple(s.space().row_triple(r));
      EXPECT_EQ(tr.b, tr.c) << "row " << r;
    }
    // Columns are channel-local states of the row's own triple.
    for (std::int64_t r = 0; r < s.space().NumRows(); ++r) {
      const basis::Triple3Index ti = s.space().row_triple(r);
      for (std::int64_t k = rm.ptr[r]; k < rm.ptr[r + 1]; ++k) {
        EXPECT_GE(rm.col[k], s.pt->state_begin(ti));
        EXPECT_LT(rm.col[k], s.pt->state_end(ti));
        EXPECT_GE(std::abs(rm.val[k]), 1.0e-13);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Independent recomputation with angmom::Wigner6j.
// ---------------------------------------------------------------------------

int Ph(int n) { return n % 2 == 0 ? 1 : -1; }

/// The recoupling coefficient of the header comment of three_body_recoupling.h
/// (without the permutation phase), written out with Wigner6j. `p` is one of
/// "ABC", "BAC", "BCA", "CAB", "ACB", "CBA".
double RecoupRef(angmom::ThreeBodyPerm p, int a2, int b2, int c2, int jin,
                 int jab, int two_J) {
  using angmom::ThreeBodyPerm;
  if (p == ThreeBodyPerm::kABC) return jin == jab ? 1.0 : 0.0;
  if (p == ThreeBodyPerm::kBAC) {
    return jin == jab ? Ph((a2 + b2) / 2 - jab) : 0.0;
  }
  const double hh = std::sqrt((2.0 * jin + 1.0) * (2.0 * jab + 1.0));
  const auto sj = [&](int x1, int x2, int x3, int x4, int x5, int x6) {
    return angmom::Wigner6j(TwoJ{x1}, TwoJ{x2}, TwoJ{x3}, TwoJ{x4}, TwoJ{x5},
                            TwoJ{x6});
  };
  switch (p) {
    case ThreeBodyPerm::kBCA:
      return Ph((b2 + c2) / 2 + jin + 1) * hh *
             sj(a2, b2, 2 * jab, c2, two_J, 2 * jin);
    case ThreeBodyPerm::kCAB:
      return Ph((a2 + b2) / 2 - jab + 1) * hh *
             sj(b2, a2, 2 * jab, c2, two_J, 2 * jin);
    case ThreeBodyPerm::kACB:
      return Ph((b2 + c2) / 2 + jin - jab) * hh *
             sj(b2, a2, 2 * jab, c2, two_J, 2 * jin);
    default:  // kCBA
      return -hh * sj(a2, b2, 2 * jab, c2, two_J, 2 * jin);
  }
}

double PermSign(angmom::ThreeBodyPerm p) { return angmom::PermutationPhase(p); }

/// Spot-checks all R rows and E rows of every `stride`-th triple against the
/// formulas of the header written out independently: for every column or row
/// of the triple, the stored value equals the recomputed one (0 if absent).
void SpotCheck(const Case& c, int stride) {
  using angmom::ThreeBodyPerm;
  const PtFixture s(c.emax, c.e3max, c.de3max);
  const PairThirdView pv = s.pt->host_view();
  const CsrView rm = s.pt->r_map().host_view();
  const CsrView em = s.pt->e_map().host_view();
  const auto orbits = s.ms->basis().orbits();
  const auto two_j_of = [&](int o) {
    return orbits[static_cast<std::size_t>(o)].j2.twice;
  };
  double max_dev = 0.0;
  int num_entries = 0;
  for (std::int32_t t = 0; t < s.space().num_triples(); t += stride) {
    const TripleIndex x = IndexOf(s, t);
    const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
    const int a = tr.a.value();
    const int b = tr.b.value();
    const int cc = tr.c.value();
    const int two_J = s.space().channel(tr.channel).two_j.twice;
    const int a2 = two_j_of(a);
    const int b2 = two_j_of(b);
    const int c2 = two_j_of(cc);
    // R rows.
    for (std::int32_t lr = x.row_begin; lr < x.row_end; ++lr) {
      const int j1 = s.space().row_two_jab(x.flat_row0 + lr).twice / 2;
      std::map<std::int32_t, double> expect;
      for (const ThreeBodyPerm perm :
           {ThreeBodyPerm::kABC, ThreeBodyPerm::kCBA, ThreeBodyPerm::kACB}) {
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        angmom::Permute(perm, a, b, cc, i1, i2, i3);
        const int jo1 = two_j_of(i1);
        const int jo2 = two_j_of(i2);
        for (std::int32_t k = x.st_begin; k < x.st_end; ++k) {
          const std::int64_t f = x.flat_st0 + k;
          if (pv.st_p[f] != std::min(i1, i2) ||
              pv.st_q[f] != std::max(i1, i2) || pv.st_r[f] != i3) {
            continue;
          }
          const int j1p = pv.st_two_jpq[f] / 2;
          if (perm == ThreeBodyPerm::kABC && j1p != j1) continue;
          double v =
              PermSign(perm) * RecoupRef(perm, a2, b2, c2, j1p, j1, two_J);
          if (i1 > i2) v *= -Ph((jo1 + jo2) / 2 - j1p);
          if (i1 == i2) v *= constants::kSqrt2;
          expect[k] += v;
        }
      }
      std::map<std::int32_t, double> got;
      for (std::int64_t k = rm.ptr[x.flat_row0 + lr];
           k < rm.ptr[x.flat_row0 + lr + 1]; ++k) {
        got[rm.col[k]] = rm.val[k];
      }
      for (const auto& [col, v] : expect) {
        max_dev = std::max(max_dev, std::abs(v - got[col]));
        ++num_entries;
      }
      for (const auto& [col, v] : got) {
        max_dev = std::max(max_dev, std::abs(v - expect[col]));
      }
    }
    // E rows.
    for (std::int32_t k = x.st_begin; k < x.st_end; ++k) {
      const std::int64_t f = x.flat_st0 + k;
      int sa = 0;
      int sb = 0;
      int sc = 0;
      const ThreeBodyPerm perm =
          angmom::SortOrbits(pv.st_p[f], pv.st_q[f], pv.st_r[f], sa, sb, sc);
      EXPECT_EQ(sa, a);
      EXPECT_EQ(sb, b);
      EXPECT_EQ(sc, cc);
      for (std::int32_t lr = x.row_begin; lr < x.row_end; ++lr) {
        const int jab = s.space().row_two_jab(x.flat_row0 + lr).twice / 2;
        const double v =
            PermSign(perm) *
            RecoupRef(perm, a2, b2, c2, pv.st_two_jpq[f] / 2, jab, two_J);
        const double got = At(em, f, lr);
        if (std::abs(v) >= 1.0e-13) ++num_entries;
        max_dev = std::max(max_dev, std::abs(v - got));
      }
    }
  }
  EXPECT_GT(num_entries, 100);
  EXPECT_LT(max_dev, kTol);
}

TEST(PairThirdSpot, Emax2AllTriples) { SpotCheck({2, 6, kNoCut}, 1); }
TEST(PairThirdSpot, Emax3EveryFifthTriple) { SpotCheck({3, 9, kNoCut}, 5); }

// ---------------------------------------------------------------------------
// Recoupling-block identities per triple (on the kept rows).
// ---------------------------------------------------------------------------

using Matrix = std::vector<std::vector<double>>;

Matrix BlockOf(const PtFixture& s, const TripleIndex& x,
               angmom::ThreeBodyPerm p) {
  const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{x.t});
  const auto orbits = s.ms->basis().orbits();
  const int a2 = orbits[tr.a.ToSize()].j2.twice;
  const int b2 = orbits[tr.b.ToSize()].j2.twice;
  const int c2 = orbits[tr.c.ToSize()].j2.twice;
  const int two_J = s.space().channel(tr.channel).two_j.twice;
  const std::int32_t n = x.row_end - x.row_begin;
  Matrix m(static_cast<std::size_t>(n),
           std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (std::int32_t i = 0; i < n; ++i) {
    for (std::int32_t j = 0; j < n; ++j) {
      const int jin =
          s.space().row_two_jab(x.flat_row0 + x.row_begin + i).twice / 2;
      const int jab =
          s.space().row_two_jab(x.flat_row0 + x.row_begin + j).twice / 2;
      m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          PermSign(p) * RecoupRef(p, a2, b2, c2, jin, jab, two_J);
    }
  }
  return m;
}

Matrix Multiply(const Matrix& x, const Matrix& y) {
  const std::size_t n = x.size();
  Matrix z(n, std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) z[i][j] += x[i][k] * y[k][j];
    }
  }
  return z;
}

// For b == c, the ACB recoupling maps |(a b) Jab, b; J> to |(a b) Jab', b; J>
// and applying it twice is the identity.
TEST(PairThirdBlocks, AcbIsAnInvolutionForCoincidingBc) {
  const PtFixture s(3, 9, kNoCut);
  int num_checked = 0;
  double max_dev = 0.0;
  for (std::int32_t t = 0; t < s.space().num_triples(); ++t) {
    const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
    if (!(tr.a < tr.b && tr.b == tr.c)) continue;
    const TripleIndex x = IndexOf(s, t);
    const Matrix m = BlockOf(s, x, angmom::ThreeBodyPerm::kACB);
    const Matrix m2 = Multiply(m, m);
    for (std::size_t i = 0; i < m2.size(); ++i) {
      for (std::size_t j = 0; j < m2.size(); ++j) {
        max_dev = std::max(max_dev, std::abs(m2[i][j] - (i == j ? 1.0 : 0.0)));
      }
    }
    ++num_checked;
  }
  EXPECT_GT(num_checked, 0);
  EXPECT_LT(max_dev, kTol);
}

// For a == b == c, (1 + M_CBA + M_ACB) / 3 is a symmetric projector.
TEST(PairThirdBlocks, AntisymmetrizerIsAProjectorForCoincidingAbc) {
  const PtFixture s(3, 9, kNoCut);
  int num_checked = 0;
  double max_idem = 0.0;
  double max_sym = 0.0;
  for (std::int32_t t = 0; t < s.space().num_triples(); ++t) {
    const basis::OrbitTriple& tr = s.space().triple(basis::Triple3Index{t});
    if (!(tr.a == tr.b && tr.b == tr.c)) continue;
    const TripleIndex x = IndexOf(s, t);
    const Matrix cba = BlockOf(s, x, angmom::ThreeBodyPerm::kCBA);
    const Matrix acb = BlockOf(s, x, angmom::ThreeBodyPerm::kACB);
    const std::size_t n = cba.size();
    Matrix p(n, std::vector<double>(static_cast<std::size_t>(n), 0.0));
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        p[i][j] = ((i == j ? 1.0 : 0.0) + cba[i][j] + acb[i][j]) / 3.0;
      }
    }
    const Matrix p2 = Multiply(p, p);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        max_idem = std::max(max_idem, std::abs(p2[i][j] - p[i][j]));
        max_sym = std::max(max_sym, std::abs(p[i][j] - p[j][i]));
      }
    }
    ++num_checked;
  }
  EXPECT_GT(num_checked, 0);
  EXPECT_LT(max_idem, kTol);
  EXPECT_LT(max_sym, kTol);
}

// ---------------------------------------------------------------------------
// Device round trip.
// ---------------------------------------------------------------------------

template <class T>
void ExpectDeviceCopyEquals(const T* device_ptr, const T* host_ptr,
                            std::int64_t n) {
  exec::DeviceBuffer<T> out(static_cast<std::size_t>(n));
  test::CopyOnDevice(device_ptr, out);
  const std::vector<T> got = out.ToVector();
  ASSERT_EQ(static_cast<std::int64_t>(got.size()), n);
  for (std::int64_t i = 0; i < n; ++i) {
    ASSERT_EQ(got[static_cast<std::size_t>(i)], host_ptr[i]) << "index " << i;
  }
}

TEST(PairThirdDevice, DeviceViewEqualsHostViewAfterRoundTrip) {
  const PtFixture s(2, 6, 3.0);
  const PairThirdSpace& pt = *s.pt;
  const PairThirdView h = pt.host_view();
  const PairThirdView d = pt.device_view();
  EXPECT_EQ(d.num_channels, h.num_channels);
  EXPECT_EQ(d.num_triples, h.num_triples);
  EXPECT_EQ(d.num_states, h.num_states);
  const std::int64_t nch = h.num_channels;
  const std::int64_t ntr = h.num_triples;
  const std::int64_t nst = h.num_states;
  ExpectDeviceCopyEquals(d.ch_state_offset, h.ch_state_offset, nch);
  ExpectDeviceCopyEquals(d.ch_num_states, h.ch_num_states, nch);
  ExpectDeviceCopyEquals(d.tr_state_begin, h.tr_state_begin, ntr);
  ExpectDeviceCopyEquals(d.tr_state_end, h.tr_state_end, ntr);
  ExpectDeviceCopyEquals(d.st_p, h.st_p, nst);
  ExpectDeviceCopyEquals(d.st_q, h.st_q, nst);
  ExpectDeviceCopyEquals(d.st_r, h.st_r, nst);
  ExpectDeviceCopyEquals(d.st_two_jpq, h.st_two_jpq, nst);
  const std::int64_t nrows = s.space().NumRows();
  ExpectDeviceCopyEquals(d.r.ptr, h.r.ptr, nrows + 1);
  ExpectDeviceCopyEquals(d.r.col, h.r.col, pt.r_map().nonzeros());
  ExpectDeviceCopyEquals(d.r.val, h.r.val, pt.r_map().nonzeros());
  ExpectDeviceCopyEquals(d.e.ptr, h.e.ptr, nst + 1);
  ExpectDeviceCopyEquals(d.e.col, h.e.col, pt.e_map().nonzeros());
  ExpectDeviceCopyEquals(d.e.val, h.e.val, pt.e_map().nonzeros());
}

TEST(PairThirdSizes, ReportO16Emax3) {
  const PtFixture s(3, 9, kNoCut);
  const auto mb = [](std::int64_t b) { return static_cast<double>(b) / 1.0e6; };
  RecordProperty("states_emax3_e3max9", std::to_string(s.pt->num_states()));
  RecordProperty("nnz_R_emax3_e3max9",
                 std::to_string(s.pt->r_map().nonzeros()));
  RecordProperty("nnz_E_emax3_e3max9",
                 std::to_string(s.pt->e_map().nonzeros()));
  std::cout << "PairThird O16 emax 3 E3max 9: states " << s.pt->num_states()
            << " (" << mb(s.pt->StateBytes()) << " MB), nnz(R) "
            << s.pt->r_map().nonzeros() << " (" << mb(s.pt->r_map().Bytes())
            << " MB), nnz(E) " << s.pt->e_map().nonzeros() << " ("
            << mb(s.pt->e_map().Bytes()) << " MB)" << std::endl;
  EXPECT_GT(s.pt->e_map().nonzeros(), 0);
  EXPECT_GT(s.pt->r_map().nonzeros(), 0);
}

}  // namespace
}  // namespace imsrg::op
