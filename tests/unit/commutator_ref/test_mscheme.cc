// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Acceptance tests of the m-scheme commutator reference (mscheme.h): J -> m ->
// J round trip, the generic Wick engine against the validated J-scheme IMSRG(2)
// reference and against Comm223, and algebraic sanity checks of the rank-3
// parts.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator_ref/commutator_ref.h"
#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/three_body_test_support.h"

namespace {

using imsrg::Hermiticity;
using imsrg::basis::ModelSpace;
using imsrg::basis::ThreeBodyCuts;
using imsrg::op::Operator;
using imsrg::op::OperatorSpec;
using imsrg::testing::AddThreeBodySpace;
using imsrg::testing::FractionalModelSpace;
using imsrg::testing::MakeThreeBodySpace;
using imsrg::testing::RandomOperator;
using imsrg::testing::RandomThreeBodyOperator;
using imsrg::testing::RelativeDeviation;
namespace cr = imsrg::commutator_ref;
namespace msc = imsrg::commutator_ref::mscheme;

using MS = std::shared_ptr<const ModelSpace>;

struct Case {
  std::string name;
  MS ms;
};

std::vector<Case> Cases() {
  return {
      {"He4", MakeThreeBodySpace(1, "He4", ThreeBodyCuts{})},
      {"C12", MakeThreeBodySpace(1, "C12", ThreeBodyCuts{})},
      {"He4_frac",
       AddThreeBodySpace(FractionalModelSpace(1, "He4"), ThreeBodyCuts{})},
  };
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

// Spec of [X, Y] with particle rank `rank` (the hermiticity rule of
// commutator_ref).
OperatorSpec ResultSpec(const Operator& x, const Operator& y, int rank) {
  OperatorSpec s = cr::CommutatorSpec(x.spec(), y.spec());
  s.particle_rank = rank;
  return s;
}

// The engine parts [nA, nB]->nC summed over nA, nB in 1..2, as an operator.
Operator EngineParts(const msc::MOperator& a, const msc::MOperator& b,
                     const msc::MBasis& mb, const MS& ms,
                     const OperatorSpec& spec, int n_c) {
  msc::MOperator z(mb.size());
  for (int n_a = 1; n_a <= 2; ++n_a) {
    for (int n_b = 1; n_b <= 2; ++n_b) {
      z.Axpy(1.0, msc::CommutatorPart(a, n_a, b, n_b, n_c, mb));
    }
  }
  return msc::Contract(z, ms, spec);
}

TEST(MSchemeTest, MOperatorLabelsInAnyOrder) {
  msc::MOperator a(6);
  const int o[3] = {4, 1, 3};
  const int q[3] = {0, 5, 2};
  a.Set(3, o, q, 2.5);
  const int os[3] = {1, 3, 4};  // (4 1 3) -> (1 3 4) is an even permutation
  const int qs[3] = {0, 2, 5};  // (0 5 2) -> (0 2 5) is odd
  EXPECT_DOUBLE_EQ(a.Get(3, os, qs), -2.5);
  EXPECT_DOUBLE_EQ(a.Get(3, o, q), 2.5);
  const int rep[3] = {1, 1, 3};
  EXPECT_DOUBLE_EQ(a.Get(3, rep, qs), 0.0);
  a.Add(3, os, qs, 1.0);
  EXPECT_DOUBLE_EQ(a.Get(3, o, q), 1.5);
}

TEST(MSchemeTest, RoundTripOfRank3Operators) {
  for (const Case& c : Cases()) {
    const imsrg::commutator::ThreeBodyContext ctx(c.ms);
    const msc::MBasis mb(*c.ms);
    EXPECT_EQ(mb.size(), 16);
    for (const Hermiticity h :
         {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
          Hermiticity::kNonHermitian}) {
      const Operator x = RandomThreeBodyOperator(ctx, h, 5);
      ASSERT_TRUE(x.three_body().has_value());
      const auto t0 = std::chrono::steady_clock::now();
      const msc::MOperator a = msc::Expand(x, mb);
      const Operator y = msc::Contract(a, c.ms, x.spec());
      const double dev = RelativeDeviation(y, x);
      std::printf("[mscheme] round trip %s h=%d dev %.2e (%.2f s)\n",
                  c.name.c_str(), static_cast<int>(h), dev, SecondsSince(t0));
      EXPECT_LE(dev, 1.0e-12) << c.name;
      EXPECT_GT(a.MaxAbs(3), 0.0);
    }
  }
}

TEST(MSchemeTest, EngineMatchesJSchemeImsrg2) {
  for (const Case& c : Cases()) {
    const msc::MBasis mb(*c.ms);
    for (const Hermiticity hx :
         {Hermiticity::kAntiHermitian, Hermiticity::kNonHermitian}) {
      const Operator x = RandomOperator(c.ms, hx, 31);
      const Operator y = RandomOperator(c.ms, Hermiticity::kHermitian, 32);
      const msc::MOperator a = msc::Expand(x, mb);
      const msc::MOperator b = msc::Expand(y, mb);
      const auto t0 = std::chrono::steady_clock::now();
      const OperatorSpec spec = ResultSpec(x, y, 2);
      {
        Operator ref(c.ms, spec);
        cr::Comm110ss(x, y, ref);
        cr::Comm220ss(x, y, ref);
        const double dev =
            RelativeDeviation(EngineParts(a, b, mb, c.ms, spec, 0), ref);
        std::printf("[mscheme] %s h=%d ->0 dev %.2e\n", c.name.c_str(),
                    static_cast<int>(hx), dev);
        EXPECT_LE(dev, 1.0e-10) << c.name << " rank 0";
      }
      {
        Operator ref(c.ms, spec);
        cr::Comm111ss(x, y, ref);
        cr::Comm121ss(x, y, ref);
        cr::Comm221ss(x, y, ref);
        const double dev =
            RelativeDeviation(EngineParts(a, b, mb, c.ms, spec, 1), ref);
        std::printf("[mscheme] %s h=%d ->1 dev %.2e\n", c.name.c_str(),
                    static_cast<int>(hx), dev);
        EXPECT_LE(dev, 1.0e-10) << c.name << " rank 1";
      }
      {
        Operator ref(c.ms, spec);
        cr::Comm122ss(x, y, ref);
        cr::Comm222PpHhss(x, y, ref);
        cr::Comm222Phss(x, y, ref);
        const double dev =
            RelativeDeviation(EngineParts(a, b, mb, c.ms, spec, 2), ref);
        std::printf("[mscheme] %s h=%d ->2 dev %.2e (%.2f s)\n", c.name.c_str(),
                    static_cast<int>(hx), dev, SecondsSince(t0));
        EXPECT_LE(dev, 1.0e-10) << c.name << " rank 2";
      }
    }
  }
}

TEST(MSchemeTest, Comm222To3MatchesComm223) {
  for (const Case& c : Cases()) {
    const imsrg::commutator::ThreeBodyContext ctx(c.ms);
    const msc::MBasis mb(*c.ms);
    const Operator x = RandomOperator(c.ms, Hermiticity::kAntiHermitian, 41);
    const Operator y = RandomOperator(c.ms, Hermiticity::kHermitian, 42);
    imsrg::op::ThreeBody z3 = imsrg::commutator::Comm223(x, y, ctx);
    const OperatorSpec spec = OperatorSpec::Scalar(3, z3.hermiticity());
    Operator ref(c.ms, spec);
    *ref.three_body() = std::move(z3);
    const auto t0 = std::chrono::steady_clock::now();
    const msc::MOperator a = msc::Expand(x, mb);
    const msc::MOperator b = msc::Expand(y, mb);
    const msc::MOperator z = msc::CommutatorPart(a, 2, b, 2, 3, mb);
    const Operator zj = msc::Contract(z, c.ms, spec);
    const double dev = RelativeDeviation(zj, ref);
    std::printf("[mscheme] %s [2,2]->3 dev %.2e (%.2f s)\n", c.name.c_str(),
                dev, SecondsSince(t0));
    EXPECT_LE(dev, 1.0e-10) << c.name;
    EXPECT_GT(z.MaxAbs(3), 0.0);
  }
}

// Largest |Z_{out;in} - Z_{in;out}| of the rank-`rank` part.
double Asymmetry(const msc::MOperator& z, int rank, int n) {
  double asym = 0.0;
  auto upd = [&](const int* o, const int* q) {
    const std::span<const int> so(o, static_cast<std::size_t>(rank));
    const std::span<const int> sq(q, static_cast<std::size_t>(rank));
    asym = std::max(asym, std::abs(z.Get(rank, so, sq) - z.Get(rank, sq, so)));
  };
  if (rank == 1) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const int o[1] = {i};
        const int q[1] = {j};
        upd(o, q);
      }
    }
  } else if (rank == 2) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          for (int l = k + 1; l < n; ++l) {
            const int o[2] = {i, j};
            const int q[2] = {k, l};
            upd(o, q);
          }
        }
      }
    }
  } else if (rank == 3) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        for (int k = j + 1; k < n; ++k) {
          for (int l = 0; l < n; ++l) {
            for (int m = l + 1; m < n; ++m) {
              for (int r = m + 1; r < n; ++r) {
                const int o[3] = {i, j, k};
                const int q[3] = {l, m, r};
                upd(o, q);
              }
            }
          }
        }
      }
    }
  }
  return asym;
}

// [Y, X] = -[X, Y], and for X Hermitian and Y anti-Hermitian [X, Y] is
// Hermitian: the sum of the parts [nA, nB]->nC and [nB, nA]->nC of the product
// is a symmetric m-scheme matrix. The second property is a nontrivial test of
// the signs of the Wick patterns of ranks that the J-scheme IMSRG(2) reference
// does not cover.
TEST(MSchemeTest, RankThreePartsAreAntisymmetricAndHermitian) {
  // C12: holes in the s and p3/2 shells, so that every part is nonzero. The
  // Hermitian operator comes first: an anti-Hermitian one-body operator of a
  // spherical basis with one orbit per (l, j, tz) vanishes at emax 1.
  const Case c = Cases()[1];
  const imsrg::commutator::ThreeBodyContext ctx(c.ms);
  const msc::MBasis mb(*c.ms);
  const Operator x = RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 7);
  const Operator y =
      RandomThreeBodyOperator(ctx, Hermiticity::kAntiHermitian, 8);
  const msc::MOperator a = msc::Expand(x, mb);
  const msc::MOperator b = msc::Expand(y, mb);
  struct Part {
    int n_a;
    int n_b;
    int n_c;
  };
  const std::vector<Part> parts = {{3, 3, 1}, {2, 3, 2}, {3, 3, 0},
                                   {2, 3, 1}, {1, 3, 3}, {2, 3, 3}};
  for (const Part& p : parts) {
    const auto t0 = std::chrono::steady_clock::now();
    const msc::MOperator xy =
        msc::CommutatorPart(a, p.n_a, b, p.n_b, p.n_c, mb);
    const msc::MOperator yx =
        msc::CommutatorPart(b, p.n_b, a, p.n_a, p.n_c, mb);
    msc::MOperator sum = xy;
    sum.Axpy(1.0, yx);
    const double scale = std::max(1.0, xy.MaxAbs(p.n_c));
    EXPECT_LE(sum.MaxAbs(p.n_c) / scale, 1.0e-12);
    msc::MOperator full = xy;
    if (p.n_a != p.n_b) {
      full.Axpy(1.0, msc::CommutatorPart(a, p.n_b, b, p.n_a, p.n_c, mb));
    }
    const double asym = p.n_c == 0 ? 0.0 : Asymmetry(full, p.n_c, mb.size());
    std::printf(
        "[mscheme] [%d,%d]->%d anti %.2e hermiticity %.2e (|Z| %.2e, "
        "%.2f s)\n",
        p.n_a, p.n_b, p.n_c, sum.MaxAbs(p.n_c), asym, xy.MaxAbs(p.n_c),
        SecondsSince(t0));
    EXPECT_GT(xy.MaxAbs(p.n_c), 0.0);
    EXPECT_LE(asym / scale, 1.0e-12);
  }
}

// A diagonal one-body operator with energies e_s: [A1, B]^(n) = (sum_out e -
// sum_in e) B^(n) for the rank-n part of any B, and no contraction survives.
// An analytic check of the engine independent of the J-scheme code.
TEST(MSchemeTest, DiagonalOneBodyOperatorGivesEnergyDifferences) {
  const Case c = Cases()[2];
  const imsrg::commutator::ThreeBodyContext ctx(c.ms);
  const msc::MBasis mb(*c.ms);
  const int n = mb.size();
  msc::MOperator a(n);
  std::vector<double> e(static_cast<std::size_t>(n));
  for (int s = 0; s < n; ++s) {
    e[static_cast<std::size_t>(s)] = 0.37 * s * s - 1.1 * s + 0.2;
    const int o[1] = {s};
    a.Set(1, o, o, e[static_cast<std::size_t>(s)]);
  }
  const Operator y = RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 9);
  const msc::MOperator b = msc::Expand(y, mb);
  for (int rank = 1; rank <= 3; ++rank) {
    const msc::MOperator z = msc::CommutatorPart(a, 1, b, rank, rank, mb);
    msc::MOperator expected(n);
    double worst = 0.0;
    auto check = [&](const int* o, const int* q) {
      double de = 0.0;
      for (int t = 0; t < rank; ++t) {
        de += e[static_cast<std::size_t>(o[t])] -
              e[static_cast<std::size_t>(q[t])];
      }
      const std::span<const int> so(o, static_cast<std::size_t>(rank));
      const std::span<const int> sq(q, static_cast<std::size_t>(rank));
      worst = std::max(
          worst, std::abs(z.Get(rank, so, sq) - de * b.Get(rank, so, sq)));
    };
    // Spot checks over sorted labels.
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        for (int k = j + 1; k < n; ++k) {
          for (int l = 0; l < n; ++l) {
            for (int m = l + 1; m < n; ++m) {
              if (rank == 1) {
                const int o[1] = {i};
                const int q[1] = {l};
                check(o, q);
              } else if (rank == 2) {
                const int o[2] = {i, j};
                const int q[2] = {l, m};
                check(o, q);
              } else {
                for (int r = m + 1; r < n; ++r) {
                  const int o[3] = {i, j, k};
                  const int q[3] = {l, m, r};
                  check(o, q);
                }
              }
            }
          }
        }
      }
    }
    std::printf("[mscheme] [1,%d]->%d energy-difference dev %.2e\n", rank, rank,
                worst);
    EXPECT_LE(worst, 1.0e-10 * std::max(1.0, b.MaxAbs(rank)));
  }
}

}  // namespace
