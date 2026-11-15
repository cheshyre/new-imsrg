// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Algebraic property tests of the reference commutator on random operators
// (design/30 section 2): antisymmetry, linearity, element-wise hermiticity, the
// Jacobi identity for one-body operators and consistency for non-Hermitian
// operands.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator_ref/commutator_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"

namespace {

using imsrg::Hermiticity;
using imsrg::op::Operator;
using imsrg::testing::FractionalModelSpace;
using imsrg::testing::MakeModelSpace;
using imsrg::testing::RandomOperator;
using imsrg::testing::RelativeDeviation;
namespace cr = imsrg::commutator_ref;

constexpr double kTol = 1.0e-12;

using MS = std::shared_ptr<const imsrg::basis::ModelSpace>;

struct Case {
  std::string name;
  MS ms;
  std::vector<std::uint64_t> seeds;
};

std::vector<Case> Cases() {
  return {
      {"He4_emax1", MakeModelSpace(1, "He4"), {11, 12}},
      {"He4_frac_emax1", FractionalModelSpace(1, "He4"), {21, 22}},
      {"O16_emax2", MakeModelSpace(2, "O16"), {31}},
      {"O18_emax2", MakeModelSpace(2, "O18"), {41}},
      {"O16_frac_emax2", FractionalModelSpace(2, "O16"), {51}},
  };
}

using TermFn = void (*)(const Operator&, const Operator&, Operator&);
struct Term {
  const char* name;
  TermFn fn;
};
const Term kTerms[] = {
    {"comm110ss", &cr::Comm110ss},           {"comm220ss", &cr::Comm220ss},
    {"comm111ss", &cr::Comm111ss},           {"comm121ss", &cr::Comm121ss},
    {"comm221ss", &cr::Comm221ss},           {"comm122ss", &cr::Comm122ss},
    {"comm222_pp_hhss", &cr::Comm222PpHhss}, {"comm222_phss", &cr::Comm222Phss},
};

Operator Apply(const Term& t, const Operator& X, const Operator& Y) {
  Operator Z(X.model_space_ptr(), cr::CommutatorSpec(X.spec(), Y.spec()));
  t.fn(X, Y, Z);
  return Z;
}

struct HPair {
  const char* name;
  Hermiticity hx;
  Hermiticity hy;
};
const HPair kHPairs[] = {
    {"HH", Hermiticity::kHermitian, Hermiticity::kHermitian},
    {"HA", Hermiticity::kHermitian, Hermiticity::kAntiHermitian},
    {"AH", Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
    {"AA", Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian}};

/// Largest |Z_ij - h Z_ji| / max(1, max|Z|) over the one-body and all two-body
/// blocks.
double HermiticityViolation(const Operator& Z, double h) {
  const auto& ms = Z.model_space();
  double maxabs = 1.0;
  double dev = 0.0;
  for (int i = 0; i < ms.num_orbits(); ++i) {
    for (const auto j : ms.basis().SameLjtz(imsrg::OrbitIndex{i})) {
      const double a = Z.one_body().get(imsrg::OrbitIndex{i}, j);
      const double b = Z.one_body().get(j, imsrg::OrbitIndex{i});
      maxabs = std::max(maxabs, std::abs(a));
      dev = std::max(dev, std::abs(a - h * b));
    }
  }
  if (Z.two_body().allocated()) {
    const auto& chans = ms.two_body();
    for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
      const imsrg::ChannelIndex ch{c};
      const int nk = chans.channel(ch).num_kets;
      for (int i = 0; i < nk; ++i) {
        for (int j = 0; j < nk; ++j) {
          const double a = Z.two_body().get(ch, ch, imsrg::LocalKetIndex{i},
                                            imsrg::LocalKetIndex{j});
          const double b = Z.two_body().get(ch, ch, imsrg::LocalKetIndex{j},
                                            imsrg::LocalKetIndex{i});
          maxabs = std::max(maxabs, std::abs(a));
          dev = std::max(dev, std::abs(a - h * b));
        }
      }
    }
  }
  return dev / maxabs;
}

double SignOf(Hermiticity h) {
  return h == Hermiticity::kHermitian
             ? 1.0
             : (h == Hermiticity::kAntiHermitian ? -1.0 : 0.0);
}

TEST(commutator_ref_properties, antisymmetry) {
  for (const auto& c : Cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X = RandomOperator(c.ms, hp.hx, seed);
        const Operator Y = RandomOperator(c.ms, hp.hy, seed + 1000);
        for (const auto& t : kTerms) {
          const Operator zxy = Apply(t, X, Y);
          const Operator zyx = Apply(t, Y, X);
          const double dev = RelativeDeviation(zxy, -zyx);
          std::printf("[prop] antisym %s %s seed %" PRIu64 " %s: %.3e\n",
                      c.name.c_str(), hp.name, static_cast<std::uint64_t>(seed),
                      t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
        const double dev =
            RelativeDeviation(cr::Commutator(X, Y), -cr::Commutator(Y, X));
        std::printf("[prop] antisym %s %s seed %" PRIu64 " full: %.3e\n",
                    c.name.c_str(), hp.name, static_cast<std::uint64_t>(seed),
                    dev);
        EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " full";
      }
    }
  }
}

TEST(commutator_ref_properties, linearity) {
  for (const auto& c : Cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X1 = RandomOperator(c.ms, hp.hx, seed);
        const Operator X2 = RandomOperator(c.ms, hp.hx, seed + 500);
        const Operator Y = RandomOperator(c.ms, hp.hy, seed + 1000);
        const Operator Xs = 0.7 * X1 - 1.3 * X2;
        for (const auto& t : kTerms) {
          const Operator lhs = Apply(t, Xs, Y);
          const Operator rhs = 0.7 * Apply(t, X1, Y) - 1.3 * Apply(t, X2, Y);
          const double dev = RelativeDeviation(lhs, rhs);
          std::printf("[prop] linear %s %s seed %" PRIu64 " %s: %.3e\n",
                      c.name.c_str(), hp.name, static_cast<std::uint64_t>(seed),
                      t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
      }
    }
  }
}

TEST(commutator_ref_properties, hermiticity) {
  for (const auto& c : Cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X = RandomOperator(c.ms, hp.hx, seed);
        const Operator Y = RandomOperator(c.ms, hp.hy, seed + 1000);
        // [H,H] and [A,A] are anti-Hermitian, [A,H] and [H,A] are Hermitian.
        const double h = (hp.hx == hp.hy) ? -1.0 : 1.0;
        for (const auto& t : kTerms) {
          const Operator Z = Apply(t, X, Y);
          if (std::string(t.name) == "comm110ss" ||
              std::string(t.name) == "comm220ss")
            continue;
          const double dev = HermiticityViolation(Z, h);
          std::printf("[prop] herm %s %s seed %" PRIu64 " %s: %.3e\n",
                      c.name.c_str(), hp.name, static_cast<std::uint64_t>(seed),
                      t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
        const Operator Z = cr::Commutator(X, Y);
        EXPECT_EQ(SignOf(Z.hermiticity()), h);
        EXPECT_LE(HermiticityViolation(Z, h), kTol)
            << c.name << " " << hp.name << " full";
      }
    }
  }
}

TEST(commutator_ref_properties, jacobi_one_body) {
  for (const auto& c : Cases()) {
    for (const auto seed : c.seeds) {
      const Operator X = RandomOperator(c.ms, Hermiticity::kHermitian, seed, 1);
      const Operator Y =
          RandomOperator(c.ms, Hermiticity::kHermitian, seed + 1, 1);
      const Operator W =
          RandomOperator(c.ms, Hermiticity::kHermitian, seed + 2, 1);
      Operator J = cr::Commutator(X, cr::Commutator(Y, W));
      J += cr::Commutator(Y, cr::Commutator(W, X));
      J += cr::Commutator(W, cr::Commutator(X, Y));
      Operator zero(c.ms, J.spec());
      const double dev = RelativeDeviation(J, zero);
      std::printf("[prop] jacobi %s seed %" PRIu64 ": %.3e\n", c.name.c_str(),
                  static_cast<std::uint64_t>(seed), dev);
      EXPECT_LE(dev, kTol) << c.name;
    }
  }
}

TEST(commutator_ref_properties, non_hermitian_consistency) {
  for (const auto& c : Cases()) {
    for (const auto seed : c.seeds) {
      const Operator XH = RandomOperator(c.ms, Hermiticity::kHermitian, seed);
      const Operator XA =
          RandomOperator(c.ms, Hermiticity::kAntiHermitian, seed + 500);
      Operator X = XH + XA;
      X.set_hermiticity(Hermiticity::kNonHermitian);
      for (const auto& hp :
           {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
        const Operator Y = RandomOperator(c.ms, hp, seed + 1000);
        for (const auto& t : kTerms) {
          Operator lhs = Apply(t, X, Y);
          Operator rhs = Apply(t, XH, Y);
          rhs.set_hermiticity(Hermiticity::kNonHermitian);
          Operator part = Apply(t, XA, Y);
          part.set_hermiticity(Hermiticity::kNonHermitian);
          rhs += part;
          const double dev = RelativeDeviation(lhs, rhs);
          std::printf("[prop] nonherm %s seed %" PRIu64 " %s: %.3e\n",
                      c.name.c_str(), static_cast<std::uint64_t>(seed), t.name,
                      dev);
          EXPECT_LE(dev, kTol) << c.name << " " << t.name;
        }
        EXPECT_EQ(cr::Commutator(X, Y).hermiticity(),
                  Hermiticity::kNonHermitian);
      }
    }
  }
}

// Validity of the reference for GENERAL X and Y (M7-Q2). Every term is
// bilinear in (X, Y), and the reference agrees with imsrg++'s goldens and with
// the symmetry-based fast kernels for every (anti-)Hermitian pair (the HH, HA,
// AH, AA pairs of the term tests), which span all operators (X = (X + X^+)/2 +
// (X - X^+)/2). So a general X and a general Y must give the sum of the four
// part commutators; this also covers general-by-general, which
// non_hermitian_consistency does not.
TEST(commutator_ref_properties, general_by_general_bilinearity) {
  for (const auto& c : Cases()) {
    const auto seed = c.seeds.front();
    const Operator XH = RandomOperator(c.ms, Hermiticity::kHermitian, seed + 1);
    const Operator XA =
        RandomOperator(c.ms, Hermiticity::kAntiHermitian, seed + 2);
    const Operator YH = RandomOperator(c.ms, Hermiticity::kHermitian, seed + 3);
    const Operator YA =
        RandomOperator(c.ms, Hermiticity::kAntiHermitian, seed + 4);
    Operator X = XH + XA;
    Operator Y = YH + YA;
    X.set_hermiticity(Hermiticity::kNonHermitian);
    Y.set_hermiticity(Hermiticity::kNonHermitian);
    for (const auto& t : kTerms) {
      Operator sum(c.ms, cr::CommutatorSpec(X.spec(), Y.spec()));
      for (const Operator* px : {&XH, &XA}) {
        for (const Operator* py : {&YH, &YA}) {
          Operator part = Apply(t, *px, *py);
          part.set_hermiticity(Hermiticity::kNonHermitian);
          sum += part;
        }
      }
      const double dev = RelativeDeviation(Apply(t, X, Y), sum);
      std::printf("[prop] general-by-general %s %s: %.3e\n", c.name.c_str(),
                  t.name, dev);
      EXPECT_LE(dev, kTol) << c.name << " " << t.name;
    }
  }
}

}  // namespace
