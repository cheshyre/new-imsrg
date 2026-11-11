// Algebraic property tests of the reference commutator on random operators (design/30 section 2):
// antisymmetry, linearity, element-wise hermiticity, the Jacobi identity for one-body operators
// and consistency for non-Hermitian operands.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/commutator_ref/commutator_ref.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/operator.hpp"

namespace {

using imsrg::Hermiticity;
using imsrg::op::Operator;
using imsrg::testing::fractional_model_space;
using imsrg::testing::model_space;
using imsrg::testing::random_operator;
using imsrg::testing::relative_deviation;
namespace cr = imsrg::commutator_ref;

constexpr double kTol = 1e-12;

using MS = std::shared_ptr<const imsrg::basis::ModelSpace>;

struct Case {
  std::string name;
  MS ms;
  std::vector<std::uint64_t> seeds;
};

std::vector<Case> cases() {
  return {
      {"He4_emax1", model_space(1, "He4"), {11, 12}},
      {"He4_frac_emax1", fractional_model_space(1, "He4"), {21, 22}},
      {"O16_emax2", model_space(2, "O16"), {31}},
      {"O18_emax2", model_space(2, "O18"), {41}},
      {"O16_frac_emax2", fractional_model_space(2, "O16"), {51}},
  };
}

using TermFn = void (*)(const Operator&, const Operator&, Operator&);
struct Term {
  const char* name;
  TermFn fn;
};
const Term kTerms[] = {
    {"comm110ss", &cr::comm110ss},
    {"comm220ss", &cr::comm220ss},
    {"comm111ss", &cr::comm111ss},
    {"comm121ss", &cr::comm121ss},
    {"comm221ss", &cr::comm221ss},
    {"comm122ss", &cr::comm122ss},
    {"comm222_pp_hhss", &cr::comm222_pp_hhss},
    {"comm222_phss", &cr::comm222_phss},
};

Operator apply(const Term& t, const Operator& X, const Operator& Y) {
  Operator Z(X.model_space_ptr(), cr::commutator_spec(X.spec(), Y.spec()));
  t.fn(X, Y, Z);
  return Z;
}

struct HPair {
  const char* name;
  Hermiticity hx;
  Hermiticity hy;
};
const HPair kHPairs[] = {{"HH", Hermiticity::Hermitian, Hermiticity::Hermitian},
                         {"HA", Hermiticity::Hermitian, Hermiticity::AntiHermitian},
                         {"AH", Hermiticity::AntiHermitian, Hermiticity::Hermitian},
                         {"AA", Hermiticity::AntiHermitian, Hermiticity::AntiHermitian}};

/// Largest |Z_ij - h Z_ji| / max(1, max|Z|) over the one-body and all two-body blocks.
double hermiticity_violation(const Operator& Z, double h) {
  const auto& ms = Z.model_space();
  double maxabs = 1.0;
  double dev = 0.0;
  for (int i = 0; i < ms.num_orbits(); ++i) {
    for (const auto j : ms.basis().same_ljtz(imsrg::OrbitIndex{i})) {
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
          const double a =
              Z.two_body().get(ch, ch, imsrg::LocalKetIndex{i}, imsrg::LocalKetIndex{j});
          const double b =
              Z.two_body().get(ch, ch, imsrg::LocalKetIndex{j}, imsrg::LocalKetIndex{i});
          maxabs = std::max(maxabs, std::abs(a));
          dev = std::max(dev, std::abs(a - h * b));
        }
      }
    }
  }
  return dev / maxabs;
}

double sign_of(Hermiticity h) {
  return h == Hermiticity::Hermitian ? 1.0 : (h == Hermiticity::AntiHermitian ? -1.0 : 0.0);
}

TEST(commutator_ref_properties, antisymmetry) {
  for (const auto& c : cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X = random_operator(c.ms, hp.hx, seed);
        const Operator Y = random_operator(c.ms, hp.hy, seed + 1000);
        for (const auto& t : kTerms) {
          const Operator zxy = apply(t, X, Y);
          const Operator zyx = apply(t, Y, X);
          const double dev = relative_deviation(zxy, -zyx);
          std::printf("[prop] antisym %s %s seed %llu %s: %.3e\n", c.name.c_str(), hp.name,
                      static_cast<unsigned long long>(seed), t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
        const double dev = relative_deviation(cr::commutator(X, Y), -cr::commutator(Y, X));
        std::printf("[prop] antisym %s %s seed %llu full: %.3e\n", c.name.c_str(), hp.name,
                    static_cast<unsigned long long>(seed), dev);
        EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " full";
      }
    }
  }
}

TEST(commutator_ref_properties, linearity) {
  for (const auto& c : cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X1 = random_operator(c.ms, hp.hx, seed);
        const Operator X2 = random_operator(c.ms, hp.hx, seed + 500);
        const Operator Y = random_operator(c.ms, hp.hy, seed + 1000);
        const Operator Xs = 0.7 * X1 - 1.3 * X2;
        for (const auto& t : kTerms) {
          const Operator lhs = apply(t, Xs, Y);
          const Operator rhs = 0.7 * apply(t, X1, Y) - 1.3 * apply(t, X2, Y);
          const double dev = relative_deviation(lhs, rhs);
          std::printf("[prop] linear %s %s seed %llu %s: %.3e\n", c.name.c_str(), hp.name,
                      static_cast<unsigned long long>(seed), t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
      }
    }
  }
}

TEST(commutator_ref_properties, hermiticity) {
  for (const auto& c : cases()) {
    for (const auto seed : c.seeds) {
      for (const auto& hp : kHPairs) {
        const Operator X = random_operator(c.ms, hp.hx, seed);
        const Operator Y = random_operator(c.ms, hp.hy, seed + 1000);
        // [H,H] and [A,A] are anti-Hermitian, [A,H] and [H,A] are Hermitian.
        const double h = (hp.hx == hp.hy) ? -1.0 : 1.0;
        for (const auto& t : kTerms) {
          const Operator Z = apply(t, X, Y);
          if (std::string(t.name) == "comm110ss" || std::string(t.name) == "comm220ss") continue;
          const double dev = hermiticity_violation(Z, h);
          std::printf("[prop] herm %s %s seed %llu %s: %.3e\n", c.name.c_str(), hp.name,
                      static_cast<unsigned long long>(seed), t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << hp.name << " " << t.name;
        }
        const Operator Z = cr::commutator(X, Y);
        EXPECT_EQ(sign_of(Z.hermiticity()), h);
        EXPECT_LE(hermiticity_violation(Z, h), kTol) << c.name << " " << hp.name << " full";
      }
    }
  }
}

TEST(commutator_ref_properties, jacobi_one_body) {
  for (const auto& c : cases()) {
    for (const auto seed : c.seeds) {
      const Operator X = random_operator(c.ms, Hermiticity::Hermitian, seed, 1);
      const Operator Y = random_operator(c.ms, Hermiticity::Hermitian, seed + 1, 1);
      const Operator W = random_operator(c.ms, Hermiticity::Hermitian, seed + 2, 1);
      Operator J = cr::commutator(X, cr::commutator(Y, W));
      J += cr::commutator(Y, cr::commutator(W, X));
      J += cr::commutator(W, cr::commutator(X, Y));
      Operator zero(c.ms, J.spec());
      const double dev = relative_deviation(J, zero);
      std::printf("[prop] jacobi %s seed %llu: %.3e\n", c.name.c_str(),
                  static_cast<unsigned long long>(seed), dev);
      EXPECT_LE(dev, kTol) << c.name;
    }
  }
}

TEST(commutator_ref_properties, non_hermitian_consistency) {
  for (const auto& c : cases()) {
    for (const auto seed : c.seeds) {
      const Operator XH = random_operator(c.ms, Hermiticity::Hermitian, seed);
      const Operator XA = random_operator(c.ms, Hermiticity::AntiHermitian, seed + 500);
      Operator X = XH + XA;
      X.set_hermiticity(Hermiticity::NonHermitian);
      for (const auto& hp : {Hermiticity::Hermitian, Hermiticity::AntiHermitian}) {
        const Operator Y = random_operator(c.ms, hp, seed + 1000);
        for (const auto& t : kTerms) {
          Operator lhs = apply(t, X, Y);
          Operator rhs = apply(t, XH, Y);
          rhs.set_hermiticity(Hermiticity::NonHermitian);
          Operator part = apply(t, XA, Y);
          part.set_hermiticity(Hermiticity::NonHermitian);
          rhs += part;
          const double dev = relative_deviation(lhs, rhs);
          std::printf("[prop] nonherm %s seed %llu %s: %.3e\n", c.name.c_str(),
                      static_cast<unsigned long long>(seed), t.name, dev);
          EXPECT_LE(dev, kTol) << c.name << " " << t.name;
        }
        EXPECT_EQ(cr::commutator(X, Y).hermiticity(), Hermiticity::NonHermitian);
      }
    }
  }
}

}  // namespace
