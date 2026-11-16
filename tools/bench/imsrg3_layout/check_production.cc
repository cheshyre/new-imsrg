// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Checks of the production [2,2]->3 term against the benchmark reference
// kernels (see imsrg3_layout_check.cc): `commutator::Comm223` element by
// element against K0 (`Comm223Reference`, the loop-by-loop port of imsrg++
// comm223ss) and `commutator::PerturbativeTriplesEnergy` against
// `TriplesReference`, on the spaces of check_tiled.cc (including the dE3max
// cut), relative tolerance 1e-12. Suite `Production*`.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/three_body.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/production.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

constexpr double kNoCut = 1.0e10;
constexpr double kTol = 1.0e-12;

/// The benchmark tables and the production context of one space.
struct ProdFixture {
  ProdFixture(const char* ref, int emax, int e3max, double de3max)
      : ms(basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, ref,
                                        ref)
               ->WithThreeBody(basis::ThreeBodyCuts{e3max})),
        space(ms, de3max, -1.0),
        sixj(emax),
        pt(space, sixj),
        kctx(space, pt, sixj),
        pctx(ProductionModelSpace(ms, de3max, -1.0)) {}
  std::shared_ptr<const basis::ModelSpace> ms;
  CanonicalSpace space;
  SixJ3Table sixj;
  PairThirdSpace pt;
  KernelContext kctx;
  commutator::ThreeBodyContext pctx;

  op::Operator Random(Hermiticity h, std::uint64_t seed) const {
    return op::RandomOperator(ms, op::OperatorSpec::Scalar(2, h), {seed});
  }
};

/// The spaces of check_tiled.cc: reference, emax, E3max, dE3max.
struct Case {
  const char* ref;
  int emax;
  int e3max;
  double de3max;
};
constexpr Case kCases[] = {{"He4", 2, 6, kNoCut},
                           {"O16", 2, 6, kNoCut},
                           {"O16", 3, 7, kNoCut},
                           {"O16", 3, 9, 3.0}};

/// epsilon_i = (n_i > 0.5 ? -30 : 10) + 2 e_i + 0.1 (i % 7).
std::vector<double> Epsilon(const basis::ModelSpace& ms) {
  const auto orbits = ms.basis().orbits();
  const auto occ = ms.reference().occupations();
  std::vector<double> eps(orbits.size());
  for (std::size_t i = 0; i < orbits.size(); ++i) {
    eps[i] = (occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * orbits[i].e() +
             0.1 * static_cast<double>(i % 7);
  }
  return eps;
}

double RelScalar(double x, double y) {
  return std::abs(x - y) / std::max(1.0e-300, std::abs(y));
}

void CheckCase(const Case& cs, Hermiticity hx, Hermiticity hy) {
  const ProdFixture f(cs.ref, cs.emax, cs.e3max, cs.de3max);
  const op::Operator x = f.Random(hx, 101);
  const op::Operator y = f.Random(hy, 202);
  std::vector<double> z0(static_cast<std::size_t>(f.space.total_size()), 0.0);
  Comm223Reference(f.kctx, x, y, z0, true);
  const op::ThreeBody z = commutator::Comm223(x, y, f.pctx);
  const std::vector<double> zp = ToCanonicalDense(z, f.space);
  double diff = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < z0.size(); ++i) {
    diff = std::max(diff, std::abs(zp[i] - z0[i]));
    scale = std::max(scale, std::abs(z0[i]));
  }
  const double n0 = ImsrgppNorm(f.space, z0);
  const double np = op::ImsrgppNorm(z);
  const std::vector<double> eps = Epsilon(*f.ms);
  const double e0 = TriplesReference(f.kctx, x, y, eps);
  const double ep = commutator::PerturbativeTriplesEnergy(x, y, eps, f.pctx);
  std::cout << "  production " << cs.ref << " emax " << cs.emax << " e3max "
            << cs.e3max << " de3max " << cs.de3max << ": max|z| " << scale
            << ", |prod-K0|/max " << diff / scale << ", norm rel "
            << RelScalar(np, n0) << ", triples " << ep << " rel "
            << RelScalar(ep, e0) << "\n";
  EXPECT_GT(scale, 1.0);
  EXPECT_LE(diff, kTol * scale);
  EXPECT_LE(RelScalar(np, n0), kTol);
  if (e0 != 0.0) {
    EXPECT_LE(RelScalar(ep, e0), kTol);
  } else {
    EXPECT_EQ(ep, 0.0);
  }
}

TEST(ProductionComm223, MatchesK0HermitianAntiHermitian) {
  for (const Case& cs : kCases) {
    CheckCase(cs, Hermiticity::kAntiHermitian, Hermiticity::kHermitian);
  }
}

TEST(ProductionComm223, MatchesK0BothAntiHermitian) {
  for (const Case& cs : {kCases[0], kCases[1], kCases[3]}) {
    CheckCase(cs, Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian);
  }
}

}  // namespace
}  // namespace imsrg::layout3
