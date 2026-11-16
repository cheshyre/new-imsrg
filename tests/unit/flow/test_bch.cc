// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// BCH transform and product (M8 task T2): exactness on one-body operators
// against dense matrix exponentials, invertibility, the BCH product against
// expm(X) expm(Y), the thresholds and hermiticity.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg {
namespace {

using commutator::CommutatorOptions;
using commutator::Workspace;
using flow::BchOptions;
using flow::BchReport;
using op::Operator;
using op::OperatorSpec;

// ---- dense helpers (host only)
// -------------------------------------------------------------

struct Dense {
  int n = 0;
  std::vector<double> a;  // row-major
  Dense() = default;
  explicit Dense(int dim)
      : n(dim),
        a(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0) {}
  double& operator()(int i, int j) { return a[index(i, j)]; }
  double operator()(int i, int j) const { return a[index(i, j)]; }
  std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(j);
  }
};

Dense Identity(int n) {
  Dense d(n);
  for (int i = 0; i < n; ++i) d(i, i) = 1.0;
  return d;
}
Dense Mul(const Dense& x, const Dense& y) {
  Dense z(x.n);
  for (int i = 0; i < x.n; ++i)
    for (int k = 0; k < x.n; ++k)
      for (int j = 0; j < x.n; ++j) z(i, j) += x(i, k) * y(k, j);
  return z;
}
Dense Transpose(const Dense& x) {
  Dense z(x.n);
  for (int i = 0; i < x.n; ++i)
    for (int j = 0; j < x.n; ++j) z(i, j) = x(j, i);
  return z;
}
Dense Add(const Dense& x, const Dense& y, double c = 1.0) {
  Dense z = x;
  for (std::size_t i = 0; i < z.a.size(); ++i) z.a[i] += c * y.a[i];
  return z;
}
Dense Comm(const Dense& x, const Dense& y) {
  return Add(Mul(x, y), Mul(y, x), -1.0);
}
double MaxAbsDiff(const Dense& x, const Dense& y) {
  double m = 0;
  for (std::size_t i = 0; i < x.a.size(); ++i)
    m = std::max(m, std::abs(x.a[i] - y.a[i]));
  return m;
}
double MaxAbs(const Dense& x) {
  double m = 0;
  for (double v : x.a) m = std::max(m, std::abs(v));
  return m;
}

/// exp(A) by scaling and squaring with a 30-term Taylor series.
Dense Expm(const Dense& A) {
  int s = 0;
  double m = MaxAbs(A) * A.n;
  while (m > 0.1) {
    m /= 2;
    ++s;
  }
  Dense As = A;
  for (double& v : As.a) v /= std::pow(2.0, s);
  Dense result = Identity(A.n), term = Identity(A.n);
  for (int k = 1; k <= 30; ++k) {
    term = Mul(term, As);
    for (double& v : term.a) v /= k;
    result = Add(result, term);
  }
  for (int i = 0; i < s; ++i) result = Mul(result, result);
  return result;
}

/// Dense one-body matrix over all orbits (zero between different (l, 2j, tz)
/// blocks).
Dense OneBodyDense(Operator& o) {
  o.SyncToHost();
  const int n = o.model_space().basis().num_orbits();
  Dense d(n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      d(i, j) = o.one_body().get(OrbitIndex{i}, OrbitIndex{j});
  return d;
}

std::shared_ptr<const basis::ModelSpace> MakeSpace(int emax,
                                                   const char* nucleus) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, nucleus, nucleus);
}

/// Random operator with the given symmetry, rank and norm `target` (zero-body
/// part untouched).
Operator RandOp(const std::shared_ptr<const basis::ModelSpace>& ms,
                Hermiticity h, std::uint64_t seed, int rank, double target) {
  op::RandomOperatorOptions ro;
  ro.seed = seed;
  Operator o = op::RandomOperator(ms, OperatorSpec::Scalar(rank, h), ro);
  op::Scale(o, target / op::Norm(o));
  return o;
}

/// Largest part-wise absolute difference (all parts).
double MaxDev(const Operator& a, const Operator& b) {
  return op::MaxAbsDifference(a, b).max();
}

// ---- tests
// -----------------------------------------------------------------------------

TEST(flow_bch, one_body_exactness) {
  const auto ms = MakeSpace(2, "O16");
  Workspace ws(ms);
  const int n = ms->basis().num_orbits();
  const auto occ = ms->reference().occupations();
  for (int rank : {1, 2}) {
    for (std::uint64_t seed : {11u, 12u, 13u}) {
      Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, seed, 1, 0.3);
      op::RandomOperatorOptions ro;
      ro.seed = seed + 100;
      ro.zero_body_scale = 1.0;
      Operator X = op::RandomOperator(
          ms, OperatorSpec::Scalar(1, Hermiticity::kHermitian), ro);
      // The series truncation error is of order eps0 = ||X|| e^{-2||Om||} thr /
      // (2||Om||), so thr = 1e-12 makes the 1e-10 criterion a statement about
      // the algebra, not the cut.
      BchOptions bo;
      bo.transform_threshold = 1.0e-12;
      BchReport rep;
      Operator Z = flow::BchTransform(flow::WithParticleRank(X, rank),
                                      flow::WithParticleRank(Om, rank), bo,
                                      CommutatorOptions{}, ws, &rep);
      const Dense U = Expm(OneBodyDense(Om));
      const Dense X1 = OneBodyDense(X);
      const Dense Zref = Mul(Mul(U, X1), Transpose(U));
      const Dense Z1 = OneBodyDense(Z);
      const double dev1 = MaxAbsDiff(Z1, Zref);
      double e0 = X.zero_body();
      for (int a = 0; a < n; ++a)
        e0 += occ[static_cast<std::size_t>(a)] *
              (ms->basis().orbit(OrbitIndex{a}).j2.twice + 1) *
              (Zref(a, a) - X1(a, a));
      const double dev0 = std::abs(Z.zero_body() - e0);
      const double eps0 = op::Norm(X) * std::exp(-0.6) * 1.0e-12 / 0.6;
      std::printf("[bch one-body] rank %d seed %" PRIu64
                  ": n_comm %d, dev1 %.2e dev0 %.2e "
                  "(eps0 %.2e)\n",
                  rank, static_cast<std::uint64_t>(seed),
                  rep.nested_commutators, dev1, dev0, eps0);
      EXPECT_LT(dev1, 1.0e-10);
      EXPECT_LT(dev0, 1.0e-10 * std::max(1.0, std::abs(e0)));
      EXPECT_LT(dev1, eps0);
      EXPECT_EQ(Z.spec().particle_rank,
                2);  // the transform always returns rank >= 2
      EXPECT_EQ(op::TwoBodyNorm(Z), 0.0);
    }
  }
}

TEST(flow_bch, invertibility) {
  const auto ms = MakeSpace(2, "O16");
  Workspace ws(ms);
  op::RandomOperatorOptions ro;
  ro.seed = 21;
  ro.zero_body_scale = 3.0;
  Operator H = op::RandomOperator(
      ms, OperatorSpec::Scalar(2, Hermiticity::kHermitian), ro);
  op::Scale(H, 5.0 / op::Norm(H));
  const Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, 22, 2, 0.1);
  const Operator neg_Om = -Om;
  BchReport r1, r2;
  const Operator Z =
      flow::BchTransform(H, Om, BchOptions{}, CommutatorOptions{}, ws, &r1);
  const Operator Hb =
      flow::BchTransform(Z, neg_Om, BchOptions{}, CommutatorOptions{}, ws, &r2);
  // The truncated commutator is bilinear, so exp(ad_Om) exp(-ad_Om) = 1 exactly
  // in the limit; what remains is the series truncation: each pass stops once
  // ||nested_i|| < eps0 (i+1)!, with eps0 = ||X|| e^{-2||Om||} thr / (2||Om||),
  // so each leaves an error of order eps0.
  const double eps0_h = op::Norm(H) * std::exp(-0.2) * 1.0e-9 / 0.2;
  const double eps0_z = op::Norm(Z) * std::exp(-0.2) * 1.0e-9 / 0.2;
  const double bound = 2 * (eps0_h + eps0_z);
  Operator d = Hb;
  op::Axpy(-1.0, H, d);
  const double dnorm = op::Norm(d);
  const double dmax = MaxDev(Hb, H);
  std::printf(
      "[bch invert] ||H|| %.3f, n_comm %d/%d, ||dH|| %.2e (bound %.2e), max "
      "%.2e\n",
      op::Norm(H), r1.nested_commutators, r2.nested_commutators, dnorm, bound,
      dmax);
  EXPECT_LT(dnorm, bound);
  EXPECT_LT(dmax, 1.0e-8);
  EXPECT_TRUE(r1.converged && r2.converged);
}

TEST(flow_bch, product_one_body) {
  // emax 4: the s, p, d... blocks are up to 3x3, so anti-Hermitian one-body
  // operators of different blocks' rotations do not commute (2x2 antisymmetric
  // matrices would).
  const auto ms = MakeSpace(4, "O16");
  Workspace ws(ms);
  for (std::uint64_t seed : {31u, 32u, 33u}) {
    Operator X = RandOp(ms, Hermiticity::kAntiHermitian, seed, 1, 1.0e-2);
    Operator Y = RandOp(ms, Hermiticity::kAntiHermitian, seed + 50, 1, 0.2);
    Operator Z = flow::BchProduct(X, Y, BchOptions{}, CommutatorOptions{}, ws);
    const Dense Xd = OneBodyDense(X), Yd = OneBodyDense(Y),
                Zd = OneBodyDense(Z);
    const double dev = MaxAbsDiff(Expm(Zd), Mul(Expm(Xd), Expm(Yd)));
    // The omitted term of the exact series is -1/24 [Y,[X,[X,Y]]].
    const double omitted = MaxAbs(Comm(Yd, Comm(Xd, Comm(Xd, Yd)))) / 24.0;
    std::printf("[bch product] seed %" PRIu64
                ": max|expm(Z)-eXeY| = %.2e, omitted 1/24 term "
                "%.2e\n",
                static_cast<std::uint64_t>(seed), dev, omitted);
    // Deviation = the omitted -1/24 term + the higher terms dropped by the 1e-4
    // threshold and the fixed order (observed ~1e-7, dominated by the latter,
    // e.g. O(X^3 Y)); both are far below the product_threshold of 1e-4, which
    // is the asserted bound.
    EXPECT_LT(dev, 2 * omitted + 1.0e-4);
    EXPECT_LT(dev, 1.0e-4);
    EXPECT_EQ(Z.spec().particle_rank, 1);
    EXPECT_EQ(Z.hermiticity(), Hermiticity::kAntiHermitian);
  }
  // Neutral elements.
  const Operator X = RandOp(ms, Hermiticity::kAntiHermitian, 41, 2, 1.0e-2);
  const Operator zero(ms, OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
  const Operator Zx =
      flow::BchProduct(X, zero, BchOptions{}, CommutatorOptions{}, ws);
  const Operator Zy =
      flow::BchProduct(zero, X, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(MaxDev(Zx, X), 0.0);
  EXPECT_EQ(MaxDev(Zy, X), 0.0);
}

TEST(flow_bch, thresholds_and_report) {
  const auto ms = MakeSpace(1, "He4");
  Workspace ws(ms);
  const BchOptions o;
  // ||X|| <= transform_threshold: returned unchanged (promoted to rank 2), no
  // commutator.
  {
    const Operator X = RandOp(ms, Hermiticity::kHermitian, 51, 1, 1.0e-10);
    const Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, 52, 2, 0.5);
    BchReport rep;
    const Operator Z =
        flow::BchTransform(X, Om, o, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 0);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(Z.spec().particle_rank, 2);
    EXPECT_EQ(MaxDev(Z, flow::WithParticleRank(X, 2)), 0.0);
  }
  // Counts grow with ||Omega||.
  const Operator X = RandOp(ms, Hermiticity::kHermitian, 53, 2, 3.0);
  const Operator Om0 = RandOp(ms, Hermiticity::kAntiHermitian, 54, 2, 1.0);
  int first = 0, last = 0;
  for (double nrm : {0.02, 0.2, 1.0, 3.0}) {
    Operator Om = Om0;
    op::Scale(Om, nrm);
    BchReport rep;
    flow::BchTransform(X, Om, o, CommutatorOptions{}, ws, &rep);
    std::printf("[bch report] ||Omega|| %.2f: %d nested, converged %d\n", nrm,
                rep.nested_commutators, rep.converged);
    EXPECT_GE(rep.nested_commutators, last);
    if (first == 0) first = rep.nested_commutators;
    last = rep.nested_commutators;
  }
  EXPECT_GT(last, first);
  // Max iterations without convergence.
  {
    Operator Om = Om0;
    op::Scale(Om, 3.0);
    BchOptions few = o;
    few.max_iterations = 3;
    BchReport rep;
    flow::BchTransform(X, Om, few, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 3);
    EXPECT_FALSE(rep.converged);
  }
  // Omega = 0: one commutator (epsilon = +inf), result equals X.
  {
    const Operator Om(ms, OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
    BchReport rep;
    const Operator Z =
        flow::BchTransform(X, Om, o, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 1);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(MaxDev(Z, X), 0.0);
  }
}

// A NaN Omega stops the transform after the first nested commutator instead
// of running all max_iterations (flow review item 2).
TEST(flow_bch, non_finite_omega_stops_the_series) {
  const auto ms = MakeSpace(1, "He4");
  Workspace ws(ms);
  const Operator X = RandOp(ms, Hermiticity::kHermitian, 61, 2, 1.0);
  Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, 62, 2, 0.5);
  op::Scale(Om, std::numeric_limits<double>::quiet_NaN());
  BchReport rep;
  flow::BchTransform(X, Om, BchOptions{}, CommutatorOptions{}, ws, &rep);
  EXPECT_LE(rep.nested_commutators, 2);
  EXPECT_FALSE(rep.converged);
}

// skip_first_commutator (imsrg++ SetBCHSkipiEq1, perturbative triples):
// Z = X + sum_{k>=1} ad_Omega^k(X) / (k+1)!, with the imsrg++ termination
// rule (epsilon starts at ||X|| exp(-2||Omega||) thr / (2||Omega||) and is
// multiplied by j + 2 after the term j, not by 2 at the skipped step).
TEST(flow_bch, skip_first_commutator_series) {
  const auto ms = MakeSpace(1, "He4");
  Workspace ws(ms);
  const CommutatorOptions copts;
  const Operator X = RandOp(ms, Hermiticity::kHermitian, 71, 2, 2.0);
  const Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, 72, 2, 0.4);
  BchOptions opts;
  opts.skip_first_commutator = true;
  BchReport rep;
  const Operator Z = flow::BchTransform(X, Om, opts, copts, ws, &rep);

  // Explicit series with the imsrg++ termination rule.
  const double nx = op::Norm(X);
  const double ny = op::Norm(Om);
  Operator expected = flow::WithParticleRank(X, 2);
  Operator nested = expected;
  double eps = nx * std::exp(-2 * ny) * opts.transform_threshold / (2 * ny);
  double fact = 1.0;  // 1 / (j + 1)!
  int count = 0;
  for (int j = 1; j <= opts.max_iterations - 1; ++j) {
    nested = commutator::Commutator(Om, nested, copts, ws);
    ++count;
    fact /= j + 1;
    op::Axpy(fact, nested, expected);
    eps *= j + 2;
    if (op::Norm(nested) < eps) break;
  }
  EXPECT_EQ(rep.nested_commutators, count);
  EXPECT_TRUE(rep.converged);
  EXPECT_GT(count, 3);
  EXPECT_LT(MaxDev(Z, expected), 1.0e-13);

  // Relation to the full transform: [Omega, Z_skip] = Z_full - X (up to the
  // series truncation, about 1e-9).
  const Operator full =
      flow::BchTransform(X, Om, BchOptions{}, copts, ws, nullptr);
  Operator lhs = commutator::Commutator(Om, Z, copts, ws);
  Operator rhs = full;
  op::Axpy(-1.0, flow::WithParticleRank(X, 2), rhs);
  EXPECT_LT(MaxDev(lhs, rhs), 1.0e-8);

  // The skipped step also costs one iteration of the cap.
  opts.max_iterations = 4;
  flow::BchTransform(X, Om, opts, copts, ws, &rep);
  EXPECT_EQ(rep.nested_commutators, 3);
  EXPECT_FALSE(rep.converged);
}

TEST(flow_bch, hermiticity) {
  const auto ms = MakeSpace(2, "O16");
  Workspace ws(ms);
  const Operator H = RandOp(ms, Hermiticity::kHermitian, 61, 2, 3.0);
  const Operator Om = RandOp(ms, Hermiticity::kAntiHermitian, 62, 2, 0.2);
  Operator Z = flow::BchTransform(H, Om, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(Z.hermiticity(), Hermiticity::kHermitian);
  const Dense Z1 = OneBodyDense(Z);
  const double hz = MaxAbsDiff(Z1, Transpose(Z1));
  EXPECT_LT(hz, 1.0e-12);
  EXPECT_GT(MaxAbs(Z1), 0.0);

  const Operator A = RandOp(ms, Hermiticity::kAntiHermitian, 63, 2, 0.05);
  Operator P = flow::BchProduct(A, Om, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(P.hermiticity(), Hermiticity::kAntiHermitian);
  const Dense P1 = OneBodyDense(P);
  const double hp = MaxAbsDiff(P1, Add(Dense(P1.n), Transpose(P1), -1.0));
  std::printf("[bch hermiticity] transform %.2e, product %.2e\n", hz, hp);
  EXPECT_LT(hp, 1.0e-12);
  EXPECT_EQ(P.zero_body(), 0.0);
}

}  // namespace
}  // namespace imsrg
