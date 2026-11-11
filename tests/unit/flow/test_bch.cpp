// BCH transform and product (M8 task T2): exactness on one-body operators against dense matrix
// exponentials, invertibility, the BCH product against expm(X) expm(Y), the thresholds and
// hermiticity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/flow/bch.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/random_operator.hpp"

namespace {

using namespace imsrg;
using commutator::CommutatorOptions;
using commutator::Workspace;
using flow::BchOptions;
using flow::BchReport;
using op::Operator;
using op::OperatorSpec;

// ---- dense helpers (host only) -------------------------------------------------------------

struct Dense {
  int n = 0;
  std::vector<double> a;  // row-major
  Dense() = default;
  explicit Dense(int n_) : n(n_), a(static_cast<std::size_t>(n_) * n_, 0.0) {}
  double& operator()(int i, int j) { return a[static_cast<std::size_t>(i) * n + j]; }
  double operator()(int i, int j) const { return a[static_cast<std::size_t>(i) * n + j]; }
};

Dense identity(int n) {
  Dense d(n);
  for (int i = 0; i < n; ++i) d(i, i) = 1.0;
  return d;
}
Dense mul(const Dense& x, const Dense& y) {
  Dense z(x.n);
  for (int i = 0; i < x.n; ++i)
    for (int k = 0; k < x.n; ++k)
      for (int j = 0; j < x.n; ++j) z(i, j) += x(i, k) * y(k, j);
  return z;
}
Dense transpose(const Dense& x) {
  Dense z(x.n);
  for (int i = 0; i < x.n; ++i)
    for (int j = 0; j < x.n; ++j) z(i, j) = x(j, i);
  return z;
}
Dense add(const Dense& x, const Dense& y, double c = 1.0) {
  Dense z = x;
  for (std::size_t i = 0; i < z.a.size(); ++i) z.a[i] += c * y.a[i];
  return z;
}
Dense comm(const Dense& x, const Dense& y) { return add(mul(x, y), mul(y, x), -1.0); }
double max_abs_diff(const Dense& x, const Dense& y) {
  double m = 0;
  for (std::size_t i = 0; i < x.a.size(); ++i) m = std::max(m, std::abs(x.a[i] - y.a[i]));
  return m;
}
double max_abs(const Dense& x) {
  double m = 0;
  for (double v : x.a) m = std::max(m, std::abs(v));
  return m;
}

/// exp(A) by scaling and squaring with a 30-term Taylor series.
Dense expm(const Dense& A) {
  int s = 0;
  double m = max_abs(A) * A.n;
  while (m > 0.1) {
    m /= 2;
    ++s;
  }
  Dense As = A;
  for (double& v : As.a) v /= std::pow(2.0, s);
  Dense result = identity(A.n), term = identity(A.n);
  for (int k = 1; k <= 30; ++k) {
    term = mul(term, As);
    for (double& v : term.a) v /= k;
    result = add(result, term);
  }
  for (int i = 0; i < s; ++i) result = mul(result, result);
  return result;
}

/// Dense one-body matrix over all orbits (zero between different (l, 2j, tz) blocks).
Dense one_body_dense(Operator& o) {
  o.sync_to_host();
  const int n = o.model_space().basis().num_orbits();
  Dense d(n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) d(i, j) = o.one_body().get(OrbitIndex{i}, OrbitIndex{j});
  return d;
}

std::shared_ptr<const basis::ModelSpace> make_space(int emax, const char* nucleus) {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = emax}, 16.0, nucleus,
                                       nucleus);
}

/// Random operator with the given symmetry, rank and norm `target` (zero-body part untouched).
Operator rand_op(const std::shared_ptr<const basis::ModelSpace>& ms, Hermiticity h,
                 std::uint64_t seed, int rank, double target) {
  op::RandomOperatorOptions ro;
  ro.seed = seed;
  Operator o = op::random_operator(ms, OperatorSpec::scalar(rank, h), ro);
  op::scale(o, target / op::norm(o));
  return o;
}

/// Largest part-wise absolute difference (all parts).
double max_dev(const Operator& a, const Operator& b) { return op::max_abs_difference(a, b).max(); }

// ---- tests ----------------------------------------------------------------------------------

TEST(flow_bch, one_body_exactness) {
  const auto ms = make_space(2, "O16");
  Workspace ws(ms);
  const int n = ms->basis().num_orbits();
  const auto occ = ms->reference().occupations();
  for (int rank : {1, 2}) {
    for (std::uint64_t seed : {11, 12, 13}) {
      Operator Om = rand_op(ms, Hermiticity::AntiHermitian, seed, 1, 0.3);
      op::RandomOperatorOptions ro;
      ro.seed = seed + 100;
      ro.zero_body_scale = 1.0;
      Operator X = op::random_operator(ms, OperatorSpec::scalar(1, Hermiticity::Hermitian), ro);
      // The series truncation error is of order eps0 = ||X|| e^{-2||Om||} thr / (2||Om||), so
      // thr = 1e-12 makes the 1e-10 criterion a statement about the algebra, not the cut.
      BchOptions bo;
      bo.transform_threshold = 1e-12;
      BchReport rep;
      Operator Z = flow::bch_transform(flow::with_particle_rank(X, rank),
                                       flow::with_particle_rank(Om, rank), bo,
                                       CommutatorOptions{}, ws, &rep);
      const Dense U = expm(one_body_dense(Om));
      const Dense X1 = one_body_dense(X);
      const Dense Zref = mul(mul(U, X1), transpose(U));
      const Dense Z1 = one_body_dense(Z);
      const double dev1 = max_abs_diff(Z1, Zref);
      double e0 = X.zero_body();
      for (int a = 0; a < n; ++a)
        e0 += occ[a] * (ms->basis().orbit(OrbitIndex{a}).j2.twice + 1) * (Zref(a, a) - X1(a, a));
      const double dev0 = std::abs(Z.zero_body() - e0);
      const double eps0 = op::norm(X) * std::exp(-0.6) * 1e-12 / 0.6;
      std::printf("[bch one-body] rank %d seed %llu: n_comm %d, dev1 %.2e dev0 %.2e (eps0 %.2e)\n",
                  rank, static_cast<unsigned long long>(seed), rep.nested_commutators, dev1, dev0,
                  eps0);
      EXPECT_LT(dev1, 1e-10);
      EXPECT_LT(dev0, 1e-10 * std::max(1.0, std::abs(e0)));
      EXPECT_LT(dev1, eps0);
      EXPECT_EQ(Z.spec().particle_rank, 2);  // the transform always returns rank >= 2
      EXPECT_EQ(op::two_body_norm(Z), 0.0);
    }
  }
}

TEST(flow_bch, invertibility) {
  const auto ms = make_space(2, "O16");
  Workspace ws(ms);
  op::RandomOperatorOptions ro;
  ro.seed = 21;
  ro.zero_body_scale = 3.0;
  Operator H = op::random_operator(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian), ro);
  op::scale(H, 5.0 / op::norm(H));
  const Operator Om = rand_op(ms, Hermiticity::AntiHermitian, 22, 2, 0.1);
  const Operator mOm = -Om;
  BchReport r1, r2;
  const Operator Z = flow::bch_transform(H, Om, BchOptions{}, CommutatorOptions{}, ws, &r1);
  const Operator Hb = flow::bch_transform(Z, mOm, BchOptions{}, CommutatorOptions{}, ws, &r2);
  // The truncated commutator is bilinear, so exp(ad_Om) exp(-ad_Om) = 1 exactly in the limit;
  // what remains is the series truncation: each pass stops once ||nested_i|| < eps0 (i+1)!,
  // with eps0 = ||X|| e^{-2||Om||} thr / (2||Om||), so each leaves an error of order eps0.
  const double eps0_h = op::norm(H) * std::exp(-0.2) * 1e-9 / 0.2;
  const double eps0_z = op::norm(Z) * std::exp(-0.2) * 1e-9 / 0.2;
  const double bound = 2 * (eps0_h + eps0_z);
  Operator d = Hb;
  op::axpy(-1.0, H, d);
  const double dnorm = op::norm(d);
  const double dmax = max_dev(Hb, H);
  std::printf("[bch invert] ||H|| %.3f, n_comm %d/%d, ||dH|| %.2e (bound %.2e), max %.2e\n",
              op::norm(H), r1.nested_commutators, r2.nested_commutators, dnorm, bound, dmax);
  EXPECT_LT(dnorm, bound);
  EXPECT_LT(dmax, 1e-8);
  EXPECT_TRUE(r1.converged && r2.converged);
}

TEST(flow_bch, product_one_body) {
  // emax 4: the s, p, d... blocks are up to 3x3, so anti-Hermitian one-body operators of
  // different blocks' rotations do not commute (2x2 antisymmetric matrices would).
  const auto ms = make_space(4, "O16");
  Workspace ws(ms);
  for (std::uint64_t seed : {31, 32, 33}) {
    Operator X = rand_op(ms, Hermiticity::AntiHermitian, seed, 1, 1e-2);
    Operator Y = rand_op(ms, Hermiticity::AntiHermitian, seed + 50, 1, 0.2);
    Operator Z = flow::bch_product(X, Y, BchOptions{}, CommutatorOptions{}, ws);
    const Dense Xd = one_body_dense(X), Yd = one_body_dense(Y), Zd = one_body_dense(Z);
    const double dev = max_abs_diff(expm(Zd), mul(expm(Xd), expm(Yd)));
    // The omitted term of the exact series is -1/24 [Y,[X,[X,Y]]].
    const double omitted = max_abs(comm(Yd, comm(Xd, comm(Xd, Yd)))) / 24.0;
    std::printf("[bch product] seed %llu: max|expm(Z)-eXeY| = %.2e, omitted 1/24 term %.2e\n",
                static_cast<unsigned long long>(seed), dev, omitted);
    // Deviation = the omitted -1/24 term + the higher terms dropped by the 1e-4 threshold and
    // the fixed order (observed ~1e-7, dominated by the latter, e.g. O(X^3 Y)); both are far
    // below the product_threshold of 1e-4, which is the asserted bound.
    EXPECT_LT(dev, 2 * omitted + 1e-4);
    EXPECT_LT(dev, 1e-4);
    EXPECT_EQ(Z.spec().particle_rank, 1);
    EXPECT_EQ(Z.hermiticity(), Hermiticity::AntiHermitian);
  }
  // Neutral elements.
  const Operator X = rand_op(ms, Hermiticity::AntiHermitian, 41, 2, 1e-2);
  const Operator zero(ms, OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
  const Operator Zx = flow::bch_product(X, zero, BchOptions{}, CommutatorOptions{}, ws);
  const Operator Zy = flow::bch_product(zero, X, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(max_dev(Zx, X), 0.0);
  EXPECT_EQ(max_dev(Zy, X), 0.0);
}

TEST(flow_bch, thresholds_and_report) {
  const auto ms = make_space(1, "He4");
  Workspace ws(ms);
  const BchOptions o;
  // ||X|| <= transform_threshold: returned unchanged (promoted to rank 2), no commutator.
  {
    const Operator X = rand_op(ms, Hermiticity::Hermitian, 51, 1, 1e-10);
    const Operator Om = rand_op(ms, Hermiticity::AntiHermitian, 52, 2, 0.5);
    BchReport rep;
    const Operator Z = flow::bch_transform(X, Om, o, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 0);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(Z.spec().particle_rank, 2);
    EXPECT_EQ(max_dev(Z, flow::with_particle_rank(X, 2)), 0.0);
  }
  // Counts grow with ||Omega||.
  const Operator X = rand_op(ms, Hermiticity::Hermitian, 53, 2, 3.0);
  const Operator Om0 = rand_op(ms, Hermiticity::AntiHermitian, 54, 2, 1.0);
  int first = 0, last = 0;
  for (double nrm : {0.02, 0.2, 1.0, 3.0}) {
    Operator Om = Om0;
    op::scale(Om, nrm);
    BchReport rep;
    flow::bch_transform(X, Om, o, CommutatorOptions{}, ws, &rep);
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
    op::scale(Om, 3.0);
    BchOptions few = o;
    few.max_iterations = 3;
    BchReport rep;
    flow::bch_transform(X, Om, few, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 3);
    EXPECT_FALSE(rep.converged);
  }
  // Omega = 0: one commutator (epsilon = +inf), result equals X.
  {
    const Operator Om(ms, OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
    BchReport rep;
    const Operator Z = flow::bch_transform(X, Om, o, CommutatorOptions{}, ws, &rep);
    EXPECT_EQ(rep.nested_commutators, 1);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(max_dev(Z, X), 0.0);
  }
}

TEST(flow_bch, hermiticity) {
  const auto ms = make_space(2, "O16");
  Workspace ws(ms);
  const Operator H = rand_op(ms, Hermiticity::Hermitian, 61, 2, 3.0);
  const Operator Om = rand_op(ms, Hermiticity::AntiHermitian, 62, 2, 0.2);
  Operator Z = flow::bch_transform(H, Om, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(Z.hermiticity(), Hermiticity::Hermitian);
  const Dense Z1 = one_body_dense(Z);
  const double hz = max_abs_diff(Z1, transpose(Z1));
  EXPECT_LT(hz, 1e-12);
  EXPECT_GT(max_abs(Z1), 0.0);

  const Operator A = rand_op(ms, Hermiticity::AntiHermitian, 63, 2, 0.05);
  Operator P = flow::bch_product(A, Om, BchOptions{}, CommutatorOptions{}, ws);
  EXPECT_EQ(P.hermiticity(), Hermiticity::AntiHermitian);
  const Dense P1 = one_body_dense(P);
  const double hp = max_abs_diff(P1, add(Dense(P1.n), transpose(P1), -1.0));
  std::printf("[bch hermiticity] transform %.2e, product %.2e\n", hz, hp);
  EXPECT_LT(hp, 1e-12);
  EXPECT_EQ(P.zero_body(), 0.0);
}

}  // namespace
