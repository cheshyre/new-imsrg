// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark (kernels); see
// imsrg3_layout_check.cc. K0 (`Comm223Reference`) and the reference triples
// (`TriplesReference`): mirror, antisymmetry and linearity of the commutator,
// zero diagonals for two anti-Hermitian operands, null kets, and the triples
// energy against a host recomputation from the K0 blocks.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/random_operator.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

/// A model space with its canonical space, tables and kernel context.
struct KFixture {
  KFixture(const char* ref, int emax, int e3max, double de3max)
      : ms(basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, ref,
                                        ref)
               ->WithThreeBody(basis::ThreeBodyCuts{e3max})),
        space(ms, de3max, -1.0),
        sixj(emax),
        pt(space, sixj),
        ctx(space, pt, sixj) {}
  std::shared_ptr<const basis::ModelSpace> ms;
  CanonicalSpace space;
  SixJ3Table sixj;
  PairThirdSpace pt;
  KernelContext ctx;
};

/// Random scalar two-body operator of hermiticity `h`.
op::Operator Random(const KFixture& f, Hermiticity h, std::uint64_t seed,
                    double scale = 1.0) {
  op::RandomOperatorOptions ro;
  ro.seed = seed;
  ro.one_body_scale = scale;
  ro.two_body_scale = scale;
  return op::RandomOperator(f.ms, op::OperatorSpec::Scalar(2, h), ro);
}

std::vector<double> K0(const KFixture& f, const op::Operator& X,
                       const op::Operator& Y, bool both, double* seconds) {
  std::vector<double> z(U(f.space.total_size()), 1.0e300);
  const auto t0 = std::chrono::steady_clock::now();
  Comm223Reference(f.ctx, X, Y, z, both);
  if (seconds != nullptr) {
    *seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
  }
  return z;
}

double MaxAbs(std::span<const double> x) {
  double m = 0.0;
  for (const double v : x) m = std::max(m, std::abs(v));
  return m;
}

/// max |x - s y| / scale.
double Dev(std::span<const double> x, std::span<const double> y, double s,
           double scale) {
  double d = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    d = std::max(d, std::abs(x[i] - s * y[i]));
  }
  return d / scale;
}

/// epsilon_i = (n_i > 0.5 ? -30 : 10) + 2 e_i + 0.1 (i % 7).
std::vector<double> Epsilon(const KFixture& f) {
  const auto orbits = f.ms->basis().orbits();
  const auto occ = f.ms->reference().occupations();
  std::vector<double> eps(orbits.size());
  for (std::size_t i = 0; i < orbits.size(); ++i) {
    eps[i] = (occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * orbits[i].e() +
             0.1 * static_cast<double>(i % 7);
  }
  return eps;
}

/// Host recomputation of the triples energy from the full K0 blocks `z`:
/// E = 1/2 sum_ch (2J+1) sum_{i != j} w_i w_j z_ij^2 f_ij.
double HostTriples(const KFixture& f, std::span<const double> z,
                   std::span<const double> eps) {
  const CanonicalView cv = f.space.host_view();
  const auto occ = f.ms->reference().occupations();
  double e_tot = 0.0;
  for (std::int32_t c = 0; c < cv.num_channels; ++c) {
    const std::int64_t n = cv.ch_dim[c];
    const std::int64_t r0 = cv.ch_row_offset[c];
    const double* blk = z.data() + cv.ch_block_offset[c];
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int32_t ti = cv.row_triple[r0 + i];
      const double ni =
          occ[U(cv.tr_a[ti])] * occ[U(cv.tr_b[ti])] * occ[U(cv.tr_c[ti])];
      const double ui = (1.0 - occ[U(cv.tr_a[ti])]) *
                        (1.0 - occ[U(cv.tr_b[ti])]) *
                        (1.0 - occ[U(cv.tr_c[ti])]);
      const double ei =
          eps[U(cv.tr_a[ti])] + eps[U(cv.tr_b[ti])] + eps[U(cv.tr_c[ti])];
      for (std::int64_t j = 0; j < n; ++j) {
        if (i == j) continue;
        const std::int32_t tj = cv.row_triple[r0 + j];
        const double nj =
            occ[U(cv.tr_a[tj])] * occ[U(cv.tr_b[tj])] * occ[U(cv.tr_c[tj])];
        const double uj = (1.0 - occ[U(cv.tr_a[tj])]) *
                          (1.0 - occ[U(cv.tr_b[tj])]) *
                          (1.0 - occ[U(cv.tr_c[tj])]);
        const double ej =
            eps[U(cv.tr_a[tj])] + eps[U(cv.tr_b[tj])] + eps[U(cv.tr_c[tj])];
        if (ni < 1.0e-8 && ui < 1.0e-8) continue;
        const double num = ni * uj - nj * ui;
        if (std::abs(num) < 1.0e-8) continue;
        const double zij = blk[i + j * n];
        sum += cv.tr_sym[ti] / 6.0 * cv.tr_sym[tj] / 6.0 * zij * zij * num /
               (ei - ej);
      }
    }
    e_tot += 0.5 * (cv.ch_two_j[c] + 1.0) * sum;
  }
  return e_tot;
}

/// Runs all K0 and triples checks on one space.
void CheckSpace(const char* name, const char* ref, int emax, int e3max,
                double de3max, bool print_item6) {
  KFixture f(ref, emax, e3max, de3max);
  const op::Operator X = Random(f, Hermiticity::kAntiHermitian, 1);
  const op::Operator Y = Random(f, Hermiticity::kHermitian, 2);
  const op::Operator X2 = Random(f, Hermiticity::kAntiHermitian, 1, 2.0);
  const op::Operator Ya = Random(f, Hermiticity::kAntiHermitian, 3);

  double secs = 0.0;
  const std::vector<double> z = K0(f, X, Y, false, &secs);
  const double zmax = MaxAbs(z);
  ASSERT_GT(zmax, 0.0);
  const std::vector<double> zb = K0(f, X, Y, true, nullptr);
  const std::vector<double> zyx = K0(f, Y, X, false, nullptr);
  const std::vector<double> z2 = K0(f, X2, Y, false, nullptr);

  const double d_mirror = Dev(z, zb, 1.0, zmax);
  const double d_anti = Dev(zyx, z, -1.0, zmax);
  const double d_lin = Dev(z2, z, 2.0, zmax);
  EXPECT_LT(d_mirror, 1.0e-12);
  EXPECT_LT(d_anti, 1.0e-12);
  EXPECT_LT(d_lin, 1.0e-13);

  // Two anti-Hermitian operands: h_Z = -1, the diagonal of every block is 0.
  EXPECT_EQ(CommutatorHermiticity(X, Ya), -1);
  const std::vector<double> za = K0(f, X, Ya, true, nullptr);
  const double zamax = MaxAbs(za);
  const CanonicalView cv = f.space.host_view();
  double diag = 0.0;
  for (std::int32_t c = 0; c < cv.num_channels; ++c) {
    const std::int64_t n = cv.ch_dim[c];
    for (std::int64_t i = 0; i < n; ++i) {
      diag = std::max(diag, std::abs(za[U(cv.ch_block_offset[c] + i + i * n)]));
    }
  }
  EXPECT_GT(zamax, 0.0);
  EXPECT_LT(diag / zamax, 1.0e-12);

  // Null kets (no pair-third state): zero rows and columns.
  const CsrView rv = f.pt.RMap().host_view();
  double null_max = 0.0;
  std::int64_t num_null = 0;
  for (std::int64_t row = 0; row < cv.num_rows; ++row) {
    if (rv.ptr[row] != rv.ptr[row + 1]) continue;
    ++num_null;
    const std::int32_t c = cv.tr_channel[cv.row_triple[row]];
    const std::int64_t n = cv.ch_dim[c];
    const std::int64_t loc = row - cv.ch_row_offset[c];
    const double* blk = zb.data() + cv.ch_block_offset[c];
    for (std::int64_t k = 0; k < n; ++k) {
      null_max = std::max(null_max, std::abs(blk[loc + k * n]));
      null_max = std::max(null_max, std::abs(blk[k + loc * n]));
    }
  }
  EXPECT_LT(null_max / zmax, 1.0e-12);

  // Triples.
  const std::vector<double> eps = Epsilon(f);
  const double e_ref = TriplesReference(f.ctx, X, Y, eps);
  const double e_host = HostTriples(f, z, eps);
  EXPECT_NE(e_ref, 0.0);
  const double d_tri = std::abs(e_ref - e_host) / std::abs(e_host);
  EXPECT_LT(d_tri, 1.0e-12);

  std::printf("[ K0 %s ] rows %" PRId64 "  null rows %" PRId64
              "  K0 time %.2f s  max|z| %.6e  "
              "dev mirror %.2e antisym %.2e linear %.2e diag %.2e null %.2e  "
              "E3 %.12e (host %.12e, rel %.2e)\n",
              name, cv.num_rows, num_null, secs, zmax, d_mirror, d_anti, d_lin,
              diag / zamax, null_max / zmax, e_ref, e_host, d_tri);
  if (print_item6) {
    std::printf(
        "[ K0 item6 ] O16 emax 2 seed 1: max|z| %.12e  ImsrgppNorm %.12e  "
        "triples E %.12e\n",
        zmax, ImsrgppNorm(f.space, z), e_ref);
  }
}

TEST(Comm223Ref, He4Emax2) {
  CheckSpace("He4 emax2 E3max6", "He4", 2, 6, 1.0e10, false);
}

TEST(Comm223Ref, O16Emax2) {
  CheckSpace("O16 emax2 E3max6", "O16", 2, 6, 1.0e10, true);
}

TEST(Comm223Ref, O16Emax3De3max3) {
  CheckSpace("O16 emax3 E3max9 de3max3", "O16", 3, 9, 3.0, false);
}

// K0 against imsrg++ itself: values printed by `refharness comm223` (old
// imsrg-devel library, tools/reference/rh_imsrg3.cpp) for the same operators
// (seeds 1 and 2, written with `imsrg3_layout_bench --write-dumps`) on
// 2026-11-15: the three-body norm of comm223ss and its perturbative-triples
// energy.
TEST(Comm223Ref, MatchesImsrgDevel) {
  struct Golden {
    const char* ref;
    int emax;
    int e3max;
    double norm;
    double triples;
  };
  const Golden cases[] = {
      {"O16", 2, 6, 340.71339955428823, -64.610022003823062},
      {"He4", 2, 6, 340.71339955428823, -3.1864715462234221},
      {"O16", 3, 9, 2174.8102180586943, -1451.9275912120786}};
  for (const Golden& g : cases) {
    KFixture f(g.ref, g.emax, g.e3max, 1.0e10);
    const op::Operator X = Random(f, Hermiticity::kAntiHermitian, 1);
    const op::Operator Y = Random(f, Hermiticity::kHermitian, 2);
    const std::vector<double> z = K0(f, X, Y, false, nullptr);
    EXPECT_NEAR(ImsrgppNorm(f.space, z) / g.norm, 1.0, 1.0e-12) << g.ref;
    const double e = TriplesReference(f.ctx, X, Y, Epsilon(f));
    EXPECT_NEAR(e / g.triples, 1.0, 1.0e-12) << g.ref << " " << g.emax;
  }
}

TEST(TriplesRef, RejectsWrongEpsilonSize) {
  KFixture f("He4", 2, 6, 1.0e10);
  const op::Operator X = Random(f, Hermiticity::kAntiHermitian, 1);
  const op::Operator Y = Random(f, Hermiticity::kHermitian, 2);
  const std::vector<double> eps(3, 1.0);
  EXPECT_ANY_THROW(TriplesReference(f.ctx, X, Y, eps));
}

}  // namespace
}  // namespace imsrg::layout3
