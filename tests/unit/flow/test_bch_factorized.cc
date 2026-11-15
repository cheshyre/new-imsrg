// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The IMSRG(3f2) correction of the BCH transform and product
// (`BchOptions::factorized`, D32): imsrg++ goldens (`refharness bch`), the
// nested_{i-2} bookkeeping against series built by hand from `Commutator` and
// `FactorizedDoubleCommutator`, and bit-identity with the plain series when
// every topology is switched off.
#include <array>
#include <cstddef>
#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/factorized.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"
#include "tests/support/commutator_test_support.h"

namespace imsrg {
namespace {

using commutator::CommutatorOptions;
using commutator::Workspace;
using flow::BchOptions;
using flow::BchReport;
using imsrg::testing::RandomOperator;
using imsrg::testing::ReadGolden;
using imsrg::testing::RelativeDeviation;
using op::Operator;

/// One golden case of tests/golden/factorized/bch_emax2_O16 (README there).
struct BchGolden {
  const char* file;
  bool transform;  ///< BchTransform(a, b) or BchProduct(a, b).
  const char* a;
  double scale_a;
  const char* b;
  double scale_b;
  bool factorized;
};

const BchGolden kBchGoldens[] = {
    {"transform_H__eta_white_x0.0929__d32", true, "H", 1.0, "eta_white", 0.0929,
     true},
    {"transform_H__eta_white_x1.86__d32", true, "H", 1.0, "eta_white", 1.86,
     true},
    {"transform_H__eta_white_x1.86__none", true, "H", 1.0, "eta_white", 1.86,
     false},
    {"product_eta_white_x0.0464__random_A_x0.000415__d32", false, "eta_white",
     0.0464, "random_A", 0.000415, true},
    {"product_eta_white_x0.0929__random_A_x0.00829__d32", false, "eta_white",
     0.0929, "random_A", 0.00829, true},
    {"product_eta_white_x0.0929__random_A_x0.00829__none", false, "eta_white",
     0.0929, "random_A", 0.00829, false},
};

std::string Input(const char* name) {
  return std::string("commutator/emax2_O16/") + name + ".dump";
}

TEST(flow_bch_factorized, golden) {
  const auto ms = imsrg::testing::GoldenModelSpace(Input("H"));
  Workspace ws(ms);
  const CommutatorOptions copts;
  for (const auto& g : kBchGoldens) {
    const Operator a = g.scale_a * ReadGolden(Input(g.a), ms);
    const Operator b = g.scale_b * ReadGolden(Input(g.b), ms);
    BchOptions opts;
    opts.factorized = g.factorized;
    BchReport report;
    const Operator z = g.transform
                           ? flow::BchTransform(a, b, opts, copts, ws, &report)
                           : flow::BchProduct(a, b, opts, copts, ws);
    const std::string file =
        std::string("factorized/bch_emax2_O16/") + g.file + ".dump";
    const double dev = RelativeDeviation(z, ReadGolden(file, ms));
    std::printf("[golden] %s: %.3e", g.file, dev);
    if (g.transform) {
      std::printf(" (%d nested, %d corrections)", report.nested_commutators,
                  report.factorized_corrections);
    }
    std::printf("\n");
    EXPECT_LE(dev, 1.0e-10) << g.file;
  }
}

/// F(eta, gamma) with the D32 options.
Operator F(const Operator& eta, const Operator& gamma, Workspace& ws) {
  Operator z;
  commutator::FactorizedDoubleCommutator(eta, gamma, z, {}, ws);
  return z;
}

TEST(flow_bch_factorized, transform_series_by_hand) {
  const auto ms = imsrg::testing::FractionalModelSpace(2, "O16");
  Workspace ws(ms);
  const CommutatorOptions copts;
  const Operator X = RandomOperator(ms, Hermiticity::kHermitian, 41);
  const Operator R = RandomOperator(ms, Hermiticity::kAntiHermitian, 42);
  const Operator omega = (0.3 / op::Norm(R)) * R;
  // nested_1 = [O, X], nested_2 = [O, n1] + F(O, X), nested_3 = [O, n2] +
  // F(O, n1), nested_4 = [O, n3] + F(O, n2) (the first F of a corrected
  // term); the series is truncated by max_iterations (no epsilon stop).
  const Operator n1 = commutator::Commutator(omega, X, copts, ws);
  const Operator n2 = commutator::Commutator(omega, n1, copts, ws) +
                      F(omega, flow::WithParticleRank(X, 2), ws);
  const Operator n3 =
      commutator::Commutator(omega, n2, copts, ws) + F(omega, n1, ws);
  const Operator n4 =
      commutator::Commutator(omega, n3, copts, ws) + F(omega, n2, ws);
  BchOptions opts;
  opts.factorized = true;
  opts.transform_threshold = 1.0e-300;
  for (const int iterations : {2, 3, 4}) {
    opts.max_iterations = iterations;
    BchReport report;
    const Operator z = flow::BchTransform(X, omega, opts, copts, ws, &report);
    Operator expected = X + n1 + 0.5 * n2;
    if (iterations >= 3) expected = expected + (1.0 / 6) * n3;
    if (iterations >= 4) expected = expected + (1.0 / 24) * n4;
    const double dev = RelativeDeviation(z, expected);
    std::printf("[by-hand] transform, %d terms: %.3e\n", iterations, dev);
    EXPECT_LE(dev, 1.0e-14) << iterations;
    EXPECT_EQ(report.nested_commutators, iterations);
    EXPECT_EQ(report.factorized_corrections, iterations - 1);
  }
}

TEST(flow_bch_factorized, product_series_by_hand) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "He6");
  Workspace ws(ms);
  const CommutatorOptions copts;
  const Operator A = RandomOperator(ms, Hermiticity::kAntiHermitian, 51);
  const Operator B = RandomOperator(ms, Hermiticity::kAntiHermitian, 52);
  const Operator X = (0.05 / op::Norm(A)) * A;
  const Operator Y = (0.4 / op::Norm(B)) * B;
  // Every threshold off: the loop adds k = 1, 2, 4, 6, 8 and stops at k = 9.
  constexpr std::array<double, 9> bernoulli = {
      1.0, -0.5, 1.0 / 6, 0.0, -1.0 / 30, 0.0, 1.0 / 42, 0.0, -1.0 / 30};
  constexpr std::array<double, 9> factorial = {
      1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0};
  std::array<Operator, 9> n;
  n[0] = X;
  n[1] = commutator::Commutator(Y, X, copts, ws);
  for (std::size_t k = 2; k <= 8; ++k)
    n[k] = commutator::Commutator(Y, n[k - 1], copts, ws) + F(Y, n[k - 2], ws);
  Operator expected =
      X + Y +
      (1.0 / 12) * (commutator::Commutator(n[1], X, copts, ws) + F(X, Y, ws));
  for (const std::size_t k : {1U, 2U, 4U, 6U, 8U})
    expected = expected + (bernoulli[k] / factorial[k]) * n[k];
  BchOptions opts;
  opts.factorized = true;
  opts.product_threshold = 1.0e-300;
  const double dev =
      RelativeDeviation(flow::BchProduct(X, Y, opts, copts, ws), expected);
  std::printf("[by-hand] product: %.3e\n", dev);
  EXPECT_LE(dev, 1.0e-14);
}

TEST(flow_bch_factorized, all_topologies_off_is_plain) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "O16");
  Workspace ws(ms);
  const CommutatorOptions copts;
  const Operator H = RandomOperator(ms, Hermiticity::kHermitian, 61);
  const Operator R = RandomOperator(ms, Hermiticity::kAntiHermitian, 62);
  const Operator omega = (0.5 / op::Norm(R)) * R;
  const Operator small = (0.05 / op::Norm(R)) * R;
  BchOptions plain;
  BchOptions off;
  off.factorized = true;
  off.factorized_terms.one_body_intermediates = false;
  BchReport report;
  EXPECT_EQ(
      RelativeDeviation(flow::BchTransform(H, omega, off, copts, ws, &report),
                        flow::BchTransform(H, omega, plain, copts, ws)),
      0.0);
  EXPECT_EQ(report.factorized_corrections, 0);  // F vanishes, skipped
  EXPECT_EQ(RelativeDeviation(flow::BchProduct(small, omega, off, copts, ws),
                              flow::BchProduct(small, omega, plain, copts, ws)),
            0.0);
}

TEST(flow_bch_factorized, rank_one_operands_skip_the_correction) {
  // F needs two-body parts; with rank-1 X and Y the corrected series equals
  // the plain one (and stays rank 1).
  const auto ms = imsrg::testing::MakeModelSpace(2, "O16");
  Workspace ws(ms);
  const CommutatorOptions copts;
  const Operator A = RandomOperator(ms, Hermiticity::kAntiHermitian, 81, 1);
  const Operator B = RandomOperator(ms, Hermiticity::kAntiHermitian, 82, 1);
  const Operator X = (0.1 / op::Norm(A)) * A;
  const Operator Y = (0.5 / op::Norm(B)) * B;
  BchOptions plain;
  BchOptions corrected;
  corrected.factorized = true;
  const Operator z = flow::BchProduct(X, Y, corrected, copts, ws);
  EXPECT_EQ(z.spec().particle_rank, 1);
  EXPECT_EQ(RelativeDeviation(z, flow::BchProduct(X, Y, plain, copts, ws)),
            0.0);
}

}  // namespace
}  // namespace imsrg
