// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The factorized double commutator of IMSRG(3f2) (D32, one-body intermediates
// only): golden acceptance against imsrg++ (`refharness factorized d32`), the
// fast GEMM form against the plain-loop transcription in commutator_ref on
// closed-shell and fractional references, the options and preconditions, and
// an emax-4 timing report next to a full IMSRG(2) commutator.
#include <chrono>
#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/factorized.h"
#include "imsrg/commutator_ref/factorized_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg {
namespace {

using commutator::FactorizedDoubleCommutator;
using commutator::FactorizedOptions;
using commutator::Workspace;
using imsrg::testing::RandomOperator;
using imsrg::testing::ReadGolden;
using imsrg::testing::RelativeDeviation;
using op::Operator;

/// A golden case: directory of the inputs, of the result, eta and gamma.
struct FactorizedGolden {
  const char* input_dir;
  const char* result_dir;
  const char* eta;
  const char* gamma;
};

const FactorizedGolden kGoldens[] = {
    {"factorized/emax2_He6", "factorized/emax2_He6", "random_A1", "random_H1"},
    {"factorized/emax2_He6", "factorized/emax2_He6", "random_A1", "random_A2"},
    {"commutator/emax2_O16", "factorized/emax2_O16", "eta_white", "H"},
    {"commutator/emax2_O16", "factorized/emax2_O16", "random_A", "random_H"},
    {"commutator/emax2_O16", "factorized/emax2_O16", "eta_white", "random_A"},
};

std::string Dump(const char* dir, const std::string& name) {
  return std::string(dir) + "/" + name + ".dump";
}

/// Fast F(eta, gamma) with the D32 options into a new operator.
Operator Fast(const Operator& eta, const Operator& gamma, Workspace& ws) {
  Operator Z;
  FactorizedDoubleCommutator(eta, gamma, Z, FactorizedOptions{}, ws);
  return Z;
}

/// commutator_ref F(eta, gamma) into a new operator with gamma's hermiticity.
Operator Reference(const Operator& eta, const Operator& gamma) {
  Operator Z(eta.model_space_ptr(),
             op::OperatorSpec::Scalar(2, gamma.hermiticity()));
  commutator_ref::FactorizedDoubleCommutator(eta, gamma, Z);
  return Z;
}

TEST(commutator_factorized, golden) {
  for (const auto& g : kGoldens) {
    const auto ms = imsrg::testing::GoldenModelSpace(Dump(g.input_dir, g.eta));
    Workspace ws(ms);
    const Operator eta = ReadGolden(Dump(g.input_dir, g.eta), ms);
    const Operator gamma = ReadGolden(Dump(g.input_dir, g.gamma), ms);
    const Operator Z = Fast(eta, gamma, ws);
    const std::string file =
        Dump(g.result_dir, std::string("Z_d32__") + g.eta + "__" + g.gamma);
    const double dev = RelativeDeviation(Z, ReadGolden(file, ms));
    std::printf("[golden] %s d32(%s, %s): %.3e\n", g.result_dir, g.eta, g.gamma,
                dev);
    EXPECT_LE(dev, commutator_test::kTol) << file;
  }
}

TEST(commutator_factorized, fast_vs_reference) {
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    const Operator eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 13);
    for (const Hermiticity hg :
         {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
      const Operator gamma = RandomOperator(ms, hg, 31);
      const double dev =
          RelativeDeviation(Fast(eta, gamma, ws), Reference(eta, gamma));
      std::printf("[fast-vs-ref] %s gamma %s: %.3e\n", sc.label.c_str(),
                  hg == Hermiticity::kHermitian ? "H" : "A", dev);
      EXPECT_LE(dev, 1.0e-12) << sc.label;
    }
  }
}

TEST(commutator_factorized, reuses_and_overwrites_z) {
  const auto ms = imsrg::testing::FractionalModelSpace(2, "O16");
  Workspace ws(ms);
  const Operator eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 3);
  const Operator h = RandomOperator(ms, Hermiticity::kHermitian, 4);
  const Operator a = RandomOperator(ms, Hermiticity::kAntiHermitian, 5);
  const Operator first = Fast(eta, h, ws);
  Operator Z = first;
  const double* storage = Z.two_body().matrix().buffer().device_data();
  FactorizedDoubleCommutator(eta, h, Z, FactorizedOptions{}, ws);
  EXPECT_EQ(Z.two_body().matrix().buffer().device_data(), storage);
  EXPECT_EQ(RelativeDeviation(Z, first), 0.0);
  // Another hermiticity of gamma replaces Z by an operator of the new spec.
  FactorizedDoubleCommutator(eta, a, Z, FactorizedOptions{}, ws);
  EXPECT_EQ(Z.hermiticity(), Hermiticity::kAntiHermitian);
  EXPECT_EQ(RelativeDeviation(Z, Fast(eta, a, ws)), 0.0);
}

TEST(commutator_factorized, scaling_and_linearity) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "He6");
  Workspace ws(ms);
  const Operator eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 7);
  const Operator g1 = RandomOperator(ms, Hermiticity::kHermitian, 8);
  const Operator g2 = RandomOperator(ms, Hermiticity::kHermitian, 9);
  const Operator f1 = Fast(eta, g1, ws);
  // F(2 eta, gamma) = 4 F(eta, gamma).
  EXPECT_LE(RelativeDeviation(Fast(2.0 * eta, g1, ws), 4.0 * f1), 1.0e-14);
  // F(eta, g1 + 2 g2) = F(eta, g1) + 2 F(eta, g2).
  EXPECT_LE(RelativeDeviation(Fast(eta, g1 + 2.0 * g2, ws),
                              f1 + 2.0 * Fast(eta, g2, ws)),
            1.0e-13);
}

TEST(commutator_factorized, options_and_trivial_inputs) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "O16");
  Workspace ws(ms);
  const Operator eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 21);
  const Operator h = RandomOperator(ms, Hermiticity::kHermitian, 22);
  Operator Z;
  FactorizedOptions none;
  none.one_body_intermediates = false;
  FactorizedDoubleCommutator(eta, h, Z, none, ws);
  EXPECT_EQ(op::Norm(Z), 0.0);
  EXPECT_EQ(Z.hermiticity(), Hermiticity::kHermitian);
  // Every topology needs eta_2 and gamma_2.
  const Operator eta1 =
      RandomOperator(ms, Hermiticity::kAntiHermitian, 23, /*particle_rank=*/1);
  FactorizedDoubleCommutator(eta1, h, Z, FactorizedOptions{}, ws);
  EXPECT_EQ(op::Norm(Z), 0.0);
  // A zero eta (allocated two-body part) gives zero.
  FactorizedDoubleCommutator(0.0 * eta, h, Z, FactorizedOptions{}, ws);
  EXPECT_EQ(op::Norm(Z), 0.0);
}

TEST(commutator_factorized, precomputed_eta_is_bit_identical) {
  const auto ms = imsrg::testing::FractionalModelSpace(2, "O16");
  Workspace ws(ms);
  const Operator eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 71);
  const commutator::FactorizedEta pre(eta, ws);
  EXPECT_EQ(&pre.eta(), &eta);
  // Several gammas with one set of eta intermediates (the BCH use), with a
  // plain call for another eta in between (the workspace scratch is reused).
  const Operator other = RandomOperator(ms, Hermiticity::kAntiHermitian, 72);
  for (const Hermiticity hg :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    const Operator gamma = RandomOperator(ms, hg, 73);
    const Operator plain = Fast(eta, gamma, ws);
    (void)Fast(other, gamma, ws);
    Operator Z;
    FactorizedDoubleCommutator(pre, gamma, Z, FactorizedOptions{}, ws);
    EXPECT_EQ(RelativeDeviation(Z, plain), 0.0);
  }
}

TEST(commutator_factorized, rejected_inputs) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator a = RandomOperator(ms, Hermiticity::kAntiHermitian, 1);
  const Operator h = RandomOperator(ms, Hermiticity::kHermitian, 2);
  const Operator n = RandomOperator(ms, Hermiticity::kNonHermitian, 3);
  Operator Z;
  EXPECT_THROW(FactorizedDoubleCommutator(h, h, Z, FactorizedOptions{}, ws),
               NotImplementedError);
  EXPECT_THROW(FactorizedDoubleCommutator(n, h, Z, FactorizedOptions{}, ws),
               NotImplementedError);
  EXPECT_THROW(FactorizedDoubleCommutator(a, n, Z, FactorizedOptions{}, ws),
               NotImplementedError);
  FactorizedOptions both;
  both.two_body_intermediates = true;
  EXPECT_THROW(FactorizedDoubleCommutator(a, h, Z, both, ws),
               NotImplementedError);
  Operator alias = a;
  EXPECT_THROW(
      FactorizedDoubleCommutator(a, alias, alias, FactorizedOptions{}, ws),
      InvariantError);
}

TEST(commutator_factorized, timing_emax3) {
  const auto ms = imsrg::testing::MakeModelSpace(3, "O16");
  Workspace ws(ms);
  const Operator A = RandomOperator(ms, Hermiticity::kAntiHermitian, 1);
  const Operator H = RandomOperator(ms, Hermiticity::kHermitian, 2);
  Operator F;
  Operator C;
  FactorizedDoubleCommutator(A, H, F, FactorizedOptions{}, ws);  // warm-up
  commutator::Commutator(A, H, C, commutator::CommutatorOptions{}, ws);
  const auto t0 = std::chrono::steady_clock::now();
  FactorizedDoubleCommutator(A, H, F, FactorizedOptions{}, ws);
  const auto t1 = std::chrono::steady_clock::now();
  commutator::Commutator(A, H, C, commutator::CommutatorOptions{}, ws);
  const auto t2 = std::chrono::steady_clock::now();
  const double tf = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double tc = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::printf(
      "[timing] emax 3 O16: factorized %.1f ms, IMSRG(2) commutator %.1f ms "
      "(ratio %.2f)\n",
      tf, tc, tf / tc);
}

}  // namespace
}  // namespace imsrg
