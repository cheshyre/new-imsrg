// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Three-body part of the white generator (imsrg++
// ConstructGenerator_SingleRef_3body) against the goldens of
// tests/golden/n7 (H = RandomThreeBodyOperator(ctx, Hermitian, 13)).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/flow/generator.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg {
namespace {

using flow::GeneratorKind;
using flow::GeneratorOptions;
using op::Operator;

void CheckCase(const char* reference, int e3max, const char* file) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  const commutator::ThreeBodyContext ctx(
      testing::MakeThreeBodySpace(2, reference, cuts));
  const Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 13);
  const std::string path =
      std::string(IMSRG_GOLDEN_DIR) + "/n7/" + reference + "_emax2/" + file;
  const Operator eta_ref = op::ReadOperatorDump(path, ctx.model_space_ptr());
  GeneratorOptions opts;
  opts.kind = GeneratorKind::kWhite;
  const Operator eta = flow::ComputeGenerator(H, opts);
  ASSERT_TRUE(eta.three_body().has_value());
  const op::PartMaxima ref = op::MaxAbs(eta_ref);
  const op::PartMaxima dev = op::MaxAbsDifference(eta, eta_ref);
  const double tol = 1.0e-10 * std::max(1.0, ref.max());
  std::printf(
      "[generator3 %s e3max %d] max|d eta|: 1B %.3e 2B %.3e 3B %.3e (tol %.3e, "
      "max|eta3| %.3e)\n",
      reference, e3max, dev.one_body, dev.two_body, dev.three_body, tol,
      ref.three_body);
  EXPECT_LE(dev.one_body, tol);
  EXPECT_LE(dev.two_body, tol);
  EXPECT_LE(dev.three_body, tol);
  if (e3max >= 6) {
    EXPECT_GT(ref.three_body, 0.0);
  }
}

TEST(Generator3, He4E3max3) { CheckCase("He4", 3, "eta_white.dump"); }
TEST(Generator3, He4E3max6) { CheckCase("He4", 6, "eta_white_e3max6.dump"); }
TEST(Generator3, O16E3max3) { CheckCase("O16", 3, "eta_white.dump"); }
TEST(Generator3, O16E3max6) { CheckCase("O16", 6, "eta_white_e3max6.dump"); }

TEST(Generator3, ShellModelRank3IsNotImplemented) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = 3;
  const commutator::ThreeBodyContext ctx(
      testing::MakeThreeBodySpace(2, "O16", cuts));
  const Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 13);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::kShellModelAtan;
  EXPECT_THROW(flow::ComputeGenerator(H, opts), NotImplementedError);
}

TEST(Generator3, Only2bEtaHasNoThreeBodyPart) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = 6;
  const commutator::ThreeBodyContext ctx(
      testing::MakeThreeBodySpace(2, "He4", cuts));
  const Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 13);
  GeneratorOptions opts;
  opts.kind = GeneratorKind::kWhite;
  opts.only_2b_eta = true;
  const Operator eta = flow::ComputeGenerator(H, opts);
  EXPECT_EQ(op::MaxAbs(eta).three_body, 0.0);
}

// D49 / M10-Q60: Delta3 has the sign-keeping floor (imsrg++: none, inf/NaN).
TEST(Generator3, VanishingDelta3IsFloored) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = 6;
  const commutator::ThreeBodyContext ctx(
      testing::MakeThreeBodySpace(2, "He4", cuts));
  Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 13);
  H.one_body().set_zero();  // every Delta3 is exactly zero
  H.SyncToDevice();
  GeneratorOptions opts;
  opts.kind = GeneratorKind::kWhite;
  opts.denominator_cutoff = 1.0e-3;
  const OrbitIndex o0{0};
  {
    const flow::Denominators den(H, opts);
    EXPECT_EQ(den.three_body(o0, o0, o0, o0, o0, o0), 1.0e-3);
  }
  // Delta3 = f_00 - 0 = -1e-9 and +1e-9: floored to -+cutoff, sign kept.
  H.one_body().set(o0, o0, -1.0e-9);
  H.SyncToDevice();
  {
    const flow::Denominators den(H, opts);
    EXPECT_EQ(
        den.three_body(o0, o0, o0, OrbitIndex{1}, OrbitIndex{1}, OrbitIndex{1}),
        -1.0e-3);
    EXPECT_EQ(
        den.three_body(OrbitIndex{1}, OrbitIndex{1}, OrbitIndex{1}, o0, o0, o0),
        1.0e-3);
  }
  H.one_body().set_zero();
  H.SyncToDevice();
  const Operator eta_a = flow::ComputeGenerator(H, opts);
  opts.denominator_cutoff = 1.0e-2;
  const Operator eta_b = flow::ComputeGenerator(H, opts);
  const op::PartMaxima ma = op::MaxAbs(eta_a);
  const op::PartMaxima mb = op::MaxAbs(eta_b);
  std::printf("[generator3 floor] max|eta3| %.6e (cutoff 1e-3) %.6e (1e-2)\n",
              ma.three_body, mb.three_body);
  ASSERT_TRUE(std::isfinite(ma.three_body));
  ASSERT_GT(ma.three_body, 0.0);
  // eta3 = H3 / cutoff on the ppp-hhh block: scales as 1/cutoff.
  EXPECT_NEAR(ma.three_body * 1.0e-3, mb.three_body * 1.0e-2,
              1.0e-10 * ma.three_body * 1.0e-3);
}

}  // namespace
}  // namespace imsrg
