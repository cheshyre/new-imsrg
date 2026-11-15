// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Three-body part of the white generator (imsrg++
// ConstructGenerator_SingleRef_3body) against the goldens of
// tests/golden/n7 (H = RandomThreeBodyOperator(ctx, Hermitian, 13)).
#include <algorithm>
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
  if (e3max >= 6) EXPECT_GT(ref.three_body, 0.0);
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

}  // namespace
}  // namespace imsrg
