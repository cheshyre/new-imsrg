// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// BCH transform of tensor operators with a scalar anti-Hermitian Omega (the
// Magnus step of M15) against the imsrg-devel goldens of
// tests/golden/tensor/emax{1,2}_He4/bch.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/tensor_dump.h"

namespace imsrg::flow {
namespace {

namespace fs = std::filesystem;

fs::path Golden(const std::string& rel) {
  return fs::path(IMSRG_GOLDEN_DIR) / "tensor" / rel;
}

/// Omega = scale * random_A with imsrg++'s scale (result.json, chosen so that
/// ||Omega|| = 0.3); Y from `y_file`; compares BchTransform(Y, Omega) with
/// the golden.
void Check(const std::string& dir, double scale, const std::string& y_file,
           const std::string& y_name) {
  SCOPED_TRACE(dir + " " + y_name);
  const auto ms = op::ModelSpaceForDump(
      op::ReadOperatorDumpHeader(Golden(dir + "/inputs/H_HF.dump")));
  op::Operator omega =
      op::ReadOperatorDump(Golden(dir + "/inputs/random_A.dump"), ms);
  op::Scale(omega, scale);
  EXPECT_NEAR(op::Norm(omega), 0.3, 1.0e-12);
  const op::Operator Y = op::ReadTensorDump(Golden(y_file), ms);
  commutator::Workspace ws(ms);
  BchReport report;
  op::Operator out = BchTransform(Y, omega, BchOptions{},
                                  commutator::CommutatorOptions{}, ws, &report);
  out.SyncToHost();
  const op::Operator ref = op::ReadTensorDump(
      Golden(dir + "/bch/Y_" + y_name + "_transformed.tdump"), ms);
  ASSERT_TRUE(ref.spec().SameSymmetry(out.spec()));
  const double s = std::max(1.0, op::MaxAbs(ref).max());
  const auto d = op::MaxAbsDifference(out, ref);
  std::printf(
      "tensor BCH %s %-16s nested %d max|d| 1b %.3e 2b %.3e (max %.3e)\n",
      dir.c_str(), y_name.c_str(), report.nested_commutators, d.one_body,
      d.two_body, s);
  EXPECT_TRUE(report.converged);
  EXPECT_LE(d.one_body, 1.0e-10 * s);
  EXPECT_LE(d.two_body, 1.0e-10 * s);
}

TEST(TensorBch, Emax1MatchesImsrgDevel) {
  const double scale = 0.008376855048297011;  // result.json runs.bch1
  Check("emax1_He4", scale, "emax1_He4/inputs/E2.tdump", "E2");
  Check("emax1_He4", scale, "emax1_He4/inputs/rand_2_0_0_H_3.tdump",
        "rand_2_0_0_H_3");
  Check("emax1_He4", scale, "emax1_He4/inputs/rand_1_1_0_H_4.tdump",
        "rand_1_1_0_H_4");
}

TEST(TensorBch, Emax2E2MatchesImsrgDevel) {
  Check("emax2_He4", 0.0013147746310740072, "emax2_He4/HF/E2.tdump", "E2");
}

}  // namespace
}  // namespace imsrg::flow
