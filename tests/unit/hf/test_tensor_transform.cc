// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tensor operators in the Hartree-Fock basis (M15): the harmonic-oscillator
// tensor goldens of imsrg++ transformed with our HF coefficients against the
// imsrg++ HF-basis goldens (He4 from the me2j file, O16 from the Hamiltonian
// dump).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "imsrg/io/me2j.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/tensor_dump.h"
#include "imsrg/opslib/center_of_mass.h"
#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::hftest {

namespace fs = std::filesystem;
namespace {

fs::path TensorGolden(const std::string& rel) {
  return fs::path(IMSRG_GOLDEN_DIR) / "tensor" / rel;
}

/// Transforms every operator of `names` in `dir/HO` with the HF coefficients
/// of `r` and compares with `dir/HF`: every part must agree to `tol *
/// max(1, max|ref|)`. Prints the deviations.
void CompareOperators(const std::string& dir,
                      const std::shared_ptr<const basis::ModelSpace>& ms,
                      const hf::HFResult& r,
                      const std::vector<std::string>& names, double tol) {
  for (const std::string& name : names) {
    SCOPED_TRACE(dir + "/" + name);
    const op::Operator ho =
        op::ReadTensorDump(TensorGolden(dir + "/HO/" + name + ".tdump"), ms);
    const op::Operator ref =
        op::ReadTensorDump(TensorGolden(dir + "/HF/" + name + ".tdump"), ms);
    const op::Operator out = op::TransformBasis(ho, r.C.const_view());
    const op::PartMaxima dev = op::MaxAbsDifference(out, ref);
    const double ref_max = op::MaxAbs(ref).max();
    const double scale = std::max(1.0, ref_max);
    std::cout << "[tensor-transform] " << dir << " " << name << ": max|ref| "
              << ref_max << "  dev 0B " << dev.zero_body << " 1B "
              << dev.one_body << " 2B " << dev.two_body << "\n";
    EXPECT_LE(dev.zero_body, tol * scale);
    EXPECT_LE(dev.one_body, tol * scale);
    EXPECT_LE(dev.two_body, tol * scale);
  }
}

const std::vector<std::string>& HeOps() {
  static const auto* const kOps =
      new std::vector<std::string>{"E1",
                                   "E2",
                                   "E3",
                                   "M1",
                                   "Fermi",
                                   "GamowTeller",
                                   "M0nu_GT_7.72_none",
                                   "M0nu_F_7.72_none",
                                   "DGT"};
  return *kOps;
}

const std::vector<std::string>& O16Ops() {
  static const auto* const kOps =
      new std::vector<std::string>{"E2",
                                   "M1",
                                   "Fermi",
                                   "GamowTeller",
                                   "M0nu_GT_7.72_none",
                                   "M0nu_F_7.72_none"};
  return *kOps;
}

// Tolerance of the transformation on fixed inputs (design/30 section 2). Our
// HF coefficients agree with the imsrg++ ones far better than the 1e-7 of the
// scalar HF golden tests for these closed-shell cases (measured deviations
// ~1e-15), so the stricter tolerance holds.
constexpr double kTolOwnHf = 1.0e-10;

TEST(TensorTransformGolden, He4Emax2) {
  const fs::path me2j = fs::path(IMSRG_TEST_DATA_DIR) /
                        "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  if (!fs::exists(me2j)) GTEST_SKIP() << "me2j file not found";
  const fs::path hdump = TensorGolden("emax2_He4/inputs/H_HF.dump");
  const auto ms = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(hdump));
  const op::Operator v = io::ReadMe2j(me2j, {.file = {14, 28, 14}}, ms);
  const op::Operator H = opslib::TRel(ms, 4) + v;
  const hf::HFResult r = hf::Solve(H, nullptr);
  EXPECT_TRUE(r.converged);
  const op::Operator href = op::ReadOperatorDump(hdump, ms);
  std::cout << "[tensor-transform] He4 E_HF " << r.e_hf
            << "  dE_HF vs H_HF.dump zero-body " << r.e_hf - href.zero_body()
            << "\n";
  EXPECT_NEAR(r.e_hf, href.zero_body(), 1.0e-8);
  CompareOperators("emax2_He4", ms, r, HeOps(), kTolOwnHf);
}

TEST(TensorTransformGolden, O16Emax2) {
  const fs::path dir = fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16";
  const op::Operator H = LoadHamiltonian(dir);
  const hf::HFResult r = hf::Solve(H, nullptr);
  EXPECT_TRUE(r.converged);
  CompareOperators("emax2_O16", H.model_space_ptr(), r, O16Ops(), kTolOwnHf);
}

}  // namespace
}  // namespace imsrg::hftest
