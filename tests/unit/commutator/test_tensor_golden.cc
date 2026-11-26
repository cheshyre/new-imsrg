// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Scalar-tensor commutator terms against the imsrg-devel goldens of
// tests/golden/tensor (M15 acceptance, design/30 section 2: max|dZ| <= 1e-10
// max(1, max|Z|)).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/tensor_dump.h"

namespace imsrg::commutator {
namespace {

namespace fs = std::filesystem;

fs::path Golden(const std::string& rel) {
  return fs::path(IMSRG_GOLDEN_DIR) / rel;
}

/// The term selection of imsrg++'s term names.
TensorOptions TermOptions(const std::string& term) {
  TensorOptions o;
  o.comm111 = o.comm121 = o.comm122 = o.comm221 = o.comm222_pp_hh =
      o.comm222_ph = false;
  if (term == "comm111st") o.comm111 = true;
  if (term == "comm121st") o.comm121 = true;
  if (term == "comm122st") o.comm122 = true;
  if (term == "comm221st") o.comm221 = true;
  if (term == "comm222_pp_hhst") o.comm222_pp_hh = true;
  if (term == "comm222_pp_hh_221st") o.comm221 = o.comm222_pp_hh = true;
  if (term == "comm222_phst") o.comm222_ph = true;
  if (term == "all") o = TensorOptions{};
  return o;
}

struct Case {
  std::string dir;     ///< Golden directory of the Z dumps.
  std::string x_name;  ///< H or A.
  std::string x_file;  ///< OperatorDump v1 of X.
  std::string y_name;
  std::string y_file;  ///< tdump of Y.
  std::string space;   ///< Dump whose header defines the model space.
  std::vector<std::string> terms;
};

void RunCase(const Case& c) {
  const auto ms =
      op::ModelSpaceForDump(op::ReadOperatorDumpHeader(Golden(c.space)));
  const op::Operator X = op::ReadOperatorDump(Golden(c.x_file), ms);
  const op::Operator Y = op::ReadTensorDump(Golden(c.y_file), ms);
  TensorWorkspace ws(ms);
  for (const auto& term : c.terms) {
    SCOPED_TRACE(c.dir + " " + term + " X=" + c.x_name + " Y=" + c.y_name);
    const fs::path zf = Golden(c.dir) / ("Z_" + term + "__" + c.x_name + "__" +
                                         c.y_name + ".tdump");
    op::Operator Z = TensorCommutator(X, Y, TermOptions(term), ws);
    Z.SyncToHost();
    if (!fs::exists(zf)) {
      // imsrg++ wrote no file: the term vanishes identically.
      EXPECT_LE(op::MaxAbs(Z).max(), 1.0e-10) << "expected a zero term";
      continue;
    }
    op::Operator ref = op::ReadTensorDump(zf, ms);
    ASSERT_TRUE(ref.spec().SameSymmetry(Z.spec()));
    if (ref.spec().particle_rank != Z.spec().particle_rank ||
        ref.hermiticity() != Z.hermiticity()) {
      // compare the parts present in both
      op::Operator tmp(ms, Z.spec());
      tmp.one_body() = ref.one_body();
      if (ref.two_body().allocated() && tmp.two_body().allocated())
        tmp.two_body().matrix().buffer() = ref.two_body().matrix().buffer();
      tmp.one_body().set_hermiticity(Z.hermiticity());
      ref = tmp;
    }
    const double scale = std::max(1.0, op::MaxAbs(ref).max());
    const auto d = op::MaxAbsDifference(Z, ref);
    std::printf(
        "tensor golden %-22s X=%s Y=%-16s max|dZ| 1b %.3e 2b %.3e "
        "(max|Z| %.3e)\n",
        term.c_str(), c.x_name.c_str(), c.y_name.c_str(), d.one_body,
        d.two_body, scale);
    EXPECT_LE(d.one_body, 1.0e-10 * scale);
    EXPECT_LE(d.two_body, 1.0e-10 * scale);
  }
}

const std::vector<std::string> kAllTerms = {
    "comm111st",       "comm121st",           "comm122st",    "comm221st",
    "comm222_pp_hhst", "comm222_pp_hh_221st", "comm222_phst", "all"};

TEST(TensorCommutatorGolden, Emax1RandomTensors) {
  const std::string in = "tensor/emax1_He4/inputs/";
  const std::vector<std::pair<std::string, std::string>> combos = {
      {"H", "rand_2_0_0_H_3"},
      {"H", "rand_1_0_1_A_5"},
      {"A", "rand_1_1_0_H_4"},
      {"A", "rand_1_0_1_A_5"}};
  for (const auto& [x, y] : combos) {
    RunCase(Case{"tensor/emax1_He4/commutator", x,
                 in + (x == "H" ? "H_HF.dump" : "random_A.dump"), y,
                 in + y + ".tdump", in + "H_HF.dump", kAllTerms});
  }
}

TEST(TensorCommutatorGolden, Emax2OneBodyOperators) {
  const std::vector<std::string> terms = {"comm111st", "comm121st", "all"};
  for (const char* y : {"E2", "GamowTeller"}) {
    RunCase(Case{"tensor/emax2_He4/commutator", "H",
                 "tensor/emax2_He4/inputs/H_HF.dump", y,
                 std::string("tensor/emax2_He4/HF/") + y + ".tdump",
                 "tensor/emax2_He4/inputs/H_HF.dump", terms});
    RunCase(Case{"tensor/emax2_O16/commutator", "H", "hf/emax2_O16_2N/HNO.dump",
                 y, std::string("tensor/emax2_O16/HF/") + y + ".tdump",
                 "hf/emax2_O16_2N/HNO.dump", terms});
  }
  RunCase(Case{"tensor/emax2_He4/commutator", "A",
               "tensor/emax2_He4/inputs/random_A.dump", "E2",
               "tensor/emax2_He4/HF/E2.tdump",
               "tensor/emax2_He4/inputs/H_HF.dump", terms});
}

}  // namespace
}  // namespace imsrg::commutator
