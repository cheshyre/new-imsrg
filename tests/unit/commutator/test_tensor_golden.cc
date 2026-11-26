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

/// The term selection of imsrg++'s term names, on top of `base` (thresholds
/// and compat switches).
TensorOptions TermOptions(const std::string& term,
                          const TensorOptions& base = TensorOptions{}) {
  TensorOptions o = base;
  o.comm111 = o.comm121 = o.comm122 = o.comm221 = o.comm222_pp_hh =
      o.comm222_ph = false;
  if (term == "comm111st") o.comm111 = true;
  if (term == "comm121st") o.comm121 = true;
  if (term == "comm122st") o.comm122 = true;
  if (term == "comm221st") o.comm221 = true;
  if (term == "comm222_pp_hhst") o.comm222_pp_hh = true;
  if (term == "comm222_pp_hh_221st") o.comm221 = o.comm222_pp_hh = true;
  if (term == "comm222_phst") o.comm222_ph = true;
  if (term == "all") o = base;
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
  TensorOptions base{};  ///< Threshold and compat switches of every term.
  /// The single-term dumps hold only the upper one-body triangle (a <= b;
  /// the fork's terms fill the lower one in CommutatorScalarTensor's final
  /// Symmetrize): compare the one-body part of single terms there only.
  bool upper_one_body_terms = false;
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
    op::Operator Z = TensorCommutator(X, Y, TermOptions(term, c.base), ws);
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
    if (c.upper_one_body_terms && term != "all") {
      const std::int32_t n = Z.one_body().num_orbits();
      for (std::int32_t a = 0; a < n; ++a)
        for (std::int32_t b = 0; b < a; ++b) {
          Z.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, 0.0);
          ref.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, 0.0);
        }
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

// The fork's 9j skip threshold (1e-8, design/50 section 5) as an option: the
// result moves by far less than the 1e-4 reproduction target.
TEST(TensorCommutatorGolden, ForkNineJThresholdIsAnOption) {
  const std::string in = "tensor/emax1_He4/inputs/";
  const auto ms = op::ModelSpaceForDump(
      op::ReadOperatorDumpHeader(Golden(in + "H_HF.dump")));
  const op::Operator X = op::ReadOperatorDump(Golden(in + "H_HF.dump"), ms);
  const op::Operator Y =
      op::ReadTensorDump(Golden(in + "rand_2_0_0_H_3.tdump"), ms);
  TensorWorkspace ws(ms);
  TensorOptions devel;
  TensorOptions fork;
  fork.ninej_threshold = 1.0e-8;
  const op::Operator a = TensorCommutator(X, Y, devel, ws);
  const op::Operator b = TensorCommutator(X, Y, fork, ws);
  const double d = op::MaxAbsDifference(a, b).max();
  std::printf("9j threshold 1e-8 vs 1e-10: max|dZ| %.3e\n", d);
  EXPECT_LE(d, 1.0e-7 * std::max(1.0, op::MaxAbs(a).max()));
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

// The terms of the 79e9330 fork (tests/golden/tensor/fork-79e9330, generated
// by tools/reference/gen_golden_tensor_fork.sh) with the fork-compat options
// (9j threshold 1e-8, the odd-rank phase errors of the fork's comm222_phst,
// QUESTIONS.md AL-F1): random anti-Hermitian scalar X, random Hermitian Y of
// rank 1, 3, 1 (odd parity) and 2 (even control) with a two-body part.
TEST(TensorCommutatorGolden, Fork79e9330CompatRandomTensors) {
  const std::string dir = "tensor/fork-79e9330/emax2_He4/";
  const std::vector<std::string> terms = {"comm111st",    "comm121st",
                                          "comm122st",    "comm222_pp_hh_221st",
                                          "comm222_phst", "all"};
  TensorOptions fork;
  fork.ninej_threshold = 1.0e-8;
  fork.fork_ph_missing_rank_phase = true;
  for (const char* y : {"rand_1_0_0_H_4", "rand_3_0_0_H_5", "rand_1_0_1_H_6",
                        "rand_2_0_0_H_3"}) {
    RunCase(Case{dir + "commutator", "A", dir + "inputs/random_A.dump", y,
                 dir + "inputs/" + y + ".tdump", dir + "inputs/random_A.dump",
                 terms, fork, true});
  }
}

// Without the compat switch the new code differs from the fork for odd rank
// only in comm222_phst: the even-rank control and the other terms agree with
// the default phases.
TEST(TensorCommutatorGolden, Fork79e9330DefaultPathEvenRankAndNonPhTerms) {
  const std::string dir = "tensor/fork-79e9330/emax2_He4/";
  TensorOptions fork_threshold;
  fork_threshold.ninej_threshold = 1.0e-8;
  RunCase(Case{dir + "commutator",
               "A",
               dir + "inputs/random_A.dump",
               "rand_2_0_0_H_3",
               dir + "inputs/rand_2_0_0_H_3.tdump",
               dir + "inputs/random_A.dump",
               {"comm222_phst", "all"},
               fork_threshold,
               true});
  for (const char* y : {"rand_1_0_0_H_4", "rand_3_0_0_H_5"}) {
    RunCase(Case{dir + "commutator",
                 "A",
                 dir + "inputs/random_A.dump",
                 y,
                 dir + "inputs/" + y + ".tdump",
                 dir + "inputs/random_A.dump",
                 {"comm111st", "comm121st", "comm122st", "comm222_pp_hh_221st"},
                 fork_threshold,
                 true});
  }
}

}  // namespace
}  // namespace imsrg::commutator
