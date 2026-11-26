// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The m-scheme tensor commutator (ExpandTensor, CommutatorPart with a
// TensorShift, ContractTensor) against the imsrg-devel scalar-tensor goldens of
// tests/golden/tensor/emax1_He4. This fixes the convention of the reduced
// elements relative to the Wigner-Eckart theorem (see `ExpandTensor`).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/tensor_dump.h"
#include "tests/unit/commutator_ref/tensor_mscheme_support.h"

namespace {

namespace fs = std::filesystem;
namespace msc = imsrg::commutator_ref::mscheme;
using imsrg::TwoM;
using imsrg::op::Operator;

fs::path Golden(const std::string& rel) {
  return fs::path(IMSRG_GOLDEN_DIR) / rel;
}

struct Combo {
  std::string x;  // H or A
  std::string y;  // rand_...
};

const char kIn[] = "tensor/emax1_He4/inputs/";
const char kOut[] = "tensor/emax1_He4/commutator/";

const std::vector<Combo>& Combos() {
  static const auto* const kCombos =
      new std::vector<Combo>{{"H", "rand_2_0_0_H_3"},
                             {"H", "rand_1_0_1_A_5"},
                             {"A", "rand_1_1_0_H_4"},
                             {"A", "rand_1_0_1_A_5"}};
  return *kCombos;
}

struct Inputs {
  std::shared_ptr<const imsrg::basis::ModelSpace> ms;
  Operator x;
  Operator y;
};

Inputs Load(const Combo& c) {
  const auto ms =
      imsrg::op::ModelSpaceForDump(imsrg::op::ReadOperatorDumpHeader(
          Golden(std::string(kIn) + "H_HF.dump")));
  Inputs in{
      ms,
      imsrg::op::ReadOperatorDump(
          Golden(std::string(kIn) + (c.x == "H" ? "H_HF" : "random_A") +
                 ".dump"),
          ms),
      imsrg::op::ReadTensorDump(Golden(std::string(kIn) + c.y + ".tdump"), ms)};
  return in;
}

/// The golden Z of `term`, or a zero operator with the spec of `like` if
/// imsrg++ wrote no file (a vanishing term).
Operator ReadZ(const Inputs& in, const Combo& c, const std::string& term,
               const imsrg::op::OperatorSpec& like) {
  const fs::path f = Golden(std::string(kOut) + "Z_" + term + "__" + c.x +
                            "__" + c.y + ".tdump");
  if (!fs::exists(f)) return Operator(in.ms, like);
  return imsrg::op::ReadTensorDump(f, in.ms);
}

/// Larger of the two-part deviations relative to max(1, max|ref|).
double Deviation(const Operator& z, const Operator& ref, double* d1,
                 double* d2) {
  const auto d = imsrg::op::MaxAbsDifference(z, ref);
  const double scale = std::max(1.0, imsrg::op::MaxAbs(ref).max());
  *d1 = d.one_body / scale;
  *d2 = d.two_body / scale;
  return std::max(*d1, *d2);
}

/// Parts of Z that the term maps to (n_c_mask, parts_mask of
/// `MSchemeTensorCommutator`).
struct TermMap {
  const char* term;
  unsigned n_c;
  unsigned parts;
};

// comm111 = [X1,Y1]->1; comm121 = ([X1,Y2] + [X2,Y1])->1; comm122 = same ->2;
// comm221 = [X2,Y2]->1; comm222_pp_hh + comm222_ph = [X2,Y2]->2.
const TermMap kTerms[] = {{"comm111st", 2U, 1U},
                          {"comm121st", 2U, 6U},
                          {"comm122st", 4U, 6U},
                          {"comm221st", 2U, 8U}};

TEST(MSchemeTensorGolden, AllTermsAndRanksEmax1) {
  for (const Combo& c : Combos()) {
    const Inputs in = Load(c);
    const auto zspec =
        imsrg::op::ReadTensorDump(
            Golden(std::string(kOut) + "Z_all__" + c.x + "__" + c.y + ".tdump"),
            in.ms)
            .spec();
    const int lam = in.y.spec().rank_J;
    for (const int two_mu : {0, 2, -2}) {
      if (std::abs(two_mu) > 2 * lam) continue;
      SCOPED_TRACE("X=" + c.x + " Y=" + c.y + " 2mu=" + std::to_string(two_mu));
      // Total, and per output rank.
      const Operator ref = ReadZ(in, c, "all", zspec);
      for (const unsigned n_c : {6U, 2U, 4U}) {
        const Operator z = imsrg::testing::MSchemeTensorCommutator(
            in.x, in.y, zspec, TwoM{two_mu}, {}, n_c, 15U);
        Operator want = ref;
        // Restrict the reference to the same output ranks.
        if ((n_c & 2U) == 0U) want.one_body().set_zero();
        if ((n_c & 4U) == 0U && want.two_body().allocated()) {
          want.two_body().set_zero();
        }
        double d1 = 0.0;
        double d2 = 0.0;
        const double dev = Deviation(z, want, &d1, &d2);
        std::printf(
            "[mscheme-tensor] %s %-16s 2mu=%+d ranks=%u: 1b %.2e 2b %.2e\n",
            c.x.c_str(), c.y.c_str(), two_mu, n_c, d1, d2);
        EXPECT_LE(dev, 1.0e-10);
      }
      // Per term (the 222 terms are summed: pp_hh + ph = [2,2] -> 2).
      for (const TermMap& t : kTerms) {
        const Operator z = imsrg::testing::MSchemeTensorCommutator(
            in.x, in.y, zspec, TwoM{two_mu}, {}, t.n_c, t.parts);
        Operator want = ReadZ(in, c, t.term, zspec);
        double d1 = 0.0;
        double d2 = 0.0;
        const double dev = Deviation(z, want, &d1, &d2);
        std::printf(
            "[mscheme-tensor] %s %-16s 2mu=%+d %-10s: 1b %.2e 2b %.2e\n",
            c.x.c_str(), c.y.c_str(), two_mu, t.term, d1, d2);
        EXPECT_LE(dev, 1.0e-10) << t.term;
      }
      {
        const Operator z = imsrg::testing::MSchemeTensorCommutator(
            in.x, in.y, zspec, TwoM{two_mu}, {}, 4U, 8U);
        Operator want = ReadZ(in, c, "comm222_pp_hhst", zspec);
        const Operator ph = ReadZ(in, c, "comm222_phst", zspec);
        want.two_body().matrix().buffer().SyncToHost();
        auto& wm = want.two_body().matrix();
        const auto& pm = ph.two_body().matrix();
        for (std::size_t k = 0; k < wm.num_blocks(); ++k) {
          auto wb = wm.host_block(k);
          const auto pb = pm.host_block(k);
          for (std::int64_t i = 0; i < wm.block_info(k).rows; ++i) {
            for (std::int64_t j = 0; j < wm.block_info(k).cols; ++j) {
              wb(i, j) += pb(i, j);
            }
          }
        }
        // Host writes into `want`: publish them (Deviation may sync to host).
        want.two_body().matrix().buffer().SyncToDevice();
        double d1 = 0.0;
        double d2 = 0.0;
        const double dev = Deviation(z, want, &d1, &d2);
        // The one-body part of z is zero by the mask; compare 2B only.
        std::printf(
            "[mscheme-tensor] %s %-16s 2mu=%+d 222(pp_hh+ph): 2b %.2e\n",
            c.x.c_str(), c.y.c_str(), two_mu, d2);
        (void)dev;
        EXPECT_LE(d2, 1.0e-10);
      }
    }
  }
}

// Documentation of the convention search: the Z_all of every combo with each
// of the eight phase gauges. Only the default gauge reproduces the goldens.
TEST(MSchemeTensorGolden, OnlyTheDefaultGaugeReproducesTheGoldens) {
  int matching = 0;
  for (const bool g1 : {false, true}) {
    for (const bool g2 : {false, true}) {
      for (const bool gr : {false, true}) {
        msc::TensorGauge gauge;
        gauge.one_body_phase = g1;
        gauge.two_body_phase = g2;
        gauge.two_body_rank_phase = gr;
        double worst = 0.0;
        for (const Combo& c : Combos()) {
          const Inputs in = Load(c);
          const Operator ref =
              imsrg::op::ReadTensorDump(Golden(std::string(kOut) + "Z_all__" +
                                               c.x + "__" + c.y + ".tdump"),
                                        in.ms);
          const Operator z = imsrg::testing::MSchemeTensorCommutator(
              in.x, in.y, ref.spec(), TwoM{0}, gauge);
          double d1 = 0.0;
          double d2 = 0.0;
          worst = std::max(worst, Deviation(z, ref, &d1, &d2));
        }
        std::printf("[mscheme-tensor] gauge 1B %d 2B %d rank %d: worst %.3e\n",
                    g1 ? 1 : 0, g2 ? 1 : 0, gr ? 1 : 0, worst);
        const bool is_default = !g1 && !g2 && !gr;
        if (worst <= 1.0e-10) ++matching;
        EXPECT_EQ(worst <= 1.0e-10, is_default);
      }
    }
  }
  EXPECT_EQ(matching, 1);
}

}  // namespace
