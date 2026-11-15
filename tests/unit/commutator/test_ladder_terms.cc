// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The pp/hh ladder terms (comm222_pp_hhss, comm221ss and their fusion
// comm222_pp_hh_221ss) of the fast commutator: golden acceptance, comparison
// with commutator_ref on random operators, fused versus separate evaluation, an
// emax-4 timing report and the NonHermitian guard.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator_ref/commutator_ref.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg {
namespace {

using commutator::Workspace;
using commutator_test::kTol;
using imsrg::testing::RandomOperator;
using imsrg::testing::ReadGolden;
using imsrg::testing::RelativeDeviation;
using op::Operator;

using FastFn = void (*)(const Operator&, const Operator&, Operator&,
                        Workspace&);
using RefFn = void (*)(const Operator&, const Operator&, Operator&);

struct TermCase {
  const char* name;
  FastFn fast;
  RefFn ref;
};

const TermCase kTerms[] = {
    {"comm222_pp_hhss", &commutator::terms::Comm222PpHhss,
     &commutator_ref::Comm222PpHhss},
    {"comm221ss", &commutator::terms::Comm221ss, &commutator_ref::Comm221ss},
    {"comm222_pp_hh_221ss", &commutator::terms::Comm222PpHh221ss,
     &commutator_ref::Comm222PpHh221ss},
};

TEST(commutator_ladder_terms, golden) {
  for (const auto& p : commutator_test::GoldenPairs()) {
    const auto ms =
        imsrg::testing::GoldenModelSpace(commutator_test::DumpPath(p, p.y));
    Workspace ws(ms);
    const Operator X = ReadGolden(commutator_test::DumpPath(p, p.x), ms);
    const Operator Y = ReadGolden(commutator_test::DumpPath(p, p.y), ms);
    for (const auto& t : kTerms) {
      const std::string file = commutator_test::ResultPath(p, t.name);
      if (!std::filesystem::exists(imsrg::testing::GoldenPath(file))) {
        std::printf("[golden] %s %s: %s, %s: no golden file, skipped\n",
                    p.dir.c_str(), t.name, p.x.c_str(), p.y.c_str());
        continue;
      }
      Operator Z(ms, commutator::CommutatorSpec(X.spec(), Y.spec()));
      t.fast(X, Y, Z, ws);
      const double dev = RelativeDeviation(Z, ReadGolden(file, ms));
      std::printf("[golden] %s %s: %s, %s: %.3e\n", p.dir.c_str(), t.name,
                  p.x.c_str(), p.y.c_str(), dev);
      EXPECT_LE(dev, kTol) << p.dir << " " << t.name << " " << p.x << " "
                           << p.y;
    }
  }
}

TEST(commutator_ladder_terms, fast_vs_reference) {
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::HermPairs()) {
      const Operator X = RandomOperator(ms, hp.x, 11, hp.rank_x);
      const Operator Y = RandomOperator(ms, hp.y, 29, hp.rank_y);
      for (const auto& t : kTerms) {
        const auto spec = commutator::CommutatorSpec(X.spec(), Y.spec());
        Operator Z(ms, spec);
        Operator Zref(ms, spec);
        t.fast(X, Y, Z, ws);
        t.ref(X, Y, Zref);
        const double dev = RelativeDeviation(Z, Zref);
        std::printf("[fast-vs-ref] %s %s %s: %.3e\n", sc.label.c_str(),
                    hp.label, t.name, dev);
        EXPECT_LE(dev, kTol) << sc.label << " " << hp.label << " " << t.name;
      }
    }
  }
}

TEST(commutator_ladder_terms, fused_equals_separate) {
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::HermPairs()) {
      const Operator X = RandomOperator(ms, hp.x, 5, hp.rank_x);
      const Operator Y = RandomOperator(ms, hp.y, 17, hp.rank_y);
      const auto spec = commutator::CommutatorSpec(X.spec(), Y.spec());
      Operator Zf(ms, spec);
      Operator Zs(ms, spec);
      commutator::terms::Comm222PpHh221ss(X, Y, Zf, ws);
      commutator::terms::Comm222PpHhss(X, Y, Zs, ws);
      commutator::terms::Comm221ss(X, Y, Zs, ws);
      const double dev = RelativeDeviation(Zf, Zs);
      std::printf("[fused-vs-sum] %s %s: %.3e\n", sc.label.c_str(), hp.label,
                  dev);
      EXPECT_LE(dev, 1.0e-14) << sc.label << " " << hp.label;
    }
  }
}

TEST(commutator_ladder_terms, timing_emax4) {
  const auto ms = imsrg::testing::MakeModelSpace(4, "O16");
  Workspace ws(ms);
  const Operator A = RandomOperator(ms, Hermiticity::kAntiHermitian, 1);
  const Operator H = RandomOperator(ms, Hermiticity::kHermitian, 2);
  Operator Z(ms, commutator::CommutatorSpec(A.spec(), H.spec()));
  commutator::terms::Comm222PpHh221ss(A, H, Z, ws);  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  commutator::terms::Comm222PpHh221ss(A, H, Z, ws);
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("[timing] comm222_pp_hh_221ss emax 4 O16: %.1f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());
}

TEST(commutator_ladder_terms, nonhermitian_input_throws) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator X = RandomOperator(ms, Hermiticity::kNonHermitian, 3);
  const Operator Y = RandomOperator(ms, Hermiticity::kHermitian, 4);
  Operator Z(ms, op::OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
  for (const auto& t : kTerms) {
    EXPECT_THROW(t.fast(X, Y, Z, ws), NotImplementedError) << t.name;
    EXPECT_THROW(t.fast(Y, X, Z, ws), NotImplementedError) << t.name;
  }
}

}  // namespace
}  // namespace imsrg
