// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The simple terms (110, 220, 111, 121, 122) of the fast commutator: golden
// acceptance, term-by-term comparison with commutator_ref on random operators,
// and the NonHermitian guard.
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
    {"comm110ss", &commutator::terms::Comm110ss, &commutator_ref::Comm110ss},
    {"comm220ss", &commutator::terms::Comm220ss, &commutator_ref::Comm220ss},
    {"comm111ss", &commutator::terms::Comm111ss, &commutator_ref::Comm111ss},
    {"comm121ss", &commutator::terms::Comm121ss, &commutator_ref::Comm121ss},
    {"comm122ss", &commutator::terms::Comm122ss, &commutator_ref::Comm122ss},
};

TEST(commutator_simple_terms, golden) {
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

TEST(commutator_simple_terms, fast_vs_reference) {
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

TEST(commutator_simple_terms, nonhermitian_input_throws) {
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
