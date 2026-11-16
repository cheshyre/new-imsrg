// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The particle-hole (Pandya) term comm222_phss of the fast commutator: golden
// acceptance, comparison with commutator_ref on random operators, a timing
// print (the general path is tested in test_general.cc).
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/pandya.h"
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

TEST(commutator_particle_hole, golden) {
  for (const auto& p : commutator_test::GoldenPairs()) {
    const auto ms =
        imsrg::testing::GoldenModelSpace(commutator_test::DumpPath(p, p.y));
    Workspace ws(ms);
    const Operator X = ReadGolden(commutator_test::DumpPath(p, p.x), ms);
    const Operator Y = ReadGolden(commutator_test::DumpPath(p, p.y), ms);
    const std::string file = commutator_test::ResultPath(p, "comm222_phss");
    if (!std::filesystem::exists(imsrg::testing::GoldenPath(file))) {
      std::printf("[golden] %s comm222_phss: %s, %s: no golden file, skipped\n",
                  p.dir.c_str(), p.x.c_str(), p.y.c_str());
      continue;
    }
    Operator Z(ms, commutator::CommutatorSpec(X.spec(), Y.spec()));
    commutator::terms::Comm222Phss(X, Y, Z, ws);
    const double dev = RelativeDeviation(Z, ReadGolden(file, ms));
    std::printf("[golden] %s comm222_phss: %s, %s: %.3e\n", p.dir.c_str(),
                p.x.c_str(), p.y.c_str(), dev);
    EXPECT_LE(dev, kTol) << p.dir << " " << p.x << " " << p.y;
  }
}

TEST(commutator_particle_hole, fast_vs_reference) {
  double worst = 0.0;
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::HermPairs()) {
      const Operator X = RandomOperator(ms, hp.x, 11, hp.rank_x);
      const Operator Y = RandomOperator(ms, hp.y, 29, hp.rank_y);
      const auto spec = commutator::CommutatorSpec(X.spec(), Y.spec());
      Operator Z(ms, spec);
      Operator Zref(ms, spec);
      commutator::terms::Comm222Phss(X, Y, Z, ws);
      commutator_ref::Comm222Phss(X, Y, Zref);
      const double dev = RelativeDeviation(Z, Zref);
      worst = dev > worst ? dev : worst;
      std::printf("[fast-vs-ref] %s %s comm222_phss: %.3e\n", sc.label.c_str(),
                  hp.label, dev);
      EXPECT_LE(dev, kTol) << sc.label << " " << hp.label;
    }
  }
  std::printf("[fast-vs-ref] comm222_phss max deviation: %.3e\n", worst);
}

TEST(commutator_particle_hole, Timing) {
  const auto ms = imsrg::testing::MakeModelSpace(4, "O16");
  Workspace ws(ms);
  const Operator A = RandomOperator(ms, Hermiticity::kAntiHermitian, 5);
  const Operator H = RandomOperator(ms, Hermiticity::kHermitian, 6);
  Operator Z(ms, commutator::CommutatorSpec(A.spec(), H.spec()));
  commutator::terms::Comm222Phss(A, H, Z, ws);  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 20; ++i) commutator::terms::Comm222Phss(A, H, Z, ws);
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("[timing] comm222_phss emax 4 O16: %.3f ms per call\n",
              1.0e3 * std::chrono::duration<double>(t1 - t0).count() / 20.0);
  // The term accumulates into Z, so compare a single fresh call with the
  // reference.
  Operator Zfast(ms, commutator::CommutatorSpec(A.spec(), H.spec()));
  Operator Zref(ms, commutator::CommutatorSpec(A.spec(), H.spec()));
  const Operator zero(ms, commutator::CommutatorSpec(A.spec(), H.spec()));
  commutator::terms::Comm222Phss(A, H, Zfast, ws);
  commutator_ref::Comm222Phss(A, H, Zref);
  const double dev = RelativeDeviation(Zfast, Zref);
  const double mag = RelativeDeviation(Zref, zero);
  std::printf("[timing] emax 4 fast-vs-ref: %.3e (reference magnitude %.3e)\n",
              dev, mag);
  EXPECT_LE(dev, kTol);
  const auto pd = commutator::detail::BuildPandyaData(*ms);
  std::printf("[timing] Pandya scratch at emax 4: %.2f MB\n",
              static_cast<double>(pd.scratch_bytes()) / 1.0e6);
}

}  // namespace
}  // namespace imsrg
