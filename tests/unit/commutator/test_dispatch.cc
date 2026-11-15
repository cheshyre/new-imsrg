// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// commutator_spec, Z reuse, the term-mask dispatch of commutator(), and the
// full IMSRG(2) commutator against the imsrg-devel golden data and
// commutator_ref.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <gtest/gtest.h>

#include "commutator_test_cases.h"
#include "commutator_test_support.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator_ref/commutator_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"

namespace {

using namespace imsrg;
using commutator::CommutatorOptions;
using commutator::Term;
using commutator::TermMask;
using commutator::Workspace;
using op::Operator;
using op::OperatorSpec;
using op::operator-=;
using op::operator*=;

TEST(commutator_dispatch, spec_rules) {
  const auto A2 = OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian);
  const auto H2 = OperatorSpec::Scalar(2, Hermiticity::kHermitian);
  const auto A1 = OperatorSpec::Scalar(1, Hermiticity::kAntiHermitian);
  EXPECT_EQ(commutator::CommutatorSpec(A2, H2).hermiticity,
            Hermiticity::kHermitian);
  EXPECT_EQ(commutator::CommutatorSpec(H2, A2).hermiticity,
            Hermiticity::kHermitian);
  EXPECT_EQ(commutator::CommutatorSpec(H2, H2).hermiticity,
            Hermiticity::kAntiHermitian);
  EXPECT_EQ(commutator::CommutatorSpec(A2, A2).hermiticity,
            Hermiticity::kAntiHermitian);
  EXPECT_EQ(commutator::CommutatorSpec(A1, H2).particle_rank, 2);
  EXPECT_EQ(commutator::CommutatorSpec(A1, A1).particle_rank, 1);
  auto N2 = H2;
  N2.hermiticity = Hermiticity::kNonHermitian;
  EXPECT_EQ(commutator::CommutatorSpec(N2, H2).hermiticity,
            Hermiticity::kNonHermitian);
  EXPECT_THROW(commutator::CommutatorSpec(
                   OperatorSpec::Scalar(3, Hermiticity::kHermitian), H2),
               NotImplementedError);
}

TEST(commutator_dispatch, z_reuse_keeps_buffer) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator X =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 5);
  const Operator Y =
      imsrg::testing::RandomOperator(ms, Hermiticity::kHermitian, 6);
  CommutatorOptions opts;
  opts.terms = TermMask::None();
  opts.terms.set(Term::kComm110ss).set(Term::kComm111ss).set(Term::kComm122ss);
  Operator Z;
  commutator::Commutator(X, Y, Z, opts, ws);
  const double* ptr = Z.two_body().matrix().buffer().device_data();
  const Operator first = Z;
  commutator::Commutator(X, Y, Z, opts, ws);
  EXPECT_EQ(Z.two_body().matrix().buffer().device_data(), ptr);
  EXPECT_EQ(imsrg::testing::RelativeDeviation(Z, first),
            0.0);  // reset, not accumulated
}

TEST(commutator_dispatch, simple_mask_matches_golden_sum) {
  CommutatorOptions opts;
  opts.terms = TermMask::None();
  for (Term t : {Term::kComm110ss, Term::kComm220ss, Term::kComm111ss,
                 Term::kComm121ss, Term::kComm122ss})
    opts.terms.set(t);
  for (const auto& p : commutator_test::GoldenPairs()) {
    if (p.dir != "emax2_O16") continue;
    const auto ms =
        imsrg::testing::GoldenModelSpace(commutator_test::DumpPath(p, p.y));
    Workspace ws(ms);
    const Operator X =
        imsrg::testing::ReadGolden(commutator_test::DumpPath(p, p.x), ms);
    const Operator Y =
        imsrg::testing::ReadGolden(commutator_test::DumpPath(p, p.y), ms);
    const Operator Z = commutator::Commutator(X, Y, opts, ws);
    // all = the five simple terms + the ladder/221 combination + the
    // particle-hole term.
    Operator ref =
        imsrg::testing::ReadGolden(commutator_test::ResultPath(p, "all"), ms);
    ref -= imsrg::testing::ReadGolden(
        commutator_test::ResultPath(p, "comm222_pp_hh_221ss"), ms);
    ref -= imsrg::testing::ReadGolden(
        commutator_test::ResultPath(p, "comm222_phss"), ms);
    const double dev = imsrg::testing::RelativeDeviation(Z, ref);
    std::printf("[dispatch] %s, %s: %.3e\n", p.x.c_str(), p.y.c_str(), dev);
    EXPECT_LE(dev, commutator_test::kTol);
  }
}

// The full IMSRG(2) commutator (default options) against imsrg-devel's
// Commutator::Commutator.
TEST(commutator_dispatch, full_commutator_matches_golden_all) {
  for (const auto& p : commutator_test::GoldenPairs()) {
    if (p.dir != "emax2_O16") continue;
    const auto ms =
        imsrg::testing::GoldenModelSpace(commutator_test::DumpPath(p, p.y));
    Workspace ws(ms);
    const Operator X =
        imsrg::testing::ReadGolden(commutator_test::DumpPath(p, p.x), ms);
    const Operator Y =
        imsrg::testing::ReadGolden(commutator_test::DumpPath(p, p.y), ms);
    const Operator Z = commutator::Commutator(X, Y, CommutatorOptions{}, ws);
    const Operator ref =
        imsrg::testing::ReadGolden(commutator_test::ResultPath(p, "all"), ms);
    const double dev = imsrg::testing::RelativeDeviation(Z, ref);
    std::printf("[full vs golden all] %s, %s: %.3e\n", p.x.c_str(), p.y.c_str(),
                dev);
    EXPECT_LE(dev, commutator_test::kTol);
  }
}

// Largest |Z_ba - h Z_ab| over the one-body part and every two-body block (h =
// +1 Hermitian, -1 anti-Hermitian), relative to max(1, max|Z|).
double HermiticityViolation(const Operator& Z) {
  const double h = Z.hermiticity() == Hermiticity::kHermitian ? 1.0 : -1.0;
  double worst = 0.0;
  const auto z1 = Z.one_body().host_view();
  for (std::int64_t a = 0; a < z1.rows; ++a)
    for (std::int64_t b = 0; b < z1.cols; ++b)
      worst = std::max(worst, std::abs(z1(b, a) - h * z1(a, b)));
  const auto& channels = Z.two_body().channels();
  for (std::int32_t c = 0; c < channels.num_channels(); ++c) {
    const auto blk = Z.two_body().host_block(ChannelIndex{c});
    for (std::int64_t i = 0; i < blk.rows; ++i)
      for (std::int64_t j = 0; j < blk.cols; ++j)
        worst = std::max(worst, std::abs(blk(j, i) - h * blk(i, j)));
  }
  return worst / std::max(1.0, op::MaxAbs(Z).max());
}

// Fast commutator vs commutator_ref on random operators (every model space and
// hermiticity pair of the shared cases, incl. fractional occupations and a
// rank-1 X), plus [X,Y] = -[Y,X] and the hermiticity of Z for the fast path.
TEST(commutator_dispatch, full_commutator_matches_reference_and_properties) {
  double worst_ref = 0.0;
  double worst_anti = 0.0;
  double worst_herm = 0.0;
  std::uint64_t seed = 900;
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::HermPairs()) {
      const Operator X =
          imsrg::testing::RandomOperator(ms, hp.x, ++seed, hp.rank_x);
      const Operator Y =
          imsrg::testing::RandomOperator(ms, hp.y, ++seed, hp.rank_y);
      const Operator Z = commutator::Commutator(X, Y, CommutatorOptions{}, ws);
      const Operator Zref = commutator_ref::Commutator(X, Y);
      const double dref = imsrg::testing::RelativeDeviation(Z, Zref);
      Operator Zswap = commutator::Commutator(Y, X, CommutatorOptions{}, ws);
      Zswap *= -1.0;
      const double danti = imsrg::testing::RelativeDeviation(Zswap, Z);
      const double dherm = HermiticityViolation(Z);
      EXPECT_LE(dref, commutator_test::kTol) << sc.label << " " << hp.label;
      EXPECT_LE(danti, 1e-12) << sc.label << " " << hp.label;
      EXPECT_LE(dherm, 1e-12) << sc.label << " " << hp.label;
      worst_ref = std::max(worst_ref, dref);
      worst_anti = std::max(worst_anti, danti);
      worst_herm = std::max(worst_herm, dherm);
    }
  }
  std::printf(
      "[full vs ref] max %.3e; antisymmetry max %.3e; hermiticity max %.3e\n",
      worst_ref, worst_anti, worst_herm);
}

TEST(commutator_dispatch, incompatible_model_space_throws) {
  const auto ms1 = imsrg::testing::MakeModelSpace(1, "He4");
  const auto ms2 = imsrg::testing::MakeModelSpace(2, "O16");
  Workspace ws(ms1);
  const Operator X =
      imsrg::testing::RandomOperator(ms2, Hermiticity::kAntiHermitian, 1);
  const Operator Y =
      imsrg::testing::RandomOperator(ms1, Hermiticity::kHermitian, 2);
  Operator Z;
  EXPECT_THROW(commutator::Commutator(X, Y, Z, CommutatorOptions{}, ws),
               InvariantError);
  EXPECT_THROW(commutator::Commutator(Y, X, Z, CommutatorOptions{}, ws),
               InvariantError);
}

}  // namespace
