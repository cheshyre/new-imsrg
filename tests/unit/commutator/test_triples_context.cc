// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The perturbative triples on the reduced `commutator::TriplesContext`
// (triples.h) against the full `ThreeBodyContext` path (three_body.h) on the
// same operands, epsilon and cuts: the energies agree to 1e-14 relative
// (He4 and O16 at emax 2, O16 at emax 3, with a dE3max cut and with an OccNat3
// cut on synthetic natural occupations), and the counts of the reduced space
// equal those of the full space restricted to the channels with an hhh triple
// and the hhh/ppp triples in them.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/triples.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"

namespace imsrg::commutator {
namespace {

using basis::ModelSpace;
using basis::ThreeBodyCuts;
using op::Operator;
using op::OperatorSpec;

constexpr double kSkip = 1.0e-8;  // imsrg++'s occupation threshold

std::shared_ptr<const ModelSpace> PlainSpace(const char* ref, int emax) {
  return ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, ref, ref);
}

Operator Random(std::shared_ptr<const ModelSpace> ms, Hermiticity h,
                std::uint64_t seed) {
  op::RandomOperatorOptions o;
  o.seed = seed;
  o.one_body_scale = 1.0;
  o.two_body_scale = 1.0;
  return op::RandomOperator(std::move(ms), OperatorSpec::Scalar(2, h), o);
}

/// epsilon_i = (n_i > 0.5 ? -30 : 10) + 2 e_i + 0.1 (i % 7).
std::vector<double> BenchEpsilon(const ModelSpace& ms) {
  const auto orbits = ms.basis().orbits();
  const auto occ = ms.reference().occupations();
  std::vector<double> eps(orbits.size());
  for (std::size_t i = 0; i < orbits.size(); ++i) {
    eps[i] = (occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * orbits[i].e() +
             0.1 * static_cast<double>(i % 7);
  }
  return eps;
}

double Rel(double x, double y) {
  return std::abs(x - y) / std::max(1.0e-300, std::abs(y));
}

/// The counts of the full kept space `full` (cuts applied) restricted to the
/// channels with a triple of n_T >= skip and, in them, to the triples with
/// n_T >= skip or nbar_T >= skip; `states` from the full context's pair-third
/// space.
TriplesCounts RestrictedCounts(const ModelSpace& ms,
                               const basis::ThreeBodySpace& full,
                               const op::PairThirdSpace& pt) {
  const auto occ = ms.reference().occupations();
  const auto occupation = [&](const basis::OrbitTriple& t, double& o,
                              double& u) {
    const double na = occ[static_cast<std::size_t>(t.a.value())];
    const double nb = occ[static_cast<std::size_t>(t.b.value())];
    const double nc = occ[static_cast<std::size_t>(t.c.value())];
    o = na * nb * nc;
    u = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
  };
  TriplesCounts c;
  for (std::int32_t ch = 0; ch < full.num_channels(); ++ch) {
    const basis::KeptChannel3& kc = full.channel(Channel3Index{ch});
    bool has_hhh = false;
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      double o = 0.0;
      double u = 0.0;
      occupation(full.triple(basis::Triple3Index{t}), o, u);
      has_hhh = has_hhh || o >= kSkip;
    }
    if (!has_hhh) continue;
    ++c.channels;
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      const basis::OrbitTriple& tr = full.triple(basis::Triple3Index{t});
      double o = 0.0;
      double u = 0.0;
      occupation(tr, o, u);
      if (o < kSkip && u < kSkip) continue;
      ++c.triples;
      if (o >= kSkip) ++c.triples_hhh;
      if (u >= kSkip) ++c.triples_ppp;
      c.rows += tr.row_end.value() - tr.row_begin.value();
      c.states += pt.state_end(basis::Triple3Index{t}) -
                  pt.state_begin(basis::Triple3Index{t});
    }
  }
  return c;
}

/// Energy of the full-context path and of the reduced context for the same
/// operands; checks the 1e-14 agreement and that the energy is nonzero, and
/// the counts against the restricted full space. Returns the reduced
/// context's counts.
TriplesCounts CheckAgainstFull(const char* label, const char* ref, int emax,
                               const ThreeBodyCuts& cuts_in, Hermiticity hx,
                               Hermiticity hy) {
  SCOPED_TRACE(label);
  const auto ms2 = PlainSpace(ref, emax);
  ThreeBodyCuts cuts = cuts_in;
  cuts.e3max = 3 * emax;
  const auto ms3 = ms2->WithThreeBody(cuts);
  const ThreeBodyContext full_ctx(ms3);
  const TriplesContext ctx(ms2, cuts);
  const Operator X = Random(ms2, hx, 1);
  const Operator Y = Random(ms2, hy, 2);
  const std::vector<double> eps = BenchEpsilon(*ms2);
  const double e_full = PerturbativeTriplesEnergy(X, Y, eps, full_ctx);
  const double e_new = PerturbativeTriplesEnergy(X, Y, eps, ctx);
  const double rel = Rel(e_new, e_full);
  std::printf("[triples] %s: full %.17g new %.17g rel %.2e\n", label, e_full,
              e_new, rel);
  EXPECT_NE(e_full, 0.0);
  EXPECT_LE(rel, 1.0e-14);

  const TriplesCounts want = RestrictedCounts(*ms2, *ms3->three_body_space(),
                                              full_ctx.impl().pair_third);
  const TriplesCounts got = ctx.Counts();
  std::printf("[triples] %s: channels %" PRId64 " triples %" PRId64
              " (hhh %" PRId64 " ppp %" PRId64 ") rows %" PRId64
              " states %" PRId64 " bytes %" PRId64 "\n",
              label, got.channels, got.triples, got.triples_hhh,
              got.triples_ppp, got.rows, got.states, ctx.TableBytes());
  EXPECT_GT(got.triples, 0);
  EXPECT_EQ(got.channels, want.channels);
  EXPECT_EQ(got.triples, want.triples);
  EXPECT_EQ(got.triples_hhh, want.triples_hhh);
  EXPECT_EQ(got.triples_ppp, want.triples_ppp);
  EXPECT_EQ(got.rows, want.rows);
  EXPECT_EQ(got.states, want.states);
  // The reduced context is smaller than the full one.
  EXPECT_LT(ctx.TableBytes(), full_ctx.TableBytes());
  return got;
}

TEST(TriplesContext, He4Emax2) {
  CheckAgainstFull("He4 emax 2", "He4", 2, {}, Hermiticity::kAntiHermitian,
                   Hermiticity::kHermitian);
}

TEST(TriplesContext, O16Emax2) {
  CheckAgainstFull("O16 emax 2", "O16", 2, {}, Hermiticity::kAntiHermitian,
                   Hermiticity::kHermitian);
}

TEST(TriplesContext, O16Emax2HermitianPair) {
  CheckAgainstFull("O16 emax 2 HH", "O16", 2, {}, Hermiticity::kHermitian,
                   Hermiticity::kHermitian);
}

TEST(TriplesContext, O16Emax3) {
  CheckAgainstFull("O16 emax 3", "O16", 3, {}, Hermiticity::kAntiHermitian,
                   Hermiticity::kHermitian);
}

/// dE3max cut at emax 3 (the ppp triples of shells 2 and 3 have dE = 3..6): the
/// cut removes some kets, the energy and the counts still agree.
TEST(TriplesContext, DE3maxCut) {
  ThreeBodyCuts cuts;
  cuts.de3max = 4.0;
  const TriplesCounts c =
      CheckAgainstFull("O16 emax 3 dE3max 4", "O16", 3, cuts,
                       Hermiticity::kAntiHermitian, Hermiticity::kHermitian);
  EXPECT_LT(c.rows, c.rows_before_cuts);
}

/// The kets before the cuts of the cut context equal the rows of the uncut
/// context.
TEST(TriplesContext, RowsBeforeCutsAreTheUncutRows) {
  const auto ms2 = PlainSpace("O16", 3);
  ThreeBodyCuts uncut;
  uncut.e3max = 9;
  ThreeBodyCuts cut = uncut;
  cut.de3max = 4.0;
  const TriplesContext a(ms2, uncut);
  const TriplesContext b(ms2, cut);
  EXPECT_EQ(a.Counts().rows, a.Counts().rows_before_cuts);
  EXPECT_EQ(b.Counts().rows_before_cuts, a.Counts().rows);
  EXPECT_LT(b.Counts().rows, a.Counts().rows);
}

/// OccNat3 cut with synthetic natural occupations (the product n (1 - n) of a
/// triple varies with the orbit index).
TEST(TriplesContext, OccNat3Cut) {
  const auto ms2 = PlainSpace("O16", 2);
  const auto occ = ms2->reference().occupations();
  ThreeBodyCuts cuts;
  cuts.occnat3_cut = 2.0e-4;
  for (std::size_t i = 0; i < occ.size(); ++i) {
    const double k = static_cast<double>(i % 5);
    cuts.natural_occupations.push_back(occ[i] > 0.5 ? 0.97 - 0.05 * k
                                                    : 0.02 + 0.04 * k);
  }
  const TriplesCounts c =
      CheckAgainstFull("O16 emax 2 OccNat3", "O16", 2, cuts,
                       Hermiticity::kAntiHermitian, Hermiticity::kHermitian);
  EXPECT_GT(c.rows, 0);
  EXPECT_LT(c.rows, c.rows_before_cuts);
}

TEST(TriplesContext, ErrorsAndMoves) {
  const auto ms2 = PlainSpace("He4", 2);
  ThreeBodyCuts cuts;
  cuts.e3max = 6;
  TriplesContext ctx(ms2, cuts);
  const Operator X = Random(ms2, Hermiticity::kAntiHermitian, 1);
  const Operator Y = Random(ms2, Hermiticity::kHermitian, 2);
  const std::vector<double> eps = BenchEpsilon(*ms2);
  const double e = PerturbativeTriplesEnergy(X, Y, eps, ctx);
  const std::vector<double> bad(eps.size() - 1, 0.0);
  EXPECT_THROW((void)PerturbativeTriplesEnergy(X, Y, bad, ctx), InputError);
  EXPECT_THROW(TriplesContext(nullptr, cuts), InputError);
  EXPECT_GT(ctx.ScratchBytesPerTeam(), 0);
  EXPECT_GT(ctx.TileWork().w, 0.0);
  EXPECT_GT(ctx.TileWork().epilogue, 0.0);
  TriplesContext moved(std::move(ctx));
  // The sum is a parallel reduction: equal up to its order.
  EXPECT_LE(Rel(PerturbativeTriplesEnergy(X, Y, eps, moved), e), 1.0e-14);
  EXPECT_EQ(ctx.TableBytes(), 0);  // NOLINT(bugprone-use-after-move)
}

}  // namespace
}  // namespace imsrg::commutator
