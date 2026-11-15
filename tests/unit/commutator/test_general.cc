// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The general (non-Hermitian) paths of the fast IMSRG(2) commutator (design/00
// D36, question M7-Q2): every combination of Hermitian, anti-Hermitian and
// general X and Y, term by term and for the full commutator, against
// commutator_ref; the general kernels against the (anti-)Hermitian kernels on
// the same data; antisymmetry and bilinearity of the fast commutator.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator_ref/commutator_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "tests/support/commutator_test_support.h"

namespace imsrg {
namespace {

using commutator::CommutatorOptions;
using commutator::Workspace;
using imsrg::testing::RandomOperator;
using imsrg::testing::RelativeDeviation;
using op::Operator;

/// Tolerance of the general paths against commutator_ref (relative, per term
/// and for the full commutator).
constexpr double kGeneralTol = 1.0e-12;

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
    {"comm221ss", &commutator::terms::Comm221ss, &commutator_ref::Comm221ss},
    {"comm122ss", &commutator::terms::Comm122ss, &commutator_ref::Comm122ss},
    {"comm222_pp_hhss", &commutator::terms::Comm222PpHhss,
     &commutator_ref::Comm222PpHhss},
    {"comm222_phss", &commutator::terms::Comm222Phss,
     &commutator_ref::Comm222Phss},
    {"comm222_pp_hh_221ss", &commutator::terms::Comm222PpHh221ss,
     &commutator_ref::Comm222PpHh221ss},
};

struct SpaceCase {
  const char* label;
  int emax;
  const char* reference;
  bool fractional;
};

/// He4 (closed shell) and He6 (fractional occupation of the neutron p3/2),
/// plus an ensemble reference with fractional holes and particles, at emax 1
/// and 2.
const SpaceCase kSpaces[] = {
    {"He4_emax1", 1, "He4", false},           {"He6_emax1", 1, "He6", false},
    {"He4_emax1_fractional", 1, "He4", true}, {"He4_emax2", 2, "He4", false},
    {"He6_emax2", 2, "He6", false},
};

std::shared_ptr<const basis::ModelSpace> MakeSpace(const SpaceCase& c) {
  return c.fractional
             ? imsrg::testing::FractionalModelSpace(c.emax, c.reference)
             : imsrg::testing::MakeModelSpace(c.emax, c.reference);
}

struct HermCase {
  const char* label;
  Hermiticity x;
  Hermiticity y;
  int rank_x;
  int rank_y;
};

constexpr Hermiticity kH = Hermiticity::kHermitian;
constexpr Hermiticity kA = Hermiticity::kAntiHermitian;
constexpr Hermiticity kN = Hermiticity::kNonHermitian;

/// All nine combinations of H, A, N (rank 2 both) and rank-1 operands with a
/// general partner.
const HermCase kCases[] = {
    {"H_N", kH, kN, 2, 2},  {"A_N", kA, kN, 2, 2},  {"N_H", kN, kH, 2, 2},
    {"N_A", kN, kA, 2, 2},  {"N_N", kN, kN, 2, 2},  {"N1_H", kN, kH, 1, 2},
    {"A_N1", kA, kN, 2, 1}, {"N1_N", kN, kN, 1, 2}, {"N_N1", kN, kN, 2, 1},
};

/// The largest deviation seen per term, for the report line.
std::map<std::string, double>& Worst() {
  static auto* const kWorst = new std::map<std::string, double>();  // NOLINT
  return *kWorst;
}

// Every scalar term (and the fused ladder/221 term) with general X and/or Y,
// against commutator_ref, on closed- and open-shell spaces.
TEST(commutator_general, terms_match_reference) {
  std::uint64_t seed = 7000;
  for (const auto& sc : kSpaces) {
    const auto ms = MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& hc : kCases) {
      const Operator X = RandomOperator(ms, hc.x, ++seed, hc.rank_x);
      const Operator Y = RandomOperator(ms, hc.y, ++seed, hc.rank_y);
      const auto spec = commutator::CommutatorSpec(X.spec(), Y.spec());
      EXPECT_EQ(spec.hermiticity, Hermiticity::kNonHermitian);
      for (const auto& t : kTerms) {
        Operator Z(ms, spec);
        Operator Zref(ms, spec);
        t.fast(X, Y, Z, ws);
        t.ref(X, Y, Zref);
        const double dev = RelativeDeviation(Z, Zref);
        double& w = Worst()[t.name];
        w = std::max(w, dev);
        EXPECT_LE(dev, kGeneralTol)
            << sc.label << " " << hc.label << " " << t.name;
      }
    }
  }
  for (const auto& [name, dev] : Worst())
    std::printf("[general-vs-ref] %s max deviation: %.3e\n", name.c_str(), dev);
}

// The full commutator (default options, fused ladder/221) with general X and
// Y, against commutator_ref, with Z reused between calls.
TEST(commutator_general, full_commutator_matches_reference) {
  std::uint64_t seed = 8000;
  double worst = 0.0;
  for (const auto& sc : kSpaces) {
    const auto ms = MakeSpace(sc);
    Workspace ws(ms);
    Operator Z;
    for (const auto& hc : kCases) {
      const Operator X = RandomOperator(ms, hc.x, ++seed, hc.rank_x);
      const Operator Y = RandomOperator(ms, hc.y, ++seed, hc.rank_y);
      commutator::Commutator(X, Y, Z, CommutatorOptions{}, ws);
      const Operator Zref = commutator_ref::Commutator(X, Y);
      const double dev = RelativeDeviation(Z, Zref);
      worst = std::max(worst, dev);
      EXPECT_EQ(Z.hermiticity(), Hermiticity::kNonHermitian);
      EXPECT_LE(dev, kGeneralTol) << sc.label << " " << hc.label;
    }
  }
  std::printf("[general-vs-ref] full commutator max deviation: %.3e\n", worst);
}

// The general kernels on (anti-)Hermitian DATA (flag set to NonHermitian) must
// reproduce the (anti-)Hermitian kernels term by term: the same numbers by two
// different algorithms (explicit Y P X products, full Pandya matrices, both
// triangles computed independently).
TEST(commutator_general, general_kernels_match_symmetric_kernels) {
  const std::pair<Hermiticity, Hermiticity> herms[] = {
      {kA, kH}, {kH, kA}, {kH, kH}, {kA, kA}};
  std::uint64_t seed = 9000;
  double worst = 0.0;
  for (const auto& sc : kSpaces) {
    const auto ms = MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& [hx, hy] : herms) {
      const Operator X = RandomOperator(ms, hx, ++seed);
      const Operator Y = RandomOperator(ms, hy, ++seed);
      Operator Xg = X;
      Operator Yg = Y;
      Xg.set_hermiticity(Hermiticity::kNonHermitian);
      Yg.set_hermiticity(Hermiticity::kNonHermitian);
      for (const auto& t : kTerms) {
        Operator Zs(ms, commutator::CommutatorSpec(X.spec(), Y.spec()));
        Operator Zg(ms, commutator::CommutatorSpec(Xg.spec(), Yg.spec()));
        t.fast(X, Y, Zs, ws);
        t.fast(Xg, Yg, Zg, ws);
        Zg.set_hermiticity(Zs.hermiticity());
        const double dev = RelativeDeviation(Zg, Zs);
        worst = std::max(worst, dev);
        EXPECT_LE(dev, kGeneralTol) << sc.label << " " << t.name;
      }
    }
  }
  std::printf("[general-vs-symmetric] max deviation: %.3e\n", worst);
}

// [X, Y] = -[Y, X] and bilinearity with the fast general path: for X = XH + XA
// and Y = YH + YA (general), [X, Y] equals the sum of the four commutators of
// the (anti-)Hermitian parts, which run on the symmetric kernels.
TEST(commutator_general, antisymmetry_and_bilinearity) {
  std::uint64_t seed = 9500;
  double worst_anti = 0.0;
  double worst_bilinear = 0.0;
  for (const auto& sc : kSpaces) {
    const auto ms = MakeSpace(sc);
    Workspace ws(ms);
    const Operator XH = RandomOperator(ms, kH, ++seed);
    const Operator XA = RandomOperator(ms, kA, ++seed);
    const Operator YH = RandomOperator(ms, kH, ++seed);
    const Operator YA = RandomOperator(ms, kA, ++seed);
    Operator X = XH + XA;
    Operator Y = YH + YA;
    X.set_hermiticity(Hermiticity::kNonHermitian);
    Y.set_hermiticity(Hermiticity::kNonHermitian);
    const Operator Z = commutator::Commutator(X, Y, CommutatorOptions{}, ws);
    Operator Zswap = commutator::Commutator(Y, X, CommutatorOptions{}, ws);
    Zswap *= -1.0;
    const double danti = RelativeDeviation(Zswap, Z);
    worst_anti = std::max(worst_anti, danti);
    EXPECT_LE(danti, kGeneralTol) << sc.label;

    Operator sum(ms, commutator::CommutatorSpec(X.spec(), Y.spec()));
    for (const Operator* px : {&XH, &XA}) {
      for (const Operator* py : {&YH, &YA}) {
        Operator part =
            commutator::Commutator(*px, *py, CommutatorOptions{}, ws);
        part.set_hermiticity(Hermiticity::kNonHermitian);
        sum += part;
      }
    }
    const double dbil = RelativeDeviation(Z, sum);
    worst_bilinear = std::max(worst_bilinear, dbil);
    EXPECT_LE(dbil, kGeneralTol) << sc.label;
  }
  std::printf("[general] antisymmetry max %.3e, bilinearity max %.3e\n",
              worst_anti, worst_bilinear);
}

// A general commutator of operators without a two-body part and the term mask
// of a flow step also work: one-body-only operands, and masks without the
// fused ladder term.
TEST(commutator_general, one_body_and_masked_terms) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He6");
  Workspace ws(ms);
  const Operator X = RandomOperator(ms, kN, 31, 1);
  const Operator Y = RandomOperator(ms, kN, 32, 1);
  const Operator Z = commutator::Commutator(X, Y, CommutatorOptions{}, ws);
  const Operator Zref = commutator_ref::Commutator(X, Y);
  EXPECT_LE(RelativeDeviation(Z, Zref), kGeneralTol);
  EXPECT_EQ(Z.spec().particle_rank, 1);

  const Operator X2 = RandomOperator(ms, kN, 33, 2);
  const Operator Y2 = RandomOperator(ms, kA, 34, 2);
  CommutatorOptions opts;
  opts.terms = commutator::TermMask::None();
  opts.terms.set(commutator::Term::kComm221ss)
      .set(commutator::Term::kComm222Phss);
  const Operator Zm = commutator::Commutator(X2, Y2, opts, ws);
  Operator Zr(ms, commutator::CommutatorSpec(X2.spec(), Y2.spec()));
  commutator_ref::Comm221ss(X2, Y2, Zr);
  commutator_ref::Comm222Phss(X2, Y2, Zr);
  EXPECT_LE(RelativeDeviation(Zm, Zr), kGeneralTol);
}

// The general scratch is allocated lazily: (anti-)Hermitian use leaves it
// empty, the first general call allocates it (counted in scratch_bytes()), and
// later calls reuse it.
TEST(commutator_general, scratch_is_lazy) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator A = RandomOperator(ms, kA, 41);
  const Operator H = RandomOperator(ms, kH, 42);
  const Operator N = RandomOperator(ms, kN, 43);
  (void)commutator::Commutator(A, H, CommutatorOptions{}, ws);
  const auto& pd = ws.impl().pandya;
  EXPECT_FALSE(pd.general.allocated());
  EXPECT_EQ(pd.general.scratch_bytes(), 0u);
  const std::size_t before = pd.scratch_bytes();
  (void)commutator::Commutator(N, H, CommutatorOptions{}, ws);
  EXPECT_TRUE(pd.general.allocated());
  EXPECT_GT(pd.general.scratch_bytes(), 0u);
  EXPECT_EQ(pd.scratch_bytes(), before + pd.general.scratch_bytes());
  // General Ncc (4 nk + 4 Ncc) vs Hermitian Ncc (2 nk + 2 Ncc) doubles per cc
  // channel (no ZB buffer and half of YK since M12a): exactly twice.
  EXPECT_LE(pd.general.scratch_bytes(), 2 * before + 64);
  const double* z = pd.general.z.device_data();
  (void)commutator::Commutator(H, N, CommutatorOptions{}, ws);
  EXPECT_EQ(pd.general.z.device_data(), z);
}

// Z must carry the hermiticity of [X, Y]: general for a general input, and not
// general for definite inputs.
TEST(commutator_general, wrong_z_hermiticity_throws) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator N = RandomOperator(ms, kN, 51);
  const Operator H = RandomOperator(ms, kH, 52);
  const Operator A = RandomOperator(ms, kA, 53);
  for (const Hermiticity hz : {kH, kA}) {
    Operator Z(ms, op::OperatorSpec::Scalar(2, hz));
    for (const auto& t : kTerms) {
      EXPECT_THROW(t.fast(N, H, Z, ws), InvariantError) << t.name;
      EXPECT_THROW(t.fast(A, N, Z, ws), InvariantError) << t.name;
    }
  }
  Operator Zg(ms, op::OperatorSpec::Scalar(2, kN));
  for (const auto& t : kTerms) {
    EXPECT_THROW(t.fast(A, H, Zg, ws), InvariantError) << t.name;
    EXPECT_THROW(t.fast(H, H, Zg, ws), InvariantError) << t.name;
  }
}

}  // namespace
}  // namespace imsrg
