// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Input validation and consistency checks of the fast commutator and
// commutator_ref: rank-1 operands versus the same operators promoted to rank 2,
// Z hermiticity and aliasing checks, the non-fused dispatcher branches, and the
// scratch footprint of a Workspace.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator_ref/commutator_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg {
namespace {

using commutator::CommutatorOptions;
using commutator::Term;
using commutator::TermMask;
using commutator::Workspace;
using op::Operator;
using op::OperatorSpec;

using FastTerm = void (*)(const Operator&, const Operator&, Operator&,
                          Workspace&);
using RefTerm = void (*)(const Operator&, const Operator&, Operator&);

struct NamedTerm {
  const char* name;
  FastTerm fast;
  RefTerm ref;
};

const std::vector<NamedTerm>& AllTerms() {
  static const auto* const kT = new std::vector<NamedTerm>{
      {"comm110ss", commutator::terms::Comm110ss, commutator_ref::Comm110ss},
      {"comm220ss", commutator::terms::Comm220ss, commutator_ref::Comm220ss},
      {"comm111ss", commutator::terms::Comm111ss, commutator_ref::Comm111ss},
      {"comm121ss", commutator::terms::Comm121ss, commutator_ref::Comm121ss},
      {"comm221ss", commutator::terms::Comm221ss, commutator_ref::Comm221ss},
      {"comm122ss", commutator::terms::Comm122ss, commutator_ref::Comm122ss},
      {"comm222_pp_hhss", commutator::terms::Comm222PpHhss,
       commutator_ref::Comm222PpHhss},
      {"comm222_phss", commutator::terms::Comm222Phss,
       commutator_ref::Comm222Phss},
      {"comm222_pp_hh_221ss", commutator::terms::Comm222PpHh221ss,
       commutator_ref::Comm222PpHh221ss},
  };
  return *kT;
}

// The same operator with particle rank 2 and a zero two-body part (x1 has rank
// 1).
Operator PromoteToRank2(const Operator& x1,
                        const std::shared_ptr<const basis::ModelSpace>& ms) {
  Operator x2(ms, OperatorSpec::Scalar(2, x1.hermiticity()));
  const std::int32_t norb = ms->num_orbits();
  x2.zero_body() = x1.zero_body();
  const auto v = x1.one_body().host_view();
  for (std::int32_t a = 0; a < norb; ++a)
    for (std::int32_t b = 0; b < norb; ++b)
      x2.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v(a, b));
  x2.SyncToDevice();
  return x2;
}

Operator ZeroZ(const std::shared_ptr<const basis::ModelSpace>& ms,
               const Operator& X, const Operator& Y) {
  return Operator(ms, commutator::CommutatorSpec(X.spec(), Y.spec()));
}

// A rank-1 operand and the same values promoted to rank 2 with an empty
// two-body part must give the same [X, Y] for every term, fast and reference,
// for (A,H) and (H,A), the rank-1 operand on either side. (Regression:
// comm121ss returned early unless BOTH operands had a two-body part.)
TEST(commutator_robustness, rank1_equals_promoted_rank2) {
  const std::vector<std::pair<Hermiticity, Hermiticity>> herms = {
      {Hermiticity::kAntiHermitian, Hermiticity::kHermitian},
      {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}};
  std::uint64_t seed = 4200;
  for (const auto& sc : commutator_test::SpaceCases()) {
    const auto ms = commutator_test::MakeSpace(sc);
    Workspace ws(ms);
    for (const auto& [hx, hy] : herms) {
      for (const bool x_is_rank1 : {true, false}) {
        const Operator X1 =
            imsrg::testing::RandomOperator(ms, hx, ++seed, x_is_rank1 ? 1 : 2);
        const Operator Y1 =
            imsrg::testing::RandomOperator(ms, hy, ++seed, x_is_rank1 ? 2 : 1);
        const Operator X2 = x_is_rank1 ? PromoteToRank2(X1, ms) : X1;
        const Operator Y2 = x_is_rank1 ? Y1 : PromoteToRank2(Y1, ms);
        for (const auto& t : AllTerms()) {
          Operator Za = ZeroZ(ms, X1, Y1);
          Operator Zb = ZeroZ(ms, X2, Y2);
          t.fast(X1, Y1, Za, ws);
          t.fast(X2, Y2, Zb, ws);
          EXPECT_LE(imsrg::testing::RelativeDeviation(Za, Zb), 1.0e-13)
              << "fast " << t.name << " " << sc.label
              << " x_rank1=" << x_is_rank1;
          Operator Ra = ZeroZ(ms, X1, Y1);
          Operator Rb = ZeroZ(ms, X2, Y2);
          t.ref(X1, Y1, Ra);
          t.ref(X2, Y2, Rb);
          EXPECT_LE(imsrg::testing::RelativeDeviation(Ra, Rb), 1.0e-13)
              << "ref " << t.name << " " << sc.label
              << " x_rank1=" << x_is_rank1;
          EXPECT_LE(imsrg::testing::RelativeDeviation(Za, Ra),
                    commutator_test::kTol)
              << "fast vs ref " << t.name << " " << sc.label
              << " x_rank1=" << x_is_rank1;
        }
      }
    }
  }
}

// Z must have the hermiticity of [X, Y]; a different one is rejected by every
// term.
TEST(commutator_robustness, wrong_z_hermiticity_throws) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  const Operator X =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 11, 2);
  const Operator Y =
      imsrg::testing::RandomOperator(ms, Hermiticity::kHermitian, 12, 2);
  // [A, H] is Hermitian, so an anti-Hermitian Z is wrong.
  Operator Z_wrong(ms, OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
  for (const auto& t : AllTerms())
    EXPECT_THROW(t.fast(X, Y, Z_wrong, ws), InvariantError) << t.name;
  // [A, A] is anti-Hermitian, so a Hermitian Z is wrong.
  const Operator X2 =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 13, 2);
  Operator Z_wrong2(ms, OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  for (const auto& t : AllTerms())
    EXPECT_THROW(t.fast(X, X2, Z_wrong2, ws), InvariantError) << t.name;
  // The right hermiticity is accepted.
  Operator Z_ok(ms, OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  EXPECT_NO_THROW(commutator::terms::Comm122ss(X, Y, Z_ok, ws));
}

// Z may not alias X or Y.
TEST(commutator_robustness, z_aliasing_operand_throws) {
  const auto ms = imsrg::testing::MakeModelSpace(1, "He4");
  Workspace ws(ms);
  Operator A1 =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 21, 2);
  Operator A2 =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 22, 2);
  // [A, A] is anti-Hermitian, so the alias has the right hermiticity and only
  // the alias is wrong.
  for (const auto& t : AllTerms()) {
    EXPECT_THROW(t.fast(A1, A2, A1, ws), InvariantError) << t.name << " Z = X";
    EXPECT_THROW(t.fast(A1, A2, A2, ws), InvariantError) << t.name << " Z = Y";
  }
}

// The dispatcher fuses comm222_pp_hhss and comm221ss only when both are
// requested; each alone must equal the individual term, and the fused pair
// their sum.
TEST(commutator_robustness, dispatcher_non_fused_branches) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "O16");
  Workspace ws(ms);
  const Operator X =
      imsrg::testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 31, 2);
  const Operator Y =
      imsrg::testing::RandomOperator(ms, Hermiticity::kHermitian, 32, 2);
  auto run = [&](TermMask mask) {
    CommutatorOptions o;
    o.terms = mask;
    return commutator::Commutator(X, Y, o, ws);
  };
  Operator pp = ZeroZ(ms, X, Y);
  commutator::terms::Comm222PpHhss(X, Y, pp, ws);
  Operator s221 = ZeroZ(ms, X, Y);
  commutator::terms::Comm221ss(X, Y, s221, ws);
  Operator fused = ZeroZ(ms, X, Y);
  commutator::terms::Comm222PpHh221ss(X, Y, fused, ws);

  EXPECT_LE(imsrg::testing::RelativeDeviation(
                run(TermMask::Only(Term::kComm222PpHhss)), pp),
            1.0e-13);
  EXPECT_LE(imsrg::testing::RelativeDeviation(
                run(TermMask::Only(Term::kComm221ss)), s221),
            1.0e-13);
  const TermMask both =
      TermMask::Only(Term::kComm222PpHhss) | TermMask::Only(Term::kComm221ss);
  EXPECT_LE(imsrg::testing::RelativeDeviation(run(both), fused), 1.0e-13);
  Operator sum = pp;
  op::operator+=(sum, s221);
  EXPECT_LE(imsrg::testing::RelativeDeviation(fused, sum), 1.0e-12);
  // Against the reference as well.
  Operator rpp = ZeroZ(ms, X, Y);
  commutator_ref::Comm222PpHhss(X, Y, rpp);
  Operator r221 = ZeroZ(ms, X, Y);
  commutator_ref::Comm221ss(X, Y, r221);
  EXPECT_LE(imsrg::testing::RelativeDeviation(
                run(TermMask::Only(Term::kComm222PpHhss)), rpp),
            commutator_test::kTol);
  EXPECT_LE(imsrg::testing::RelativeDeviation(
                run(TermMask::Only(Term::kComm221ss)), r221),
            commutator_test::kTol);
}

// Prints the scratch memory of a Workspace relative to one two-body operator
// (O16, emax 4, 6).
TEST(commutator_robustness, scratch_footprint) {
  for (const int emax : {4, 6}) {
    const auto ms = imsrg::testing::MakeModelSpace(emax, "O16");
    Workspace ws(ms);
    const auto& impl = ws.impl();
    const double op_bytes =
        static_cast<double>(impl.two_body_size) * sizeof(double);
    const double lad = static_cast<double>(impl.ladder.scratch_bytes());
    const double pan = static_cast<double>(impl.pandya.scratch_bytes());
    std::printf(
        "[footprint] O16 emax=%d: two-body operator %.1f MB; ladder scratch "
        "%.1f MB (%.2fx); "
        "Pandya scratch %.1f MB (%.2fx)\n",
        emax, op_bytes / 1.0e6, lad / 1.0e6, lad / op_bytes, pan / 1.0e6,
        pan / op_bytes);
    EXPECT_GT(lad, 0.0);
    EXPECT_GT(pan, 0.0);
  }
}

// Closed-shell reference: the pp projector has direct channels (no gathered
// operands) and the hh gather stays; the scratch is smaller than the fully
// gathered layout. The results of the direct and gathered paths against
// commutator_ref are covered by commutator_ladder_terms / commutator_general
// (O16 and the fractional-occupation references there).
TEST(commutator_robustness, ladder_direct_channels) {
  const auto ms = imsrg::testing::MakeModelSpace(4, "O16");
  Workspace ws(ms);
  const auto& pp = ws.impl().ladder.pp;
  int num_direct = 0;
  std::int64_t full = 0;
  for (std::size_t ch = 0; ch < pp.count_host.size(); ++ch) {
    const std::int32_t k0 = pp.direct_first_host[ch];
    if (k0 < 0) continue;
    ++num_direct;
    EXPECT_GT(pp.count_host[ch], 0);
    EXPECT_EQ(pp.gather_offset_host[ch + 1], pp.gather_offset_host[ch]);
    const auto n =
        static_cast<std::int64_t>(ms->two_body().channels()[ch].num_kets);
    full += n * pp.count_host[ch];
  }
  EXPECT_GT(num_direct, 0);
  EXPECT_GT(full, 0);
  EXPECT_EQ(static_cast<std::int64_t>(pp.xg.size()), [&] {
    std::int64_t t = 0;
    for (std::size_t ch = 0; ch < pp.count_host.size(); ++ch)
      if (pp.direct_first_host[ch] < 0)
        t += ms->two_body().channels()[ch].num_kets *
             static_cast<std::int64_t>(pp.count_host[ch]);
    return t;
  }());
}

}  // namespace
}  // namespace imsrg
