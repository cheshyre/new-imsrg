// Input validation and consistency checks of the fast commutator and commutator_ref: rank-1
// operands versus the same operators promoted to rank 2, Z hermiticity and aliasing checks, the
// non-fused dispatcher branches, and the scratch footprint of a Workspace.
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "commutator_test_cases.hpp"
#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"
#include "imsrg/commutator_ref/commutator_ref.hpp"
#include "imsrg/operator/arithmetic.hpp"

namespace {

using namespace imsrg;
using commutator::CommutatorOptions;
using commutator::Term;
using commutator::TermMask;
using commutator::Workspace;
using op::Operator;
using op::OperatorSpec;

using FastTerm = void (*)(const Operator&, const Operator&, Operator&, Workspace&);
using RefTerm = void (*)(const Operator&, const Operator&, Operator&);

struct NamedTerm {
  const char* name;
  FastTerm fast;
  RefTerm ref;
};

const std::vector<NamedTerm>& all_terms() {
  static const std::vector<NamedTerm> t = {
      {"comm110ss", commutator::terms::comm110ss, commutator_ref::comm110ss},
      {"comm220ss", commutator::terms::comm220ss, commutator_ref::comm220ss},
      {"comm111ss", commutator::terms::comm111ss, commutator_ref::comm111ss},
      {"comm121ss", commutator::terms::comm121ss, commutator_ref::comm121ss},
      {"comm221ss", commutator::terms::comm221ss, commutator_ref::comm221ss},
      {"comm122ss", commutator::terms::comm122ss, commutator_ref::comm122ss},
      {"comm222_pp_hhss", commutator::terms::comm222_pp_hhss, commutator_ref::comm222_pp_hhss},
      {"comm222_phss", commutator::terms::comm222_phss, commutator_ref::comm222_phss},
      {"comm222_pp_hh_221ss", commutator::terms::comm222_pp_hh_221ss,
       commutator_ref::comm222_pp_hh_221ss},
  };
  return t;
}

// The same operator with particle rank 2 and a zero two-body part (x1 has rank 1).
Operator promote_to_rank2(const Operator& x1, const std::shared_ptr<const basis::ModelSpace>& ms) {
  Operator x2(ms, OperatorSpec::scalar(2, x1.hermiticity()));
  const std::int32_t norb = ms->num_orbits();
  x2.zero_body() = x1.zero_body();
  const auto v = x1.one_body().host_view();
  for (std::int32_t a = 0; a < norb; ++a)
    for (std::int32_t b = 0; b < norb; ++b)
      x2.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v(a, b));
  x2.sync_to_device();
  return x2;
}

Operator zero_z(const std::shared_ptr<const basis::ModelSpace>& ms, const Operator& X,
                const Operator& Y) {
  return Operator(ms, commutator::commutator_spec(X.spec(), Y.spec()));
}

// A rank-1 operand and the same values promoted to rank 2 with an empty two-body part must give
// the same [X, Y] for every term, fast and reference, for (A,H) and (H,A), the rank-1 operand on
// either side. (Regression: comm121ss returned early unless BOTH operands had a two-body part.)
TEST(commutator_robustness, rank1_equals_promoted_rank2) {
  const std::vector<std::pair<Hermiticity, Hermiticity>> herms = {
      {Hermiticity::AntiHermitian, Hermiticity::Hermitian},
      {Hermiticity::Hermitian, Hermiticity::AntiHermitian}};
  std::uint64_t seed = 4200;
  for (const auto& sc : commutator_test::space_cases()) {
    const auto ms = commutator_test::make_space(sc);
    Workspace ws(ms);
    for (const auto& [hx, hy] : herms) {
      for (const bool x_is_rank1 : {true, false}) {
        const Operator X1 = imsrg::testing::random_operator(ms, hx, ++seed, x_is_rank1 ? 1 : 2);
        const Operator Y1 = imsrg::testing::random_operator(ms, hy, ++seed, x_is_rank1 ? 2 : 1);
        const Operator X2 = x_is_rank1 ? promote_to_rank2(X1, ms) : X1;
        const Operator Y2 = x_is_rank1 ? Y1 : promote_to_rank2(Y1, ms);
        for (const auto& t : all_terms()) {
          Operator Za = zero_z(ms, X1, Y1);
          Operator Zb = zero_z(ms, X2, Y2);
          t.fast(X1, Y1, Za, ws);
          t.fast(X2, Y2, Zb, ws);
          EXPECT_LE(imsrg::testing::relative_deviation(Za, Zb), 1e-13)
              << "fast " << t.name << " " << sc.label << " x_rank1=" << x_is_rank1;
          Operator Ra = zero_z(ms, X1, Y1);
          Operator Rb = zero_z(ms, X2, Y2);
          t.ref(X1, Y1, Ra);
          t.ref(X2, Y2, Rb);
          EXPECT_LE(imsrg::testing::relative_deviation(Ra, Rb), 1e-13)
              << "ref " << t.name << " " << sc.label << " x_rank1=" << x_is_rank1;
          EXPECT_LE(imsrg::testing::relative_deviation(Za, Ra), commutator_test::kTol)
              << "fast vs ref " << t.name << " " << sc.label << " x_rank1=" << x_is_rank1;
        }
      }
    }
  }
}

// Z must have the hermiticity of [X, Y]; a different one is rejected by every term.
TEST(commutator_robustness, wrong_z_hermiticity_throws) {
  const auto ms = imsrg::testing::model_space(1, "He4");
  Workspace ws(ms);
  const Operator X = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 11, 2);
  const Operator Y = imsrg::testing::random_operator(ms, Hermiticity::Hermitian, 12, 2);
  // [A, H] is Hermitian, so an anti-Hermitian Z is wrong.
  Operator Zwrong(ms, OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
  for (const auto& t : all_terms())
    EXPECT_THROW(t.fast(X, Y, Zwrong, ws), InvariantError) << t.name;
  // [A, A] is anti-Hermitian, so a Hermitian Z is wrong.
  const Operator X2 = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 13, 2);
  Operator Zwrong2(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  for (const auto& t : all_terms())
    EXPECT_THROW(t.fast(X, X2, Zwrong2, ws), InvariantError) << t.name;
  // The right hermiticity is accepted.
  Operator Zok(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  EXPECT_NO_THROW(commutator::terms::comm122ss(X, Y, Zok, ws));
}

// Z may not alias X or Y.
TEST(commutator_robustness, z_aliasing_operand_throws) {
  const auto ms = imsrg::testing::model_space(1, "He4");
  Workspace ws(ms);
  Operator A1 = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 21, 2);
  Operator A2 = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 22, 2);
  // [A, A] is anti-Hermitian, so the alias has the right hermiticity and only the alias is wrong.
  for (const auto& t : all_terms()) {
    EXPECT_THROW(t.fast(A1, A2, A1, ws), InvariantError) << t.name << " Z = X";
    EXPECT_THROW(t.fast(A1, A2, A2, ws), InvariantError) << t.name << " Z = Y";
  }
}

// The dispatcher fuses comm222_pp_hhss and comm221ss only when both are requested; each alone
// must equal the individual term, and the fused pair their sum.
TEST(commutator_robustness, dispatcher_non_fused_branches) {
  const auto ms = imsrg::testing::model_space(2, "O16");
  Workspace ws(ms);
  const Operator X = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 31, 2);
  const Operator Y = imsrg::testing::random_operator(ms, Hermiticity::Hermitian, 32, 2);
  auto run = [&](TermMask mask) {
    CommutatorOptions o;
    o.terms = mask;
    return commutator::commutator(X, Y, o, ws);
  };
  Operator pp = zero_z(ms, X, Y);
  commutator::terms::comm222_pp_hhss(X, Y, pp, ws);
  Operator s221 = zero_z(ms, X, Y);
  commutator::terms::comm221ss(X, Y, s221, ws);
  Operator fused = zero_z(ms, X, Y);
  commutator::terms::comm222_pp_hh_221ss(X, Y, fused, ws);

  EXPECT_LE(imsrg::testing::relative_deviation(run(TermMask::only(Term::comm222_pp_hhss)), pp),
            1e-13);
  EXPECT_LE(imsrg::testing::relative_deviation(run(TermMask::only(Term::comm221ss)), s221), 1e-13);
  const TermMask both = TermMask::only(Term::comm222_pp_hhss) | TermMask::only(Term::comm221ss);
  EXPECT_LE(imsrg::testing::relative_deviation(run(both), fused), 1e-13);
  Operator sum = pp;
  op::operator+=(sum, s221);
  EXPECT_LE(imsrg::testing::relative_deviation(fused, sum), 1e-12);
  // Against the reference as well.
  Operator rpp = zero_z(ms, X, Y);
  commutator_ref::comm222_pp_hhss(X, Y, rpp);
  Operator r221 = zero_z(ms, X, Y);
  commutator_ref::comm221ss(X, Y, r221);
  EXPECT_LE(imsrg::testing::relative_deviation(run(TermMask::only(Term::comm222_pp_hhss)), rpp),
            commutator_test::kTol);
  EXPECT_LE(imsrg::testing::relative_deviation(run(TermMask::only(Term::comm221ss)), r221),
            commutator_test::kTol);
}

// Prints the scratch memory of a Workspace relative to one two-body operator (O16, emax 4, 6).
TEST(commutator_robustness, scratch_footprint) {
  for (const int emax : {4, 6}) {
    const auto ms = imsrg::testing::model_space(emax, "O16");
    Workspace ws(ms);
    const auto& impl = ws.impl();
    const double op_bytes = static_cast<double>(impl.two_body_size) * sizeof(double);
    const double lad = static_cast<double>(impl.ladder.scratch_bytes());
    const double pan = static_cast<double>(impl.pandya.scratch_bytes());
    std::printf(
        "[footprint] O16 emax=%d: two-body operator %.1f MB; ladder scratch %.1f MB (%.2fx); "
        "Pandya scratch %.1f MB (%.2fx)\n",
        emax, op_bytes / 1e6, lad / 1e6, lad / op_bytes, pan / 1e6, pan / op_bytes);
    EXPECT_GT(lad, 0.0);
    EXPECT_GT(pan, 0.0);
  }
}

}  // namespace
