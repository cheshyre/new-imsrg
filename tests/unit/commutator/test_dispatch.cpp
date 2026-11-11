// commutator_spec, Z reuse and the term-mask dispatch of commutator().
#include <cstdio>

#include "commutator_test_cases.hpp"
#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/operator/arithmetic.hpp"

namespace {

using namespace imsrg;
using commutator::CommutatorOptions;
using commutator::Term;
using commutator::TermMask;
using commutator::Workspace;
using op::Operator;
using op::OperatorSpec;
using op::operator-=;

TEST(commutator_dispatch, spec_rules) {
  const auto A2 = OperatorSpec::scalar(2, Hermiticity::AntiHermitian);
  const auto H2 = OperatorSpec::scalar(2, Hermiticity::Hermitian);
  const auto A1 = OperatorSpec::scalar(1, Hermiticity::AntiHermitian);
  EXPECT_EQ(commutator::commutator_spec(A2, H2).hermiticity, Hermiticity::Hermitian);
  EXPECT_EQ(commutator::commutator_spec(H2, A2).hermiticity, Hermiticity::Hermitian);
  EXPECT_EQ(commutator::commutator_spec(H2, H2).hermiticity, Hermiticity::AntiHermitian);
  EXPECT_EQ(commutator::commutator_spec(A2, A2).hermiticity, Hermiticity::AntiHermitian);
  EXPECT_EQ(commutator::commutator_spec(A1, H2).particle_rank, 2);
  EXPECT_EQ(commutator::commutator_spec(A1, A1).particle_rank, 1);
  auto N2 = H2;
  N2.hermiticity = Hermiticity::NonHermitian;
  EXPECT_EQ(commutator::commutator_spec(N2, H2).hermiticity, Hermiticity::NonHermitian);
  EXPECT_THROW(commutator::commutator_spec(OperatorSpec::scalar(3, Hermiticity::Hermitian), H2),
               NotImplementedError);
}

TEST(commutator_dispatch, z_reuse_keeps_buffer) {
  const auto ms = imsrg::testing::model_space(1, "He4");
  Workspace ws(ms);
  const Operator X = imsrg::testing::random_operator(ms, Hermiticity::AntiHermitian, 5);
  const Operator Y = imsrg::testing::random_operator(ms, Hermiticity::Hermitian, 6);
  CommutatorOptions opts;
  opts.terms = TermMask::none();
  opts.terms.set(Term::comm110ss).set(Term::comm111ss).set(Term::comm122ss);
  Operator Z;
  commutator::commutator(X, Y, Z, opts, ws);
  const double* ptr = Z.two_body().matrix().buffer().device_data();
  const Operator first = Z;
  commutator::commutator(X, Y, Z, opts, ws);
  EXPECT_EQ(Z.two_body().matrix().buffer().device_data(), ptr);
  EXPECT_EQ(imsrg::testing::relative_deviation(Z, first), 0.0);  // reset, not accumulated
}

TEST(commutator_dispatch, simple_mask_matches_golden_sum) {
  CommutatorOptions opts;
  opts.terms = TermMask::none();
  for (Term t :
       {Term::comm110ss, Term::comm220ss, Term::comm111ss, Term::comm121ss, Term::comm122ss})
    opts.terms.set(t);
  for (const auto& p : commutator_test::golden_pairs()) {
    if (p.dir != "emax2_O16") continue;
    const auto ms = imsrg::testing::golden_model_space(commutator_test::dump_path(p, p.y));
    Workspace ws(ms);
    const Operator X = imsrg::testing::read_golden(commutator_test::dump_path(p, p.x), ms);
    const Operator Y = imsrg::testing::read_golden(commutator_test::dump_path(p, p.y), ms);
    const Operator Z = commutator::commutator(X, Y, opts, ws);
    // all = the five simple terms + the ladder/221 combination + the particle-hole term.
    Operator ref = imsrg::testing::read_golden(commutator_test::result_path(p, "all"), ms);
    ref -= imsrg::testing::read_golden(commutator_test::result_path(p, "comm222_pp_hh_221ss"), ms);
    ref -= imsrg::testing::read_golden(commutator_test::result_path(p, "comm222_phss"), ms);
    const double dev = imsrg::testing::relative_deviation(Z, ref);
    std::printf("[dispatch] %s, %s: %.3e\n", p.x.c_str(), p.y.c_str(), dev);
    EXPECT_LE(dev, commutator_test::kTol);
  }
}

TEST(commutator_dispatch, incompatible_model_space_throws) {
  const auto ms1 = imsrg::testing::model_space(1, "He4");
  const auto ms2 = imsrg::testing::model_space(2, "O16");
  Workspace ws(ms1);
  const Operator X = imsrg::testing::random_operator(ms2, Hermiticity::AntiHermitian, 1);
  const Operator Y = imsrg::testing::random_operator(ms1, Hermiticity::Hermitian, 2);
  Operator Z;
  EXPECT_THROW(commutator::commutator(X, Y, Z, CommutatorOptions{}, ws), InvariantError);
  EXPECT_THROW(commutator::commutator(Y, X, Z, CommutatorOptions{}, ws), InvariantError);
}

}  // namespace
