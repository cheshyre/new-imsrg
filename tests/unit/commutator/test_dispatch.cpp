// commutator_spec, Z reuse, the term-mask dispatch of commutator(), and the full IMSRG(2)
// commutator against the imsrg-devel golden data and commutator_ref.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "commutator_test_cases.hpp"
#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator_ref/commutator_ref.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/compare.hpp"

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

// The full IMSRG(2) commutator (default options) against imsrg-devel's Commutator::Commutator.
TEST(commutator_dispatch, full_commutator_matches_golden_all) {
  for (const auto& p : commutator_test::golden_pairs()) {
    if (p.dir != "emax2_O16") continue;
    const auto ms = imsrg::testing::golden_model_space(commutator_test::dump_path(p, p.y));
    Workspace ws(ms);
    const Operator X = imsrg::testing::read_golden(commutator_test::dump_path(p, p.x), ms);
    const Operator Y = imsrg::testing::read_golden(commutator_test::dump_path(p, p.y), ms);
    const Operator Z = commutator::commutator(X, Y, CommutatorOptions{}, ws);
    const Operator ref = imsrg::testing::read_golden(commutator_test::result_path(p, "all"), ms);
    const double dev = imsrg::testing::relative_deviation(Z, ref);
    std::printf("[full vs golden all] %s, %s: %.3e\n", p.x.c_str(), p.y.c_str(), dev);
    EXPECT_LE(dev, commutator_test::kTol);
  }
}

// Largest |Z_ba - h Z_ab| over the one-body part and every two-body block (h = +1 Hermitian,
// -1 anti-Hermitian), relative to max(1, max|Z|).
double hermiticity_violation(const Operator& Z) {
  const double h = Z.hermiticity() == Hermiticity::Hermitian ? 1.0 : -1.0;
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
  return worst / std::max(1.0, op::max_abs(Z).max());
}

// Fast commutator vs commutator_ref on random operators (every model space and hermiticity pair
// of the shared cases, incl. fractional occupations and a rank-1 X), plus [X,Y] = -[Y,X] and the
// hermiticity of Z for the fast path.
TEST(commutator_dispatch, full_commutator_matches_reference_and_properties) {
  double worst_ref = 0.0;
  double worst_anti = 0.0;
  double worst_herm = 0.0;
  std::uint64_t seed = 900;
  for (const auto& sc : commutator_test::space_cases()) {
    const auto ms = commutator_test::make_space(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::herm_pairs()) {
      const Operator X = imsrg::testing::random_operator(ms, hp.x, ++seed, hp.rank_x);
      const Operator Y = imsrg::testing::random_operator(ms, hp.y, ++seed, hp.rank_y);
      const Operator Z = commutator::commutator(X, Y, CommutatorOptions{}, ws);
      const Operator Zref = commutator_ref::commutator(X, Y);
      const double dref = imsrg::testing::relative_deviation(Z, Zref);
      Operator Zswap = commutator::commutator(Y, X, CommutatorOptions{}, ws);
      Zswap *= -1.0;
      const double danti = imsrg::testing::relative_deviation(Zswap, Z);
      const double dherm = hermiticity_violation(Z);
      EXPECT_LE(dref, commutator_test::kTol) << sc.label << " " << hp.label;
      EXPECT_LE(danti, 1e-12) << sc.label << " " << hp.label;
      EXPECT_LE(dherm, 1e-12) << sc.label << " " << hp.label;
      worst_ref = std::max(worst_ref, dref);
      worst_anti = std::max(worst_anti, danti);
      worst_herm = std::max(worst_herm, dherm);
    }
  }
  std::printf("[full vs ref] max %.3e; antisymmetry max %.3e; hermiticity max %.3e\n", worst_ref,
              worst_anti, worst_herm);
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
