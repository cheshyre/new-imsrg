// The particle-hole (Pandya) term comm222_phss of the fast commutator: golden acceptance,
// comparison with commutator_ref on random operators, a timing print and the NonHermitian guard.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

#include "commutator_test_cases.hpp"
#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/pandya.hpp"
#include "imsrg/commutator_ref/commutator_ref.hpp"

namespace {

using namespace imsrg;
using commutator::Workspace;
using commutator_test::kTol;
using imsrg::testing::random_operator;
using imsrg::testing::read_golden;
using imsrg::testing::relative_deviation;
using op::Operator;

TEST(commutator_particle_hole, golden) {
  for (const auto& p : commutator_test::golden_pairs()) {
    const auto ms = imsrg::testing::golden_model_space(commutator_test::dump_path(p, p.y));
    Workspace ws(ms);
    const Operator X = read_golden(commutator_test::dump_path(p, p.x), ms);
    const Operator Y = read_golden(commutator_test::dump_path(p, p.y), ms);
    const std::string file = commutator_test::result_path(p, "comm222_phss");
    if (!std::filesystem::exists(imsrg::testing::golden_path(file))) {
      std::printf("[golden] %s comm222_phss: %s, %s: no golden file, skipped\n", p.dir.c_str(),
                  p.x.c_str(), p.y.c_str());
      continue;
    }
    Operator Z(ms, commutator::commutator_spec(X.spec(), Y.spec()));
    commutator::terms::comm222_phss(X, Y, Z, ws);
    const double dev = relative_deviation(Z, read_golden(file, ms));
    std::printf("[golden] %s comm222_phss: %s, %s: %.3e\n", p.dir.c_str(), p.x.c_str(), p.y.c_str(),
                dev);
    EXPECT_LE(dev, kTol) << p.dir << " " << p.x << " " << p.y;
  }
}

TEST(commutator_particle_hole, fast_vs_reference) {
  double worst = 0.0;
  for (const auto& sc : commutator_test::space_cases()) {
    const auto ms = commutator_test::make_space(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::herm_pairs()) {
      const Operator X = random_operator(ms, hp.x, 11, hp.rank_x);
      const Operator Y = random_operator(ms, hp.y, 29, hp.rank_y);
      const auto spec = commutator::commutator_spec(X.spec(), Y.spec());
      Operator Z(ms, spec);
      Operator Zref(ms, spec);
      commutator::terms::comm222_phss(X, Y, Z, ws);
      commutator_ref::comm222_phss(X, Y, Zref);
      const double dev = relative_deviation(Z, Zref);
      worst = dev > worst ? dev : worst;
      std::printf("[fast-vs-ref] %s %s comm222_phss: %.3e\n", sc.label.c_str(), hp.label, dev);
      EXPECT_LE(dev, kTol) << sc.label << " " << hp.label;
    }
  }
  std::printf("[fast-vs-ref] comm222_phss max deviation: %.3e\n", worst);
}

TEST(commutator_particle_hole, nonhermitian_input_throws) {
  const auto ms = imsrg::testing::model_space(1, "He4");
  Workspace ws(ms);
  const Operator X = random_operator(ms, Hermiticity::NonHermitian, 3);
  const Operator Y = random_operator(ms, Hermiticity::Hermitian, 4);
  Operator Z(ms, op::OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
  EXPECT_THROW(commutator::terms::comm222_phss(X, Y, Z, ws), NotImplementedError);
  EXPECT_THROW(commutator::terms::comm222_phss(Y, X, Z, ws), NotImplementedError);
}

TEST(commutator_particle_hole, Timing) {
  const auto ms = imsrg::testing::model_space(4, "O16");
  Workspace ws(ms);
  const Operator A = random_operator(ms, Hermiticity::AntiHermitian, 5);
  const Operator H = random_operator(ms, Hermiticity::Hermitian, 6);
  Operator Z(ms, commutator::commutator_spec(A.spec(), H.spec()));
  commutator::terms::comm222_phss(A, H, Z, ws);  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 20; ++i) commutator::terms::comm222_phss(A, H, Z, ws);
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("[timing] comm222_phss emax 4 O16: %.3f ms per call\n",
              1.0e3 * std::chrono::duration<double>(t1 - t0).count() / 20.0);
  // The term accumulates into Z, so compare a single fresh call with the reference.
  Operator Zfast(ms, commutator::commutator_spec(A.spec(), H.spec()));
  Operator Zref(ms, commutator::commutator_spec(A.spec(), H.spec()));
  const Operator zero(ms, commutator::commutator_spec(A.spec(), H.spec()));
  commutator::terms::comm222_phss(A, H, Zfast, ws);
  commutator_ref::comm222_phss(A, H, Zref);
  const double dev = relative_deviation(Zfast, Zref);
  const double mag = relative_deviation(Zref, zero);
  std::printf("[timing] emax 4 fast-vs-ref: %.3e (reference magnitude %.3e)\n", dev, mag);
  EXPECT_LE(dev, kTol);
  const auto pd = commutator::detail::build_pandya_data(*ms);
  std::printf("[timing] Pandya scratch at emax 4: %.2f MB\n",
              static_cast<double>(pd.scratch_bytes()) / 1.0e6);
}

}  // namespace
