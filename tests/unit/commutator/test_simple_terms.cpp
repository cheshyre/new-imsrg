// The simple terms (110, 220, 111, 121, 122) of the fast commutator: golden acceptance,
// term-by-term comparison with commutator_ref on random operators, and the NonHermitian guard.
#include <cstdio>
#include <filesystem>
#include <string>

#include "commutator_test_cases.hpp"
#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator_ref/commutator_ref.hpp"

namespace {

using namespace imsrg;
using commutator::Workspace;
using commutator_test::kTol;
using imsrg::testing::random_operator;
using imsrg::testing::read_golden;
using imsrg::testing::relative_deviation;
using op::Operator;

using FastFn = void (*)(const Operator&, const Operator&, Operator&, Workspace&);
using RefFn = void (*)(const Operator&, const Operator&, Operator&);

struct TermCase {
  const char* name;
  FastFn fast;
  RefFn ref;
};

const TermCase kTerms[] = {
    {"comm110ss", &commutator::terms::comm110ss, &commutator_ref::comm110ss},
    {"comm220ss", &commutator::terms::comm220ss, &commutator_ref::comm220ss},
    {"comm111ss", &commutator::terms::comm111ss, &commutator_ref::comm111ss},
    {"comm121ss", &commutator::terms::comm121ss, &commutator_ref::comm121ss},
    {"comm122ss", &commutator::terms::comm122ss, &commutator_ref::comm122ss},
};

TEST(commutator_simple_terms, golden) {
  for (const auto& p : commutator_test::golden_pairs()) {
    const auto ms = imsrg::testing::golden_model_space(commutator_test::dump_path(p, p.y));
    Workspace ws(ms);
    const Operator X = read_golden(commutator_test::dump_path(p, p.x), ms);
    const Operator Y = read_golden(commutator_test::dump_path(p, p.y), ms);
    for (const auto& t : kTerms) {
      const std::string file = commutator_test::result_path(p, t.name);
      if (!std::filesystem::exists(imsrg::testing::golden_path(file))) {
        std::printf("[golden] %s %s: %s, %s: no golden file, skipped\n", p.dir.c_str(), t.name,
                    p.x.c_str(), p.y.c_str());
        continue;
      }
      Operator Z(ms, commutator::commutator_spec(X.spec(), Y.spec()));
      t.fast(X, Y, Z, ws);
      const double dev = relative_deviation(Z, read_golden(file, ms));
      std::printf("[golden] %s %s: %s, %s: %.3e\n", p.dir.c_str(), t.name, p.x.c_str(), p.y.c_str(),
                  dev);
      EXPECT_LE(dev, kTol) << p.dir << " " << t.name << " " << p.x << " " << p.y;
    }
  }
}

TEST(commutator_simple_terms, fast_vs_reference) {
  for (const auto& sc : commutator_test::space_cases()) {
    const auto ms = commutator_test::make_space(sc);
    Workspace ws(ms);
    for (const auto& hp : commutator_test::herm_pairs()) {
      const Operator X = random_operator(ms, hp.x, 11, hp.rank_x);
      const Operator Y = random_operator(ms, hp.y, 29, hp.rank_y);
      for (const auto& t : kTerms) {
        const auto spec = commutator::commutator_spec(X.spec(), Y.spec());
        Operator Z(ms, spec);
        Operator Zref(ms, spec);
        t.fast(X, Y, Z, ws);
        t.ref(X, Y, Zref);
        const double dev = relative_deviation(Z, Zref);
        std::printf("[fast-vs-ref] %s %s %s: %.3e\n", sc.label.c_str(), hp.label, t.name, dev);
        EXPECT_LE(dev, kTol) << sc.label << " " << hp.label << " " << t.name;
      }
    }
  }
}

TEST(commutator_simple_terms, nonhermitian_input_throws) {
  const auto ms = imsrg::testing::model_space(1, "He4");
  Workspace ws(ms);
  const Operator X = random_operator(ms, Hermiticity::NonHermitian, 3);
  const Operator Y = random_operator(ms, Hermiticity::Hermitian, 4);
  Operator Z(ms, op::OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
  for (const auto& t : kTerms) {
    EXPECT_THROW(t.fast(X, Y, Z, ws), NotImplementedError) << t.name;
    EXPECT_THROW(t.fast(Y, X, Z, ws), NotImplementedError) << t.name;
  }
}

}  // namespace
