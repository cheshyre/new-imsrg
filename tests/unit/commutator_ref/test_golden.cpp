// Golden acceptance of the reference commutator terms against imsrg-devel dumps (design/30
// section 2): relative_deviation <= 1e-10 for every term and pair.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "commutator_test_support.hpp"
#include <gtest/gtest.h>

#include "imsrg/commutator_ref/commutator_ref.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/operator.hpp"

namespace {

using imsrg::op::operator-=;

using imsrg::op::Operator;
using imsrg::testing::golden_model_space;
using imsrg::testing::read_golden;
using imsrg::testing::relative_deviation;
namespace cr = imsrg::commutator_ref;

using TermFn = void (*)(const Operator&, const Operator&, Operator&);

struct Term {
  const char* name;
  TermFn fn;
};

constexpr double kTol = 1e-10;

const Term kTerms[] = {
    {"comm110ss", &cr::comm110ss},
    {"comm220ss", &cr::comm220ss},
    {"comm111ss", &cr::comm111ss},
    {"comm121ss", &cr::comm121ss},
    {"comm221ss", &cr::comm221ss},
    {"comm122ss", &cr::comm122ss},
    {"comm222_pp_hhss", &cr::comm222_pp_hhss},
    {"comm222_phss", &cr::comm222_phss},
    {"comm222_pp_hh_221ss", &cr::comm222_pp_hh_221ss},
};

struct Pair {
  const char* x;
  const char* y;
};
const Pair kPairs[] = {{"eta_white", "H"}, {"random_A", "random_H"}};

std::string dump_path(const std::string& name) { return "commutator/emax2_O16/" + name + ".dump"; }

double max_abs_one_body(const Operator& Z) {
  const auto& basis = Z.model_space().basis();
  double m = 0.0;
  for (int i = 0; i < Z.model_space().num_orbits(); ++i) {
    for (const auto j : basis.same_ljtz(imsrg::OrbitIndex{i})) {
      m = std::max(m, std::abs(Z.one_body().get(imsrg::OrbitIndex{i}, j)));
    }
  }
  return m;
}

void check_term(const Term& t, const Pair& p) {
  const auto ms = golden_model_space(dump_path("H"));
  const Operator X = read_golden(dump_path(p.x), ms);
  const Operator Y = read_golden(dump_path(p.y), ms);
  Operator Z(ms, cr::commutator_spec(X.spec(), Y.spec()));
  t.fn(X, Y, Z);
  const std::string zname = std::string("Z_") + t.name + "__" + p.x + "__" + p.y;
  const std::string zfile = "commutator/emax2_O16/" + zname + ".dump";
  if (!std::filesystem::exists(imsrg::testing::golden_path(zfile))) {
    // Z_comm111ss__eta_white__H.dump is absent by design (its tiny entries fail the dump
    // reader's relative hermiticity check). The README claims the term is ~1e-16 noise, but the
    // reference value is ~1e-8 (the one-body parts of eta_white and H nearly commute). The term
    // is therefore checked against "all" minus the other golden terms, which is exactly the
    // README consistency relation (all = 110 + 220 + 111 + 121 + 122 + 222_pp_hh_221 + 222_ph).
    ASSERT_STREQ(t.name, "comm111ss");
    const std::string base = std::string("__") + p.x + "__" + p.y + ".dump";
    Operator rest = read_golden("commutator/emax2_O16/Z_all" + base, ms);
    for (const char* other : {"comm110ss", "comm220ss", "comm121ss", "comm122ss",
                              "comm222_pp_hh_221ss", "comm222_phss"}) {
      rest -= read_golden(std::string("commutator/emax2_O16/Z_") + other + base, ms);
    }
    const double m = max_abs_one_body(Z);
    const double dev = relative_deviation(Z, rest);
    std::printf("[golden] %s %s,%s: %.3e (derived reference from all; max|Z1| = %.3e)\n", t.name,
                p.x, p.y, dev, m);
    EXPECT_LE(dev, kTol);
    return;
  }
  const Operator ref = read_golden(zfile, ms);
  const double dev = relative_deviation(Z, ref);
  std::printf("[golden] %s %s,%s: %.3e\n", t.name, p.x, p.y, dev);
  EXPECT_LE(dev, kTol);
}

TEST(commutator_ref_golden, comm110ss_eta_white_H) { check_term(kTerms[0], kPairs[0]); }
TEST(commutator_ref_golden, comm110ss_random) { check_term(kTerms[0], kPairs[1]); }
TEST(commutator_ref_golden, comm220ss_eta_white_H) { check_term(kTerms[1], kPairs[0]); }
TEST(commutator_ref_golden, comm220ss_random) { check_term(kTerms[1], kPairs[1]); }
TEST(commutator_ref_golden, comm111ss_eta_white_H) { check_term(kTerms[2], kPairs[0]); }
TEST(commutator_ref_golden, comm111ss_random) { check_term(kTerms[2], kPairs[1]); }
TEST(commutator_ref_golden, comm121ss_eta_white_H) { check_term(kTerms[3], kPairs[0]); }
TEST(commutator_ref_golden, comm121ss_random) { check_term(kTerms[3], kPairs[1]); }
TEST(commutator_ref_golden, comm221ss_eta_white_H) { check_term(kTerms[4], kPairs[0]); }
TEST(commutator_ref_golden, comm221ss_random) { check_term(kTerms[4], kPairs[1]); }
TEST(commutator_ref_golden, comm122ss_eta_white_H) { check_term(kTerms[5], kPairs[0]); }
TEST(commutator_ref_golden, comm122ss_random) { check_term(kTerms[5], kPairs[1]); }
TEST(commutator_ref_golden, comm222_pp_hhss_eta_white_H) { check_term(kTerms[6], kPairs[0]); }
TEST(commutator_ref_golden, comm222_pp_hhss_random) { check_term(kTerms[6], kPairs[1]); }
TEST(commutator_ref_golden, comm222_phss_eta_white_H) { check_term(kTerms[7], kPairs[0]); }
TEST(commutator_ref_golden, comm222_phss_random) { check_term(kTerms[7], kPairs[1]); }
TEST(commutator_ref_golden, comm222_pp_hh_221ss_eta_white_H) { check_term(kTerms[8], kPairs[0]); }
TEST(commutator_ref_golden, comm222_pp_hh_221ss_random) { check_term(kTerms[8], kPairs[1]); }

void check_full(const Pair& p) {
  const auto ms = golden_model_space(dump_path("H"));
  const Operator X = read_golden(dump_path(p.x), ms);
  const Operator Y = read_golden(dump_path(p.y), ms);
  const Operator Z = cr::commutator(X, Y);
  const Operator ref =
      read_golden(std::string("commutator/emax2_O16/Z_all__") + p.x + "__" + p.y + ".dump", ms);
  const double dev = relative_deviation(Z, ref);
  std::printf("[golden] all %s,%s: %.3e\n", p.x, p.y, dev);
  EXPECT_LE(dev, kTol);
}

TEST(commutator_ref_golden, full_eta_white_H) { check_full(kPairs[0]); }
TEST(commutator_ref_golden, full_random) { check_full(kPairs[1]); }

}  // namespace
