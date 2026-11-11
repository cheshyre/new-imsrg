// TermMask and term-name helpers.
#include <gtest/gtest.h>

#include "imsrg/commutator/options.hpp"

namespace {

using imsrg::commutator::CommutatorOptions;
using imsrg::commutator::Term;
using imsrg::commutator::TermMask;

TEST(commutator_options, set_reset_test) {
  TermMask m;
  EXPECT_TRUE(m.empty());
  m.set(Term::comm110ss).set(Term::comm122ss);
  EXPECT_TRUE(m.test(Term::comm110ss));
  EXPECT_TRUE(m.test(Term::comm122ss));
  EXPECT_FALSE(m.test(Term::comm220ss));
  m.reset(Term::comm110ss);
  EXPECT_FALSE(m.test(Term::comm110ss));
  EXPECT_EQ(m, TermMask::only(Term::comm122ss));
}

TEST(commutator_options, union_intersection_presets) {
  const TermMask a = TermMask::only(Term::comm110ss) | TermMask::only(Term::comm220ss);
  const TermMask b = TermMask::only(Term::comm220ss) | TermMask::only(Term::comm111ss);
  EXPECT_EQ(a & b, TermMask::only(Term::comm220ss));
  EXPECT_TRUE(TermMask::none().empty());
  const TermMask all = TermMask::imsrg2();
  for (int t = 0; t < imsrg::commutator::kNumTerms; ++t)
    EXPECT_TRUE(all.test(static_cast<Term>(t)));
  EXPECT_EQ(CommutatorOptions{}.terms, all);
}

TEST(commutator_options, name_round_trip) {
  for (int t = 0; t < imsrg::commutator::kNumTerms; ++t) {
    const auto term = static_cast<Term>(t);
    const auto back = imsrg::commutator::term_from_name(imsrg::commutator::term_name(term));
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, term);
  }
  EXPECT_FALSE(imsrg::commutator::term_from_name("comm999ss").has_value());
}

}  // namespace
