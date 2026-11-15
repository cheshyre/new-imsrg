// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// TermMask and term-name helpers.
#include <gtest/gtest.h>

#include "imsrg/commutator/options.h"

namespace {

using imsrg::commutator::CommutatorOptions;
using imsrg::commutator::Term;
using imsrg::commutator::TermMask;

TEST(commutator_options, set_reset_test) {
  TermMask m;
  EXPECT_TRUE(m.empty());
  m.set(Term::kComm110ss).set(Term::kComm122ss);
  EXPECT_TRUE(m.test(Term::kComm110ss));
  EXPECT_TRUE(m.test(Term::kComm122ss));
  EXPECT_FALSE(m.test(Term::kComm220ss));
  m.reset(Term::kComm110ss);
  EXPECT_FALSE(m.test(Term::kComm110ss));
  EXPECT_EQ(m, TermMask::Only(Term::kComm122ss));
}

TEST(commutator_options, union_intersection_presets) {
  const TermMask a =
      TermMask::Only(Term::kComm110ss) | TermMask::Only(Term::kComm220ss);
  const TermMask b =
      TermMask::Only(Term::kComm220ss) | TermMask::Only(Term::kComm111ss);
  EXPECT_EQ(a & b, TermMask::Only(Term::kComm220ss));
  EXPECT_TRUE(TermMask::None().empty());
  const TermMask all = TermMask::Imsrg2();
  for (int t = 0; t < imsrg::commutator::kNumTerms; ++t)
    EXPECT_TRUE(all.test(static_cast<Term>(t)));
  EXPECT_EQ(CommutatorOptions{}.terms, all);
}

TEST(commutator_options, name_round_trip) {
  for (int t = 0; t < imsrg::commutator::kNumTerms; ++t) {
    const auto term = static_cast<Term>(t);
    const auto back =
        imsrg::commutator::TermFromName(imsrg::commutator::TermName(term));
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, term);
  }
  EXPECT_FALSE(imsrg::commutator::TermFromName("comm999ss").has_value());
}

}  // namespace
