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
    EXPECT_EQ(all.test(static_cast<Term>(t)),
              t < imsrg::commutator::kNumImsrg2Terms);
  EXPECT_EQ(CommutatorOptions{}.terms, all);
  EXPECT_EQ(CommutatorOptions{}.threebody_threshold, 0.0);
  EXPECT_TRUE(all.ThreeBodyTerms().empty());
}

// The N7 preset is imsrg++'s SetUseIMSRG3N7 set (Commutator.cc:111-129): the
// IMSRG(2) terms plus comm330, 331, 231, 132, 232, 133, 223; the full preset
// adds the six N8/N9 terms.
TEST(commutator_options, imsrg3_presets) {
  const TermMask n7 = TermMask::Imsrg3N7();
  const TermMask expected_3b =
      TermMask::Only(Term::kComm330ss) | TermMask::Only(Term::kComm331ss) |
      TermMask::Only(Term::kComm231ss) | TermMask::Only(Term::kComm132ss) |
      TermMask::Only(Term::kComm232ss) | TermMask::Only(Term::kComm133ss) |
      TermMask::Only(Term::kComm223ss);
  EXPECT_EQ(n7, TermMask::Imsrg2() | expected_3b);
  EXPECT_EQ(n7.ThreeBodyTerms(), expected_3b);
  const TermMask full = TermMask::Imsrg3Full();
  for (int t = 0; t < imsrg::commutator::kNumTerms; ++t)
    EXPECT_TRUE(full.test(static_cast<Term>(t)));
  EXPECT_FALSE(n7.test(Term::kComm233PpHhss));
  EXPECT_FALSE(n7.test(Term::kComm333PphHhpss));
  EXPECT_EQ(imsrg::commutator::TermName(Term::kComm332PpphHhhpss),
            "comm332_ppph_hhhpss");
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
