// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Wall-clock timings of the two-body constructions at emax 6 (printed, not
// asserted).
#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/radius.h"

namespace imsrg {
namespace {

TEST(OpslibTiming, Emax6TrelAndRp2) {
  const auto ms = basis::ModelSpace::FromNames({.emax = 6}, 16.0, "O16", "O16");
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const op::Operator trel = opslib::TRel(ms, 16);
  const auto t1 = Clock::now();
  const op::Operator rp2 = opslib::Rp2(ms, 16, 8);
  const auto t2 = Clock::now();
  std::cout << "[emax6] t_rel "
            << std::chrono::duration<double>(t1 - t0).count() << " s, rp2 "
            << std::chrono::duration<double>(t2 - t1).count() << " s\n";
  EXPECT_GT(op::MaxAbs(trel).two_body, 0.0);
  EXPECT_GT(op::MaxAbs(rp2).two_body, 0.0);
}

}  // namespace
}  // namespace imsrg
