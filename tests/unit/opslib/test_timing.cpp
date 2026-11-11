// Wall-clock timings of the two-body constructions at emax 6 (printed, not asserted).
#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/opslib/center_of_mass.hpp"
#include "imsrg/opslib/radius.hpp"

namespace {

using namespace imsrg;

TEST(OpslibTiming, Emax6TrelAndRp2) {
  const auto ms = basis::ModelSpace::from_names({.emax = 6}, 16.0, "O16", "O16");
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const op::Operator trel = opslib::t_rel(ms, 16);
  const auto t1 = clock::now();
  const op::Operator rp2 = opslib::rp2(ms, 16, 8);
  const auto t2 = clock::now();
  std::cout << "[emax6] t_rel " << std::chrono::duration<double>(t1 - t0).count() << " s, rp2 "
            << std::chrono::duration<double>(t2 - t1).count() << " s\n";
  EXPECT_GT(op::max_abs(trel).two_body, 0.0);
  EXPECT_GT(op::max_abs(rp2).two_body, 0.0);
}

}  // namespace
