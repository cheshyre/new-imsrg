// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// `output.isospin_average_operators`: the write mode of scalar operators.
#include <gtest/gtest.h>

#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/pipeline/operator_frame.h"
#include "imsrg/io/valence_writers.h"

namespace imsrg::driver::pipeline {
namespace {

TEST(IsospinAverageOperators, WriteModeFollowsConfig) {
  RunConfig c;
  EXPECT_TRUE(c.output.isospin_average_operators);
  EXPECT_EQ(ScalarOperatorWriteMode(c),
            io::ValenceWriteMode::kIsospinAveragedOperator);
  c.output.isospin_average_operators = false;
  EXPECT_EQ(ScalarOperatorWriteMode(c), io::ValenceWriteMode::kHamiltonian);
}

}  // namespace
}  // namespace imsrg::driver::pipeline
