// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Work distribution of run.task = "transform" (design/50 C10): work-list
// reader, cost model, static assignment, failure manifest.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/transform/farm.h"

namespace imsrg::driver::transform {
namespace {

namespace fs = std::filesystem;

fs::path TempFile(const std::string& name) {
  return fs::temp_directory_path() / ("imsrg_farm_" + name);
}

TEST(TransformFarm, ReadWorkListSkipsCommentsAndReasons) {
  const fs::path p = TempFile("work.txt");
  {
    std::ofstream o(p);
    o << "# header\n  FM0p_15  \n\nTCM\tsome reason\n#x\r\nFDelta5n_1050\r\n";
  }
  EXPECT_EQ(ReadWorkList(p),
            (std::vector<std::string>{"FM0p_15", "TCM", "FDelta5n_1050"}));
  fs::remove(p);
  EXPECT_THROW(ReadWorkList(TempFile("does_not_exist")), InputError);
}

TEST(TransformFarm, FailureManifestIsAWorkList) {
  const fs::path p = TempFile("failed.txt");
  const std::vector<Failure> f{{"FM2p_30", "bad\nthing\twith tab"},
                               {"TCM", "x"}};
  WriteFailureManifest(p, f);
  EXPECT_EQ(ReadWorkList(p), (std::vector<std::string>{"FM2p_30", "TCM"}));
  WriteFailureManifest(p, {});
  EXPECT_TRUE(ReadWorkList(p).empty());
  fs::remove(p);
}

TEST(TransformFarm, CommutatorWeightScalarAndTensor) {
  const auto ms = basis::ModelSpace::FromNames({.emax = 3, .lmax = 3}, 16.0,
                                               "O16", "sd-shell");
  double scalar = 0.0;
  for (const auto& ch : ms->two_body().channels()) {
    scalar += 2.0 * ch.num_kets * ch.num_kets * ch.num_kets;
  }
  EXPECT_DOUBLE_EQ(CommutatorWeight(*ms, 0, Parity::kEven), scalar);
  // Rank 1 has more channel pairs than a scalar; at emax 3 the high ranks
  // run out of channel pairs (at emax 10 the weight grows up to L = 5,
  // design/50 section 6.1), so only positivity is checked beyond L = 1.
  EXPECT_GT(CommutatorWeight(*ms, 1, Parity::kEven), scalar);
  for (int L = 1; L <= 5; ++L) {
    const double w = CommutatorWeight(*ms, L, Parity::kEven);
    std::printf("rank %d weight / scalar = %.3f\n", L, w / scalar);
    EXPECT_GT(w, 0.0);
  }
  // Parity-odd operators connect channels of opposite parity only.
  EXPECT_GT(CommutatorWeight(*ms, 1, Parity::kOdd), 0.0);
}

TEST(TransformFarm, ItemCostUsesTheShorthandRank) {
  const auto ms = basis::ModelSpace::FromNames({.emax = 2, .lmax = 2}, 16.0,
                                               "O16", "sd-shell");
  const auto req = [](const std::string& s) {
    return ParseOperatorSpec(s, ModelSpaceInfo{.emax = 2}).value().front();
  };
  EXPECT_DOUBLE_EQ(ItemCost(req("FM0p_15"), *ms, 2),
                   2.0 * CommutatorWeight(*ms, 0, Parity::kEven));
  EXPECT_DOUBLE_EQ(
      ItemCost(req("FSigmap3n_15"), *ms, 3),
      3.0 * CommutatorWeight(*ms, 3, Parity::kEven) * TensorCostFactor(3));
  // TCM: its preparation at basis.emax dominates (AL-F3).
  EXPECT_DOUBLE_EQ(
      ItemCost(req("TCM"), *ms, 1),
      (1.0 + kTcmPreparationCost) * CommutatorWeight(*ms, 0, Parity::kEven));
}

TEST(TransformFarm, TensorCostFactorGrowsWithTheRank) {
  EXPECT_DOUBLE_EQ(TensorCostFactor(0), 1.0);
  for (int L = 1; L <= 5; ++L) {
    EXPECT_GT(TensorCostFactor(L), TensorCostFactor(L - 1));
  }
  EXPECT_NEAR(TensorCostFactor(5), 8.82, 1.0e-12);
}

TEST(TransformFarm, AssignStaticIsBalancedAndDeterministic) {
  // 34 operators x 71 q like the 27Al farm, with rank-dependent costs.
  std::vector<double> costs;
  for (int op = 0; op < 34; ++op) {
    for (int q = 0; q < 71; ++q) costs.push_back(1.0 + (op % 6));
  }
  for (const int ranks : {1, 7, 64, 2414, 3000}) {
    const std::vector<int> owner = AssignStatic(costs, ranks);
    EXPECT_EQ(owner, AssignStatic(costs, ranks));
    std::vector<double> load(static_cast<std::size_t>(ranks), 0.0);
    for (std::size_t i = 0; i < costs.size(); ++i) {
      ASSERT_GE(owner[i], 0);
      ASSERT_LT(owner[i], ranks);
      load[static_cast<std::size_t>(owner[i])] += costs[i];
    }
    const double total = std::accumulate(costs.begin(), costs.end(), 0.0);
    const double max = *std::max_element(load.begin(), load.end());
    // LPT: makespan <= mean + largest item.
    EXPECT_LE(max, total / ranks + 6.0) << ranks;
  }
}

}  // namespace
}  // namespace imsrg::driver::transform
