// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Golden OperatorDump files: byte-exact round trip through our reader and
// writer.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/operator/operator_dump.h"

namespace {

using namespace imsrg;
using namespace imsrg::op;
namespace fs = std::filesystem;

const char* const kDirs[] = {"hamiltonian/emax2_O16", "hf/emax2_O16_2N",
                             "hf/emax2_O16_2N3N", "commutator/emax2_O16",
                             "operator/emax2"};

std::vector<fs::path> GoldenDumps() {
  std::vector<fs::path> out;
  for (const char* d : kDirs) {
    const fs::path dir = fs::path(IMSRG_GOLDEN_DIR) / d;
    if (!fs::exists(dir)) continue;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() == ".dump") out.push_back(e.path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string Slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

TEST(GoldenDumps, EveryDumpRoundTripsByteForByte) {
  const auto files = GoldenDumps();
  ASSERT_GE(files.size(), 37U);
  for (const auto& p : files) {
    SCOPED_TRACE(p.string());
    const OperatorDumpHeader h = ReadOperatorDumpHeader(p);
    const auto ms = ModelSpaceForDump(h);
    const Operator op = ReadOperatorDump(p, ms);
    std::ostringstream os;
    WriteOperatorDump(
        os, op,
        OperatorDumpWriteOptions{.source = h.source, .reference = h.reference});
    EXPECT_EQ(os.str(), Slurp(p));
  }
}

class GoldenH : public ::testing::Test {
 protected:
  void SetUp() override {
    const fs::path p =
        fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/H.dump";
    ms_ = ModelSpaceForDump(ReadOperatorDumpHeader(p));
    op_ = std::make_unique<Operator>(ReadOperatorDump(p, ms_));
  }
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::unique_ptr<Operator> op_;
};

TEST_F(GoldenH, OneBodySpotValues) {
  EXPECT_EQ(op_->one_body().get(OrbitIndex{0}, OrbitIndex{0}), 11.25);
  EXPECT_EQ(op_->one_body().get(OrbitIndex{0}, OrbitIndex{10}),
            9.1855865354369168);
}

TEST_F(GoldenH, TwoBodyNormalizedInBothOrbitOrders) {
  // File line: "0 8 0 8 1 1 0.17978553473949432" (p < q, r < s).
  const double v = 0.17978553473949432;
  const OrbitIndex a{0};
  const OrbitIndex b{8};
  const TwoJ J = TwoJ::FromInteger(1);
  const auto& basis = ms_->basis();
  const int ph = basis::ExchangePhase(basis.orbit(a).j2, basis.orbit(b).j2, J);
  const auto& tb = op_->two_body();
  EXPECT_EQ(tb.get_normalized(a, b, a, b, J), v);
  EXPECT_EQ(tb.get_normalized(b, a, a, b, J), ph * v);
  EXPECT_EQ(tb.get_normalized(a, b, b, a, J), ph * v);
  EXPECT_EQ(tb.get_normalized(b, a, b, a, J), v);
}

}  // namespace
