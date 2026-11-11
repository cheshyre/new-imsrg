// Golden OperatorDump files: byte-exact round trip through our reader and writer.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;
namespace fs = std::filesystem;

const char* const kDirs[] = {"hamiltonian/emax2_O16", "hf/emax2_O16_2N", "hf/emax2_O16_2N3N",
                             "commutator/emax2_O16", "operator/emax2"};

std::vector<fs::path> golden_dumps() {
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

std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

TEST(GoldenDumps, EveryDumpRoundTripsByteForByte) {
  const auto files = golden_dumps();
  ASSERT_GE(files.size(), 37U);
  for (const auto& p : files) {
    SCOPED_TRACE(p.string());
    const OperatorDumpHeader h = read_operator_dump_header(p);
    const auto ms = model_space_for_dump(h);
    const Operator op = read_operator_dump(p, ms);
    std::ostringstream os;
    write_operator_dump(os, op,
                        OperatorDumpWriteOptions{.source = h.source, .reference = h.reference});
    EXPECT_EQ(os.str(), slurp(p));
  }
}

class GoldenH : public ::testing::Test {
 protected:
  void SetUp() override {
    const fs::path p = fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/H.dump";
    ms = model_space_for_dump(read_operator_dump_header(p));
    op = std::make_unique<Operator>(read_operator_dump(p, ms));
  }
  std::shared_ptr<const basis::ModelSpace> ms;
  std::unique_ptr<Operator> op;
};

TEST_F(GoldenH, OneBodySpotValues) {
  EXPECT_EQ(op->one_body().get(OrbitIndex{0}, OrbitIndex{0}), 11.25);
  EXPECT_EQ(op->one_body().get(OrbitIndex{0}, OrbitIndex{10}), 9.1855865354369168);
}

TEST_F(GoldenH, TwoBodyNormalizedInBothOrbitOrders) {
  // File line: "0 8 0 8 1 1 0.17978553473949432" (p < q, r < s).
  const double v = 0.17978553473949432;
  const OrbitIndex a{0};
  const OrbitIndex b{8};
  const TwoJ J = TwoJ::from_integer(1);
  const auto& basis = ms->basis();
  const int ph = basis::exchange_phase(basis.orbit(a).j2, basis.orbit(b).j2, J);
  const auto& tb = op->two_body();
  EXPECT_EQ(tb.get_normalized(a, b, a, b, J), v);
  EXPECT_EQ(tb.get_normalized(b, a, a, b, J), ph * v);
  EXPECT_EQ(tb.get_normalized(a, b, b, a, J), ph * v);
  EXPECT_EQ(tb.get_normalized(b, a, b, a, J), v);
}

}  // namespace
