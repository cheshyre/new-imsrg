// The real emax-14 interaction read into the golden model spaces, compared with the harness dumps.
#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/io/me2j.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace {

using namespace imsrg;
namespace fs = std::filesystem;

const fs::path kMe2j =
    fs::path(IMSRG_TEST_DATA_DIR) / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";

/// Reads kMe2j into the model space of `dump` and compares the two-body part with the dump.
void compare_with_dump(const fs::path& dump, const std::string& tag) {
  if (!fs::exists(kMe2j)) GTEST_SKIP() << "me2j file not found: " << kMe2j;
  if (!fs::exists(dump)) GTEST_SKIP() << "golden dump not found: " << dump;
  const auto header = op::read_operator_dump_header(dump);
  const auto ms = op::model_space_for_dump(header);
  io::Me2jReadReport rep;
  const op::Operator V = io::read_me2j(kMe2j, {.file = {14, 28, 14}}, ms, &rep);
  const op::Operator ref = op::read_operator_dump(dump, ms);
  const double dev = op::max_abs_difference(V, ref).two_body;
  const double scale = op::max_abs(ref).two_body;
  std::cout << "[" << tag << "] values read " << rep.values_read << " in " << rep.seconds
            << " s; max |V - dump| = " << dev << " (max |V| = " << scale << ")\n";
  ::testing::Test::RecordProperty(tag + "_max_dev", std::to_string(dev));
  ::testing::Test::RecordProperty(tag + "_read_seconds", std::to_string(rep.seconds));
  ::testing::Test::RecordProperty(tag + "_values_read", std::to_string(rep.values_read));
  EXPECT_LT(dev, 1e-10);
  EXPECT_GT(scale, 1.0);  // the comparison is not vacuous
}

TEST(Me2jGolden, Emax2MatchesTheHarnessV2Dump) {
  compare_with_dump(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/V2.dump", "emax2");
}

TEST(Me2jGolden, Emax4MatchesTheGeneratedV2Dump) {
  compare_with_dump(fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16/V2.dump",
                    "emax4");
}

}  // namespace
