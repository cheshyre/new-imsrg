// Slow: consumes the whole emax-14 file (168066656 values) and checks the format's value count.
#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/io/me2j.hpp"

namespace {

using namespace imsrg;
namespace fs = std::filesystem;

TEST(Me2jFullFile, WholeFileHasTheExpectedValueCount) {
  const fs::path file =
      fs::path(IMSRG_TEST_DATA_DIR) / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  if (!fs::exists(file)) GTEST_SKIP() << "me2j file not found: " << file;
  const auto ms = basis::ModelSpace::from_names({.emax = 2}, 16.0, "O16", "O16");
  io::Me2jReadReport rep;
  const auto H = io::read_me2j(file, {.file = {14, 28, 14}, .read_whole_file = true}, ms, &rep);
  (void)H;
  std::cout << "[full file] " << rep.values_read << " values in " << rep.seconds << " s\n";
  ::testing::Test::RecordProperty("full_file_seconds", std::to_string(rep.seconds));
  EXPECT_EQ(rep.values_read, 168066656);
}

}  // namespace
