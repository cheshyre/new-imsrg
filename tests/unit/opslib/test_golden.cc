// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// T_rel and R_p^2 against the harness dumps (imsrg++ output) at emax 2 and emax
// 4.
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/radius.h"

namespace imsrg {
namespace {

namespace fs = std::filesystem;

constexpr int kA = 16;
constexpr int kZ = 8;

/// Compares `built` with the dump at `dir/name`; prints the per-part
/// deviations.
void ExpectMatches(const op::Operator& built,
                   const std::shared_ptr<const basis::ModelSpace>& ms,
                   const fs::path& dump, const std::string& tag) {
  const op::Operator ref = op::ReadOperatorDump(dump, ms);
  const op::PartMaxima dev = op::MaxAbsDifference(built, ref);
  const op::PartMaxima scale = op::MaxAbs(ref);
  std::cout << "[" << tag << "] max |built - dump|: zero " << dev.zero_body
            << ", one-body " << dev.one_body << ", two-body " << dev.two_body
            << " (max |dump| " << scale.max() << ")\n";
  ::testing::Test::RecordProperty(tag + "_max_dev", std::to_string(dev.max()));
  EXPECT_LT(dev.max(), 1.0e-10);
  EXPECT_GT(scale.max(), 0.1);  // not a vacuous comparison
}

void CheckDir(const fs::path& dir, const std::string& tag) {
  const fs::path t_dump = dir / "T.dump";
  const fs::path rp2_dump = dir / "Rp2.dump";
  if (!fs::exists(t_dump) || !fs::exists(rp2_dump)) {
    GTEST_SKIP() << "golden dumps not found in " << dir;
  }
  const auto ms = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(t_dump));
  ExpectMatches(opslib::TRel(ms, kA), ms, t_dump, tag + "_Trel");
  ExpectMatches(opslib::Rp2(ms, kA, kZ), ms, rp2_dump, tag + "_Rp2");
}

TEST(OpslibGolden, Emax2TrelAndRp2) {
  CheckDir(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16", "emax2");
}

TEST(OpslibGolden, Emax4TrelAndRp2) {
  CheckDir(fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16",
           "emax4");
}

}  // namespace
}  // namespace imsrg
