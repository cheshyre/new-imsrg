// SPDX-License-Identifier: GPL-3.0-only
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

/// The spin-orbit correction against the patched imsrg++ `Rso` dump (D35),
/// relative to the largest dump element.
TEST(OpslibGolden, Emax2RsoRelative1e10) {
  const fs::path dir = fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16";
  const auto ms =
      op::ModelSpaceForDump(op::ReadOperatorDumpHeader(dir / "T.dump"));
  const op::Operator ref = op::ReadOperatorDump(dir / "Rso.dump", ms);
  const op::Operator built = opslib::RpSpinOrbitCorrection(ms);
  const op::PartMaxima dev = op::MaxAbsDifference(built, ref);
  const double scale = op::MaxAbs(ref).max();
  std::cout << "[emax2_Rso] max |built - dump| " << dev.max() << " (max |dump| "
            << scale << ", relative " << dev.max() / scale << ")\n";
  ::testing::Test::RecordProperty("emax2_Rso_max_dev",
                                  std::to_string(dev.max()));
  EXPECT_GT(scale, 0.1);
  EXPECT_LT(dev.max(), 1.0e-10 * scale);
}

/// D35-2: the original imsrg++ Rp2 equals the new Rp2 plus Rso/Z. The numbers
/// are diagonal elements of the pre-D35 `Rp2.dump` (pristine imsrg++, before
/// the regeneration; design/30 section 6).
TEST(OpslibGolden, OldRp2IsNewRp2PlusRsoOverZ) {
  const fs::path dir = fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16";
  const auto ms =
      op::ModelSpaceForDump(op::ReadOperatorDumpHeader(dir / "T.dump"));
  const op::Operator rp2 = op::ReadOperatorDump(dir / "Rp2.dump", ms);
  const op::Operator rso = op::ReadOperatorDump(dir / "Rso.dump", ms);
  struct Old {
    int n, l, j2, tz2;
    double value;
  };
  const Old kOld[] = {{0, 2, 3, -1, 0.98968387},
                      {0, 1, 3, -1, 0.74670537},
                      {0, 1, 3, +1, 0.01475002}};
  for (const Old& o : kOld) {
    bool found = false;
    for (std::int32_t i = 0; i < ms->num_orbits(); ++i) {
      const basis::Orbit& orb = ms->basis().orbit(OrbitIndex{i});
      if (orb.n == o.n && orb.l == o.l && orb.j2.twice == o.j2 &&
          orb.tz2.twice == o.tz2) {
        const double v = rp2.one_body().get(OrbitIndex{i}, OrbitIndex{i}) +
                         rso.one_body().get(OrbitIndex{i}, OrbitIndex{i}) / kZ;
        EXPECT_NEAR(v, o.value, 5.0e-9);
        found = true;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(OpslibGolden, Emax4TrelAndRp2) {
  CheckDir(fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16",
           "emax4");
}

}  // namespace
}  // namespace imsrg
