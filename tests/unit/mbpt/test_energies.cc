// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// MP2 / MP3 energies against the imsrg-devel goldens (hf.json EMP2, EMP3_*).
#include "../hf/hf_test_support.h"
#include "imsrg/mbpt/energies.h"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

void Check(const fs::path& ham_dir, const fs::path& hfjson,
           const op::No2bThreeBody* v3, const std::string& tag, double tol) {
  op::Operator H = LoadHamiltonian(ham_dir);
  const hf::HFResult r = hf::Solve(H, v3);
  op::Operator hno = hf::NormalOrderedHamiltonian(H, v3, r);
  const Json j = ReadJson(hfjson);
  const double mp2 = mbpt::Mp2Energy(hno);
  const mbpt::Mp3Energy mp3 = mbpt::ComputeMp3Energy(hno);
  std::cout << "[" << tag << "] dMP2 " << mp2 - j.at("EMP2").number << " dpp "
            << mp3.pp - j.at("EMP3_pp").number << " dhh "
            << mp3.hh - j.at("EMP3_hh").number << " dph "
            << mp3.ph - j.at("EMP3_ph").number << " dMP3 "
            << mp3.total - j.at("EMP3").number << "\n";
  EXPECT_NEAR(mp2, j.at("EMP2").number, tol);
  EXPECT_NEAR(mp3.pp, j.at("EMP3_pp").number, tol);
  EXPECT_NEAR(mp3.hh, j.at("EMP3_hh").number, tol);
  EXPECT_NEAR(mp3.ph, j.at("EMP3_ph").number, tol);
  EXPECT_NEAR(mp3.total, j.at("EMP3").number, tol);
}

TEST(MbptEnergies, Emax2TwoNucleon) {
  Check(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
        fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N/hf.json", nullptr,
        "emax2_2N", 1.0e-9);
}

TEST(MbptEnergies, Emax2TwoPlusThreeNucleon) {
  const fs::path file = No2bFile();
  if (file.empty())
    GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = LoadV3(file, {2, 4, 12, 2});
  Check(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
        fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N3N/hf.json", &v3,
        "emax2_2N3N", 1.0e-9);
}

}  // namespace
