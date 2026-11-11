// MP2 / MP3 energies against the imsrg-devel goldens (hf.json EMP2, EMP3_*).
#include "../hf/hf_test_support.hpp"

#include "imsrg/mbpt/energies.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

void check(const fs::path& ham_dir, const fs::path& hfjson, const op::No2bThreeBody* v3,
           const std::string& tag, double tol) {
  op::Operator H = load_hamiltonian(ham_dir);
  const hf::HFResult r = hf::solve(H, v3);
  op::Operator hno = hf::normal_ordered_hamiltonian(H, v3, r);
  const Json j = read_json(hfjson);
  const double mp2 = mbpt::mp2_energy(hno);
  const mbpt::Mp3Energy mp3 = mbpt::mp3_energy(hno);
  std::cout << "[" << tag << "] dMP2 " << mp2 - j.at("EMP2").number << " dpp "
            << mp3.pp - j.at("EMP3_pp").number << " dhh " << mp3.hh - j.at("EMP3_hh").number
            << " dph " << mp3.ph - j.at("EMP3_ph").number << " dMP3 "
            << mp3.total - j.at("EMP3").number << "\n";
  EXPECT_NEAR(mp2, j.at("EMP2").number, tol);
  EXPECT_NEAR(mp3.pp, j.at("EMP3_pp").number, tol);
  EXPECT_NEAR(mp3.hh, j.at("EMP3_hh").number, tol);
  EXPECT_NEAR(mp3.ph, j.at("EMP3_ph").number, tol);
  EXPECT_NEAR(mp3.total, j.at("EMP3").number, tol);
}

TEST(MbptEnergies, Emax2TwoNucleon) {
  check(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
        fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N/hf.json", nullptr, "emax2_2N", 1e-9);
}

TEST(MbptEnergies, Emax2TwoPlusThreeNucleon) {
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = load_v3(file, {2, 4, 12, 2});
  check(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
        fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N3N/hf.json", &v3, "emax2_2N3N", 1e-9);
}

}  // namespace
