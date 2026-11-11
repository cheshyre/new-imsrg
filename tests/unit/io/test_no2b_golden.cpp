// The real NO2B 3N file read into the golden model spaces, normal-ordered and compared with the
// harness dumps; and a check against an independent transcription of imsrg++ ReadFile.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/io/no2b.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/normal_ordering.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;
namespace fs = std::filesystem;

const fs::path kFile =
    fs::path(IMSRG_TEST_DATA_DIR) / "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";
constexpr io::No2bFileTruncation kFileTrunc{6, 12, 18, 6};

void compare_with_dump(const fs::path& dump, const No2bTruncation& tr, const std::string& tag) {
  if (!fs::exists(kFile)) GTEST_SKIP() << "NO2B file not found: " << kFile;
  if (!fs::exists(dump)) GTEST_SKIP() << "golden dump not found: " << dump;
  No2bThreeBody v3(tr);
  io::No2bReadReport rep;
  io::read_no2b(kFile, {.file = kFileTrunc}, v3, &rep);
  const auto ms = model_space_for_dump(read_operator_dump_header(dump));
  const Operator ref = read_operator_dump(dump, ms);
  const Operator zero(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const Operator got = normal_order(zero, v3);
  const PartMaxima dev = max_abs_difference(got, ref);
  const PartMaxima rm = max_abs(ref);
  const double tol = 1e-10 * std::max(1.0, rm.max());
  std::cout << "[" << tag << "] read " << rep.values_read << " values (" << rep.values_stored
            << " stored) in " << rep.seconds << " s; max deviation 0b " << dev.zero_body << ", 1b "
            << dev.one_body << ", 2b " << dev.two_body << " (max |ref| " << rm.max() << ")\n";
  ::testing::Test::RecordProperty(tag + "_max_dev", std::to_string(dev.max()));
  ::testing::Test::RecordProperty(tag + "_read_seconds", std::to_string(rep.seconds));
  EXPECT_LT(dev.max(), tol);
  EXPECT_GT(rm.max(), 1.0);  // not vacuous
}

TEST(No2bReaderGolden, Emax2MatchesTheHarnessDump) {
  compare_with_dump(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/V3_NO2B.dump",
                    No2bTruncation{2, 4, 6, 2}, "emax2");
}

TEST(No2bReaderGolden, Emax4MatchesTheGeneratedDump) {
  compare_with_dump(fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16/V3_NO2B.dump",
                    No2bTruncation{4, 8, 12, 4}, "emax4");
}

/// Independent compact transcription of imsrg++ ReadFile for a binary float file (whole-file
/// slurp, no skipping tricks), as in the M4 operator test.
std::int64_t reference_read(const fs::path& path, No2bThreeBody& v3) {
  struct O {
    int n, l, j, e;
  };
  const auto& f = kFileTrunc;
  std::vector<O> orbs;
  for (int e = 0; e <= f.e1max; ++e) {
    for (int l = e % 2; l <= std::min(e, f.lmax); l += 2) {
      for (int j = std::abs(2 * l - 1); j <= 2 * l + 1; j += 2)
        orbs.push_back({(e - l) / 2, l, j, e});
    }
  }
  std::ifstream in(path, std::ios::binary);
  std::vector<float> all(static_cast<std::size_t>(fs::file_size(path) / 4));
  in.read(reinterpret_cast<char*>(all.data()), static_cast<std::streamsize>(all.size() * 4));
  std::size_t pos = 0;
  const No2bTruncation tr = v3.truncation();
  const auto idx = [&](const O& o) { return *v3.isospin_orbit_index(o.n, o.l, TwoJ{o.j}); };
  std::int64_t stored = 0;
  for (const O& o1 : orbs) {
    if (o1.e > tr.emax) break;
    for (const O& o2 : orbs) {
      if (&o2 > &o1) break;
      if (o1.e + o2.e > f.e2max) continue;
      for (const O& o3 : orbs) {
        if (o2.e + o3.e > f.e2max || o1.e + o3.e > f.e2max || o1.e + o2.e + o3.e > f.e3max) {
          continue;
        }
        for (const O& o4 : orbs) {
          if (&o4 > &o1) break;
          for (const O& o5 : orbs) {
            if (&o5 > &o4) break;
            if (o4.e + o5.e > f.e2max) continue;
            for (const O& o6 : orbs) {
              if (o6.j != o3.j || o6.l != o3.l) continue;
              if (o4.e + o6.e > f.e2max || o5.e + o6.e > f.e2max) continue;
              if (o4.e + o5.e + o6.e > f.e3max) continue;
              if ((o1.l + o2.l + o3.l) % 2 != (o4.l + o5.l + o6.l) % 2) continue;
              const int jmin = std::max(std::abs(o1.j - o2.j), std::abs(o4.j - o5.j)) / 2;
              const int jmax = std::min(o1.j + o2.j, o4.j + o5.j) / 2;
              for (int J = jmin; J <= jmax; ++J) {
                for (int T12 : {0, 1}) {
                  for (int T45 : {0, 1}) {
                    for (int T3 = 1; T3 <= std::min(2 * T12 + 1, 2 * T45 + 1); T3 += 2) {
                      const double v = all[pos++];
                      if (std::max({o1.e, o2.e, o3.e, o4.e, o5.e, o6.e}) > tr.emax) continue;
                      if (o1.e + o2.e > tr.e2max || o1.e + o3.e > tr.e2max ||
                          o2.e + o3.e > tr.e2max || o4.e + o5.e > tr.e2max ||
                          o4.e + o6.e > tr.e2max || o5.e + o6.e > tr.e2max) {
                        continue;
                      }
                      if (o1.e + o2.e + o3.e > tr.e3max || o4.e + o5.e + o6.e > tr.e3max) continue;
                      if ((&o1 == &o2 && (J + T12) % 2 == 0) ||
                          (&o4 == &o5 && (J + T45) % 2 == 0)) {
                        continue;
                      }
                      if (std::abs(v) > 1e-8) {
                        v3.set_isospin(idx(o1), idx(o2), idx(o3), T12, idx(o4), idx(o5), idx(o6),
                                       T45, J, T3, v);
                        ++stored;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return stored;
}

TEST(No2bReaderGolden, FillsTheSameContainerAsTheTranscription) {
  if (!fs::exists(kFile)) GTEST_SKIP() << "NO2B file not found: " << kFile;
  const No2bTruncation tr{2, 4, 6, 2};
  No2bThreeBody mine(tr), theirs(tr);
  io::No2bReadReport rep;
  io::read_no2b(kFile, {.file = kFileTrunc}, mine, &rep);
  const std::int64_t stored = reference_read(kFile, theirs);
  EXPECT_EQ(rep.values_stored, stored);
  EXPECT_EQ(mine.size(), theirs.size());
  EXPECT_EQ(mine.norm(), theirs.norm());
  EXPECT_GT(stored, 1000);
}

TEST(No2bReaderGolden, FullEmax6ReadIsFast) {
  if (!fs::exists(kFile)) GTEST_SKIP() << "NO2B file not found: " << kFile;
  No2bThreeBody v3(No2bTruncation{6, 12, 18, 6});
  io::No2bReadReport rep;
  io::read_no2b(kFile, {.file = kFileTrunc}, v3, &rep);
  std::cout << "[emax6] read " << rep.values_read << " values (" << rep.values_stored
            << " stored) in " << rep.seconds << " s\n";
  ::testing::Test::RecordProperty("emax6_read_seconds", std::to_string(rep.seconds));
  EXPECT_EQ(rep.values_read, 32685840);
  EXPECT_LT(rep.seconds, 5.0);  // generous: includes set_isospin for every stored value
}

}  // namespace
