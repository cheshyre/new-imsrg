// Normal ordering of the NO2B three-nucleon force against imsrg-devel (V3_NO2B.dump) with the
// real 3N input file. Skips itself if the file (outside the repository) is absent.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/normal_ordering.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/operator_dump.hpp"

#ifndef IMSRG_GOLDEN_DIR
#error "IMSRG_GOLDEN_DIR must be defined"
#endif
#ifndef IMSRG_TEST_DATA_DIR
#error "IMSRG_TEST_DATA_DIR must be defined"
#endif

namespace {

using namespace imsrg;
using namespace imsrg::op;

struct FileOrbit {
  std::int32_t n, l, j, e;
};

/// Reads the number after `"key":` in the first object following `"section"` in `text`.
double json_number(const std::string& text, const std::string& section, const std::string& key) {
  const std::size_t sec = text.find("\"" + section + "\"");
  EXPECT_NE(sec, std::string::npos) << section;
  const std::size_t k = text.find("\"" + key + "\"", sec);
  EXPECT_NE(k, std::string::npos) << key;
  const std::size_t colon = text.find(':', k);
  return std::strtod(text.c_str() + colon + 1, nullptr);
}

/// Transcription of imsrg++ ThreeBodyStorage_no2b::ReadFile (mode "bin": raw little-endian
/// floats, read sequentially), filling `v3` with what its truncations can hold. Every matrix
/// element of the file consumes one float. Returns the number of floats consumed.
std::size_t read_no2b_file(const std::filesystem::path& path, No2bThreeBody& v3,
                           std::int32_t emax_file, std::int32_t e2max_file, std::int32_t e3max_file,
                           std::int32_t lmax_file) {
  const No2bTruncation tr = v3.truncation();
  std::vector<FileOrbit> orbs;
  for (std::int32_t e = 0; e <= emax_file; ++e) {
    for (std::int32_t l = e % 2; l <= std::min(e, lmax_file); l += 2) {
      const std::int32_t n = (e - l) / 2;
      for (std::int32_t j = std::abs(2 * l - 1); j <= 2 * l + 1; j += 2) {
        orbs.push_back({n, l, j, e});
      }
    }
  }
  std::ifstream in(path, std::ios::binary);
  EXPECT_TRUE(in.good());
  const auto next = [&]() {
    float f = 0.0F;
    in.read(reinterpret_cast<char*>(&f), sizeof(float));
    EXPECT_TRUE(in.good()) << "unexpected end of file";
    return f;
  };
  const auto idx = [&](const FileOrbit& o) {
    const auto i = v3.isospin_orbit_index(o.n, o.l, TwoJ{o.j});
    return i ? *i : -1;
  };
  const auto norb = static_cast<std::int32_t>(orbs.size());
  std::size_t count = 0;
  for (std::int32_t f1 = 0; f1 < norb; ++f1) {
    const FileOrbit& o1 = orbs[static_cast<std::size_t>(f1)];
    if (o1.e > tr.emax) {
      break;  // shells are ascending: no later first orbit is needed
    }
    for (std::int32_t f2 = 0; f2 <= f1; ++f2) {
      const FileOrbit& o2 = orbs[static_cast<std::size_t>(f2)];
      if (o1.e + o2.e > e2max_file) continue;
      for (std::int32_t f3 = 0; f3 < norb; ++f3) {
        const FileOrbit& o3 = orbs[static_cast<std::size_t>(f3)];
        if (o2.e + o3.e > e2max_file || o1.e + o3.e > e2max_file) continue;
        if (o1.e + o2.e + o3.e > e3max_file) continue;
        for (std::int32_t f4 = 0; f4 <= f1; ++f4) {
          const FileOrbit& o4 = orbs[static_cast<std::size_t>(f4)];
          for (std::int32_t f5 = 0; f5 <= f4; ++f5) {
            const FileOrbit& o5 = orbs[static_cast<std::size_t>(f5)];
            if (o4.e + o5.e > e2max_file) continue;
            for (std::int32_t f6 = 0; f6 < norb; ++f6) {
              const FileOrbit& o6 = orbs[static_cast<std::size_t>(f6)];
              if (o6.j != o3.j || o6.l != o3.l) continue;
              if (o4.e + o6.e > e2max_file || o5.e + o6.e > e2max_file) continue;
              if (o4.e + o5.e + o6.e > e3max_file) continue;
              if ((o1.l + o2.l + o3.l) % 2 != (o4.l + o5.l + o6.l) % 2) continue;
              const std::int32_t jmin = std::max(std::abs(o1.j - o2.j), std::abs(o4.j - o5.j)) / 2;
              const std::int32_t jmax = std::min(o1.j + o2.j, o4.j + o5.j) / 2;
              for (std::int32_t J = jmin; J <= jmax; ++J) {
                for (const std::int32_t T12 : {0, 1}) {
                  for (const std::int32_t T45 : {0, 1}) {
                    for (std::int32_t T3 = 1; T3 <= std::min(2 * T12 + 1, 2 * T45 + 1); T3 += 2) {
                      const double v = next();
                      ++count;
                      const std::int32_t i1 = idx(o1), i2 = idx(o2), i3 = idx(o3);
                      const std::int32_t i4 = idx(o4), i5 = idx(o5), i6 = idx(o6);
                      if (std::max({o1.e, o2.e, o3.e, o4.e, o5.e, o6.e}) > tr.emax) continue;
                      if (o1.e + o2.e > tr.e2max || o1.e + o3.e > tr.e2max ||
                          o2.e + o3.e > tr.e2max || o4.e + o5.e > tr.e2max ||
                          o4.e + o6.e > tr.e2max || o5.e + o6.e > tr.e2max) {
                        continue;
                      }
                      if (o1.e + o2.e + o3.e > tr.e3max || o4.e + o5.e + o6.e > tr.e3max) continue;
                      if ((i1 == i2 && (J + T12) % 2 == 0) || (i4 == i5 && (J + T45) % 2 == 0)) {
                        continue;
                      }
                      if (std::abs(v) > 1e-8) {
                        v3.set_isospin(i1, i2, i3, T12, i4, i5, i6, T45, J, T3, v);
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
  return count;
}

TEST(No2bGolden, V3NormalOrderedO16) {
  const std::filesystem::path file =
      std::filesystem::path(IMSRG_TEST_DATA_DIR) /
      "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";
  if (!std::filesystem::exists(file)) {
    GTEST_SKIP() << "3N input file not found: " << file;
  }
  No2bThreeBody v3(No2bTruncation{2, 4, 6, 2});
  const std::size_t nread = read_no2b_file(file, v3, 6, 12, 18, 6);
  std::printf("[ info ] floats read: %zu, stored values: %zu\n", nread, v3.size());

  const std::filesystem::path dump =
      std::filesystem::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/V3_NO2B.dump";
  const auto ms = model_space_for_dump(read_operator_dump_header(dump));
  const Operator ref = read_operator_dump(dump, ms);
  const Operator zero(ms, OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const Operator got = normal_order(zero, v3);

  const PartMaxima rm = max_abs(ref);
  const PartMaxima dev = max_abs_difference(got, ref);
  const double tol = 1e-10 * std::max(1.0, rm.max());
  std::printf("[ dev ] V3_NO2B zero %.3e  one %.3e  two %.3e  (tol %.1e)\n", dev.zero_body,
              dev.one_body, dev.two_body, tol);
  EXPECT_LE(dev.zero_body, tol);
  EXPECT_LE(dev.one_body, tol);
  EXPECT_LE(dev.two_body, tol);

  std::ifstream js(std::filesystem::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/norms.json");
  std::stringstream ss;
  ss << js.rdbuf();
  const std::string text = ss.str();
  const double z_ref = json_number(text, "V3_NO2B", "zero_body");
  const double n1_ref = json_number(text, "V3_NO2B", "one_body_norm");
  const double n2_ref = json_number(text, "V3_NO2B", "two_body_norm");
  const double n1 = one_body_norm(got);
  const double n2 = two_body_norm(got);
  std::printf("[ dev ] norms (relative): zero %.3e  one %.3e  two %.3e\n",
              std::abs(got.zero_body() - z_ref) / std::abs(z_ref), std::abs(n1 - n1_ref) / n1_ref,
              std::abs(n2 - n2_ref) / n2_ref);
  EXPECT_NEAR(got.zero_body(), z_ref, 1e-10 * std::abs(z_ref));
  EXPECT_NEAR(n1, n1_ref, 1e-10 * n1_ref);
  EXPECT_NEAR(n2, n2_ref, 1e-10 * n2_ref);
  EXPECT_NEAR(z_ref, -58.795424620310463, 1e-9);
}

}  // namespace
