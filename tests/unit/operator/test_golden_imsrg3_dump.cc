// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The committed imsrg++ comm223ss result Z = [X, Y]_3 (tests/golden/imsrg3,
// written by tools/reference/gen_golden_imsrg3.sh through `refharness comm223
// out=Z3.dump`) read with our OperatorDump reader on a model space with a
// three-body space of the dump's E3max. The three-body norm of the operator
// read must equal the norm imsrg++ printed (relative 1e-12): this checks the
// dump path (canonical Ket3 labels, both triangles, Hermitian check) and our
// `ThreeBodyNorm` against the imsrg++ element storage.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"

namespace fs = std::filesystem;

namespace imsrg::op {
namespace {

std::string Slurp(const fs::path& p) {
  std::ifstream f(p);
  EXPECT_TRUE(f.good()) << p;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/// The `three_body_norm_str` of the values.json entry that has a dump (the
/// entries are written by gen_golden_imsrg3.sh in a fixed layout), and its
/// e3max.
struct Golden {
  int e3max = -1;
  double norm = 0.0;
};

Golden DumpEntry(const std::string& json) {
  static const auto* const kRe = new std::regex(
      R"re("e3max":\s*(\d+),\s*"three_body_norm":\s*[^,\s]+,\s*)re"
      R"re("three_body_norm_str":\s*"([^"]+)",\s*"triples_energy":\s*[^,\s]+,\s*)re"
      R"re("triples_energy_str":\s*"[^"]+",\s*"dump":\s*"Z3.dump")re");
  std::smatch m;
  EXPECT_TRUE(std::regex_search(json, m, *kRe)) << "no entry with a dump";
  return {std::stoi(m[1].str()), std::stod(m[2].str())};
}

void CheckCase(const char* name, const char* reference) {
  SCOPED_TRACE(name);
  const fs::path dir = fs::path(IMSRG_GOLDEN_DIR) / "imsrg3" / name;
  const Golden g = DumpEntry(Slurp(dir / "values.json"));
  ASSERT_GE(g.e3max, 0);
  const auto ms =
      basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                   reference, reference)
          ->WithThreeBody(basis::ThreeBodyCuts{.e3max = g.e3max});
  const Operator z = ReadOperatorDump(dir / "Z3.dump", ms);
  EXPECT_EQ(z.spec().particle_rank, 3);
  EXPECT_EQ(z.hermiticity(), Hermiticity::kHermitian);
  ASSERT_TRUE(z.three_body().has_value());
  const double n3 = ThreeBodyNorm(z);
  EXPECT_GT(n3, 0.0);
  EXPECT_NEAR(n3, g.norm, 1.0e-12 * g.norm)
      << "imsrg++ norm " << g.norm << ", ours " << n3;
  // The one- and two-body parts of this Z are identically zero.
  EXPECT_EQ(OneBodyNorm(z), 0.0);
  EXPECT_EQ(TwoBodyNorm(z), 0.0);
}

TEST(GoldenImsrg3Dump, He4Emax2) { CheckCase("He4_emax2", "He4"); }

TEST(GoldenImsrg3Dump, O16Emax2) { CheckCase("O16_emax2", "O16"); }

}  // namespace
}  // namespace imsrg::op
