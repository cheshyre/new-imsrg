// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Writes a random scalar particle-rank-3 operator with a PHYSICAL three-body
// part (tests/support/three_body_test_support.h) as an OperatorDump v1, the
// input of the IMSRG(3)-N7 reference goldens (tools/reference/gen_golden_n7.sh,
// design/14 section 4 item 2):
//
//   imsrg_n7_inputs --emax E --reference R --e3max M --seed S
//                   --herm A|H|N --out FILE
//
// The operator is `RandomThreeBodyOperator(ThreeBodyContext(
// MakeThreeBodySpace(E, R, {e3max = M})), herm, S)`. Exit codes: 0 success,
// 2 usage error, 3 other error.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "tests/support/three_body_test_support.h"

namespace {

using imsrg::Hermiticity;

constexpr char kUsage[] =
    "usage: imsrg_n7_inputs --emax E --reference R --e3max M --seed S\n"
    "                       --herm A|H|N --out FILE\n";

struct Options {
  int emax = -1;
  std::string reference;
  int e3max = -1;
  std::uint64_t seed = 0;
  Hermiticity herm = Hermiticity::kHermitian;
  bool have_herm = false;
  bool have_seed = false;
  std::string out;
};

/// Parses the command line; throws std::runtime_error on any error.
Options ParseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    try {
      if (a == "--emax") {
        o.emax = std::stoi(value());
      } else if (a == "--reference") {
        o.reference = value();
      } else if (a == "--e3max") {
        o.e3max = std::stoi(value());
      } else if (a == "--seed") {
        o.seed = std::stoull(value());
        o.have_seed = true;
      } else if (a == "--herm") {
        const std::string h = value();
        if (h == "A") {
          o.herm = Hermiticity::kAntiHermitian;
        } else if (h == "H") {
          o.herm = Hermiticity::kHermitian;
        } else if (h == "N") {
          o.herm = Hermiticity::kNonHermitian;
        } else {
          throw std::runtime_error("--herm must be A, H or N");
        }
        o.have_herm = true;
      } else if (a == "--out") {
        o.out = value();
      } else {
        throw std::runtime_error("unknown option " + a);
      }
    } catch (const std::logic_error&) {  // std::stoi / std::stoull
      throw std::runtime_error("bad value for " + a);
    }
  }
  if (o.emax < 0 || o.reference.empty() || o.e3max < 0 || !o.have_seed ||
      !o.have_herm || o.out.empty()) {
    throw std::runtime_error(
        "--emax, --reference, --e3max, --seed, --herm and --out are required");
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = ParseOptions(argc, argv);
  } catch (const std::runtime_error& e) {
    std::fprintf(stderr, "imsrg_n7_inputs: %s\n%s", e.what(), kUsage);
    return 2;
  }
  try {
    imsrg::exec::Runtime runtime(argc, argv);
    imsrg::basis::ThreeBodyCuts cuts;
    cuts.e3max = o.e3max;
    const imsrg::commutator::ThreeBodyContext ctx(
        imsrg::testing::MakeThreeBodySpace(o.emax, o.reference, cuts));
    const imsrg::op::Operator x =
        imsrg::testing::RandomThreeBodyOperator(ctx, o.herm, o.seed);
    imsrg::op::OperatorDumpWriteOptions wo;
    wo.source = "imsrg n7_inputs RandomThreeBodyOperator seed " +
                std::to_string(o.seed);
    imsrg::op::WriteOperatorDump(o.out, x, wo);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "imsrg_n7_inputs: %s\n", e.what());
    return 3;
  }
}
