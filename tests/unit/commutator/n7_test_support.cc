// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/unit/commutator/n7_test_support.h"

#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/operator_spec.h"
#include "tests/support/commutator_test_support.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg::testing {

namespace ms = commutator_ref::mscheme;
using commutator::Term;

N7Inputs RandomN7Inputs(std::shared_ptr<const basis::ModelSpace> ms3,
                        Hermiticity hx, Hermiticity hy, std::uint64_t seed) {
  N7Inputs in;
  in.ms3 = std::move(ms3);
  in.ctx = std::make_unique<commutator::ThreeBodyContext>(in.ms3);
  in.x = RandomThreeBodyOperator(*in.ctx, hx, seed);
  in.y = RandomThreeBodyOperator(*in.ctx, hy, seed + 1);
  return in;
}

N7Inputs GoldenN7Inputs(int emax, std::string_view reference, int e3max) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = e3max;
  N7Inputs in;
  in.ms3 = MakeThreeBodySpace(emax, reference, cuts);
  in.ctx = std::make_unique<commutator::ThreeBodyContext>(in.ms3);
  in.x = RandomThreeBodyOperator(*in.ctx, Hermiticity::kAntiHermitian, 11);
  in.y = RandomThreeBodyOperator(*in.ctx, Hermiticity::kHermitian, 12);
  return in;
}

op::Operator ZeroN7Output(const op::Operator& X, const op::Operator& Y,
                          std::shared_ptr<const basis::ModelSpace> ms3) {
  op::OperatorSpec spec = commutator::CommutatorSpec(X.spec(), Y.spec());
  spec.particle_rank = 3;
  return op::Operator(std::move(ms3), spec);
}

op::Operator EvaluateN7Term(Term t, const op::Operator& X,
                            const op::Operator& Y,
                            const commutator::ThreeBodyContext& ctx) {
  op::Operator z = ZeroN7Output(X, Y, ctx.model_space_ptr());
  namespace tm = commutator::terms;
  switch (t) {
    case Term::kComm330ss:
      tm::Comm330ss(X, Y, z, ctx);
      break;
    case Term::kComm331ss:
      tm::Comm331ss(X, Y, z, ctx);
      break;
    case Term::kComm231ss:
      tm::Comm231ss(X, Y, z, ctx);
      break;
    case Term::kComm132ss:
      tm::Comm132ss(X, Y, z, ctx);
      break;
    case Term::kComm232ss:
      tm::Comm232ss(X, Y, z, ctx);
      break;
    case Term::kComm133ss:
      tm::Comm133ss(X, Y, z, ctx);
      break;
    case Term::kComm223ss:
      tm::Comm223ss(X, Y, z, ctx);
      break;
    default:
      IMSRG_CHECK(false, "EvaluateN7Term: not an N7 term");
  }
  return z;
}

op::Operator BruteForceN7Term(Term t, const op::Operator& X,
                              const op::Operator& Y,
                              std::shared_ptr<const basis::ModelSpace> ms3) {
  const ms::MBasis basis(*ms3);
  const ms::MOperator a = ms::Expand(X, basis);
  const ms::MOperator b = ms::Expand(Y, basis);
  ms::MOperator z(basis.size());
  const auto add = [&](int na, int nb, int nc) {
    z.Axpy(1.0, ms::CommutatorPart(a, na, b, nb, nc, basis));
  };
  switch (t) {
    case Term::kComm330ss:
      add(3, 3, 0);
      break;
    case Term::kComm331ss:
      add(3, 3, 1);
      break;
    case Term::kComm231ss:
      add(2, 3, 1);
      add(3, 2, 1);
      break;
    case Term::kComm132ss:
      add(1, 3, 2);
      add(3, 1, 2);
      break;
    case Term::kComm232ss:
      add(2, 3, 2);
      add(3, 2, 2);
      break;
    case Term::kComm133ss:
      add(1, 3, 3);
      add(3, 1, 3);
      break;
    case Term::kComm223ss:
      add(2, 2, 3);
      break;
    default:
      IMSRG_CHECK(false, "BruteForceN7Term: not an N7 term");
  }
  op::OperatorSpec spec = commutator::CommutatorSpec(X.spec(), Y.spec());
  spec.particle_rank = 3;
  return ms::Contract(z, std::move(ms3), spec);
}

op::Operator ReadN7Golden(std::string_view case_dir, std::string_view file,
                          std::shared_ptr<const basis::ModelSpace> ms3) {
  return ReadGolden("n7/" + std::string(case_dir) + "/" + std::string(file),
                    std::move(ms3));
}

double N7GoldenValue(std::string_view case_dir, int e3max,
                     std::string_view name, std::string_view field) {
  const auto path = GoldenPath("n7/" + std::string(case_dir) + "/values.json");
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string s = buf.str();
  const auto fail = [&](const std::string& what) {
    ADD_FAILURE() << path << ": " << what;
    return std::numeric_limits<double>::quiet_NaN();
  };
  const std::size_t p0 = s.find("\"e3max\": " + std::to_string(e3max) + ",");
  if (p0 == std::string::npos) return fail("no case with this e3max");
  const std::size_t p1 = s.find("\"" + std::string(name) + "\": {", p0);
  if (p1 == std::string::npos) return fail("no entry " + std::string(name));
  const std::string key = "\"" + std::string(field) + "_str\": \"";
  const std::size_t p2 = s.find(key, p1);
  if (p2 == std::string::npos) return fail("no field " + std::string(field));
  const std::size_t b = p2 + key.size();
  const std::size_t e = s.find('"', b);
  return std::stod(s.substr(b, e - b));
}

}  // namespace imsrg::testing
