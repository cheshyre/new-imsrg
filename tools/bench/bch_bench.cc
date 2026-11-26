// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// bch_bench -- one BCH transform e^Omega H e^-Omega (M14 device residency).
//
//   bch_bench --emax N [--omega-norm W] [--reps R] [--reference O16]
//
// Random anti-Hermitian Omega (scaled to ||Omega|| = W, default 0.5) and
// Hermitian H (scalar, 1B + 2B); times `flow::BchTransform` (IMSRG(2)
// commutators) after one warm-up call and prints the number of nested
// commutators and the time per commutator. Run it under `rocprofv3
// --memory-copy-trace` to check that a BCH series moves no operator data
// between host and device (design/12 section 2, M14).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/random_operator.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace op = imsrg::op;
namespace cm = imsrg::commutator;

double Ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  imsrg::exec::Runtime runtime(argc, argv);
  int emax = 4;
  int reps = 1;
  double omega_norm = 0.5;
  std::string reference = "O16";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const char* v = argv[i + 1];
    if (k == "--emax") emax = std::atoi(v);
    if (k == "--reps") reps = std::atoi(v);
    if (k == "--omega-norm") omega_norm = std::atof(v);
    if (k == "--reference") reference = v;
  }
  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{emax}, 16.0, reference, reference);
  cm::Workspace ws(ms);
  op::RandomOperatorOptions ro;
  ro.seed = 11;
  op::Operator omega = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kAntiHermitian), ro);
  op::Scale(omega, omega_norm / op::Norm(omega));
  ro.seed = 12;
  const op::Operator H = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kHermitian), ro);
  const imsrg::flow::BchOptions opts;
  const cm::CommutatorOptions copts;
  imsrg::flow::BchReport report;
  op::Operator Z =
      imsrg::flow::BchTransform(H, omega, opts, copts, ws, &report);
  imsrg::exec::Fence("bch warm-up");
  double best = 1.0e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    Z = imsrg::flow::BchTransform(H, omega, opts, copts, ws, &report);
    imsrg::exec::Fence("bch");
    const double t = Ms(t0);
    if (t < best) best = t;
  }
  std::printf(
      "bch_bench emax %d reference %s ||Omega|| %.3g: %d nested "
      "commutators, best %.1f ms (%.2f ms per commutator), E0 %.12e\n",
      emax, reference.c_str(), omega_norm, report.nested_commutators, best,
      best / report.nested_commutators, Z.zero_body());
  return 0;
}
