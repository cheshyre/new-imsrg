// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// tensor_commutator_bench -- scalar-tensor IMSRG(2) commutator benchmark
// (M15, design/30 section 5).
//
//   tensor_commutator_bench --emax N --rank L [--parity 0|1] [--reps R]
//                           [--reference O16] [--valence O16]
//
// Times each scalar-tensor term (`commutator::tensor_terms`) and the full
// `TensorCommutator` for a random anti-Hermitian scalar X and a random
// Hermitian rank-L tensor Y (1B + 2B), after a first call that builds the
// rank-L tables (timed separately: the 9j table dominates). Prints the peak
// resident set size at the end.

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/two_body.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace op = imsrg::op;
namespace cm = imsrg::commutator;

double Ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Uniform doubles in [-1, 1) from splitmix64.
class Rng {
 public:
  explicit Rng(std::uint64_t s) : s_(s) {}
  double operator()() {
    std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return 2.0 * static_cast<double>(z >> 11) * 0x1.0p-53 - 1.0;
  }

 private:
  std::uint64_t s_;
};

/// A random Hermitian tensor (hermiticity rules of `op::OperatorSpec`).
op::Operator RandomTensor(std::shared_ptr<const imsrg::basis::ModelSpace> ms,
                          const op::OperatorSpec& spec, std::uint64_t seed) {
  Rng u(seed);
  op::Operator t(ms, spec);
  const auto& b = ms->basis();
  for (int i = 0; i < b.num_orbits(); ++i)
    for (int j = i; j < b.num_orbits(); ++j) {
      const auto& oi = b.orbit(imsrg::OrbitIndex{i});
      const auto& oj = b.orbit(imsrg::OrbitIndex{j});
      if (!op::ConnectsOrbits(spec, oi, oj)) continue;
      const double v = u();
      const int ph = ((oi.j2.twice - oj.j2.twice) / 2) % 2 == 0 ? 1 : -1;
      t.one_body().set_element(imsrg::OrbitIndex{i}, imsrg::OrbitIndex{j}, v);
      t.one_body().set_element(imsrg::OrbitIndex{j}, imsrg::OrbitIndex{i},
                               ph * v);
    }
  auto& m = t.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto& info = m.block_info(k);
    auto blk = m.host_block(k);
    const bool diag = info.row_block == info.col_block;
    for (std::int64_t c = 0; c < info.cols; ++c)
      for (std::int64_t r = diag ? c : 0; r < info.rows; ++r) {
        const double v = u();
        blk(r, c) = v;
        if (diag) blk(c, r) = v;
      }
  }
  t.SyncToDevice();
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  imsrg::exec::Runtime runtime(argc, argv);
  int emax = 4;
  int rank = 2;
  int parity = 0;
  int reps = 1;
  std::string reference = "O16";
  std::string valence = "O16";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const char* v = argv[i + 1];
    if (k == "--emax") emax = std::atoi(v);
    if (k == "--rank") rank = std::atoi(v);
    if (k == "--parity") parity = std::atoi(v);
    if (k == "--reps") reps = std::atoi(v);
    if (k == "--reference") reference = v;
    if (k == "--valence") valence = v;
  }
  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{emax}, 16.0, reference, valence);
  op::RandomOperatorOptions ro;
  ro.seed = 7;
  const op::Operator X = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kAntiHermitian), ro);
  op::OperatorSpec ys = op::OperatorSpec::Scalar(2);
  ys.rank_J = rank;
  ys.parity = parity == 0 ? imsrg::Parity::kEven : imsrg::Parity::kOdd;
  const op::Operator Y = RandomTensor(ms, ys, 11);
  std::printf(
      "tensor_commutator_bench emax %d rank %d parity %d reference %s "
      "norb %d threads %d Y blocks %zu Y 2B doubles %.0f\n",
      emax, rank, parity, reference.c_str(), ms->num_orbits(),
      imsrg::exec::Runtime::Concurrency(), Y.two_body().matrix().num_blocks(),
      static_cast<double>(Y.two_body().matrix().total_size()));
  auto t0 = Clock::now();
  cm::TensorWorkspace ws(ms);
  std::printf("  workspace            %10.1f ms\n", Ms(t0));
  t0 = Clock::now();
  op::Operator Z = cm::TensorCommutator(X, Y, cm::TensorOptions{}, ws);
  std::printf("  first call (tables)  %10.1f ms\n", Ms(t0));
  const char* names[] = {"comm111st", "comm121st",       "comm122st",
                         "comm221st", "comm222_pp_hhst", "comm222_phst",
                         "all"};
  for (int t = 0; t < 7; ++t) {
    cm::TensorOptions o;
    o.comm111 = (t == 0 || t == 6);
    o.comm121 = (t == 1 || t == 6);
    o.comm122 = (t == 2 || t == 6);
    o.comm221 = (t == 3 || t == 6);
    o.comm222_pp_hh = (t == 4 || t == 6);
    o.comm222_ph = (t == 5 || t == 6);
    double best = 1e300;
    double sum = 0.0;
    for (int r = 0; r < reps; ++r) {
      t0 = Clock::now();
      cm::TensorCommutator(X, Y, Z, o, ws);
      const double ms_t = Ms(t0);
      best = ms_t < best ? ms_t : best;
      sum += ms_t;
    }
    std::printf("  %-20s %10.1f ms (mean %.1f)\n", names[t], best, sum / reps);
  }
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  std::printf("  peak RSS %.2f GB\n", static_cast<double>(ru.ru_maxrss) / 1e6);
  return 0;
}
