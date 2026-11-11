// commutator_bench -- IMSRG(2) commutator micro-benchmark (design/30 section 5).
//
//   commutator_bench --emax N --reps R [--reference O16] [--seed S]
//
// Times every `commutator::terms::` function and the full `commutator()` on random scalar
// two-body operators (X anti-Hermitian, Y Hermitian), after one warm-up call per term. The term
// functions add into Z (as in imsrg++), so Z accumulates over the repetitions; the full
// `commutator()` resets Z itself. The Workspace construction is timed separately. The
// counterpart for imsrg++ is `refharness bench-commutator` (tools/reference/rh_bench.cpp),
// which prints the same table.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/options.hpp"
#include "imsrg/commutator/workspace.hpp"
#include "imsrg/exec/kokkos_backend.hpp"
#include "imsrg/exec/runtime.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_spec.hpp"
#include "imsrg/operator/random_operator.hpp"

#ifndef IMSRG_BENCH_GIT_HASH
#define IMSRG_BENCH_GIT_HASH "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
namespace cm = imsrg::commutator;
namespace op = imsrg::op;

struct Row {
  std::string name;
  double mean_ms = 0.0;
  double min_ms = 0.0;
};

/// One warm-up call, then `reps` fenced timed calls of `fn`; returns mean and min in ms.
Row time_it(const std::string& name, int reps, const std::function<void()>& fn) {
  fn();
  imsrg::exec::fence("bench warm-up");
  Row r;
  r.name = name;
  r.min_ms = std::numeric_limits<double>::max();
  double total = 0.0;
  for (int i = 0; i < reps; ++i) {
    imsrg::exec::fence("bench pre");
    const auto t0 = Clock::now();
    fn();
    imsrg::exec::fence("bench post");
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    total += ms;
    if (ms < r.min_ms) r.min_ms = ms;
  }
  r.mean_ms = total / reps;
  return r;
}

using TermFn = void (*)(const op::Operator&, const op::Operator&, op::Operator&, cm::Workspace&);

struct Named {
  const char* name;
  TermFn fn;
};

}  // namespace

int main(int argc, char** argv) {
  int emax = 4;
  int reps = 10;
  std::uint64_t seed = 1;
  std::string reference = "O16";
  imsrg::exec::Runtime runtime(argc, argv);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_val = i + 1 < argc;
    if (a == "--emax" && has_val) {
      emax = std::atoi(argv[++i]);
    } else if (a == "--reps" && has_val) {
      reps = std::atoi(argv[++i]);
    } else if (a == "--reference" && has_val) {
      reference = argv[++i];
    } else if (a == "--seed" && has_val) {
      seed = std::strtoull(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr,
                   "usage: commutator_bench --emax N --reps R [--reference O16] [--seed S]\n");
      return 2;
    }
  }
  if (reps < 1) return 2;

  const auto ms = imsrg::basis::ModelSpace::from_names(imsrg::basis::BasisTruncation{emax}, 16.0,
                                                       reference, reference);

  const auto tw0 = Clock::now();
  cm::Workspace ws(ms);
  imsrg::exec::fence("workspace");
  const double ws_ms = std::chrono::duration<double, std::milli>(Clock::now() - tw0).count();

  op::RandomOperatorOptions ro;
  ro.seed = seed;
  const op::Operator X =
      op::random_operator(ms, op::OperatorSpec::scalar(2, imsrg::Hermiticity::AntiHermitian), ro);
  ro.seed = seed + 1;
  const op::Operator Y =
      op::random_operator(ms, op::OperatorSpec::scalar(2, imsrg::Hermiticity::Hermitian), ro);
  op::Operator Z(ms, cm::commutator_spec(X.spec(), Y.spec()));

  const char* omp = std::getenv("OMP_NUM_THREADS");
  std::printf("# commutator_bench (this code)\n");
  std::printf("emax %d  reference %s  orbits %d  2B elements %lld  OMP_NUM_THREADS %s  git %s\n",
              emax, reference.c_str(), static_cast<int>(ms->num_orbits()),
              static_cast<long long>(X.two_body().matrix().total_size()), omp ? omp : "(unset)",
              IMSRG_BENCH_GIT_HASH);
  std::printf("reps %d\nworkspace construction: %.3f ms\n", reps, ws_ms);

  const Named named[] = {
      {"comm110ss", cm::terms::comm110ss},
      {"comm220ss", cm::terms::comm220ss},
      {"comm111ss", cm::terms::comm111ss},
      {"comm121ss", cm::terms::comm121ss},
      {"comm221ss", cm::terms::comm221ss},
      {"comm122ss", cm::terms::comm122ss},
      {"comm222_pp_hhss", cm::terms::comm222_pp_hhss},
      {"comm222_phss", cm::terms::comm222_phss},
      {"comm222_pp_hh_221ss", cm::terms::comm222_pp_hh_221ss},
  };
  std::vector<Row> rows;
  for (const Named& n : named) {
    Z.set_zero();
    rows.push_back(time_it(n.name, reps, [&] { n.fn(X, Y, Z, ws); }));
  }
  const cm::CommutatorOptions opts;
  rows.push_back(time_it("commutator (full)", reps, [&] { cm::commutator(X, Y, Z, opts, ws); }));

  std::printf("%-22s %12s %12s\n", "term", "mean ms", "min ms");
  for (const Row& r : rows)
    std::printf("%-22s %12.4f %12.4f\n", r.name.c_str(), r.mean_ms, r.min_ms);
  return 0;
}
