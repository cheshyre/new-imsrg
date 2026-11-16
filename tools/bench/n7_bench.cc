// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// n7_bench -- IMSRG(3)-N7 timing and checksum benchmark (design/30 section 5).
//
//   n7_bench --emax E [--e3max M] [--reference R] [--reps N] [--herm XY]
//            [--terms LIST] [--checksum] [--seed S]
//
// Operands X, Y = RandomThreeBodyOperator(ctx, h, S) and (ctx, h, S + 1) on
// MakeThreeBodySpace(E, R, {e3max = M}); `--herm` gives the hermiticities of X
// and Y as two letters (H, A, N). Each term of `--terms` (133, 330, 223, 232,
// 331, 231, 132, or `full`) is timed from Z = 0 after one untimed warm-up:
// `best_ms` and `mean_ms` over `--reps` fenced calls. `full` is
// `commutator::Commutator` with the IMSRG(3)-N7 term mask, so it also
// includes the IMSRG(2) terms. With `--checksum` nothing is timed; instead
// Z = term(X, Y) is computed once from Z = 0 and the FNV-1a 64 hash of the
// bytes of Z (zero-body value, one-, two-, three-body buffers) is printed, and
// the same hash of X and Y (are the inputs themselves reproducible?). This
// is the "before" tool of milestone M10 (N7 performance).

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "tests/support/three_body_test_support.h"

#ifndef IMSRG_BENCH_GIT_HASH
#define IMSRG_BENCH_GIT_HASH "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
namespace cm = imsrg::commutator;
namespace op = imsrg::op;
using imsrg::Hermiticity;

constexpr char kUsage[] =
    "usage: n7_bench --emax E [--e3max M] [--reference R] [--reps N]\n"
    "                [--herm XY] [--terms LIST] [--checksum] [--seed S]\n"
    "  XY: two letters of H, A, N (default AH); LIST: comma separated of\n"
    "  133,330,223,232,331,231,132,full\n";

struct Options {
  int emax = -1;
  int e3max = -1;
  std::string reference = "O16";
  int reps = 5;
  std::string herm = "AH";
  std::string terms = "133,330,223,232,331,231,132,full";
  bool checksum = false;
  std::uint64_t seed = 1;
};

/// Hermiticity of a letter H, A or N; throws std::runtime_error otherwise.
Hermiticity HermFromLetter(char c) {
  if (c == 'H') return Hermiticity::kHermitian;
  if (c == 'A') return Hermiticity::kAntiHermitian;
  if (c == 'N') return Hermiticity::kNonHermitian;
  throw std::runtime_error("--herm letters must be H, A or N");
}

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
      } else if (a == "--e3max") {
        o.e3max = std::stoi(value());
      } else if (a == "--reference") {
        o.reference = value();
      } else if (a == "--reps") {
        o.reps = std::stoi(value());
      } else if (a == "--herm") {
        o.herm = value();
      } else if (a == "--terms") {
        o.terms = value();
      } else if (a == "--checksum") {
        o.checksum = true;
      } else if (a == "--seed") {
        o.seed = std::stoull(value());
      } else {
        throw std::runtime_error("unknown option " + a);
      }
    } catch (const std::logic_error&) {  // std::stoi / std::stoull
      throw std::runtime_error("bad value for " + a);
    }
  }
  if (o.emax < 0) throw std::runtime_error("--emax is required");
  if (o.e3max < 0) o.e3max = 3 * o.emax;
  if (o.reps < 1) throw std::runtime_error("--reps must be >= 1");
  if (o.herm.size() != 2) throw std::runtime_error("--herm needs two letters");
  HermFromLetter(o.herm[0]);
  HermFromLetter(o.herm[1]);
  return o;
}

/// Splits `s` at commas.
std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

/// One benchmarked term: a name and a call Z (+)= term(X, Y).
struct Term {
  std::string name;
  std::function<void(const op::Operator&, const op::Operator&, op::Operator&)>
      run;
};

/// The term `name` ("133", ..., "full") on the context `ctx` and workspace
/// `ws`; throws std::runtime_error for an unknown name.
Term MakeTerm(const std::string& name, const cm::ThreeBodyContext& ctx,
              cm::Workspace& ws) {
  namespace tm = cm::terms;
  using Fn = void (*)(const op::Operator&, const op::Operator&, op::Operator&,
                      const cm::ThreeBodyContext&);
  const std::pair<const char*, Fn> table[] = {
      {"133", tm::Comm133ss}, {"330", tm::Comm330ss}, {"223", tm::Comm223ss},
      {"232", tm::Comm232ss}, {"331", tm::Comm331ss}, {"231", tm::Comm231ss},
      {"132", tm::Comm132ss}};
  for (const auto& [n, fn] : table) {
    if (name == n) {
      return {"comm" + name + "ss",
              [fn, &ctx](const op::Operator& x, const op::Operator& y,
                         op::Operator& z) { fn(x, y, z, ctx); }};
    }
  }
  if (name == "full") {
    cm::CommutatorOptions o;
    o.terms = cm::TermMask::Imsrg3N7();
    o.threebody_threshold = 0.0;
    return {"full",
            [o, &ws](const op::Operator& x, const op::Operator& y,
                     op::Operator& z) { cm::Commutator(x, y, z, o, ws); }};
  }
  throw std::runtime_error("unknown term " + name);
}

/// FNV-1a 64-bit hash of the bytes of `v`, continuing from `h`.
std::uint64_t Fnv1a(const std::vector<double>& v, std::uint64_t h) {
  for (const double d : v) {
    unsigned char b[sizeof(double)];
    std::memcpy(b, &d, sizeof(double));
    for (const unsigned char c : b) {
      h ^= c;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

/// Hash of the zero-body value and the one-, two- and three-body buffers of
/// `z` (device data copied to the host).
std::uint64_t Checksum(const op::Operator& z) {
  std::uint64_t h = 14695981039346656037ULL;
  h = Fnv1a({z.zero_body()}, h);
  h = Fnv1a(z.one_body().buffer().ToVector(), h);
  h = Fnv1a(z.two_body().matrix().buffer().ToVector(), h);
  if (z.three_body().has_value()) {
    h = Fnv1a(z.three_body()->buffer().ToVector(), h);
  }
  return h;
}

/// A zero rank-3 output with the hermiticity of [X, Y].
op::Operator ZeroOutput(const op::Operator& x, const op::Operator& y,
                        const cm::ThreeBodyContext& ctx) {
  op::OperatorSpec spec = cm::CommutatorSpec(x.spec(), y.spec());
  spec.particle_rank = 3;
  return op::Operator(ctx.model_space_ptr(), spec);
}

double ElapsedMs(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Prints the header lines (size of the problem, threads, context cost).
void PrintHeader(const Options& o, const op::Operator& x,
                 const cm::ThreeBodyContext& ctx, double ctx_ms) {
  const char* omp = std::getenv("OMP_NUM_THREADS");
  std::int64_t rows = 0;
  const op::ThreeBody& tb = *x.three_body();
  for (std::int32_t c = 0; c < tb.num_channels(); ++c) {
    rows += tb.dim(imsrg::Channel3Index{c});
  }
  std::printf("# n7_bench git %s\n", IMSRG_BENCH_GIT_HASH);
  std::printf("emax %d e3max %d reference %s herm %s seed %" PRIu64 "\n",
              o.emax, o.e3max, o.reference.c_str(), o.herm.c_str(), o.seed);
  std::printf("OMP_NUM_THREADS %s kokkos_concurrency %d space %s\n",
              omp != nullptr ? omp : "(unset)",
              imsrg::exec::Runtime::Concurrency(),
              imsrg::exec::Runtime::ExecutionSpaceName().c_str());
  std::printf("kept_3b_rows %" PRId64 " packed_doubles_per_operand %zu\n", rows,
              tb.size());
  std::printf("context_table_bytes %" PRId64 " context_build_ms %.1f\n",
              ctx.TableBytes(), ctx_ms);
}

/// Timing mode for one term: warm-up, then `reps` fenced calls from Z = 0.
void TimeTerm(const Term& t, const Options& o, const op::Operator& x,
              const op::Operator& y, op::Operator& z) {
  z.set_zero();
  t.run(x, y, z);  // untimed warm-up
  imsrg::exec::Fence("n7_bench warm-up");
  double best = std::numeric_limits<double>::max();
  double total = 0.0;
  for (int i = 0; i < o.reps; ++i) {
    z.set_zero();
    imsrg::exec::Fence("n7_bench pre");
    const auto t0 = Clock::now();
    t.run(x, y, z);
    imsrg::exec::Fence("n7_bench post");
    const double ms = ElapsedMs(t0);
    best = std::min(best, ms);
    total += ms;
  }
  std::printf("term %s best_ms %.3f mean_ms %.3f reps %d%s\n", t.name.c_str(),
              best, total / o.reps, o.reps,
              t.name == "full" ? " (includes the IMSRG(2) terms)" : "");
  std::fflush(stdout);
}

/// Checksum mode for one term: Z = term(X, Y) once from Z = 0.
void ChecksumTerm(const Term& t, const Options& o, const op::Operator& x,
                  const op::Operator& y, op::Operator& z) {
  z.set_zero();
  imsrg::exec::Fence("n7_bench pre");
  t.run(x, y, z);
  imsrg::exec::Fence("n7_bench post");
  z.SyncToHost();
  std::printf("checksum %s %s 0x%016" PRIx64 "\n", t.name.c_str(),
              o.herm.c_str(), Checksum(z));
}

/// Builds the operands and runs the selected terms.
void Run(const Options& o) {
  imsrg::basis::ThreeBodyCuts cuts;
  cuts.e3max = o.e3max;
  const auto tc0 = Clock::now();
  const cm::ThreeBodyContext ctx(
      imsrg::testing::MakeThreeBodySpace(o.emax, o.reference, cuts));
  imsrg::exec::Fence("n7_bench context");
  const double ctx_ms = ElapsedMs(tc0);
  const op::Operator x = imsrg::testing::RandomThreeBodyOperator(
      ctx, HermFromLetter(o.herm[0]), o.seed);
  const op::Operator y = imsrg::testing::RandomThreeBodyOperator(
      ctx, HermFromLetter(o.herm[1]), o.seed + 1);
  PrintHeader(o, x, ctx, ctx_ms);
  cm::Workspace ws(ctx.model_space_ptr());
  op::Operator z = ZeroOutput(x, y, ctx);
  if (o.checksum) {
    x.SyncToHost();
    y.SyncToHost();
    std::printf("checksum input_X %c 0x%016" PRIx64 "\n", o.herm[0],
                Checksum(x));
    std::printf("checksum input_Y %c 0x%016" PRIx64 "\n", o.herm[1],
                Checksum(y));
  }
  for (const std::string& name : Split(o.terms)) {
    const Term t = MakeTerm(name, ctx, ws);
    if (o.checksum) {
      ChecksumTerm(t, o, x, y, z);
    } else {
      TimeTerm(t, o, x, y, z);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = ParseOptions(argc, argv);
  } catch (const std::runtime_error& e) {
    std::fprintf(stderr, "n7_bench: %s\n%s", e.what(), kUsage);
    return 2;
  }
  try {
    imsrg::exec::Runtime runtime(argc, argv);
    Run(o);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "n7_bench: %s\n", e.what());
    return 3;
  }
}
