// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// n7_bench -- IMSRG(3)-N7 timing and checksum benchmark (design/30 section 5).
//
//   n7_bench --emax E [--e3max M] [--reference R] [--reps N] [--herm XY]
//            [--terms LIST] [--checksum] [--seed S] [--count-only]
//            [--triples]
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
//
// `--count-only` builds neither a context nor an operator: it prints the
// exact counts of the kept space (rows, triples, pair-third states, packed
// size) and the per-term multiply-add totals of `commutator::N7CostModel`
// with the time at two sustained rates and the memory of six operators.
//
// `--triples` runs the perturbative triples on the reduced
// `commutator::TriplesContext` instead of the N7 terms (no three-body space,
// so any emax is possible with `--count-only`): `--count-only` builds the
// context but no operator and runs no kernel, and prints its channels,
// triples (hhh, ppp), kept rows, pair-third states, the context bytes and the
// multiply-adds of the tiles; without it two random IMSRG(2) operands
// (`--herm`, `--seed`) are made and `PerturbativeTriplesEnergy` is timed
// (`--reps`; emax <= 3 only by the project rules).

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
#include "imsrg/commutator/n7_cost.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/triples.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
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
    "                [--count-only] [--triples]\n"
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
  bool count_only = false;
  bool triples = false;
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
      } else if (a == "--count-only") {
        o.count_only = true;
      } else if (a == "--triples") {
        o.triples = true;
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

/// `--count-only`: counts and cost model, without context or operators.
void RunCountOnly(const Options& o) {
  imsrg::basis::ThreeBodyCuts cuts;
  cuts.e3max = o.e3max;
  const auto t0 = Clock::now();
  const auto ms = imsrg::testing::MakeThreeBodySpace(o.emax, o.reference, cuts);
  const double space_ms = ElapsedMs(t0);
  const auto t1 = Clock::now();
  const cm::N7CostModel model(*ms);
  const double model_ms = ElapsedMs(t1);
  std::printf("# n7_bench --count-only git %s\n", IMSRG_BENCH_GIT_HASH);
  std::printf("emax %d e3max %d reference %s\n", o.emax, o.e3max,
              o.reference.c_str());
  std::printf("space_build_ms %.0f model_build_ms %.0f\n", space_ms, model_ms);
  std::printf("orbits %d onebody_mean %.3f\n", model.num_orbits(),
              model.OneBodyMean());
  std::printf("rows %" PRId64 " triples %" PRId64 " states %" PRId64
              " active_states %" PRId64 " max_N %d max_channel_states %" PRId64
              "\n",
              model.num_rows(), model.num_triples(), model.num_states(),
              model.NumActive(), model.MaxDim(), model.max_channel_states());
  std::printf("packed_doubles %" PRId64 " packed_GB_per_operand %.3f\n",
              model.PackedDoubles(), model.PackedBytes() / 1.0e9);
  const double twobody_gb =
      8.0e-9 * static_cast<double>(model.TwoBodyDoubles());
  std::printf("twobody_doubles %" PRId64 " twobody_GB %.3f\n",
              model.TwoBodyDoubles(), twobody_gb);
  const auto tot = model.TermTotals();
  const auto p223 = model.Comm223Parts();
  const double total = model.TotalMultiplyAdds();
  const double plan_total = total - p223[1];
  std::printf("multiply-adds per commutator (3B parts in X and Y):\n");
  std::printf("  223(W sums) H W 232 331 133 330 231+132 (plan layout)\n");
  std::printf("  %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e\n", p223[0], tot[1],
              tot[2], tot[3], tot[4], tot[5], tot[6], tot[7]);
  std::printf("  223 epilogue %.3e (in 223 below)\n", p223[1]);
  std::printf("  223 total    %.3e\n", tot[0]);
  std::printf("  total        %.3e (without 223 epilogue %.3e)\n", total,
              plan_total);
  std::printf("time_per_commutator_s at 5e10/s %.1f at 2.5e11/s %.1f\n",
              total / 5.0e10, total / 2.5e11);
  std::printf("memory_six_operators_GB %.1f (packed 3B) + 2B %.2f\n",
              6.0 * model.PackedBytes() / 1.0e9, 6.0 * twobody_gb);
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

/// `--triples`: the perturbative triples on the reduced `TriplesContext`.
void RunTriples(const Options& o) {
  imsrg::basis::ThreeBodyCuts cuts;
  cuts.e3max = o.e3max;
  const auto t0 = Clock::now();
  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{o.emax}, 16.0, o.reference, o.reference);
  const double space_ms = ElapsedMs(t0);
  const auto t1 = Clock::now();
  const cm::TriplesContext ctx(ms, cuts);
  imsrg::exec::Fence("n7_bench triples context");
  const double ctx_ms = ElapsedMs(t1);
  const cm::TriplesCounts& c = ctx.Counts();
  std::printf("# n7_bench --triples%s git %s\n",
              o.count_only ? " --count-only" : "", IMSRG_BENCH_GIT_HASH);
  std::printf("emax %d e3max %d reference %s\n", o.emax, o.e3max,
              o.reference.c_str());
  std::printf("space_build_ms %.0f context_build_ms %.0f\n", space_ms, ctx_ms);
  std::printf("channels %" PRId64 " triples %" PRId64 " hhh %" PRId64
              " ppp %" PRId64 "\n",
              c.channels, c.triples, c.triples_hhh, c.triples_ppp);
  std::printf("rows %" PRId64 " rows_before_cuts %" PRId64 " states %" PRId64
              "\n",
              c.rows, c.rows_before_cuts, c.states);
  std::printf("context_bytes %" PRId64
              " context_GB %.3f scratch_per_team %" PRId64 "\n",
              ctx.TableBytes(), ctx.TableBytes() / 1.0e9,
              ctx.ScratchBytesPerTeam());
  const auto t2 = Clock::now();
  const cm::TriplesWork work = ctx.TileWork();
  std::printf("tile_multiply_adds %.3e (W %.3e epilogue %.3e) count_ms %.0f\n",
              work.Total(), work.w, work.epilogue, ElapsedMs(t2));
  if (o.count_only) return;
  const op::Operator x = imsrg::op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, HermFromLetter(o.herm[0])),
      imsrg::op::RandomOperatorOptions{.seed = o.seed});
  const op::Operator y = imsrg::op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, HermFromLetter(o.herm[1])),
      imsrg::op::RandomOperatorOptions{.seed = o.seed + 1});
  std::vector<double> eps(static_cast<std::size_t>(ms->num_orbits()));
  const auto occ = ms->reference().occupations();
  for (std::size_t i = 0; i < eps.size(); ++i) {
    eps[i] = (occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * ms->basis().orbits()[i].e() +
             0.1 * static_cast<double>(i % 7);
  }
  double e = cm::PerturbativeTriplesEnergy(x, y, eps, ctx);  // warm-up
  double best = std::numeric_limits<double>::max();
  double sum = 0.0;
  for (int r = 0; r < o.reps; ++r) {
    const auto t = Clock::now();
    e = cm::PerturbativeTriplesEnergy(x, y, eps, ctx);
    imsrg::exec::Fence("n7_bench triples");
    const double ms_r = ElapsedMs(t);
    best = std::min(best, ms_r);
    sum += ms_r;
  }
  std::printf("triples %c%c energy %.17g best_ms %.3f mean_ms %.3f\n",
              o.herm[0], o.herm[1], e, best, sum / o.reps);
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
    if (o.triples) {
      RunTriples(o);
    } else if (o.count_only) {
      RunCountOnly(o);
    } else {
      Run(o);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "n7_bench: %s\n", e.what());
    return 3;
  }
}
