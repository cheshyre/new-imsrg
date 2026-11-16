// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// imsrg3_layout_bench -- IMSRG(3) three-body layout benchmark (design/14
// section 2).
//
//   imsrg3_layout_bench [--emax N] [--e3max N] [--de3max X] [--occnat3 X]
//       [--reference NAME] [--layouts A,B,C] [--kernels LIST] [--reps R]
//       [--ref-reps R] [--seed S] [--check] [--count-only]
//       [--write-dumps DIR] [--x-dump F --y-dump F]
//
// Full mode (emax 2 or 3): builds the canonical space, pair-third states, the
// layouts A, B, C and the kernel context (each timed), reports the memory of
// every layout, then times the kernels `ref` (K0), `tiled` / `wonly` (K1 and
// its W part per layout), `triples-ref`, `triples` (per layout) and `pack`
// (dense <-> packed triangle of layout A), and compares every layout with the
// reference results. The production mode (kernels `prod`, `prod-alloc`,
// `prod-triples`) builds a `commutator::ThreeBodyContext` for the same kept
// space and times `commutator::Comm223` (packed layout A, a new part per
// call; `prod-alloc` times the allocation of that part alone) and
// `commutator::PerturbativeTriplesEnergy`, reports their table and
// team-scratch bytes, and checks them against the same references.
// Count-only mode (any emax): sizes of the three layouts and the
// operation-count cost models from the metadata alone.
//
// Every measurement is also printed as one `JSON {...}` line for scripts.
// Exit codes: 0 success, 1 a `--check` comparison failed, 2 usage error, 3
// other error.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/three_body.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/cost_model.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/layouts.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/production.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

#ifndef IMSRG_BENCH_GIT_HASH
#define IMSRG_BENCH_GIT_HASH "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
namespace l3 = imsrg::layout3;
namespace op = imsrg::op;

constexpr double kCheckTol = 1.0e-10;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

// ------------------------------------------------------------------ options

struct Options {
  int emax = 2;
  int e3max = -1;  // -1: 3 emax
  double de3max = 1.0e10;
  double occnat3 = -1.0;
  std::string reference = "O16";
  std::vector<std::string> layouts = {"A", "B", "C"};
  std::vector<std::string> kernels = {"ref",     "tiled",       "wonly",
                                      "triples", "triples-ref", "pack",
                                      "prod",    "prod-alloc",  "prod-triples"};
  int reps = 10;
  int ref_reps = 3;
  std::uint64_t seed = 1;
  bool check = false;
  bool count_only = false;
  std::string write_dumps;
  std::string x_dump;
  std::string y_dump;

  bool WantsLayout(const std::string& l) const {
    return std::find(layouts.begin(), layouts.end(), l) != layouts.end();
  }
  bool WantsKernel(const std::string& k) const {
    return std::find(kernels.begin(), kernels.end(), k) != kernels.end();
  }
};

std::vector<std::string> SplitList(const std::string& s) {
  std::vector<std::string> out;
  std::size_t begin = 0;
  while (begin <= s.size()) {
    const std::size_t end = std::min(s.find(',', begin), s.size());
    if (end > begin) out.push_back(s.substr(begin, end - begin));
    begin = end + 1;
  }
  return out;
}

const char* kUsage =
    "usage: imsrg3_layout_bench [--emax N] [--e3max N] [--de3max X] "
    "[--occnat3 X]\n"
    "         [--reference NAME] [--layouts A,B,C] [--kernels LIST] "
    "[--reps R]\n"
    "         [--ref-reps R] [--seed S] [--check] [--count-only]\n"
    "         [--write-dumps DIR] [--x-dump F --y-dump F]\n"
    "kernels: ref,tiled,wonly,triples,triples-ref,pack,prod,prod-alloc,\n"
    "         prod-triples (default: all)\n";

/// Parses the command line; throws std::invalid_argument on any error.
Options ParseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    try {
      if (a == "--emax") {
        o.emax = std::stoi(value());
      } else if (a == "--e3max") {
        o.e3max = std::stoi(value());
      } else if (a == "--de3max") {
        o.de3max = std::stod(value());
      } else if (a == "--occnat3") {
        o.occnat3 = std::stod(value());
      } else if (a == "--reference") {
        o.reference = value();
      } else if (a == "--layouts") {
        o.layouts = SplitList(value());
      } else if (a == "--kernels") {
        o.kernels = SplitList(value());
      } else if (a == "--reps") {
        o.reps = std::stoi(value());
      } else if (a == "--ref-reps") {
        o.ref_reps = std::stoi(value());
      } else if (a == "--seed") {
        o.seed = std::stoull(value());
      } else if (a == "--check") {
        o.check = true;
      } else if (a == "--count-only") {
        o.count_only = true;
      } else if (a == "--write-dumps") {
        o.write_dumps = value();
      } else if (a == "--x-dump") {
        o.x_dump = value();
      } else if (a == "--y-dump") {
        o.y_dump = value();
      } else {
        throw std::invalid_argument("unknown option " + a);
      }
    } catch (const std::logic_error&) {
      throw std::invalid_argument("bad value for " + a);
    }
  }
  if (o.e3max < 0) o.e3max = 3 * o.emax;
  if (o.emax < 0 || o.reps < 1 || o.ref_reps < 1) {
    throw std::invalid_argument("emax >= 0, reps >= 1 and ref-reps >= 1");
  }
  for (const std::string& l : o.layouts) {
    if (l != "A" && l != "B" && l != "C") {
      throw std::invalid_argument("unknown layout " + l);
    }
  }
  for (const std::string& k : o.kernels) {
    if (k != "ref" && k != "tiled" && k != "wonly" && k != "triples" &&
        k != "triples-ref" && k != "pack" && k != "prod" && k != "prod-alloc" &&
        k != "prod-triples") {
      throw std::invalid_argument("unknown kernel " + k);
    }
  }
  if (o.x_dump.empty() != o.y_dump.empty()) {
    throw std::invalid_argument("--x-dump and --y-dump go together");
  }
  return o;
}

// --------------------------------------------------------------------- JSON

/// One `JSON {...}` line: flat object with string, integer and real fields.
class Json {
 public:
  explicit Json(const char* kind) { Str("kind", kind); }

  Json& Str(const char* key, const std::string& v) {
    Key(key);
    body_ += '"' + v + '"';
    return *this;
  }
  Json& Int(const char* key, std::int64_t v) {
    Key(key);
    body_ += std::to_string(v);
    return *this;
  }
  Json& Num(const char* key, double v) {
    Key(key);
    if (!std::isfinite(v)) {
      body_ += "null";
    } else {
      char buf[48];
      std::snprintf(buf, sizeof buf, "%.17g", v);
      body_ += buf;
    }
    return *this;
  }
  Json& Bool(const char* key, bool v) {
    Key(key);
    body_ += v ? "true" : "false";
    return *this;
  }
  /// The run parameters every measurement carries.
  Json& Params(const Options& o) {
    return Int("emax", o.emax)
        .Int("e3max", o.e3max)
        .Num("de3max", o.de3max)
        .Str("reference", o.reference)
        .Int("threads", imsrg::exec::Runtime::Concurrency());
  }
  void Print() const { std::printf("JSON {%s}\n", body_.c_str()); }

 private:
  void Key(const char* key) {
    if (!body_.empty()) body_ += ", ";
    body_ += '"';
    body_ += key;
    body_ += "\": ";
  }
  std::string body_;
};

// ------------------------------------------------------------------- timing

struct Stats {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double max_ms = 0.0;
};

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// One warm-up call, then `reps` calls of `f`, each between fences; min,
/// median and max in ms.
template <class F>
Stats TimeCalls(int reps, const F& f) {
  f();
  imsrg::exec::Fence("bench warm-up");
  std::vector<double> ms;
  for (int i = 0; i < reps; ++i) {
    imsrg::exec::Fence("bench pre");
    const auto t0 = Clock::now();
    f();
    imsrg::exec::Fence("bench post");
    ms.push_back(MsSince(t0));
  }
  std::sort(ms.begin(), ms.end());
  Stats s;
  s.min_ms = ms.front();
  s.max_ms = ms.back();
  const std::size_t n = ms.size();
  s.median_ms = n % 2 == 1 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
  return s;
}

/// Constructs `slot` in place from `args`, between fences; returns ms.
template <class T, class... Args>
double TimedEmplace(std::optional<T>& slot, Args&&... args) {
  imsrg::exec::Fence("bench build pre");
  const auto t0 = Clock::now();
  slot.emplace(std::forward<Args>(args)...);
  imsrg::exec::Fence("bench build post");
  return MsSince(t0);
}

// ---------------------------------------------------------------- reporting

constexpr double kMB = 1.0e6;
constexpr double kGB = 1.0e9;

void PrintHeader(const Options& o) {
  const char* omp = std::getenv("OMP_NUM_THREADS");
#ifdef NDEBUG
  const char* build = "NDEBUG";
#else
  const char* build = "debug (no NDEBUG)";
#endif
  std::printf("imsrg3_layout_bench  git %s  build %s\n", IMSRG_BENCH_GIT_HASH,
              build);
  std::printf(
      "  Kokkos concurrency %d, OMP_NUM_THREADS=%s\n"
      "  emax %d, e3max %d, de3max %g, occnat3 %g, reference %s, reps %d, "
      "ref-reps %d, seed %" PRIu64 "%s\n",
      imsrg::exec::Runtime::Concurrency(), omp != nullptr ? omp : "(unset)",
      o.emax, o.e3max, o.de3max, o.occnat3, o.reference.c_str(), o.reps,
      o.ref_reps, static_cast<std::uint64_t>(o.seed),
      o.count_only ? ", count-only" : "");
  Json("header")
      .Params(o)
      .Str("git", IMSRG_BENCH_GIT_HASH)
      .Str("build", build)
      .Str("omp_num_threads", omp != nullptr ? omp : "")
      .Num("occnat3", o.occnat3)
      .Int("reps", o.reps)
      .Int("ref_reps", o.ref_reps)
      .Int("seed", static_cast<std::int64_t>(o.seed))
      .Bool("count_only", o.count_only)
      .Print();
}

/// Table header; sizes in GB (`gb`) or MB.
void PrintCountsHeader(bool gb) {
  std::printf("\n%-6s %13s %17s %9s %17s %10s %10s %9s\n", "layout", "sum N",
              "sum N^2", "max N", "sum N(N+1)/2", gb ? "dense GB" : "dense MB",
              gb ? "packed GB" : "packed MB", "channels");
}

void PrintCountsRow(const char* name, const l3::SpaceCounts& c, bool gb) {
  const double unit = gb ? kGB : kMB;
  std::printf("%-6s %13lld %17lld %9d %17lld %10.3f %10.3f %9d\n", name,
              static_cast<std::int64_t>(c.sum_n),
              static_cast<std::int64_t>(c.sum_n2), c.max_n,
              static_cast<std::int64_t>(c.sum_packed),
              8.0 * static_cast<double>(c.sum_n2) / unit,
              8.0 * static_cast<double>(c.sum_packed) / unit, c.num_channels);
}

Json CountsJson(const Options& o, const char* kind, const char* name,
                const l3::SpaceCounts& c) {
  Json j(kind);
  j.Params(o)
      .Str("layout", name)
      .Int("sum_n", c.sum_n)
      .Int("sum_n2", c.sum_n2)
      .Int("max_n", c.max_n)
      .Int("sum_packed", c.sum_packed)
      .Int("num_channels", c.num_channels)
      .Int("num_triples", c.num_triples)
      .Num("dense_mb", 8.0 * static_cast<double>(c.sum_n2) / kMB)
      .Num("packed_mb", 8.0 * static_cast<double>(c.sum_packed) / kMB);
  return j;
}

// ------------------------------------------------------------- count-only

int RunCountOnly(const Options& o) {
  PrintHeader(o);
  const auto t_all = Clock::now();
  auto t0 = Clock::now();
  const auto ms =
      imsrg::basis::ModelSpace::FromNames(imsrg::basis::BasisTruncation{o.emax},
                                          16.0, o.reference, o.reference)
          ->WithThreeBody(imsrg::basis::ThreeBodyCuts{o.e3max});
  const double t_ms = MsSince(t0);
  t0 = Clock::now();
  const l3::CanonicalSpace space(ms, o.de3max, o.occnat3);
  const double t_space = MsSince(t0);
  t0 = Clock::now();
  const l3::SpaceCounts ca = space.Counts();
  const l3::SpaceCounts cc = l3::PairThirdSpace::CountOnly(space);
  const double t_c = MsSince(t0);
  t0 = Clock::now();
  const l3::SpaceCounts cb = l3::LayoutB::CountOnly(space);
  const double t_b = MsSince(t0);
  t0 = Clock::now();
  const l3::CostModels cost = l3::ComputeCostModels(space);
  const double t_cost = MsSince(t0);

  std::printf(
      "\ncount-only timings (ms): model space %.1f, canonical space %.1f, "
      "layout C counts %.1f, layout B counts %.1f, cost models %.1f\n",
      t_ms, t_space, t_c, t_b, t_cost);
  std::printf("triples %" PRId64 ", channels %d\n",
              static_cast<std::int64_t>(ca.num_triples), ca.num_channels);
  PrintCountsHeader(true);
  PrintCountsRow("A", ca, true);
  PrintCountsRow("B", cb, true);
  PrintCountsRow("C", cc, true);
  CountsJson(o, "counts", "A", ca).Print();
  CountsJson(o, "counts", "B", cb).Print();
  CountsJson(o, "counts", "C", cc).Print();

  std::printf(
      "\ncost models (multiply-adds, pairs t' <= t):\n"
      "  w_cost (K1 W work)        %.6e\n"
      "  k0_cost (K0 / imsrg++)    %.6e\n"
      "  k0_cost / w_cost          %.3f\n"
      "  epi_cost  A %.6e   B %.6e   C %.6e\n",
      cost.w_cost, cost.k0_cost,
      cost.w_cost > 0.0 ? cost.k0_cost / cost.w_cost : 0.0, cost.epi_cost_a,
      cost.epi_cost_b, cost.epi_cost_c);
  const double wall_ms = MsSince(t_all);
  std::printf("wall time %.1f ms\n", wall_ms);
  Json("cost")
      .Params(o)
      .Num("w_cost", cost.w_cost)
      .Num("k0_cost", cost.k0_cost)
      .Num("epi_cost_A", cost.epi_cost_a)
      .Num("epi_cost_B", cost.epi_cost_b)
      .Num("epi_cost_C", cost.epi_cost_c)
      .Num("model_space_ms", t_ms)
      .Num("canonical_ms", t_space)
      .Num("count_c_ms", t_c)
      .Num("count_b_ms", t_b)
      .Num("cost_ms", t_cost)
      .Num("wall_ms", wall_ms)
      .Print();
  return 0;
}

// --------------------------------------------------------------- full mode

/// Dense <-> packed upper triangle per channel (column-major packed, entry
/// (i <= j) at j (j + 1) / 2 + i): a parallel loop over channels packs the
/// dense blocks of `dense` into `packed` and a second one unpacks them
/// (writing both triangles). Cost O(sum N^2) reads and writes.
void PackUnpack(const l3::LayoutView lv, const std::int64_t* packed_offset,
                double* dense, double* packed) {
  const std::int64_t nch = lv.num_channels;
  imsrg::exec::ParallelFor(
      "bench::pack", nch, IMSRG_LAMBDA(const std::int64_t ch) {
        const std::int64_t n = lv.ch_dim[ch];
        const double* d = dense + lv.ch_block_offset[ch];
        double* p = packed + packed_offset[ch];
        for (std::int64_t j = 0; j < n; ++j) {
          for (std::int64_t i = 0; i <= j; ++i) {
            p[j * (j + 1) / 2 + i] = d[i + j * n];
          }
        }
      });
  imsrg::exec::ParallelFor(
      "bench::unpack", nch, IMSRG_LAMBDA(const std::int64_t ch) {
        const std::int64_t n = lv.ch_dim[ch];
        double* d = dense + lv.ch_block_offset[ch];
        const double* p = packed + packed_offset[ch];
        for (std::int64_t j = 0; j < n; ++j) {
          for (std::int64_t i = 0; i <= j; ++i) {
            const double v = p[j * (j + 1) / 2 + i];
            d[i + j * n] = v;
            d[j + i * n] = v;
          }
        }
      });
}

struct KernelRow {
  std::string layout;
  std::string kernel;
  int reps = 0;
  Stats stats;
};

struct Agreement {
  std::string layout;
  std::string quantity;
  double value = 0.0;      // the layout's number (NaN: none)
  double reference = 0.0;  // the reference number (NaN: none)
  double deviation = 0.0;
  bool pass = true;
};

/// Everything the per-layout runs share.
struct Run {
  Run(const Options& options, const l3::KernelContext& context,
      const op::Operator& xop, const op::Operator& yop,
      std::span<const double> epsilon)
      : opt(options), ctx(context), x(xop), y(yop), eps(epsilon) {}
  const Options& opt;
  const l3::KernelContext& ctx;
  const op::Operator& x;
  const op::Operator& y;
  std::span<const double> eps;
  bool have_ref = false;   // K0 blocks available
  std::vector<double> z0;  // K0 canonical blocks
  double norm0 = 0.0;      // ImsrgppNorm(K0)
  bool have_e0 = false;    // reference triples energy available
  double e0 = 0.0;
  std::vector<KernelRow> rows;
  std::vector<Agreement> agreements;

  void AddRow(const std::string& layout, const char* kernel, int reps,
              const Stats& s) {
    rows.push_back({layout, kernel, reps, s});
    Json("kernel")
        .Params(opt)
        .Str("layout", layout)
        .Str("kernel", kernel)
        .Int("reps", reps)
        .Num("min_ms", s.min_ms)
        .Num("median_ms", s.median_ms)
        .Num("max_ms", s.max_ms)
        .Print();
  }
};

/// Canonical image of layout blocks `z` (`ToCanonical` per channel).
template <class L>
std::vector<double> ToCanonicalAll(const L& l, const std::vector<double>& z) {
  const l3::CanonicalView cv = l.Space().host_view();
  std::vector<double> out(U(l.Space().total_size()));
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t lo = l.BlockOffsets()[U(ch)];
    const std::int64_t n = l.BlockOffsets()[U(ch) + 1] - lo;
    const std::int64_t co = cv.ch_block_offset[ch];
    const std::int64_t cn = cv.ch_block_offset[ch + 1] - co;
    l.ToCanonical(ch, std::span<const double>(z).subspan(U(lo), U(n)),
                  std::span<double>(out).subspan(U(co), U(cn)));
  }
  return out;
}

/// max |x - y| / max(1, max |y|).
double RelDiff(std::span<const double> x, std::span<const double> y) {
  double diff = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    diff = std::max(diff, std::abs(x[i] - y[i]));
    scale = std::max(scale, std::abs(y[i]));
  }
  return diff / scale;
}

double RelScalar(double x, double y) {
  return std::abs(x - y) / std::max(1.0e-300, std::abs(y));
}

/// Times and checks every requested kernel on layout `l` named `name`.
template <class L>
void RunLayout(Run* r, const char* name, const L& l) {
  const Options& o = r->opt;
  const l3::LayoutView lv = l.device_view();
  const std::int64_t total = l.total_size();
  std::vector<double> z(U(total), 0.0);
  bool have_z = false;
  const std::string lname = name;

  if (o.WantsKernel("tiled")) {
    const Stats s = TimeCalls(o.reps, [&] {
      l3::Comm223Tiled(r->ctx, lv, r->x, r->y, z, l3::TiledMode::kFull);
    });
    r->AddRow(lname, "tiled", o.reps, s);
    have_z = true;
  }
  if (o.WantsKernel("wonly")) {
    // kWOnly leaves z untouched, so the result of `tiled` survives.
    const Stats s = TimeCalls(o.reps, [&] {
      l3::Comm223Tiled(r->ctx, lv, r->x, r->y, z, l3::TiledMode::kWOnly);
    });
    r->AddRow(lname, "wonly", o.reps, s);
  }
  double e_layout = 0.0;
  bool have_e = false;
  if (o.WantsKernel("triples")) {
    const Stats s = TimeCalls(o.reps, [&] {
      e_layout = l3::TriplesTiled(r->ctx, lv, r->x, r->y, r->eps);
    });
    r->AddRow(lname, "triples", o.reps, s);
    have_e = true;
  }
  if (o.WantsKernel("pack") && L::kKind == l3::LayoutKind::kA) {
    if (!have_z) {
      for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] = 1.0e-3 * static_cast<double>(i % 1013);
      }
    }
    std::vector<std::int64_t> poff(U(lv.num_channels) + 1, 0);
    for (std::size_t c = 0; c < U(lv.num_channels); ++c) {
      const std::int64_t n = l.Dims()[c];
      poff[c + 1] = poff[c] + n * (n + 1) / 2;
    }
    imsrg::exec::DeviceBuffer<std::int64_t> poff_dev(poff.size(),
                                                     "bench::poff");
    poff_dev.CopyFromHost(std::span<const std::int64_t>(poff));
    imsrg::exec::DeviceBuffer<double> dense(z.size(), "bench::dense");
    dense.CopyFromHost(std::span<const double>(z));
    imsrg::exec::DeviceBuffer<double> packed(U(poff.back()), "bench::packed");
    const Stats s = TimeCalls(o.reps, [&] {
      PackUnpack(lv, poff_dev.device_data(), dense.device_data(),
                 packed.device_data());
    });
    r->AddRow(lname, "pack", o.reps, s);
  }

  // Agreement with the reference results.
  const auto add = [&](const char* quantity, double value, double reference,
                       double dev) {
    const bool pass = dev <= kCheckTol;
    r->agreements.push_back({lname, quantity, value, reference, dev, pass});
    Json("agreement")
        .Params(o)
        .Str("layout", lname)
        .Str("quantity", quantity)
        .Num("value", value)
        .Num("reference_value", reference)
        .Num("deviation", dev)
        .Num("tolerance", kCheckTol)
        .Bool("pass", pass)
        .Print();
  };
  const double nan = std::nan("");
  if (have_z && r->have_ref) {
    const std::vector<double> zc = ToCanonicalAll(l, z);
    add("tiled_vs_K0_maxdiff", nan, nan, RelDiff(zc, r->z0));
    const double n = l3::LayoutImsrgppNorm(l, z);
    add("imsrgpp_norm", n, r->norm0, RelScalar(n, r->norm0));
  }
  if (have_e && r->have_e0) {
    add("triples_energy", e_layout, r->e0, RelScalar(e_layout, r->e0));
  }
}

void PrintAgreements(const Run& r) {
  std::printf("\nagreement with the K0 reference (tolerance %g)\n", kCheckTol);
  std::printf("%-6s %-22s %-24s %-24s %11s %s\n", "layout", "quantity", "value",
              "reference", "deviation", "");
  for (const Agreement& a : r.agreements) {
    if (std::isnan(a.value)) {
      std::printf("%-6s %-22s %-24s %-24s %11.3e %s\n", a.layout.c_str(),
                  a.quantity.c_str(), "-", "-", a.deviation,
                  a.pass ? "ok" : "FAIL");
    } else {
      std::printf("%-6s %-22s %-24.17g %-24.17g %11.3e %s\n", a.layout.c_str(),
                  a.quantity.c_str(), a.value, a.reference, a.deviation,
                  a.pass ? "ok" : "FAIL");
    }
  }
}

/// Times and checks the production kernels (`commutator::Comm223`,
/// `commutator::PerturbativeTriplesEnergy`) on the context `pctx`, whose kept
/// space has the rows of `space`.
void RunProduction(Run* r, const imsrg::commutator::ThreeBodyContext& pctx,
                   const l3::CanonicalSpace& space) {
  const Options& o = r->opt;
  const std::string lname = "P";
  const double nan = std::nan("");
  const auto add = [&](const char* quantity, double value, double reference,
                       double dev) {
    const bool pass = dev <= kCheckTol;
    r->agreements.push_back({lname, quantity, value, reference, dev, pass});
    Json("agreement")
        .Params(o)
        .Str("layout", lname)
        .Str("quantity", quantity)
        .Num("value", value)
        .Num("reference_value", reference)
        .Num("deviation", dev)
        .Num("tolerance", kCheckTol)
        .Bool("pass", pass)
        .Print();
  };
  if (o.WantsKernel("prod")) {
    op::ThreeBody z;
    const Stats s = TimeCalls(
        o.reps, [&] { z = imsrg::commutator::Comm223(r->x, r->y, pctx); });
    r->AddRow(lname, "prod", o.reps, s);
    if (r->have_ref) {
      add("prod_vs_K0_maxdiff", nan, nan,
          RelDiff(l3::ToCanonicalDense(z, space), r->z0));
      const double n = op::ImsrgppNorm(z);
      add("imsrgpp_norm", n, r->norm0, RelScalar(n, r->norm0));
    }
  }
  if (o.WantsKernel("prod-alloc")) {
    const imsrg::Hermiticity h = r->x.hermiticity() == r->y.hermiticity()
                                     ? imsrg::Hermiticity::kAntiHermitian
                                     : imsrg::Hermiticity::kHermitian;
    op::ThreeBody z;
    const Stats s = TimeCalls(
        o.reps, [&] { z = op::ThreeBody(pctx.model_space_ptr(), h); });
    r->AddRow(lname, "prod-alloc", o.reps, s);
  }
  if (o.WantsKernel("prod-triples")) {
    double e = 0.0;
    const Stats s = TimeCalls(o.reps, [&] {
      e = imsrg::commutator::PerturbativeTriplesEnergy(r->x, r->y, r->eps,
                                                       pctx);
    });
    r->AddRow(lname, "prod-triples", o.reps, s);
    if (r->have_e0) add("triples_energy", e, r->e0, RelScalar(e, r->e0));
  }
}

int RunFull(const Options& o) {
  PrintHeader(o);

  // ---- builds (each timed) ----
  std::vector<std::pair<std::string, double>> builds;
  const auto note = [&](const char* step, double ms) {
    builds.emplace_back(step, ms);
    Json("build").Params(o).Str("step", step).Num("ms", ms).Print();
  };
  auto t0 = Clock::now();
  const auto ms =
      imsrg::basis::ModelSpace::FromNames(imsrg::basis::BasisTruncation{o.emax},
                                          16.0, o.reference, o.reference)
          ->WithThreeBody(imsrg::basis::ThreeBodyCuts{o.e3max});
  note("model_space", MsSince(t0));
  std::optional<l3::SixJ3Table> sixj;
  note("sixj_table", TimedEmplace(sixj, o.emax));
  std::optional<l3::CanonicalSpace> space;
  note("canonical_space", TimedEmplace(space, ms, o.de3max, o.occnat3));
  std::optional<l3::PairThirdSpace> pt;
  note("pair_third_space", TimedEmplace(pt, *space, *sixj));
  std::optional<l3::LayoutA> la;
  std::optional<l3::LayoutB> lb;
  std::optional<l3::LayoutC> lc;
  if (o.WantsLayout("A")) note("layout_A", TimedEmplace(la, *space, *pt));
  if (o.WantsLayout("B"))
    note("layout_B", TimedEmplace(lb, *space, *pt, *sixj));
  if (o.WantsLayout("C")) note("layout_C", TimedEmplace(lc, *space, *pt));
  std::optional<l3::KernelContext> ctx;
  note("kernel_context", TimedEmplace(ctx, *space, *pt, *sixj));
  std::optional<imsrg::commutator::ThreeBodyContext> pctx;
  if (o.WantsKernel("prod") || o.WantsKernel("prod-alloc") ||
      o.WantsKernel("prod-triples")) {
    // The production model space (cuts applied at enumeration) is part of
    // the build: ThreeBodySpace with the dE3max / OccNat3 cuts, then the
    // context tables.
    auto t_prod = Clock::now();
    const auto ms_prod = l3::ProductionModelSpace(ms, o.de3max, o.occnat3);
    note("production_model_space", MsSince(t_prod));
    note("production_context", TimedEmplace(pctx, ms_prod));
  }

  std::printf("\nbuild times\n");
  for (const auto& b : builds) {
    std::printf("  %-18s %12.2f ms\n", b.first.c_str(), b.second);
  }

  // ---- memory ----
  const l3::SpaceCounts ca = space->Counts();
  std::printf("\ncanonical space: %d channels, %d triples, %" PRId64 " rows\n",
              ca.num_channels, static_cast<int>(ca.num_triples),
              static_cast<std::int64_t>(space->NumRows()));
  PrintCountsHeader(false);
  const auto memory_row = [&](const char* name, const l3::SpaceCounts& c,
                              std::int64_t metadata) {
    PrintCountsRow(name, c, false);
    CountsJson(o, "memory", name, c).Int("metadata_bytes", metadata).Print();
  };
  if (la) memory_row("A", la->Counts(), la->MetadataBytes());
  if (lb) memory_row("B", lb->Counts(), lb->MetadataBytes());
  if (lc) memory_row("C", lc->Counts(), lc->MetadataBytes());
  std::printf("metadata MB:");
  if (la)
    std::printf("  A %.3f", static_cast<double>(la->MetadataBytes()) / kMB);
  if (lb)
    std::printf("  B %.3f", static_cast<double>(lb->MetadataBytes()) / kMB);
  if (lc)
    std::printf("  C %.3f", static_cast<double>(lc->MetadataBytes()) / kMB);
  std::printf("\n");
  const std::int64_t canon_meta = space->MetadataBytes();
  const std::int64_t pt_state = pt->StateBytes();
  const std::int64_t r_bytes = pt->RMap().Bytes();
  const std::int64_t e_bytes = pt->EMap().Bytes();
  const std::int64_t sixj_bytes = sixj->size_bytes();
  std::printf(
      "shared bytes: canonical metadata %" PRId64 ", pair-third states %" PRId64
      ", R map "
      "%" PRId64 ", E map %" PRId64 ", 6j table %" PRId64 "\n",
      static_cast<std::int64_t>(canon_meta),
      static_cast<std::int64_t>(pt_state), static_cast<std::int64_t>(r_bytes),
      static_cast<std::int64_t>(e_bytes),
      static_cast<std::int64_t>(sixj_bytes));
  Json("memory_shared")
      .Params(o)
      .Int("canonical_metadata_bytes", canon_meta)
      .Int("pair_third_state_bytes", pt_state)
      .Int("r_map_bytes", r_bytes)
      .Int("e_map_bytes", e_bytes)
      .Int("sixj_bytes", sixj_bytes)
      .Print();
  if (pctx) {
    const std::int64_t tables = pctx->TableBytes();
    const std::int64_t scratch = pctx->ScratchBytesPerTeam();
    std::printf("production context: device tables %" PRId64
                " bytes, team scratch %" PRId64 " bytes per team\n",
                static_cast<std::int64_t>(tables),
                static_cast<std::int64_t>(scratch));
    Json("memory_production")
        .Params(o)
        .Int("table_bytes", tables)
        .Int("scratch_bytes_per_team", scratch)
        .Print();
  }

  // ---- operators ----
  op::Operator x =
      o.x_dump.empty()
          ? op::RandomOperator(
                ms,
                op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kAntiHermitian),
                {o.seed})
          : op::ReadOperatorDump(o.x_dump, ms);
  op::Operator y =
      o.y_dump.empty()
          ? op::RandomOperator(
                ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kHermitian),
                {o.seed + 1})
          : op::ReadOperatorDump(o.y_dump, ms);
  std::vector<double> eps;
  {
    const auto orbits = ms->basis().orbits();
    const auto occ = ms->reference().occupations();
    for (std::size_t i = 0; i < orbits.size(); ++i) {
      eps.push_back((occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * orbits[i].e() +
                    0.1 * static_cast<double>(i % 7));
    }
  }
  if (!o.write_dumps.empty()) {
    const std::filesystem::path dir = o.write_dumps;
    std::filesystem::create_directories(dir);
    op::WriteOperatorDump(dir / "X.dump", x);
    op::WriteOperatorDump(dir / "Y.dump", y);
    std::FILE* f = std::fopen((dir / "epsilon.txt").c_str(), "w");
    if (f == nullptr) throw std::runtime_error("cannot write epsilon.txt");
    for (const double e : eps) std::fprintf(f, "%.17g\n", e);
    std::fclose(f);
    std::printf("\nwrote X.dump, Y.dump, epsilon.txt to %s\n",
                o.write_dumps.c_str());
  }

  // ---- kernels ----
  Run run(o, *ctx, x, y, eps);
  if (o.WantsKernel("ref")) {
    run.z0.assign(U(space->total_size()), 0.0);
    const Stats s = TimeCalls(
        o.ref_reps, [&] { l3::Comm223Reference(*ctx, x, y, run.z0); });
    run.AddRow("-", "ref", o.ref_reps, s);
    run.have_ref = true;
    run.norm0 = l3::ImsrgppNorm(*space, run.z0);
  }
  if (o.WantsKernel("triples-ref")) {
    const Stats s = TimeCalls(
        o.ref_reps, [&] { run.e0 = l3::TriplesReference(*ctx, x, y, eps); });
    run.AddRow("-", "triples-ref", o.ref_reps, s);
    run.have_e0 = true;
  }
  if (la) RunLayout(&run, "A", *la);
  if (lb) RunLayout(&run, "B", *lb);
  if (lc) RunLayout(&run, "C", *lc);
  if (pctx) RunProduction(&run, *pctx, *space);

  std::printf("\nkernel timings (ms; warm-up excluded)\n");
  std::printf("%-6s %-12s %5s %12s %12s %12s\n", "layout", "kernel", "reps",
              "min", "median", "max");
  for (const KernelRow& k : run.rows) {
    std::printf("%-6s %-12s %5d %12.3f %12.3f %12.3f\n", k.layout.c_str(),
                k.kernel.c_str(), k.reps, k.stats.min_ms, k.stats.median_ms,
                k.stats.max_ms);
  }
  if (run.have_ref) {
    std::printf("\nImsrgppNorm(K0) = %.17g\n", run.norm0);
  }
  if (run.have_e0) std::printf("triples energy (K0) = %.17g\n", run.e0);
  PrintAgreements(run);

  bool failed = false;
  for (const Agreement& a : run.agreements) failed = failed || !a.pass;
  if (o.check && failed) {
    std::printf("\nCHECK FAILED\n");
    return 1;
  }
  if (o.check)
    std::printf("\nCHECK PASSED (%zu comparisons)\n", run.agreements.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  imsrg::exec::Runtime runtime(argc, argv);
  Options opt;
  try {
    opt = ParseOptions(argc, argv);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "imsrg3_layout_bench: %s\n%s", e.what(), kUsage);
    return 2;
  }
  if (!opt.count_only && opt.emax != 2 && opt.emax != 3) {
    std::fprintf(stderr,
                 "imsrg3_layout_bench: full mode runs only at emax 2 or 3 "
                 "(AGENTS.md rule 3); use --count-only for larger emax\n");
    return 2;
  }
  try {
    return opt.count_only ? RunCountOnly(opt) : RunFull(opt);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "imsrg3_layout_bench: error: %s\n", e.what());
    return 3;
  }
}
