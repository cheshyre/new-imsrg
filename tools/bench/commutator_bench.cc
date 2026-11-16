// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// commutator_bench -- IMSRG(2) commutator micro-benchmark (design/30 section
// 5).
//
//   commutator_bench --emax N --reps R [--reference O16] [--seed S]
//                    [--checksum] [--only-full]
//
// Times every `commutator::terms::` function and the full `Commutator()` on
// random scalar two-body operators (X anti-Hermitian, Y Hermitian), after one
// warm-up call per term. The term functions add into Z (as in imsrg++), so Z
// accumulates over the repetitions; the full `Commutator()` resets Z itself.
// The Workspace construction is timed separately. The counterpart for imsrg++
// is `refharness bench-commutator` (tools/reference/rh_bench.cpp), which prints
// the same table. `--checksum` prints an FNV-1a hash of the bytes of [X, Y]
// (zero-, one- and two-body parts) for X and Y anti-Hermitian/Hermitian and
// for X general, so optimizations can be checked for bitwise identical
// results; `--only-full` times only the full commutator.

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/exec/runtime.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"

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

/// One warm-up call, then `reps` fenced timed calls of `fn`; returns mean and
/// min in ms.
Row TimeIt(const std::string& name, int reps, const std::function<void()>& fn) {
  fn();
  imsrg::exec::Fence("bench warm-up");
  Row r;
  r.name = name;
  r.min_ms = std::numeric_limits<double>::max();
  double total = 0.0;
  for (int i = 0; i < reps; ++i) {
    imsrg::exec::Fence("bench pre");
    const auto t0 = Clock::now();
    fn();
    imsrg::exec::Fence("bench post");
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    total += ms;
    if (ms < r.min_ms) r.min_ms = ms;
  }
  r.mean_ms = total / reps;
  return r;
}

using TermFn = void (*)(const op::Operator&, const op::Operator&, op::Operator&,
                        cm::Workspace&);

struct Named {
  const char* name;
  TermFn fn;
};

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

/// Hash of the zero-, one- and two-body parts of `z` (host copies).
std::uint64_t Checksum(const op::Operator& z) {
  std::uint64_t h = 14695981039346656037ULL;
  h = Fnv1a({z.zero_body()}, h);
  h = Fnv1a(z.one_body().buffer().ToVector(), h);
  h = Fnv1a(z.two_body().matrix().buffer().ToVector(), h);
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  int emax = 4;
  int reps = 10;
  std::uint64_t seed = 1;
  std::string reference = "O16";
  bool checksum = false;
  bool only_full = false;
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
    } else if (a == "--checksum") {
      checksum = true;
    } else if (a == "--only-full") {
      only_full = true;
    } else {
      std::fprintf(stderr,
                   "usage: commutator_bench --emax N --reps R [--reference "
                   "O16] [--seed S] [--checksum] [--only-full]\n");
      return 2;
    }
  }
  if (reps < 1) return 2;

  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{emax}, 16.0, reference, reference);

  const auto tw0 = Clock::now();
  cm::Workspace ws(ms);
  imsrg::exec::Fence("workspace");
  const double ws_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - tw0).count();

  op::RandomOperatorOptions ro;
  ro.seed = seed;
  const op::Operator X = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kAntiHermitian), ro);
  ro.seed = seed + 1;
  const op::Operator Y = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kHermitian), ro);
  op::Operator Z(ms, cm::CommutatorSpec(X.spec(), Y.spec()));

  const char* omp = std::getenv("OMP_NUM_THREADS");
  std::printf("# commutator_bench (this code)\n");
  std::printf("emax %d  reference %s  orbits %d  2B elements %" PRId64
              "  OMP_NUM_THREADS %s  "
              "git %s\n",
              emax, reference.c_str(), static_cast<int>(ms->num_orbits()),
              static_cast<std::int64_t>(X.two_body().matrix().total_size()),
              omp ? omp : "(unset)", IMSRG_BENCH_GIT_HASH);
  std::printf("reps %d\nworkspace construction: %.3f ms\n", reps, ws_ms);
  {
    // Scratch footprint relative to one two-body operator (design/benchmarks).
    const auto& impl = ws.impl();
    const double op_mb =
        static_cast<double>(impl.two_body_size) * sizeof(double) / 1.0e6;
    const double lad_mb =
        static_cast<double>(impl.ladder.scratch_bytes()) / 1.0e6;
    const double pan_mb =
        static_cast<double>(impl.pandya.scratch_bytes()) / 1.0e6;
    const double sixj_mb =
        static_cast<double>(impl.sixj.size()) * sizeof(double) / 1.0e6;
    std::printf(
        "scratch: two-body operator %.1f MB, ladder %.1f MB (%.2fx), Pandya "
        "%.1f MB (%.2fx), 6j table %.1f MB\n",
        op_mb, lad_mb, lad_mb / op_mb, pan_mb, pan_mb / op_mb, sixj_mb);
    // The same model space in the layout before M12a (gathered operands of
    // every listed channel, YK for both pair orders, a separate ZB buffer).
    double old_lad = 0.0;
    for (const auto* p : {&impl.ladder.pp, &impl.ladder.hh}) {
      old_lad += static_cast<double>(impl.two_body_size);
      for (std::size_t ch = 0; ch < p->count_host.size(); ++ch) {
        const double n = std::sqrt(static_cast<double>(
            impl.block_offsets_host[ch + 1] - impl.block_offsets_host[ch]));
        old_lad += 2.0 * n * p->count_host[ch];
      }
    }
    double old_pan = 0.0;
    for (std::size_t c = 0; c < impl.pandya.ncc_host.size(); ++c) {
      const double n = impl.pandya.ncc_host[c];
      old_pan += n * (3.0 * impl.pandya.nk_host[c] + 4.0 * n);
    }
    old_lad *= sizeof(double) / 1.0e6;
    old_pan *= sizeof(double) / 1.0e6;
    std::printf(
        "scratch before M12a (same model space): ladder %.1f MB (%.2fx), "
        "Pandya %.1f MB (%.2fx)\n",
        old_lad, old_lad / op_mb, old_pan, old_pan / op_mb);
  }

  const Named named[] = {
      {"comm110ss", cm::terms::Comm110ss},
      {"comm220ss", cm::terms::Comm220ss},
      {"comm111ss", cm::terms::Comm111ss},
      {"comm121ss", cm::terms::Comm121ss},
      {"comm221ss", cm::terms::Comm221ss},
      {"comm122ss", cm::terms::Comm122ss},
      {"comm222_pp_hhss", cm::terms::Comm222PpHhss},
      {"comm222_phss", cm::terms::Comm222Phss},
      {"comm222_pp_hh_221ss", cm::terms::Comm222PpHh221ss},
  };
  std::vector<Row> rows;
  if (!only_full) {
    for (const Named& n : named) {
      Z.set_zero();
      rows.push_back(TimeIt(n.name, reps, [&] { n.fn(X, Y, Z, ws); }));
    }
  }
  const cm::CommutatorOptions opts;
  rows.push_back(TimeIt("commutator (full)", reps,
                        [&] { cm::Commutator(X, Y, Z, opts, ws); }));

  std::printf("%-22s %12s %12s\n", "term", "mean ms", "min ms");
  for (const Row& r : rows)
    std::printf("%-22s %12.4f %12.4f\n", r.name.c_str(), r.mean_ms, r.min_ms);
  if (checksum) {
    cm::Commutator(X, Y, Z, opts, ws);
    std::printf("checksum [A,H] %016" PRIx64 "\n", Checksum(Z));
    ro.seed = seed + 2;
    const op::Operator N = op::RandomOperator(
        ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kNonHermitian), ro);
    op::Operator ZN;
    cm::Commutator(N, Y, ZN, opts, ws);
    std::printf("checksum [N,H] %016" PRIx64 "\n", Checksum(ZN));
  }
  return 0;
}
