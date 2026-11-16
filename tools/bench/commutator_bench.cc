// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// commutator_bench -- IMSRG(2) commutator micro-benchmark (design/30 section
// 5).
//
//   commutator_bench --emax N --reps R [--reference O16] [--seed S]
//                    [--checksum] [--only-full]
//   commutator_bench --model-mpi
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
// results; `--only-full` times only the full commutator. `--model-mpi` is a
// count-only cost model of the MPI work distribution (M12b): for emax 8, 10 and
// 12 it builds the model space only (no operators, no Workspace) and prints the
// work items' load imbalance on several rank counts.
//
// MPI usage (M12b phase 1: replicated 1B/2B operators, channel blocks of the
// expensive terms split across ranks, combined by reduce + broadcast):
//
//   mpiexec -n P [launcher flags] commutator_bench --emax N --reps R ...
//
// With Open MPI use `--bind-to none` (and `--oversubscribe` if P exceeds the
// core count) and OMP_NUM_THREADS=T, OMP_PROC_BIND=false, so each rank runs T
// threads. Every rank builds identical X and Y (same seed; the hashes are
// compared across ranks and the run fails on a mismatch) and a Workspace on
// the world communicator. Each timed function is bracketed by a barrier and
// its time is the maximum over the ranks (the cost of the slowest rank), per
// repetition; the "mean ms" and "min ms" columns are over repetitions of that
// maximum. The term functions time each rank's own share without the combine;
// the full `Commutator()` includes it; "combine (reduce+bcast)" times
// `ReduceBcastSum` alone over a buffer of the size of Z's zero-, one- and
// two-body parts. With more than one rank a last column "rmin mean ms" gives
// the mean over repetitions of the minimum over ranks (so max/min shows the
// imbalance). Only rank 0 prints. `--checksum` requires bitwise equal hashes
// on all ranks. With one rank the output is that of the serial benchmark plus
// the "ranks"/"threads" header fields.

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/comm/environment.h"
#include "imsrg/comm/ownership_map.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/work_items.h"
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
  double rank_min_mean_ms = 0.0;  // mean over reps of the min over ranks
};

/// Collective over `world`. One warm-up call, then `reps` timed calls of `fn`,
/// each bracketed by a barrier and fences; the time of a repetition is the
/// maximum over the ranks. Returns the mean and min over repetitions of that
/// maximum, and the mean over repetitions of the minimum over ranks, in ms
/// (identical on all ranks; all equal the serial values on one rank).
Row TimeIt(const imsrg::comm::Communicator& world, const std::string& name,
           int reps, const std::function<void()>& fn) {
  world.Barrier();
  fn();
  imsrg::exec::Fence("bench warm-up");
  Row r;
  r.name = name;
  std::vector<double> t_max(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    imsrg::exec::Fence("bench pre");
    world.Barrier();
    const auto t0 = Clock::now();
    fn();
    imsrg::exec::Fence("bench post");
    t_max[static_cast<std::size_t>(i)] =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
  std::vector<double> t_min = t_max;
  world.AllreduceMax(std::span<double>(t_max));
  world.AllreduceMin(std::span<double>(t_min));
  r.min_ms = std::numeric_limits<double>::max();
  double total = 0.0;
  double total_min = 0.0;
  for (int i = 0; i < reps; ++i) {
    const double ms = t_max[static_cast<std::size_t>(i)];
    total += ms;
    total_min += t_min[static_cast<std::size_t>(i)];
    if (ms < r.min_ms) r.min_ms = ms;
  }
  r.mean_ms = total / reps;
  r.rank_min_mean_ms = total_min / reps;
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

/// Collective. True iff `value` is the same on every rank.
bool AgreeOnAllRanks(const imsrg::comm::Communicator& world,
                     std::uint64_t value) {
  const std::vector<std::int64_t> all =
      world.Allgather(static_cast<std::int64_t>(value));
  for (const std::int64_t v : all)
    if (v != all.front()) return false;
  return true;
}

/// Sum of `cost`.
double Sum(const std::vector<double>& cost) {
  double t = 0.0;
  for (const double c : cost) t += c;
  return t;
}

/// Largest entry of `cost` as a fraction of its sum.
double LargestShare(const std::vector<double>& cost) {
  double mx = 0.0;
  for (const double c : cost) mx = c > mx ? c : mx;
  const double t = Sum(cost);
  return t > 0.0 ? mx / t : 0.0;
}

/// LPT imbalance (max / mean - 1) of `cost` on `ranks` ranks, in percent.
double ImbalancePercent(const std::vector<double>& cost, int ranks) {
  return 100.0 * imsrg::comm::OwnershipMap::Greedy(cost, ranks).Imbalance();
}

/// Prints "name  sum  largest share  imbalance on each rank count".
void PrintTermRow(const char* name, const std::vector<double>& cost,
                  const int* ranks, int num_ranks) {
  std::printf("  %-26s sum %.3e  largest %5.1f %%  imbalance %%:", name,
              Sum(cost), 100.0 * LargestShare(cost));
  for (int i = 0; i < num_ranks; ++i)
    std::printf(" %d:%.1f", ranks[i], ImbalancePercent(cost, ranks[i]));
  std::printf("\n");
}

/// The count-only cost model of one emax: channel statistics, ladder with the
/// plan's weight 2 N^3 and the implementation's 2 N^2 (k_pp + k_hh), the
/// ladder with row blocks, comm122 and the particle-hole term.
void ModelMpiRow(int emax) {
  namespace wi = imsrg::commutator::detail;
  constexpr int kRanks[] = {2, 4, 22, 64, 256};
  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{emax}, 16.0, "O16", "O16");
  const wi::WorkSizes s = wi::ComputeWorkSizes(*ms);
  std::int32_t max_n = 0;
  std::size_t nonempty = 0;
  double z2 = 0.0;
  std::vector<double> plan_cost;
  for (const std::int32_t n : s.num_kets) {
    max_n = n > max_n ? n : max_n;
    nonempty += n > 0 ? 1 : 0;
    z2 += static_cast<double>(n) * n;
    plan_cost.push_back(2.0 * n * n * n);
  }
  std::printf(
      "emax %d: %zu two-body channels (%zu non-empty), max N %d, Z2 "
      "%.2f GB\n",
      emax, s.num_kets.size(), nonempty, static_cast<int>(max_n), z2 * 8.0e-9);
  PrintTermRow("ladder, plan 2 N^3", plan_cost, kRanks, 5);
  PrintTermRow("ladder, 2 N^2 (kpp+khh)", wi::LadderWorkItems(s, 1).cost,
               kRanks, 5);
  std::printf(
      "  ladder with row blocks (implementation weight), imbalance %%:");
  for (const int r : kRanks)
    std::printf(
        " %d:%.1f", r,
        100.0 * wi::WorkAssignment::Build(s, 0, r).ladder_map().Imbalance());
  std::printf("\n");
  PrintTermRow("comm122 (whole channels)", wi::Comm122WorkItems(s).cost, kRanks,
               5);
  PrintTermRow("ph 4 Ncc^2 nk (whole)", wi::PhWorkItems(s).cost, kRanks, 5);
  // Total of the assignment: ladder + w_122 comm122 + w_ph ph loads
  // (calibrated weights in ladder units), max over ranks over the mean; the
  // real (joint) cut, then the old equal ladder cut for comparison.
  for (const bool joint : {true, false}) {
    std::printf("  %-26s w_122 %.1f  w_ph %.1f  imbalance %%:",
                joint ? "total, calibrated weights" : "total, equal ladder cut",
                wi::kComm122WeightInLadderUnits, wi::kPhWeightInLadderUnits);
    for (const int r : kRanks) {
      const wi::WorkAssignment a = wi::WorkAssignment::Build(s, 0, r);
      std::vector<double> ladder(static_cast<std::size_t>(r), 0.0);
      if (joint) {
        for (int p = 0; p < r; ++p)
          ladder[static_cast<std::size_t>(p)] = a.ladder_map().load(p);
      } else {
        const wi::LadderItems eq = wi::LadderWorkItems(s, r);
        for (std::size_t i = 0; i < eq.cost.size(); ++i)
          ladder[static_cast<std::size_t>(eq.owner[i])] += eq.cost[i];
      }
      double mx = 0.0, sum = 0.0;
      for (int p = 0; p < r; ++p) {
        const double t =
            ladder[static_cast<std::size_t>(p)] +
            wi::kComm122WeightInLadderUnits * a.comm122_map().load(p) +
            wi::kPhWeightInLadderUnits * a.ph_map().load(p);
        mx = t > mx ? t : mx;
        sum += t;
      }
      std::printf(" %d:%.1f", r, 100.0 * (mx / (sum / r) - 1.0));
    }
    std::printf("\n");
  }
}

/// Runs `ModelMpiRow` for emax 8, 10 and 12.
void ModelMpi() {
  for (const int emax : {8, 10, 12}) ModelMpiRow(emax);
}

}  // namespace

int main(int argc, char** argv) {
  int emax = 4;
  int reps = 10;
  std::uint64_t seed = 1;
  std::string reference = "O16";
  bool checksum = false;
  bool only_full = false;
  bool model_mpi = false;
  // MPI first, Kokkos second, destroyed in the reverse order (design/10 s. 4).
  imsrg::comm::Environment environment(argc, argv);
  imsrg::exec::Runtime runtime(argc, argv);
  const imsrg::comm::Communicator world = environment.world();
  const bool root = world.is_root();
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
    } else if (a == "--model-mpi") {
      model_mpi = true;
    } else {
      std::fprintf(
          stderr,
          "usage: commutator_bench --emax N --reps R [--reference "
          "O16] [--seed S] [--checksum] [--only-full] | --model-mpi\n");
      return 2;
    }
  }
  if (reps < 1) return 2;
  if (model_mpi) {
    if (root) ModelMpi();
    return 0;
  }

  const auto ms = imsrg::basis::ModelSpace::FromNames(
      imsrg::basis::BasisTruncation{emax}, 16.0, reference, reference);

  const auto tw0 = Clock::now();
  cm::Workspace ws(ms, cm::WorkspaceOptions{world});
  imsrg::exec::Fence("workspace");
  const double ws_ms = world.AllreduceMax(
      std::chrono::duration<double, std::milli>(Clock::now() - tw0).count());

  op::RandomOperatorOptions ro;
  ro.seed = seed;
  const op::Operator X = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kAntiHermitian), ro);
  ro.seed = seed + 1;
  const op::Operator Y = op::RandomOperator(
      ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kHermitian), ro);
  op::Operator Z(ms, cm::CommutatorSpec(X.spec(), Y.spec()));

  // X and Y must be identical on all ranks (same seed).
  if (!AgreeOnAllRanks(world, Checksum(X)) ||
      !AgreeOnAllRanks(world, Checksum(Y))) {
    if (root) std::fprintf(stderr, "error: X or Y differ between ranks\n");
    return 1;
  }

  const char* omp = std::getenv("OMP_NUM_THREADS");
  if (root) {
    std::printf("# commutator_bench (this code)\n");
    std::printf("emax %d  reference %s  orbits %d  2B elements %" PRId64
                "  ranks %d  threads %d  OMP_NUM_THREADS %s  "
                "git %s\n",
                emax, reference.c_str(), static_cast<int>(ms->num_orbits()),
                static_cast<std::int64_t>(X.two_body().matrix().total_size()),
                world.size(), imsrg::exec::Runtime::Concurrency(),
                omp ? omp : "(unset)", IMSRG_BENCH_GIT_HASH);
    std::printf("reps %d\nworkspace construction: %.3f ms\n", reps, ws_ms);
  }
  if (root) {
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
      rows.push_back(TimeIt(world, n.name, reps, [&] { n.fn(X, Y, Z, ws); }));
    }
  }
  const cm::CommutatorOptions opts;
  rows.push_back(TimeIt(world, "commutator (full)", reps,
                        [&] { cm::Commutator(X, Y, Z, opts, ws); }));
  const bool multi = world.size() > 1;
  if (multi) {
    // The combine alone: the buffer has the size of Z's three parts.
    std::vector<double> buf(
        1 + Z.one_body().buffer().ToVector().size() +
            static_cast<std::size_t>(Z.two_body().matrix().total_size()),
        1.0);
    rows.push_back(TimeIt(world, "combine (reduce+bcast)", reps, [&] {
      world.ReduceBcastSum(std::span<double>(buf));
    }));
  }

  if (root) {
    std::printf("%-22s %12s %12s", "term", "mean ms", "min ms");
    if (multi) std::printf(" %14s", "rmin mean ms");
    std::printf("\n");
    for (const Row& r : rows) {
      std::printf("%-22s %12.4f %12.4f", r.name.c_str(), r.mean_ms, r.min_ms);
      if (multi) std::printf(" %14.4f", r.rank_min_mean_ms);
      std::printf("\n");
    }
  }
  if (checksum) {
    cm::Commutator(X, Y, Z, opts, ws);
    const std::uint64_t h_ah = Checksum(Z);
    if (!AgreeOnAllRanks(world, h_ah)) {
      if (root) std::fprintf(stderr, "error: [A,H] differs between ranks\n");
      return 1;
    }
    if (root) std::printf("checksum [A,H] %016" PRIx64 "\n", h_ah);
    ro.seed = seed + 2;
    const op::Operator N = op::RandomOperator(
        ms, op::OperatorSpec::Scalar(2, imsrg::Hermiticity::kNonHermitian), ro);
    op::Operator ZN;
    cm::Commutator(N, Y, ZN, opts, ws);
    const std::uint64_t h_nh = Checksum(ZN);
    if (!AgreeOnAllRanks(world, h_nh)) {
      if (root) std::fprintf(stderr, "error: [N,H] differs between ranks\n");
      return 1;
    }
    if (root) std::printf("checksum [N,H] %016" PRIx64 "\n", h_nh);
  }
  return 0;
}
