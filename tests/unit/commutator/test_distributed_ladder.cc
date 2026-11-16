// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The distributed pp/hh ladder terms and comm122ss (M12b phase 1, D2a): with a
// fake split of the work assignment in one process, the sum over the parts of
// the partial results equals the serial term; stale product buffers left by
// other terms do not change a part's result; the parts really split the work.
#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/ladder.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg::commutator {
namespace {

using imsrg::testing::MakeModelSpace;
using imsrg::testing::RandomOperator;
using op::Operator;

using TermFn = void (*)(const Operator&, const Operator&, Operator&,
                        Workspace&);

struct TermCase {
  const char* name;
  TermFn fn;
};

const TermCase kTerms[] = {
    {"comm222_pp_hh_221ss", &terms::Comm222PpHh221ss},
    {"comm221ss", &terms::Comm221ss},
    {"comm222_pp_hhss", &terms::Comm222PpHhss},
    {"comm122ss", &terms::Comm122ss},
};

struct Pair {
  const char* label;
  Hermiticity x;
  Hermiticity y;
};

constexpr Hermiticity kH = Hermiticity::kHermitian;
constexpr Hermiticity kA = Hermiticity::kAntiHermitian;
constexpr Hermiticity kN = Hermiticity::kNonHermitian;

const Pair kPairs[] = {
    {"H_A", kH, kA}, {"H_H", kH, kH}, {"A_A", kA, kA},
    {"A_H", kA, kH}, {"N_H", kN, kH}, {"N_N", kN, kN},
};

struct Space {
  const char* label;
  int emax;
  const char* reference;
};

/// He4 and open-shell He6 (fractional occupations) at emax 2, O16 at emax 4.
const Space kSpaces[] = {
    {"He4_emax2", 2, "He4"},
    {"He6_emax2", 2, "He6"},
    {"O16_emax4", 4, "O16"},
};

Operator Compute(const TermCase& t, const Operator& X, const Operator& Y,
                 Workspace& ws,
                 const std::shared_ptr<const basis::ModelSpace>& ms) {
  Operator Z(ms, CommutatorSpec(X.spec(), Y.spec()));
  t.fn(X, Y, Z, ws);
  return Z;
}

/// The sum over the parts of `num_parts` of `t` (fake parts, one process).
Operator PartSum(const TermCase& t, const Operator& X, const Operator& Y,
                 Workspace& ws, int num_parts,
                 const std::shared_ptr<const basis::ModelSpace>& ms) {
  Operator sum(ms, CommutatorSpec(X.spec(), Y.spec()));
  for (int p = 0; p < num_parts; ++p) {
    detail::AssignWork(ws, p, num_parts);
    sum += Compute(t, X, Y, ws, ms);
  }
  detail::AssignWork(ws, 0, 1);
  return sum;
}

TEST(commutator_distributed_ladder, parts_sum_to_serial) {
  std::uint64_t seed = 900;
  for (const Space& sp : kSpaces) {
    const auto ms = MakeModelSpace(sp.emax, sp.reference);
    Workspace ws(ms);
    for (const Pair& pr : kPairs) {
      const Operator X = RandomOperator(ms, pr.x, ++seed);
      const Operator Y = RandomOperator(ms, pr.y, ++seed);
      for (const TermCase& t : kTerms) {
        detail::AssignWork(ws, 0, 1);
        const Operator serial = Compute(t, X, Y, ws, ms);
        const double scale = std::max(1.0, op::MaxAbs(serial).max());
        for (int parts : {2, 3, 7}) {
          const Operator sum = PartSum(t, X, Y, ws, parts, ms);
          const double dev = op::MaxAbsDifference(sum, serial).max();
          std::printf(
              "[parts-sum] %s %s %s P=%d: max|dZ| = %.3e (scale %.3e)\n",
              sp.label, pr.label, t.name, parts, dev, scale);
          EXPECT_LE(dev, 1.0e-12 * scale)
              << sp.label << " " << pr.label << " " << t.name << " P=" << parts;
        }
      }
    }
  }
}

/// Dirties the product buffers like the replicated 3f2 path and comm122ss do.
void Dirty(const Operator& A, const Operator& B, Operator& scratch,
           Workspace& ws) {
  detail::LadderProducts(A, B, ws.impl(), detail::LadderScope::kAll);
  terms::Comm122ss(A, B, scratch, ws);
  detail::LadderProducts(B, A, ws.impl(), detail::LadderScope::kAll);
}

TEST(commutator_distributed_ladder, stale_scratch_is_harmless) {
  std::uint64_t seed = 1900;
  for (const Space& sp : kSpaces) {
    const auto ms = MakeModelSpace(sp.emax, sp.reference);
    Workspace ws(ms);
    const Operator U = RandomOperator(ms, kH, 4001);
    const Operator V = RandomOperator(ms, kA, 4002);
    Operator scratch(ms, CommutatorSpec(U.spec(), V.spec()));
    for (const Pair& pr : kPairs) {
      const Operator X = RandomOperator(ms, pr.x, ++seed);
      const Operator Y = RandomOperator(ms, pr.y, ++seed);
      for (const TermCase& t : kTerms) {
        for (int p = 0; p < 2; ++p) {
          detail::AssignWork(ws, p, 2);
          const Operator clean = Compute(t, X, Y, ws, ms);
          Dirty(U, V, scratch, ws);
          const Operator dirty = Compute(t, X, Y, ws, ms);
          const double dev = op::MaxAbsDifference(dirty, clean).max();
          std::printf("[stale] %s %s %s part %d: max|dZ| = %.3e\n", sp.label,
                      pr.label, t.name, p, dev);
          EXPECT_EQ(dev, 0.0)
              << sp.label << " " << pr.label << " " << t.name << " part " << p;
        }
      }
    }
    detail::AssignWork(ws, 0, 1);
  }
}

/// Total flops (sum of rows x N per gemm of the last call) and count of the
/// gemm problems of the last ladder products.
struct GemmTotals {
  std::int64_t count = 0;
  double cells = 0.0;
};

GemmTotals LastProblems(const Workspace& ws) {
  GemmTotals g;
  for (const auto& p : ws.impl().ladder.problems) {
    ++g.count;
    g.cells += static_cast<double>(p.C.rows) * static_cast<double>(p.C.cols);
  }
  return g;
}

/// The expected problems of a part: one per (owned row block, projector) with
/// listed kets.
GemmTotals Expected(const Workspace& ws) {
  const auto& impl = ws.impl();
  GemmTotals g;
  for (const auto* proj : {&impl.ladder.pp, &impl.ladder.hh}) {
    for (const auto& b : impl.work.ladder_blocks()) {
      if (proj->count_host[static_cast<std::size_t>(b.channel)] == 0) continue;
      const auto c = static_cast<std::size_t>(b.channel);
      std::int64_t n = 0;
      while (n * n <
             impl.block_offsets_host[c + 1] - impl.block_offsets_host[c])
        ++n;
      ++g.count;
      g.cells +=
          static_cast<double>(b.row_end - b.row_begin) * static_cast<double>(n);
    }
  }
  return g;
}

TEST(commutator_distributed_ladder, work_actually_splits) {
  const auto ms = MakeModelSpace(4, "O16");
  Workspace ws(ms);
  const Operator X = RandomOperator(ms, kA, 31);
  const Operator Y = RandomOperator(ms, kH, 32);
  const TermCase& t = kTerms[0];
  detail::AssignWork(ws, 0, 1);
  Compute(t, X, Y, ws, ms);
  const GemmTotals serial = LastProblems(ws);
  GemmTotals total;
  for (int p = 0; p < 2; ++p) {
    detail::AssignWork(ws, p, 2);
    Compute(t, X, Y, ws, ms);
    const GemmTotals got = LastProblems(ws);
    const GemmTotals want = Expected(ws);
    const auto& map = ws.impl().work.ladder_map();
    double all = 0.0;
    for (int q = 0; q < 2; ++q) all += map.load(q);
    std::printf("[split] part %d: ladder cost share %.3f, %" PRId64
                " gemms, %.3e C cells "
                "(serial %" PRId64 " gemms, %.3e cells)\n",
                p, map.load(p) / all, static_cast<std::int64_t>(got.count),
                got.cells, static_cast<std::int64_t>(serial.count),
                serial.cells);
    EXPECT_EQ(got.count, want.count);
    EXPECT_EQ(got.cells, want.cells);
    EXPECT_LT(got.cells, serial.cells);
    EXPECT_GT(map.load(p) / all, 0.3);
    total.cells += got.cells;
  }
  EXPECT_EQ(total.cells, serial.cells);  // the parts tile the serial work
  detail::AssignWork(ws, 0, 1);
}

}  // namespace
}  // namespace imsrg::commutator
