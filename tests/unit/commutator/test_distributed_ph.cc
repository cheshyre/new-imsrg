// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Distributed particle-hole term comm222_phss (M12b phase 1, task D2b): the
// sum over the parts of the partial terms equals the serial term, stale
// scratch of non-owned channels is never read, and each part runs only its own
// GEMMs. The parts are faked in one process with `detail::AssignWork`.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/pandya.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/workspace.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg::commutator {
namespace {

using commutator_test::MakeSpace;
using commutator_test::SpaceCase;
using imsrg::testing::RandomOperator;
using op::Operator;

/// The two-body storage of `z` on the host.
std::vector<double> TwoBodyData(const Operator& z) {
  return z.two_body().matrix().buffer().ToVector();
}

/// The model spaces of the tests: He4 and fractional He6 at emax 2, O16 at 4.
std::vector<SpaceCase> Cases() {
  return {{"He4_emax2", 2, "He4", false},
          {"He6_emax2_fractional", 2, "He6", true},
          {"O16_emax4", 4, "O16", false}};
}

/// An operand pair of the tests.
struct Pair {
  const char* label;
  Hermiticity x;
  Hermiticity y;
};

/// (H,A), (H,H), (A,A) take the symmetric path; (N,N) and (N,H) the general.
std::vector<Pair> Pairs() {
  return {{"H,A", Hermiticity::kHermitian, Hermiticity::kAntiHermitian},
          {"H,H", Hermiticity::kHermitian, Hermiticity::kHermitian},
          {"A,A", Hermiticity::kAntiHermitian, Hermiticity::kAntiHermitian},
          {"N,N", Hermiticity::kNonHermitian, Hermiticity::kNonHermitian},
          {"N,H", Hermiticity::kNonHermitian, Hermiticity::kHermitian}};
}

/// Z of the term for part `part` of `parts` on a fresh zero Z.
std::vector<double> PartialTerm(Workspace& ws, const Operator& X,
                                const Operator& Y, int part, int parts) {
  detail::AssignWork(ws, part, parts);
  Operator Z(ws.model_space_ptr(), CommutatorSpec(X.spec(), Y.spec()));
  terms::Comm222Phss(X, Y, Z, ws);
  return TwoBodyData(Z);
}

/// max |a - b| and max |a|.
void MaxDiff(const std::vector<double>& a, const std::vector<double>& b,
             double& dmax, double& amax) {
  dmax = 0.0;
  amax = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dmax = std::max(dmax, std::abs(a[i] - b[i]));
    amax = std::max(amax, std::abs(a[i]));
  }
}

TEST(commutator_distributed_ph, sum_of_parts_equals_serial) {
  for (const SpaceCase& sc : Cases()) {
    const auto ms = MakeSpace(sc);
    Workspace ws(ms);
    for (const Pair& pr : Pairs()) {
      const Operator X = RandomOperator(ms, pr.x, 11);
      const Operator Y = RandomOperator(ms, pr.y, 29);
      const std::vector<double> serial = PartialTerm(ws, X, Y, 0, 1);
      for (const int parts : {2, 3}) {
        std::vector<double> sum(serial.size(), 0.0);
        for (int p = 0; p < parts; ++p) {
          const std::vector<double> z = PartialTerm(ws, X, Y, p, parts);
          for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += z[i];
        }
        double dmax = 0.0, amax = 0.0;
        MaxDiff(sum, serial, dmax, amax);
        std::printf("[dist-ph] %s (%s) P=%d: max|dZ| = %.3e (max|Z| = %.3e)\n",
                    sc.label.c_str(), pr.label, parts, dmax, amax);
        EXPECT_LE(dmax, 1.0e-12 * std::max(1.0, amax))
            << sc.label << " " << pr.label << " P=" << parts;
      }
    }
  }
}

TEST(commutator_distributed_ph, stale_scratch_is_never_read) {
  const auto ms = MakeSpace({"O16_emax4", 4, "O16", false});
  for (const Pair& pr : Pairs()) {
    Workspace used(ms);
    Workspace clean(ms);
    const Operator X = RandomOperator(ms, pr.x, 11);
    const Operator Y = RandomOperator(ms, pr.y, 29);
    const Operator X2 = RandomOperator(ms, pr.x, 101);
    const Operator Y2 = RandomOperator(ms, pr.y, 202);
    // Parts 1 fill the scratch of their own channels (stale for part 0).
    (void)PartialTerm(used, X2, Y2, 1, 2);
    (void)PartialTerm(used, X2, Y2, 1, 3);
    const std::vector<double> a = PartialTerm(used, X, Y, 0, 2);
    const std::vector<double> b = PartialTerm(clean, X, Y, 0, 2);
    std::printf("[dist-ph] stale scratch (%s): bitwise equal = %d\n", pr.label,
                a == b ? 1 : 0);
    EXPECT_TRUE(a == b) << pr.label;
  }
}

TEST(commutator_distributed_ph, work_actually_splits) {
  const auto ms = MakeSpace({"O16_emax4", 4, "O16", false});
  Workspace ws(ms);
  const Operator N = RandomOperator(ms, Hermiticity::kNonHermitian, 5);
  (void)PartialTerm(ws, N, N, 0, 1);  // allocates the general scratch
  const std::size_t total = ws.impl().pandya.gemms.size();
  const std::size_t total_general = ws.impl().pandya.general.gemms.size();
  ASSERT_GT(total, 1u);
  std::size_t sum = 0, sum_right = 0, sum_general = 0;
  std::int64_t kets = 0;
  double load[2] = {0.0, 0.0};
  for (int p = 0; p < 2; ++p) {
    detail::AssignWork(ws, p, 2);
    load[p] = ws.impl().work.ph_map().load(p);
  }
  for (int p = 0; p < 2; ++p) {
    detail::AssignWork(ws, p, 2);
    const auto& pd = ws.impl().pandya;
    std::printf(
        "[dist-ph] part %d/2: ph cost share %.4f, owned channels %zu, GEMMs "
        "%zu of %zu (right %zu, general %zu of %zu), owned kets %lld\n",
        p, load[p] / (load[0] + load[1]), ws.impl().work.ph_channels().size(),
        pd.gemms_owned.size(), total, pd.gemms_right_owned.size(),
        pd.general.gemms_owned.size(), total_general,
        static_cast<long long>(pd.num_owned_kets));
    EXPECT_GT(pd.gemms_owned.size(), 0u);
    EXPECT_LT(pd.gemms_owned.size(), total);
    EXPECT_EQ(pd.gemms_right_owned.size(), pd.gemms_owned.size());
    sum += pd.gemms_owned.size();
    sum_right += pd.gemms_right_owned.size();
    sum_general += pd.general.gemms_owned.size();
    kets += pd.num_owned_kets;
  }
  EXPECT_EQ(sum, total);
  EXPECT_EQ(sum_right, total);
  EXPECT_EQ(sum_general, total_general);
  std::int64_t all_kets = 0;
  // Channels without contraction pairs (|K| = 0) have no work and no owner.
  const auto& pdata = ws.impl().pandya;
  for (std::size_t c = 0; c < pdata.ncc_host.size(); ++c)
    if (pdata.nk_host[c] > 0) all_kets += pdata.ncc_host[c];
  EXPECT_EQ(kets, all_kets);
  detail::AssignWork(ws, 0, 1);
  EXPECT_FALSE(ws.impl().pandya.distributed);
  EXPECT_TRUE(ws.impl().pandya.gemms_owned.empty());
}

}  // namespace
}  // namespace imsrg::commutator
