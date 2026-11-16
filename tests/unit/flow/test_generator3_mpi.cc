// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The three-body generator kernel on distributed three-body storage (M12b
// task D8): every rank fills its own rows of eta3; the gathered eta3 equals
// the serial one bitwise. The executable runs on 1 (serial unit test), 2 and 4
// ranks.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_distribution.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/flow/generator.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/mpi_test_main.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg {
namespace {

using basis::KeptRow3Index;
using basis::RowPartition;
using basis::ThreeBodyDistribution;
using flow::GeneratorKind;
using flow::GeneratorOptions;
using op::Operator;

enum class Kind { kBalanced, kWholeChannels };

const comm::Communicator& World() { return test_support::TestWorld(); }

std::shared_ptr<const basis::ModelSpace> DistributedSpace(
    const std::shared_ptr<const basis::ModelSpace>& ms, Kind kind) {
  const basis::ThreeBodySpace& s = *ms->three_body_space();
  basis::LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  w.memory.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  const std::int32_t parts = World().size();
  auto part = std::make_shared<const RowPartition>(
      kind == Kind::kBalanced ? RowPartition::Balanced(s, w, parts)
                              : RowPartition::WholeChannels(s, w, parts));
  return ms->WithThreeBodyDistribution(ThreeBodyDistribution{part, World()});
}

// The operator `H` (serial) on the distributed model space `dms`: zero-,
// one- and two-body parts copied, the three-body part restricted to the local
// rows (packed Hermitian: the stored elements z(i >= j) of the local rows j).
Operator Distribute(const Operator& H,
                    const std::shared_ptr<const basis::ModelSpace>& dms) {
  Operator d(dms, H.spec());
  d.zero_body() = H.zero_body();
  d.one_body() = H.one_body();
  d.two_body() = H.two_body();
  const op::ThreeBody& full = *H.three_body();
  op::ThreeBody& loc = *d.three_body();
  const basis::RowSet& rows = dms->three_body_rows();
  for (std::int32_t c = 0; c < loc.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::int32_t n = loc.dim(ch);
    for (std::int32_t j = 0; j < n; ++j) {
      if (!rows.Contains(ch, KeptRow3Index{j})) continue;
      for (std::int32_t i = j; i < n; ++i) {
        loc.set(ch, KeptRow3Index{i}, KeptRow3Index{j},
                full.get(ch, KeptRow3Index{i}, KeptRow3Index{j}));
      }
    }
  }
  d.SyncToDevice();
  return d;
}

struct Case {
  const char* reference;
  int emax;
  int e3max;
};

void CheckGather(const Case& c, Kind kind, GeneratorKind gk) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = c.e3max;
  const commutator::ThreeBodyContext ctx(
      testing::MakeThreeBodySpace(c.emax, c.reference, cuts));
  const Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 29);
  GeneratorOptions opts;
  opts.kind = gk;
  opts.qtransfer_n = 2;
  const Operator eta_serial = flow::ComputeGenerator(H, opts);

  const Operator h_dist =
      Distribute(H, DistributedSpace(H.model_space_ptr(), kind));
  ASSERT_TRUE(h_dist.three_body()->distributed());
  const Operator eta_d = flow::ComputeGenerator(h_dist, opts);
  ASSERT_TRUE(eta_d.three_body()->distributed());
  EXPECT_EQ(eta_d.three_body()->size(),
            static_cast<std::size_t>(
                h_dist.model_space().three_body_rows().StoredSize(true)));
  const op::ThreeBody g = op::GatherToRoot(*eta_d.three_body());
  if (!World().is_root()) {
    EXPECT_TRUE(g.empty());
    return;
  }
  eta_serial.SyncToHost();
  g.SyncToHost();
  const auto zs = eta_serial.three_body()->host_span();
  const auto zg = g.host_span();
  ASSERT_EQ(zs.size(), zg.size());
  bool any = false;
  for (double x : zs) any = any || x != 0.0;
  EXPECT_TRUE(any);
  EXPECT_EQ(std::memcmp(zs.data(), zg.data(), zs.size() * sizeof(double)), 0);
}

void CheckAll(const Case& c) {
  for (Kind kind : {Kind::kBalanced, Kind::kWholeChannels}) {
    for (GeneratorKind gk :
         {GeneratorKind::kWhite, GeneratorKind::kAtan,
          GeneratorKind::kImaginaryTime, GeneratorKind::kWegner}) {
      SCOPED_TRACE(std::string(c.reference) +
                   (kind == Kind::kBalanced ? " balanced " : " channels ") +
                   std::to_string(static_cast<int>(gk)));
      CheckGather(c, kind, gk);
    }
  }
}

TEST(GeneratorThreeBodyMpi, He4Emax2) { CheckAll({"He4", 2, 6}); }
TEST(GeneratorThreeBodyMpi, O16Emax2) { CheckAll({"O16", 2, 6}); }

}  // namespace
}  // namespace imsrg
