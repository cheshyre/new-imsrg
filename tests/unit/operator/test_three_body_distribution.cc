// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Distributed storage of op::ThreeBody (M12b task D7): the model space's
// three-body row distribution, local sizes, ownership checks, collective norm
// and GatherToRoot. The executable runs on 1 (serial unit test), 2 and 4 ranks.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_distribution.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/mpi_test_main.h"

namespace imsrg::op {
namespace {

using basis::KeptRow3Index;
using basis::RowPartition;
using basis::ThreeBodyDistribution;

// kRootOnly: every row on part 0 (zero weights), all other ranks hold none.
enum class Kind { kBalanced, kWholeChannels, kRootOnly };

const comm::Communicator& World() { return test_support::TestWorld(); }

std::shared_ptr<const basis::ModelSpace> SerialSpace(const std::string& nuc) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      nuc, nuc)
      ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 4});
}

std::shared_ptr<const basis::ModelSpace> Distributed(
    const std::shared_ptr<const basis::ModelSpace>& ms, Kind kind) {
  const basis::ThreeBodySpace& s = *ms->three_body_space();
  basis::LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  w.memory.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  const std::int32_t parts = World().size();
  if (kind == Kind::kRootOnly) {
    w.compute.assign(w.compute.size(), 0.0);
    w.memory.assign(w.memory.size(), 0.0);
  }
  auto part = std::make_shared<const RowPartition>(
      kind == Kind::kWholeChannels ? RowPartition::WholeChannels(s, w, parts)
                                   : RowPartition::Balanced(s, w, parts));
  return ms->WithThreeBodyDistribution(ThreeBodyDistribution{part, World()});
}

/// Deterministic test value of the stored element (ch, i, j).
double Value(std::int32_t ch, std::int64_t i, std::int64_t j) {
  return std::sin(1.0 + 7.0 * ch + 0.37 * static_cast<double>(i) +
                  0.011 * static_cast<double>(j * j)) +
         0.25 * static_cast<double>(j);
}

/// True iff the stored element behind (i, j) is a free parameter of the part.
bool IsStored(const ThreeBody& t, std::int64_t i, std::int64_t j) {
  if (!t.packed()) return true;
  if (i < j) return false;
  return !(i == j && t.hermiticity() == Hermiticity::kAntiHermitian);
}

/// True iff the stored row of element (i, j) is local to this rank.
bool IsLocal(const ThreeBody& t, Channel3Index ch, std::int64_t i,
             std::int64_t j) {
  const std::int64_t row = t.packed() ? std::min(i, j) : j;
  return t.model_space_ptr()->three_body_rows().Contains(
      ch, KeptRow3Index{static_cast<std::int32_t>(row)});
}

/// Fills every stored element of `t` that this rank holds with Value().
void FillLocal(ThreeBody& t) {
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::int32_t n = t.dim(ch);
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        if (IsStored(t, i, j) && IsLocal(t, ch, i, j)) {
          t.set(ch, KeptRow3Index{i}, KeptRow3Index{j}, Value(c, i, j));
        }
      }
    }
  }
  t.SyncToDevice();
}

/// Copies the stored elements of the whole part `full` that `d` holds into `d`.
void CopyLocalFrom(const ThreeBody& full, ThreeBody& d) {
  full.SyncToHost();  // `full` may have been changed by device ops
  for (std::int32_t c = 0; c < d.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::int32_t n = d.dim(ch);
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        if (IsStored(d, i, j) && IsLocal(d, ch, i, j)) {
          d.set(ch, KeptRow3Index{i}, KeptRow3Index{j},
                full.get(ch, KeptRow3Index{i}, KeptRow3Index{j}));
        }
      }
    }
  }
  d.SyncToDevice();
}

bool BitsEqual(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

/// Compares the local rows of `d` with the whole part `full` bitwise.
void ExpectLocalRowsEqual(const ThreeBody& full, const ThreeBody& d) {
  full.SyncToHost();
  d.SyncToHost();
  std::int64_t checked = 0;
  for (std::int32_t c = 0; c < d.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::int32_t n = d.dim(ch);
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        if (!IsStored(d, i, j) || !IsLocal(d, ch, i, j)) continue;
        ASSERT_TRUE(BitsEqual(full.get(ch, KeptRow3Index{i}, KeptRow3Index{j}),
                              d.get(ch, KeptRow3Index{i}, KeptRow3Index{j})))
            << "channel " << c << " element " << i << "," << j;
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

struct Case {
  std::string nucleus;
  Kind kind;
  Hermiticity h;
};

std::string Name(const Case& c) {
  return c.nucleus + (c.kind == Kind::kBalanced ? "/balanced/" : "/channels/") +
         std::string(c.h == Hermiticity::kHermitian
                         ? "H"
                         : (c.h == Hermiticity::kAntiHermitian ? "A" : "N"));
}

std::vector<Case> Cases() {
  std::vector<Case> r;
  for (const char* nuc : {"He4", "O16"}) {
    for (Kind k : {Kind::kBalanced, Kind::kWholeChannels}) {
      for (Hermiticity h :
           {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
            Hermiticity::kNonHermitian}) {
        r.push_back({nuc, k, h});
      }
    }
  }
  return r;
}

TEST(ThreeBodyDistributionTest, LocalSizesAndAccessors) {
  for (const Case& c : Cases()) {
    SCOPED_TRACE(Name(c));
    const auto serial = SerialSpace(c.nucleus);
    const auto ms = Distributed(serial, c.kind);
    ASSERT_NE(ms->three_body_distribution(), nullptr);
    EXPECT_EQ(serial->three_body_distribution(), nullptr);
    EXPECT_NE(*ms, *serial);
    EXPECT_EQ(*ms, *Distributed(serial, c.kind));
    const ThreeBody t(ms, c.h);
    EXPECT_TRUE(t.distributed());
    EXPECT_EQ(static_cast<std::int64_t>(t.size()),
              ms->three_body_rows().StoredSize(t.packed()));
    const ThreeBody whole(serial, c.h);
    EXPECT_FALSE(whole.distributed());
    EXPECT_EQ(static_cast<std::int64_t>(whole.size()),
              serial->three_body_rows().StoredSize(whole.packed()));
    EXPECT_EQ(whole.block_offsets().back(),
              static_cast<std::int64_t>(whole.size()));
    // The local sizes of all ranks add up to the whole size.
    EXPECT_EQ(World().AllreduceSum(static_cast<std::int64_t>(t.size())),
              static_cast<std::int64_t>(whole.size()));
    EXPECT_THROW(t.block_offsets(), InvariantError);
  }
}

TEST(ThreeBodyDistributionTest, ConstructionErrors) {
  const auto serial = SerialSpace("He4");
  const auto no3b = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 2}, 16.0, "He4", "He4");
  const basis::ThreeBodySpace& s = *serial->three_body_space();
  basis::LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  w.memory = w.compute;
  auto part = std::make_shared<const RowPartition>(
      RowPartition::Balanced(s, w, World().size()));
  EXPECT_THROW(no3b->WithThreeBodyDistribution({part, World()}), InputError);
  EXPECT_THROW(serial->WithThreeBodyDistribution({nullptr, World()}),
               InputError);
  auto wrong = std::make_shared<const RowPartition>(
      RowPartition::Balanced(s, w, World().size() + 1));
  EXPECT_THROW(serial->WithThreeBodyDistribution({wrong, World()}), InputError);
  EXPECT_EQ(no3b->three_body_rows().NumRows(), 0);
}

TEST(ThreeBodyDistributionTest, OwnershipChecks) {
  for (const Case& c : Cases()) {
    SCOPED_TRACE(Name(c));
    const auto ms = Distributed(SerialSpace(c.nucleus), c.kind);
    ThreeBody t(ms, c.h);
    std::int64_t local = 0;
    std::int64_t foreign = 0;
    for (std::int32_t ch = 0; ch < t.num_channels(); ++ch) {
      const std::int32_t n = t.dim(Channel3Index{ch});
      for (std::int32_t j = 0; j < n; j += 3) {
        for (std::int32_t i = 0; i < n; i += 2) {
          const KeptRow3Index a{i};
          const KeptRow3Index b{j};
          if (IsLocal(t, Channel3Index{ch}, i, j)) {
            ++local;
            if (IsStored(t, i, j)) {
              t.set(Channel3Index{ch}, a, b, 1.5);
              EXPECT_EQ(t.get(Channel3Index{ch}, a, b), 1.5);
              t.Add(Channel3Index{ch}, a, b, 0.5);
              EXPECT_EQ(t.get(Channel3Index{ch}, a, b), 2.0);
            } else {
              EXPECT_NO_THROW(t.get(Channel3Index{ch}, a, b));
            }
          } else {
            ++foreign;
            EXPECT_THROW(t.get(Channel3Index{ch}, a, b), InvariantError);
            EXPECT_THROW(t.set(Channel3Index{ch}, a, b, 1.0), InvariantError);
            EXPECT_THROW(t.Add(Channel3Index{ch}, a, b, 1.0), InvariantError);
          }
        }
      }
    }
    EXPECT_GT(local + foreign, 0);
    if (World().size() > 1 && ms->three_body_rows().NumRows() > 0) {
      EXPECT_GT(foreign, 0);
    }
  }
}

TEST(ThreeBodyDistributionTest, NormMatchesSerialOnAllRanks) {
  for (const Case& c : Cases()) {
    SCOPED_TRACE(Name(c));
    const auto serial = SerialSpace(c.nucleus);
    ThreeBody whole(serial, c.h);
    FillLocal(whole);
    const double ref = ImsrgppNorm(whole);
    ThreeBody d(Distributed(serial, c.kind), c.h);
    CopyLocalFrom(whole, d);
    ExpectLocalRowsEqual(whole, d);
    const double got = ImsrgppNorm(d);
    EXPECT_LE(std::abs(got - ref), 1.0e-14 * std::max(1.0, std::abs(ref)));
    const std::vector<double> all = World().Allgather(got);
    for (const double x : all) EXPECT_TRUE(BitsEqual(x, got));
  }
}

TEST(ThreeBodyDistributionTest, OperatorNormIsCollective) {
  const auto serial = SerialSpace("He4");
  Operator xs(serial, OperatorSpec::Scalar(3, Hermiticity::kHermitian));
  FillLocal(*xs.three_body());
  Operator xd(Distributed(serial, Kind::kBalanced),
              OperatorSpec::Scalar(3, Hermiticity::kHermitian));
  CopyLocalFrom(*xs.three_body(), *xd.three_body());
  const double ref = Norm(xs);
  const double got = Norm(xd);
  EXPECT_LE(std::abs(got - ref), 1.0e-14 * std::max(1.0, std::abs(ref)));
  const std::vector<double> all = World().Allgather(got);
  for (const double x : all) EXPECT_TRUE(BitsEqual(x, got));
}

TEST(ThreeBodyDistributionTest, AxpyAndScaleAreLocal) {
  for (const char* nuc : {"He4", "O16"}) {
    for (Kind kind : {Kind::kBalanced, Kind::kWholeChannels}) {
      for (Hermiticity h :
           {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
        SCOPED_TRACE(std::string(nuc));
        const auto serial = SerialSpace(nuc);
        const auto dist = Distributed(serial, kind);
        const OperatorSpec spec = OperatorSpec::Scalar(3, h);
        Operator xs(serial, spec);
        Operator ys(serial, spec);
        FillLocal(*xs.three_body());
        FillLocal(*ys.three_body());
        // Distinct values in y: scale the filled operator.
        Scale(ys, -0.75);
        Operator xd(dist, spec);
        Operator yd(dist, spec);
        CopyLocalFrom(*xs.three_body(), *xd.three_body());
        CopyLocalFrom(*ys.three_body(), *yd.three_body());
        Axpy(0.3, xs, ys);
        Axpy(0.3, xd, yd);
        Scale(ys, 1.7);
        Scale(yd, 1.7);
        ExpectLocalRowsEqual(*ys.three_body(), *yd.three_body());
      }
    }
  }
}

TEST(ThreeBodyDistributionTest, GatherToRootRoundTrip) {
  for (const Case& c : Cases()) {
    SCOPED_TRACE(Name(c));
    const auto serial = SerialSpace(c.nucleus);
    ThreeBody whole(serial, c.h);
    FillLocal(whole);
    ThreeBody d(Distributed(serial, c.kind), c.h);
    CopyLocalFrom(whole, d);
    const ThreeBody g = GatherToRoot(d);
    if (World().is_root()) {
      ASSERT_FALSE(g.empty());
      EXPECT_FALSE(g.distributed());
      EXPECT_EQ(g.hermiticity(), c.h);
      ASSERT_EQ(g.size(), whole.size());
      whole.SyncToHost();
      g.SyncToHost();
      EXPECT_EQ(std::memcmp(whole.host_span().data(), g.host_span().data(),
                            whole.size() * sizeof(double)),
                0);
      EXPECT_EQ(*g.model_space_ptr(), *serial);
    } else {
      EXPECT_TRUE(g.empty());
    }
    // An undistributed part is copied on every rank.
    const ThreeBody same = GatherToRoot(whole);
    EXPECT_EQ(same.size(), whole.size());
  }
}

TEST(ThreeBodyDistributionTest, SetHermiticity) {
  const auto serial = SerialSpace("He4");
  ThreeBody d(Distributed(serial, Kind::kBalanced), Hermiticity::kHermitian);
  for (double& x : d.host_span()) x = 2.0;
  d.SyncToDevice();
  d.set_hermiticity(Hermiticity::kAntiHermitian);
  d.SyncToHost();
  const auto& rows = d.model_space_ptr()->three_body_rows();
  for (const basis::RowRange& r : rows.ranges()) {
    for (std::int32_t k = r.row_begin.value(); k < r.row_end.value(); ++k) {
      const std::int64_t off = rows.LocalRowOffset(r.channel, KeptRow3Index{k},
                                                   /*packed=*/true);
      EXPECT_EQ(d.host_span()[static_cast<std::size_t>(off)], 0.0);
      if (k + 1 < d.dim(r.channel)) {
        EXPECT_EQ(d.host_span()[static_cast<std::size_t>(off) + 1], 2.0);
      }
    }
  }
  EXPECT_THROW(d.set_hermiticity(Hermiticity::kNonHermitian),
               NotImplementedError);
  EXPECT_EQ(d.hermiticity(), Hermiticity::kAntiHermitian);
  ThreeBody dense(Distributed(serial, Kind::kBalanced),
                  Hermiticity::kNonHermitian);
  EXPECT_THROW(dense.set_hermiticity(Hermiticity::kHermitian),
               NotImplementedError);
}

// Elements above the diagonal of a packed part are not stored; get() returns
// h * (transposed element) (0 on the diagonal of an anti-Hermitian part), and
// set() of such an element writes the stored transposed one.
TEST(ThreeBodyDistributionTest, UpperTriangleGetSetValues) {
  for (const Case& c : Cases()) {
    SCOPED_TRACE(Name(c));
    const auto ms = Distributed(SerialSpace(c.nucleus), c.kind);
    ThreeBody t(ms, c.h);
    FillLocal(t);
    const double sign = c.h == Hermiticity::kAntiHermitian ? -1.0 : 1.0;
    std::int64_t checked = 0;
    for (std::int32_t ch = 0; ch < t.num_channels(); ++ch) {
      const Channel3Index cc{ch};
      const std::int32_t n = t.dim(cc);
      for (std::int32_t j = 0; j < n; ++j) {
        for (std::int32_t i = 0; i < n; ++i) {
          if (IsStored(t, i, j) || !IsLocal(t, cc, i, j)) continue;
          const KeptRow3Index a{i};
          const KeptRow3Index b{j};
          ++checked;
          if (i == j) {
            EXPECT_EQ(t.get(cc, a, b), 0.0);  // anti-Hermitian diagonal
          } else {
            EXPECT_EQ(t.get(cc, a, b), sign * Value(ch, j, i));
            t.set(cc, a, b, 3.25);
            EXPECT_EQ(t.get(cc, a, b), 3.25);
            EXPECT_EQ(t.get(cc, b, a), sign * 3.25);
          }
        }
      }
    }
    if (c.h != Hermiticity::kNonHermitian &&
        ms->three_body_rows().NumRows() > 0) {
      EXPECT_GT(World().AllreduceSum(checked), 0);
    }
  }
}

// A rank without rows (kRootOnly: all other ranks) takes part in the
// collectives and in the local arithmetic.
TEST(ThreeBodyDistributionTest, RankWithoutRows) {
  for (const Hermiticity h :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
        Hermiticity::kNonHermitian}) {
    const auto serial = SerialSpace("He4");
    const auto dist = Distributed(serial, Kind::kRootOnly);
    ThreeBody whole(serial, h);
    FillLocal(whole);
    ThreeBody d(dist, h);
    EXPECT_EQ(d.size() == 0, !World().is_root());
    CopyLocalFrom(whole, d);
    const double ref = ImsrgppNorm(whole);
    const double got = ImsrgppNorm(d);
    EXPECT_LE(std::abs(got - ref), 1.0e-14 * std::max(1.0, std::abs(ref)));
    if (h != Hermiticity::kNonHermitian) {
      const OperatorSpec spec = OperatorSpec::Scalar(3, h);
      Operator xs(serial, spec);
      Operator ys(serial, spec);
      FillLocal(*xs.three_body());
      FillLocal(*ys.three_body());
      Scale(ys, -0.75);
      Operator xd(dist, spec);
      Operator yd(dist, spec);
      CopyLocalFrom(*xs.three_body(), *xd.three_body());
      CopyLocalFrom(*ys.three_body(), *yd.three_body());
      Axpy(0.3, xs, ys);
      Axpy(0.3, xd, yd);
      Scale(yd, 1.7);
      Scale(ys, 1.7);
      if (yd.three_body()->size() > 0) {
        ExpectLocalRowsEqual(*ys.three_body(), *yd.three_body());
      }
    }
    const ThreeBody g = GatherToRoot(d);
    if (World().is_root()) {
      ASSERT_FALSE(g.empty());
      ASSERT_EQ(g.size(), whole.size());
      whole.SyncToHost();
      g.SyncToHost();
      EXPECT_EQ(std::memcmp(whole.host_span().data(), g.host_span().data(),
                            whole.size() * sizeof(double)),
                0);
    } else {
      EXPECT_TRUE(g.empty());
    }
  }
}

// Whole-channel partitions leave ranks without rows when there are more
// ranks than channels worth of work; the collectives still work.
TEST(ThreeBodyDistributionTest, WholeChannelsCollectives) {
  const auto serial = SerialSpace("He4");
  const auto dist = Distributed(serial, Kind::kWholeChannels);
  ThreeBody whole(serial, Hermiticity::kHermitian);
  FillLocal(whole);
  ThreeBody d(dist, Hermiticity::kHermitian);
  CopyLocalFrom(whole, d);
  const double ref = ImsrgppNorm(whole);
  EXPECT_LE(std::abs(ImsrgppNorm(d) - ref), 1.0e-14 * std::max(1.0, ref));
  const ThreeBody g = GatherToRoot(d);
  EXPECT_EQ(g.empty(), !World().is_root());
}

// Deriving a model space with another reference drops the distribution (the
// kept space depends on the reference).
TEST(ThreeBodyDistributionTest, WithReferenceDropsDistribution) {
  const auto serial = SerialSpace("He4");
  const auto dist = Distributed(serial, Kind::kBalanced);
  ASSERT_NE(dist->three_body_distribution(), nullptr);
  const auto derived = dist->WithReference(dist->reference());
  EXPECT_EQ(derived->three_body_distribution(), nullptr);
  EXPECT_EQ(derived->three_body_rows().NumRows(),
            serial->three_body_rows().NumRows());
  const auto derived2 =
      dist->WithReferenceAndPartition(dist->reference(), dist->partition());
  EXPECT_EQ(derived2->three_body_distribution(), nullptr);
  EXPECT_EQ(dist->WithoutThreeBodyDistribution()->three_body_distribution(),
            nullptr);
}

// The partition must cover THIS model space's kept rows.
TEST(ThreeBodyDistributionTest, PartitionOfAnotherSpaceIsRejected) {
  const auto serial = SerialSpace("He4");
  const auto bigger = basis::ModelSpace::FromNames(
                          basis::BasisTruncation{.emax = 2}, 16.0, "He4", "He4")
                          ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 6});
  const basis::ThreeBodySpace& s = *bigger->three_body_space();
  basis::LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  w.memory = w.compute;
  auto part = std::make_shared<const RowPartition>(
      RowPartition::Balanced(s, w, World().size()));
  ASSERT_NE(s.num_triples(), serial->three_body_space()->num_triples());
  EXPECT_THROW(serial->WithThreeBodyDistribution({part, World()}), InputError);
  EXPECT_NO_THROW(bigger->WithThreeBodyDistribution({part, World()}));
}

TEST(ThreeBodyDistributionTest, ReadDumpIntoDistributedSpaceThrows) {
  const auto serial = SerialSpace("He4");
  std::istringstream in("");
  EXPECT_THROW(
      ReadOperatorDump(in, "dist", Distributed(serial, Kind::kBalanced),
                       OperatorDumpReadOptions{}),
      InputError);
}

TEST(ThreeBodyDistributionTest, DumpOfDistributedPartThrows) {
  const auto serial = SerialSpace("He4");
  Operator x(Distributed(serial, Kind::kBalanced),
             OperatorSpec::Scalar(3, Hermiticity::kHermitian));
  std::ostringstream out;
  EXPECT_THROW(WriteOperatorDump(out, x, OperatorDumpWriteOptions{}),
               InputError);
}

}  // namespace
}  // namespace imsrg::op
