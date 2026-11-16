// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The device kernel of the three-body generator (task D8) against the former
// host loop (kept here as the reference): bitwise identical values, for every
// generator kind with a function, and the row-set restriction.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/flow/detail/generator_three_body.h"
#include "imsrg/flow/generator.h"
#include "imsrg/operator/operator.h"
#include "tests/support/three_body_test_support.h"

namespace imsrg {
namespace {

using flow::GeneratorKind;
using flow::GeneratorOptions;
using op::Operator;

// The former host loop of FillThreeBodyEta (generator.cc before D8), writing
// the three-body part of `eta` (zero on entry).
void HostReferenceEta3(const Operator& H, const GeneratorOptions& opts,
                       Operator& eta) {
  const flow::Denominators den(H, opts);
  const basis::ModelSpace& ms = H.model_space();
  const basis::Partition& part = ms.partition();
  const basis::ThreeBodySpace& ts = *ms.three_body_space();
  const op::ThreeBody& H3 = *H.three_body();
  op::ThreeBody& eta3 = *eta.three_body();
  using basis::OrbitClass;
  const auto all_class = [&](const basis::OrbitTriple& t, bool core) {
    const bool a = part.orbit_class(t.a) == OrbitClass::kCore;
    const bool b = part.orbit_class(t.b) == OrbitClass::kCore;
    const bool c = part.orbit_class(t.c) == OrbitClass::kCore;
    return core ? (a && b && c) : (!a && !b && !c);
  };
  for (std::int32_t c = 0; c < ts.num_channels(); ++c) {
    const Channel3Index ch{c};
    const basis::KeptChannel3& kc = ts.channel(ch);
    for (std::int32_t tb = kc.triple_begin.value(); tb < kc.triple_end.value();
         ++tb) {
      const basis::OrbitTriple& bt = ts.triple(basis::Triple3Index{tb});
      if (!all_class(bt, false)) continue;
      for (std::int32_t tk = kc.triple_begin.value();
           tk < kc.triple_end.value(); ++tk) {
        const basis::OrbitTriple& kt = ts.triple(basis::Triple3Index{tk});
        if (!all_class(kt, true)) continue;
        const double d = den.three_body(bt.a, bt.b, bt.c, kt.a, kt.b, kt.c);
        for (std::int32_t i = bt.row_begin.value(); i < bt.row_end.value();
             ++i) {
          for (std::int32_t j = kt.row_begin.value(); j < kt.row_end.value();
               ++j) {
            const basis::KeptRow3Index ri{i}, rj{j};
            eta3.set(ch, ri, rj,
                     flow::GeneratorFunction(opts.kind, H3.get(ch, ri, rj), d,
                                             opts.qtransfer_n));
          }
        }
      }
    }
  }
}

bool BitwiseEqual(std::span<const double> a, std::span<const double> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

struct Case {
  const char* reference;
  int emax;
  int e3max;
};

std::shared_ptr<const basis::ModelSpace> MakeSpace(const Case& c) {
  basis::ThreeBodyCuts cuts;
  cuts.e3max = c.e3max;
  return testing::MakeThreeBodySpace(c.emax, c.reference, cuts);
}

const GeneratorKind kKinds[] = {
    GeneratorKind::kWhite, GeneratorKind::kAtan, GeneratorKind::kImaginaryTime,
    GeneratorKind::kWegner, GeneratorKind::kQtransferAtan};

void CheckBitwise(const Case& c) {
  const commutator::ThreeBodyContext ctx(MakeSpace(c));
  const Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 29);
  for (GeneratorKind kind : kKinds) {
    GeneratorOptions opts;
    opts.kind = kind;
    opts.qtransfer_n = 2;
    Operator eta_new = flow::ComputeGenerator(H, opts);
    Operator eta_ref(H.model_space_ptr(), flow::GeneratorSpec(H.spec(), kind));
    eta_ref.set_zero();
    HostReferenceEta3(H, opts, eta_ref);
    eta_new.SyncToHost();
    const auto zn = eta_new.three_body()->host_span();
    const auto zr = eta_ref.three_body()->host_span();
    ASSERT_EQ(zn.size(), zr.size());
    bool any = false;
    for (double x : zr) any = any || x != 0.0;
    EXPECT_TRUE(any) << "reference is trivially zero";
    EXPECT_TRUE(BitwiseEqual(zn, zr)) << c.reference << " e3max " << c.e3max
                                      << " kind " << static_cast<int>(kind);
  }
}

TEST(GeneratorThreeBodyKernel, BitwiseHe4Emax2) { CheckBitwise({"He4", 2, 6}); }
TEST(GeneratorThreeBodyKernel, BitwiseO16Emax2) { CheckBitwise({"O16", 2, 6}); }
TEST(GeneratorThreeBodyKernel, BitwiseO16Emax3) { CheckBitwise({"O16", 3, 9}); }

// Row subsets: every even channel whole, the first half of the triples of the
// odd channels with at least two triples.
TEST(GeneratorThreeBodyKernel, RowSetWritesOnlyItsRows) {
  const Case c{"O16", 2, 6};
  const commutator::ThreeBodyContext ctx(MakeSpace(c));
  Operator H =
      testing::RandomThreeBodyOperator(ctx, Hermiticity::kHermitian, 31);
  H.SyncToDevice();
  GeneratorOptions opts;
  opts.kind = GeneratorKind::kAtan;
  const Operator eta_whole = flow::ComputeGenerator(H, opts);
  const basis::ModelSpace& ms = H.model_space();
  const basis::ThreeBodySpace& ts = *ms.three_body_space();

  std::vector<basis::RowRange> ranges;
  for (std::int32_t ch = 0; ch < ts.num_channels(); ++ch) {
    const basis::KeptChannel3& kc = ts.channel(Channel3Index{ch});
    if (kc.dim == 0) continue;
    std::int32_t tend = kc.triple_end.value();
    if (ch % 2 != 0) {
      if (kc.triple_end.value() - kc.triple_begin.value() < 2) continue;
      tend = kc.triple_begin.value() +
             (kc.triple_end.value() - kc.triple_begin.value()) / 2;
    }
    basis::RowRange rg;
    rg.channel = Channel3Index{ch};
    rg.triple_begin = kc.triple_begin;
    rg.triple_end = basis::Triple3Index{tend};
    rg.row_begin = ts.triple(rg.triple_begin).row_begin;
    rg.row_end = ts.triple(basis::Triple3Index{tend - 1}).row_end;
    ranges.push_back(rg);
  }
  ASSERT_GE(ranges.size(), 3u);
  const basis::RowSet rows = basis::RowSet::FromRanges(ts, ranges);

  Operator eta(H.model_space_ptr(), flow::GeneratorSpec(H.spec(), opts.kind));
  eta.set_zero();
  eta.SyncToDevice();
  const flow::Denominators den(H, opts);
  std::vector<std::int32_t> is_core(
      static_cast<std::size_t>(ms.basis().num_orbits()));
  for (std::size_t o = 0; o < is_core.size(); ++o) {
    is_core[o] =
        ms.partition().orbit_class(OrbitIndex{static_cast<std::int32_t>(o)}) ==
                basis::OrbitClass::kCore
            ? 1
            : 0;
  }
  flow::detail::FillThreeBodyGenerator(*H.three_body(), *eta.three_body(), ts,
                                       rows, den.OneBodyDiagonal(), is_core,
                                       opts.denominator_cutoff, opts.kind, 1);
  eta.SyncToHost();

  const auto zs = eta.three_body()->host_span();
  const auto zw = eta_whole.three_body()->host_span();
  std::vector<char> expect_set(zs.size(), 0);
  const auto offs = eta.three_body()->block_offsets();
  for (const basis::RowRange& rg : rows.ranges()) {
    const std::int64_t n = ts.dim(rg.channel);
    for (std::int32_t k = rg.row_begin.value(); k < rg.row_end.value(); ++k) {
      const std::int64_t col =
          offs[rg.channel.ToSize()] + op::ThreeBody::PackedIndex(n, k, k);
      for (std::int64_t m = k; m < n; ++m) {
        expect_set[static_cast<std::size_t>(col + m - k)] = 1;
      }
    }
  }
  std::size_t nset = 0, nnonzero = 0, bad = 0;
  for (std::size_t e = 0; e < zs.size(); ++e) {
    if (expect_set[e] != 0) {
      ++nset;
      if (zw[e] != 0.0) ++nnonzero;
      if (std::memcmp(&zs[e], &zw[e], sizeof(double)) != 0) ++bad;
    } else if (zs[e] != 0.0) {
      ++bad;
    }
  }
  EXPECT_EQ(bad, 0u);
  EXPECT_GT(nset, 0u);
  EXPECT_LT(nset, zs.size());
  EXPECT_GT(nnonzero, 0u);
}

}  // namespace
}  // namespace imsrg
