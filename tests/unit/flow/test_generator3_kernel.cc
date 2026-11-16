// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The device kernel of the three-body generator (task D8) against the former
// host loop (kept here as the reference): bitwise identical values, for every
// generator kind with a function.
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
#include "imsrg/commutator/three_body.h"
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

// The row-subset behaviour (a rank writes only its rows, addressed through the
// local row offsets) is tested on distributed storage in
// test_generator3_mpi.cc.

}  // namespace
}  // namespace imsrg
