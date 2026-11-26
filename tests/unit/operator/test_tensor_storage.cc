// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tensor operator storage (M15): reading the raw tensor goldens, the
// hermiticity relations of reduced elements, imsrg++'s tensor norms, the
// element getters for ch_bra > ch_ket, and the dump round trip.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/tensor_dump.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::op {
namespace {

namespace fs = std::filesystem;

fs::path Golden(const std::string& rel) {
  return fs::path(IMSRG_GOLDEN_DIR) / "tensor" / rel;
}

std::shared_ptr<const basis::ModelSpace> SpaceOf(const std::string& h_dump) {
  return ModelSpaceForDump(ReadOperatorDumpHeader(Golden(h_dump)));
}

int Phase(int x) { return (x % 2 == 0) ? 1 : -1; }

/// Max deviation from O(b,a) = h (-1)^(j_a - j_b) O(a,b) and from the
/// diagonal-block relation V(j,i) = h V(i,j).
double HermiticityDefect(const Operator& op) {
  const auto& basis = op.model_space().basis();
  const int h = HermiticitySign(op.hermiticity());
  double d = 0.0;
  for (int a = 0; a < basis.num_orbits(); ++a)
    for (int b = 0; b < basis.num_orbits(); ++b) {
      const int ja = basis.orbit(OrbitIndex{a}).j2.twice;
      const int jb = basis.orbit(OrbitIndex{b}).j2.twice;
      const double ab = op.one_body().get(OrbitIndex{a}, OrbitIndex{b});
      const double ba = op.one_body().get(OrbitIndex{b}, OrbitIndex{a});
      d = std::max(d, std::abs(ba - h * Phase((ja - jb) / 2) * ab));
    }
  const auto& m = op.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto& info = m.block_info(k);
    if (info.row_block != info.col_block) continue;
    const auto blk = m.host_block(k);
    for (std::int64_t i = 0; i < info.rows; ++i)
      for (std::int64_t j = 0; j < info.cols; ++j)
        d = std::max(d, std::abs(blk(j, i) - h * blk(i, j)));
  }
  return d;
}

struct NormCase {
  const char* file;
  double one_body_norm;
  double two_body_norm;
};

// imsrg++ norms from tests/golden/tensor/result.json.
TEST(TensorStorage, GoldenDumpsReadWithImsrgppNormsAndHermiticity) {
  const auto ms2 = SpaceOf("emax2_He4/inputs/H_HF.dump");
  const auto ms1 = SpaceOf("emax1_He4/inputs/H_HF.dump");
  const std::vector<NormCase> cases2 = {
      {"emax2_He4/HO/E1.tdump", 9.612725332261004, 0.0},
      {"emax2_He4/HO/E2.tdump", 41.830493037485006, 0.0},
      {"emax2_He4/HO/E3.tdump", 110.09043444961884, 0.0},
      {"emax2_He4/HO/M1.tdump", 29.795120743210454, 0.0},
      {"emax2_He4/HO/Fermi.tdump", 12.649110640673518, 0.0},
      {"emax2_He4/HO/GamowTeller.tdump", 24.044403368213008, 0.0},
      {"emax2_He4/HO/M0nu_GT_7.72_none.tdump", 0.0, 35.80458517910453},
      {"emax2_He4/HO/DGT.tdump", 0.0, 70.99295739719346},
      {"emax2_He4/HF/M0nu_T_7.72_none.tdump", 0.0, 3.907242532886348},
      {"emax2_He4/HF/M0nu_C_500_3.tdump", 0.0, 7.2218305711580255},
  };
  const std::vector<NormCase> cases1 = {
      {"emax1_He4/inputs/E2.tdump", 15.510515700928991, 0.0},
      {"emax1_He4/inputs/rand_2_0_0_H_3.tdump", 9.99991983070543,
       3.871078673723755},
      {"emax1_He4/inputs/rand_1_0_1_A_5.tdump", 8.468444835865084,
       4.3682301800333265},
      {"emax1_He4/inputs/rand_1_1_0_H_4.tdump", 4.128476888459966,
       5.683757345649201},
  };
  for (const auto* cases : {&cases2, &cases1}) {
    for (const auto& c : *cases) {
      SCOPED_TRACE(c.file);
      const Operator op =
          ReadTensorDump(Golden(c.file), cases == &cases2 ? ms2 : ms1);
      EXPECT_FALSE(op.spec().is_scalar());
      EXPECT_NEAR(OneBodyNorm(op), c.one_body_norm,
                  1.0e-10 * std::max(1.0, c.one_body_norm));
      EXPECT_NEAR(TwoBodyNorm(op), c.two_body_norm,
                  1.0e-10 * std::max(1.0, c.two_body_norm));
      EXPECT_LE(HermiticityDefect(op),
                1.0e-14 * std::max(1.0, MaxAbs(op).max()));
    }
  }
}

TEST(TensorStorage, ReversedChannelPairsUseTheHermiticityPhase) {
  const auto ms1 = SpaceOf("emax1_He4/inputs/H_HF.dump");
  for (const char* f : {"emax1_He4/inputs/rand_2_0_0_H_3.tdump",
                        "emax1_He4/inputs/rand_1_0_1_A_5.tdump",
                        "emax1_He4/inputs/rand_1_1_0_H_4.tdump"}) {
    SCOPED_TRACE(f);
    const Operator op = ReadTensorDump(Golden(f), ms1);
    const int h = HermiticitySign(op.hermiticity());
    const int n = ms1->num_orbits();
    const int jmax = ms1->two_body().max_J();
    int checked = 0;
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b)
        for (int c = 0; c < n; ++c)
          for (int d = 0; d < n; ++d)
            for (int J1 = 0; J1 <= jmax; ++J1)
              for (int J2 = 0; J2 <= jmax; ++J2) {
                const TwoJ tJ1{2 * J1};
                const TwoJ tJ2{2 * J2};
                const double v = op.two_body().get_normalized(
                    OrbitIndex{a}, OrbitIndex{b}, OrbitIndex{c}, OrbitIndex{d},
                    tJ1, tJ2);
                const double w = op.two_body().get_normalized(
                    OrbitIndex{c}, OrbitIndex{d}, OrbitIndex{a}, OrbitIndex{b},
                    tJ2, tJ1);
                ASSERT_EQ(w, h * Phase(J1 - J2) * v);
                if (v != 0.0) ++checked;
              }
    EXPECT_GT(checked, 0);
  }
}

TEST(TensorStorage, SetNormalizedThroughAReversedPairRoundTrips) {
  const auto ms1 = SpaceOf("emax1_He4/inputs/H_HF.dump");
  OperatorSpec s = OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian);
  s.rank_J = 1;
  s.delta_Tz = 1;
  Operator op(ms1, s);
  // <01 J=1 || O || 00 J=0>: bra channel (J=1) > ket channel (J=0).
  op.two_body().set_normalized(OrbitIndex{0}, OrbitIndex{1}, OrbitIndex{0},
                               OrbitIndex{0}, TwoJ{2}, TwoJ{0}, 0.75);
  EXPECT_EQ(
      op.two_body().get_normalized(OrbitIndex{0}, OrbitIndex{1}, OrbitIndex{0},
                                   OrbitIndex{0}, TwoJ{2}, TwoJ{0}),
      0.75);
  // stored as <00 J=0||O||01 J=1> = h (-1)^(1-0) 0.75 = +0.75 (h = -1)
  EXPECT_EQ(
      op.two_body().get_normalized(OrbitIndex{0}, OrbitIndex{0}, OrbitIndex{0},
                                   OrbitIndex{1}, TwoJ{0}, TwoJ{2}),
      0.75);
  EXPECT_EQ(
      op.two_body().get_normalized(OrbitIndex{1}, OrbitIndex{0}, OrbitIndex{0},
                                   OrbitIndex{0}, TwoJ{2}, TwoJ{0}),
      0.75 * basis::ExchangePhase(TwoJ{1}, TwoJ{1}, TwoJ{2}));
}

TEST(TensorStorage, WriteReadRoundTripIsExact) {
  const auto ms1 = SpaceOf("emax1_He4/inputs/H_HF.dump");
  const Operator op =
      ReadTensorDump(Golden("emax1_He4/inputs/rand_1_1_0_H_4.tdump"), ms1);
  std::stringstream ss;
  WriteTensorDump(ss, op, "rand");
  const Operator back = ReadTensorDump(ss, "roundtrip", ms1);
  EXPECT_EQ(back.spec(), op.spec());
  EXPECT_EQ(MaxAbsDifference(op, back).max(), 0.0);
}

TEST(TensorStorage, BlocksFollowTheAllocationRule) {
  const auto ms1 = SpaceOf("emax1_He4/inputs/H_HF.dump");
  OperatorSpec s = OperatorSpec::Scalar(2);
  s.rank_J = 2;
  s.parity = Parity::kOdd;
  const Operator op(ms1, s);
  const auto& tb = ms1->two_body();
  std::size_t expected = 0;
  for (int cb = 0; cb < tb.num_channels(); ++cb)
    for (int ck = cb; ck < tb.num_channels(); ++ck) {
      const bool conn = ConnectsChannels(s, tb.channel(ChannelIndex{cb}),
                                         tb.channel(ChannelIndex{ck}));
      EXPECT_EQ(op.two_body()
                    .block_index(ChannelIndex{cb}, ChannelIndex{ck})
                    .has_value(),
                conn);
      if (conn) ++expected;
      EXPECT_FALSE(op.two_body()
                       .block_index(ChannelIndex{ck + 1}, ChannelIndex{cb})
                       .has_value());
    }
  EXPECT_EQ(op.two_body().matrix().num_blocks(), expected);
}

}  // namespace
}  // namespace imsrg::op
