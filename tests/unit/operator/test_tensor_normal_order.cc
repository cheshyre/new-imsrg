// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Normal ordering of tensor operators (M15): round trip, hermiticity of the
// result (the one-body elements (a, b) and (b, a) are computed independently),
// and the fork-compatibility option (design/reference/06 section 4.1).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::op {
namespace {

namespace fs = std::filesystem;

int Phase(int x) { return (x % 2 == 0) ? 1 : -1; }

/// He4 emax-2 model space; with `open_shell` the 0p3/2 orbits get occupation
/// 0.5 and the 0s1/2 neutron 0.75 (an ensemble-like fractional reference).
std::shared_ptr<const basis::ModelSpace> Space(bool open_shell) {
  OperatorDumpHeader h = ReadOperatorDumpHeader(
      fs::path(IMSRG_GOLDEN_DIR) / "tensor/emax2_He4/inputs/H_HF.dump");
  if (open_shell) {
    for (auto& o : h.orbits) {
      if (o.orbit.n == 0 && o.orbit.l == 1 && o.orbit.j2.twice == 3)
        o.occupation = 0.5;
      if (o.orbit.n == 0 && o.orbit.l == 0 && o.orbit.tz2.twice == 1)
        o.occupation = 0.75;
    }
  }
  return ModelSpaceForDump(h);
}

/// Random tensor with the hermiticity rules of `OperatorSpec`.
Operator RandomTensor(std::shared_ptr<const basis::ModelSpace> ms,
                      const OperatorSpec& spec, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Operator op(ms, spec);
  const int h = HermiticitySign(spec.hermiticity);
  const auto& basis = ms->basis();
  for (int a = 0; a < basis.num_orbits(); ++a)
    for (int b = a; b < basis.num_orbits(); ++b) {
      const auto& oa = basis.orbit(OrbitIndex{a});
      const auto& ob = basis.orbit(OrbitIndex{b});
      if (!ConnectsOrbits(spec, oa, ob)) continue;
      const double v = (a == b && h < 0) ? 0.0 : u(gen);
      op.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      op.one_body().set_element(OrbitIndex{b}, OrbitIndex{a},
                                h * Phase((oa.j2.twice - ob.j2.twice) / 2) * v);
    }
  auto& m = op.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto& info = m.block_info(k);
    auto blk = m.host_block(k);
    for (std::int64_t j = 0; j < info.cols; ++j)
      for (std::int64_t i = 0; i < info.rows; ++i) {
        if (info.row_block == info.col_block && i > j) continue;
        const double v = (info.row_block == info.col_block && i == j && h < 0)
                             ? 0.0
                             : u(gen);
        blk(i, j) = v;
        if (info.row_block == info.col_block) blk(j, i) = h * v;
      }
  }
  op.SyncToDevice();
  return op;
}

double HermiticityDefect1B(const Operator& op) {
  const auto& basis = op.model_space().basis();
  const int h = HermiticitySign(op.hermiticity());
  double d = 0.0;
  for (int a = 0; a < basis.num_orbits(); ++a)
    for (int b = 0; b < basis.num_orbits(); ++b) {
      const int ja = basis.orbit(OrbitIndex{a}).j2.twice;
      const int jb = basis.orbit(OrbitIndex{b}).j2.twice;
      d = std::max(
          d, std::abs(op.one_body().get(OrbitIndex{b}, OrbitIndex{a}) -
                      h * Phase((ja - jb) / 2) *
                          op.one_body().get(OrbitIndex{a}, OrbitIndex{b})));
    }
  return d;
}

std::vector<OperatorSpec> Specs() {
  std::vector<OperatorSpec> v;
  for (const auto h : {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    for (const auto& [J, p, t] :
         std::vector<std::tuple<int, Parity, int>>{{0, Parity::kEven, 1},
                                                   {1, Parity::kOdd, 0},
                                                   {2, Parity::kEven, 0},
                                                   {3, Parity::kOdd, 0},
                                                   {5, Parity::kEven, 0},
                                                   {1, Parity::kEven, 1}}) {
      OperatorSpec s = OperatorSpec::Scalar(2, h);
      s.rank_J = J;
      s.parity = p;
      s.delta_Tz = t;
      v.push_back(s);
    }
  }
  return v;
}

TEST(TensorNormalOrder, RoundTripAndHermiticity) {
  for (const bool open : {false, true}) {
    const auto ms = Space(open);
    unsigned seed = 11;
    for (const auto& s : Specs()) {
      SCOPED_TRACE("rank " + std::to_string(s.rank_J) + " open " +
                   std::to_string(open));
      const Operator y = RandomTensor(ms, s, seed++);
      Operator no = NormalOrder(y);
      no.SyncToHost();
      const double scale = std::max(1.0, MaxAbs(no).max());
      EXPECT_LE(HermiticityDefect1B(no), 1.0e-13 * scale);
      // It acts, except for the anti-Hermitian rank 5 at emax 2, which only
      // connects d5/2 to itself: an anti-Hermitian tensor has no diagonal
      // reduced elements, so the contraction must vanish there.
      if (s.rank_J != 5 || s.hermiticity == Hermiticity::kHermitian) {
        EXPECT_GT(MaxAbsDifference(no, y).one_body, 1.0e-3);
      } else {
        EXPECT_LE(MaxAbsDifference(no, y).one_body, 1.0e-14);
      }
      EXPECT_EQ(MaxAbsDifference(no, y).two_body, 0.0);
      Operator back = UndoNormalOrder(no);
      back.SyncToHost();
      EXPECT_LE(MaxAbsDifference(back, y).max(), 1.0e-14 * scale);
    }
  }
}

TEST(TensorNormalOrder, ForkOptionChangesOnlyTheDiagonal) {
  const auto ms = Space(true);
  OperatorSpec s = OperatorSpec::Scalar(2, Hermiticity::kHermitian);
  s.rank_J = 2;
  const Operator y = RandomTensor(ms, s, 5);
  TensorNormalOrderOptions fork;
  fork.fork79e9330_diagonal_bug = true;
  Operator good = NormalOrderTensor(y);
  Operator bad = NormalOrderTensor(y, fork);
  good.SyncToHost();
  bad.SyncToHost();
  const int n = ms->num_orbits();
  double diag = 0.0;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      const double d = good.one_body().get(OrbitIndex{a}, OrbitIndex{b}) -
                       bad.one_body().get(OrbitIndex{a}, OrbitIndex{b});
      if (a == b) {
        diag = std::max(diag, std::abs(d));
      } else {
        EXPECT_EQ(d, 0.0);
      }
    }
  EXPECT_GT(diag, 1.0e-3);
  // With the option, Undo(Do(y)) still round-trips (the fork used the same
  // code in both directions).
  Operator back = UndoNormalOrderTensor(bad, fork);
  back.SyncToHost();
  EXPECT_LE(MaxAbsDifference(back, y).max(), 1.0e-14);
}

TEST(TensorNormalOrder, ReNormalOrderToAnotherReference) {
  const auto ensemble = Space(true);
  const auto core = Space(false);
  for (const auto& s : Specs()) {
    SCOPED_TRACE("rank " + std::to_string(s.rank_J));
    const Operator y = RandomTensor(ensemble, s, 77);
    Operator moved = ReNormalOrder(y, core);
    moved.SyncToHost();
    EXPECT_EQ(moved.model_space_ptr(), core);
    Operator expected =
        NormalOrderTensor(WithModelSpace(UndoNormalOrderTensor(y), core));
    expected.SyncToHost();
    EXPECT_LE(MaxAbsDifference(moved, expected).max(), 1.0e-15);
    Operator back = ReNormalOrder(moved, ensemble);
    back.SyncToHost();
    EXPECT_LE(MaxAbsDifference(back, y).max(),
              1.0e-14 * std::max(1.0, MaxAbs(y).max()));
  }
}

}  // namespace
}  // namespace imsrg::op
