// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The fast comm122st kernel (lifted one-body lists) against the element-wise
// reference implementation (`ReferenceComm122st`): random scalar X and tensor
// Y of several ranks and parities, both hermiticities of Y, integer and
// fractional occupations. max|dZ| <= 1e-12 max(1, max|Z|) (round-off only).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/two_body.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator/tensor_comm122_ref.h"

namespace imsrg::commutator {
namespace {

/// Uniform doubles in [-1, 1) from splitmix64.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  double Uniform() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) * 0x1.0p-53) - 1.0;
  }

 private:
  std::uint64_t state_;
};

/// A random tensor with the hermiticity of `spec` (one-body and two-body).
op::Operator RandomTensor(std::shared_ptr<const basis::ModelSpace> ms,
                          const op::OperatorSpec& spec, std::uint64_t seed) {
  op::Operator o(ms, spec);
  Rng rng(seed);
  const int h = op::HermiticitySign(spec.hermiticity);
  const auto& basis = ms->basis();
  const int n = ms->num_orbits();
  for (int a = 0; a < n; ++a) {
    for (int b = a; b < n; ++b) {
      const auto& oa = basis.orbit(OrbitIndex{a});
      const auto& ob = basis.orbit(OrbitIndex{b});
      if (!op::ConnectsOrbits(spec, oa, ob)) continue;
      if (a == b && h < 0) continue;
      const double v = rng.Uniform();
      o.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      if (a != b) {
        const int ph = ((oa.j2.twice - ob.j2.twice) / 2) % 2 == 0 ? 1 : -1;
        o.one_body().set_element(OrbitIndex{b}, OrbitIndex{a}, h * ph * v);
      }
    }
  }
  auto& m = o.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto info = m.block_info(k);
    auto blk = m.host_block(k);
    const bool diagonal = info.row_block == info.col_block;
    for (std::int64_t i = 0; i < info.rows; ++i) {
      for (std::int64_t j = 0; j < info.cols; ++j) {
        if (diagonal && j < i) {
          blk(i, j) = h * blk(j, i);
        } else if (diagonal && j == i && h < 0) {
          blk(i, j) = 0.0;
        } else {
          blk(i, j) = rng.Uniform();
        }
      }
    }
  }
  o.SyncToDevice();
  return o;
}

struct Kind {
  int rank_J;
  Parity parity;
  int delta_Tz;
};

void Check(const std::shared_ptr<const basis::ModelSpace>& ms, const Kind& kind,
           Hermiticity herm_y, const std::string& label) {
  op::OperatorSpec ys = op::OperatorSpec::Scalar(2);
  ys.rank_J = kind.rank_J;
  ys.parity = kind.parity;
  ys.delta_Tz = kind.delta_Tz;
  ys.hermiticity = herm_y;
  const op::Operator X =
      testing::RandomOperator(ms, Hermiticity::kAntiHermitian, 5, 2);
  const op::Operator Y = RandomTensor(ms, ys, 17);
  TensorWorkspace ws(ms);
  const op::OperatorSpec zs = TensorCommutatorSpec(X.spec(), Y.spec());
  op::Operator z_new(ms, zs);
  op::Operator z_ref(ms, zs);
  tensor_terms::Comm122st(X, Y, z_new, ws);
  test::ReferenceComm122st(X, Y, z_ref, ws);
  z_new.SyncToHost();
  z_ref.SyncToHost();
  const double scale = std::max(1.0, op::MaxAbs(z_ref).max());
  ASSERT_GT(op::MaxAbs(z_ref).max(), 0.0) << label << ": trivial reference";
  const auto d = op::MaxAbsDifference(z_new, z_ref);
  std::printf("comm122st fast vs element-wise %-28s max|dZ| 1b %.2e 2b %.2e\n",
              label.c_str(), d.one_body, d.two_body);
  EXPECT_LE(d.max(), 1.0e-12 * scale) << label;
}

TEST(TensorComm122, FastKernelMatchesElementWise) {
  const Kind kinds[] = {{0, Parity::kEven, 1},
                        {1, Parity::kOdd, 0},
                        {2, Parity::kEven, 0},
                        {3, Parity::kOdd, 0},
                        {5, Parity::kEven, 0}};
  for (const int emax : {2, 3}) {
    for (const bool fractional : {false, true}) {
      const auto ms = fractional ? testing::FractionalModelSpace(emax, "O16")
                                 : testing::MakeModelSpace(emax, "O16");
      for (const Kind& k : kinds) {
        for (const Hermiticity h :
             {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
          const std::string label =
              "emax " + std::to_string(emax) + (fractional ? " frac" : " int") +
              " L" + std::to_string(k.rank_J) +
              (k.parity == Parity::kOdd ? "-" : "+") + " dTz" +
              std::to_string(k.delta_Tz) +
              (h == Hermiticity::kHermitian ? " H" : " A");
          SCOPED_TRACE(label);
          Check(ms, k, h, label);
        }
      }
    }
  }
}

}  // namespace
}  // namespace imsrg::commutator
