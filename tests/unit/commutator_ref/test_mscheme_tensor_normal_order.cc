// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// `op::NormalOrderTensor` against the brute-force m-scheme normal ordering
// T1'_{ab} = T1_{ab} + sum_i n_i T2_{ai;bi} (one component mu, Wigner-Eckart
// expansion of the random tensor, least-squares contraction): ranks 0 to 5,
// both hermiticities, closed and fractional occupations, emax 2.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator_ref/tensor_mscheme_support.h"

namespace imsrg::commutator_ref::mscheme {
namespace {

TEST(MschemeTensorNormalOrder, MatchesNormalOrderTensor) {
  const std::vector<testing::TensorKind> kinds = {
      {0, Parity::kEven, 1, Hermiticity::kHermitian},
      {1, Parity::kOdd, 0, Hermiticity::kAntiHermitian},
      {2, Parity::kEven, 0, Hermiticity::kHermitian},
      {2, Parity::kEven, 0, Hermiticity::kAntiHermitian},
      {3, Parity::kOdd, 0, Hermiticity::kHermitian},
      {4, Parity::kEven, 0, Hermiticity::kHermitian},
      {5, Parity::kEven, 0, Hermiticity::kHermitian}};
  for (int space = 0; space < 2; ++space) {
    const auto ms = space == 0 ? testing::MakeModelSpace(2, "He4")
                               : testing::FractionalModelSpace(2, "He4");
    const MBasis mb(*ms);
    std::uint64_t seed = 101;
    for (const auto& k : kinds) {
      const op::OperatorSpec spec = testing::SpecOf(k);
      SCOPED_TRACE("space " + std::to_string(space) + " rank " +
                   std::to_string(k.rank_J));
      const op::Operator y = testing::RandomTensor(ms, spec, seed++);
      const TwoM mu{0};
      const MOperator m = ExpandTensor(y, mb, mu);
      MOperator no(mb.size(), 2);
      for (int a = 0; a < mb.size(); ++a)
        for (int b = 0; b < mb.size(); ++b) {
          const std::array<int, 1> oa{a};
          const std::array<int, 1> ob{b};
          double v = m.Get(1, oa, ob);
          for (int i = 0; i < mb.size(); ++i) {
            const double ni = mb.StateAt(i).n;
            if (ni == 0.0) continue;
            const std::array<int, 2> out{a, i};
            const std::array<int, 2> in{b, i};
            v += ni * m.Get(2, out, in);
          }
          no.Set(1, oa, ob, v);
        }
      op::OperatorSpec s1 = spec;
      s1.particle_rank = 1;
      const op::Operator ref = ContractTensor(no, ms, s1, mu);
      op::Operator got = op::NormalOrderTensor(y);
      got.SyncToHost();
      op::Operator r(ms, spec);
      r.one_body() = ref.one_body();
      r.one_body().set_hermiticity(spec.hermiticity);
      const double d = op::MaxAbsDifference(got, r).one_body;
      const double scale = std::max(1.0, op::MaxAbs(got).one_body);
      std::printf("tensor NO m-scheme space %d rank %d herm %c: max|d| %.3e\n",
                  space, k.rank_J, op::HermiticityCode(k.hermiticity), d);
      EXPECT_LE(d, 1.0e-12 * scale);
    }
  }
}

}  // namespace
}  // namespace imsrg::commutator_ref::mscheme
