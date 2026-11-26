// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// ExpandTensor / ContractTensor: the Wigner-Eckart expansion of random tensor
// operators (ranks 0 to 5, both parities, delta_Tz 0 to 2, Hermitian and
// anti-Hermitian) at emax 2 and back (the least-squares inverse), for several
// components mu.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator_ref/tensor_mscheme_support.h"

namespace {

namespace msc = imsrg::commutator_ref::mscheme;
using imsrg::Hermiticity;
using imsrg::Parity;
using imsrg::TwoM;
using imsrg::testing::TensorKind;

const std::vector<TensorKind>& Kinds() {
  static const auto* const kKinds = new std::vector<TensorKind>{
      {0, Parity::kEven, 1, Hermiticity::kHermitian},
      {0, Parity::kEven, 2, Hermiticity::kHermitian},
      {1, Parity::kOdd, 0, Hermiticity::kAntiHermitian},
      {1, Parity::kEven, 1, Hermiticity::kHermitian},
      {2, Parity::kEven, 0, Hermiticity::kHermitian},
      {2, Parity::kOdd, 1, Hermiticity::kAntiHermitian},
      {3, Parity::kOdd, 0, Hermiticity::kHermitian},
      {3, Parity::kEven, 1, Hermiticity::kAntiHermitian},
      {4, Parity::kEven, 0, Hermiticity::kAntiHermitian},
      {4, Parity::kOdd, 2, Hermiticity::kHermitian},
      {5, Parity::kOdd, 0, Hermiticity::kHermitian},
      {5, Parity::kEven, 1, Hermiticity::kAntiHermitian}};
  return *kKinds;
}

TEST(MSchemeTensorRoundTrip, RandomTensorsEmax2) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "He4");
  std::uint64_t seed = 11;
  for (const TensorKind& k : Kinds()) {
    const imsrg::op::OperatorSpec spec = imsrg::testing::SpecOf(k);
    const imsrg::op::Operator y =
        imsrg::testing::RandomTensor(ms, spec, seed++);
    const msc::MBasis mb(*ms);
    // mu = 0, the highest component and mu = -1 (those that exist).
    std::vector<int> mus = {0};
    if (k.rank_J > 0) {
      mus.push_back(2 * k.rank_J);
      if (k.rank_J > 1) mus.push_back(-2);
    }
    for (const int two_mu : mus) {
      const auto t0 = std::chrono::steady_clock::now();
      const msc::MOperator a = msc::ExpandTensor(y, mb, TwoM{two_mu});
      const imsrg::op::Operator back =
          msc::ContractTensor(a, ms, spec, TwoM{two_mu});
      const auto d = imsrg::op::MaxAbsDifference(back, y);
      const double scale = std::max(1.0, imsrg::op::MaxAbs(y).max());
      std::printf(
          "[mscheme-tensor] round trip rank %d parity %d dTz %d h %d 2mu %+d: "
          "max|d| 1b %.2e 2b %.2e (max|A| 1b %.2f 2b %.2f, %.2f s)\n",
          k.rank_J, static_cast<int>(k.parity), k.delta_Tz,
          static_cast<int>(k.hermiticity), two_mu, d.one_body, d.two_body,
          a.MaxAbs(1), a.MaxAbs(2),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count());
      EXPECT_LE(d.one_body, 1.0e-12 * scale);
      EXPECT_LE(d.two_body, 1.0e-12 * scale);
      // A one-body part exists only if the triangle and parity rules connect
      // some orbit pair (not for rank 5 of odd parity at emax 2).
      EXPECT_EQ(a.MaxAbs(1) > 0.0, imsrg::op::MaxAbs(y).one_body > 0.0);
      EXPECT_GT(a.MaxAbs(2), 0.0);
    }
  }
}

// Different components mu are different m-scheme operators, and reading a
// component as another one does not reproduce the reduced elements.
TEST(MSchemeTensorRoundTrip, ComponentsAreDistinctAndNotInterchangeable) {
  const auto ms = imsrg::testing::MakeModelSpace(2, "He4");
  const imsrg::op::OperatorSpec spec =
      imsrg::testing::SpecOf({2, Parity::kEven, 0, Hermiticity::kHermitian});
  const imsrg::op::Operator y = imsrg::testing::RandomTensor(ms, spec, 3);
  const msc::MBasis mb(*ms);
  const msc::MOperator a0 = msc::ExpandTensor(y, mb, TwoM{0});
  const msc::MOperator a1 = msc::ExpandTensor(y, mb, TwoM{2});
  EXPECT_GT(a0.MaxAbsDifference(a1, 1), 0.1);
  // Reading a mu = 1 operator as mu = 0 must not reproduce y.
  const imsrg::op::Operator wrong = msc::ContractTensor(a1, ms, spec, TwoM{0});
  EXPECT_GT(imsrg::op::MaxAbsDifference(wrong, y).one_body, 1.0e-3);
}

}  // namespace
