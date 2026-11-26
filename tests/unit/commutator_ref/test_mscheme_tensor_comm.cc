// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The scalar-tensor commutators of `commutator::TensorCommutator` (all terms)
// against the brute-force m-scheme (Wigner-Eckart expansion of the random
// tensor, Wick engine, least-squares contraction) at emax 2: the cases the
// imsrg-devel goldens do not cover (tensor ranks 3 to 5, both parities,
// delta_Tz 0 to 2, FRACTIONAL occupations of an ensemble reference).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/commutator/tensor.h"
#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"
#include "tests/unit/commutator_ref/tensor_mscheme_support.h"

namespace {

using imsrg::Hermiticity;
using imsrg::Parity;
using imsrg::TwoM;
using imsrg::testing::TensorKind;
using MS = std::shared_ptr<const imsrg::basis::ModelSpace>;

/// Model spaces: closed-shell He4, He4 with an ensemble reference (occupations
/// 0.6/0.7 in 0s1/2 and 0.25/0.15 in 0p3/2), and He6 (0p3/2 neutrons half
/// filled, occupation 0.5).
MS Space(int index) {
  switch (index) {
    case 0:
      return imsrg::testing::MakeModelSpace(2, "He4");
    case 1:
      return imsrg::testing::FractionalModelSpace(2, "He4");
    default:
      return imsrg::testing::MakeModelSpace(2, "He6");
  }
}

const char* SpaceName(int index) {
  static const char* const kNames[] = {"He4", "He4_ensemble", "He6"};
  return kNames[index];
}

/// The tensor kinds: rank 0 with delta_Tz 1, then ranks 1 to 5 with both
/// parities and hermiticities and delta_Tz 0/1/2.
const std::vector<TensorKind>& Kinds() {
  static const auto* const kKinds = new std::vector<TensorKind>{
      {0, Parity::kEven, 1, Hermiticity::kHermitian},
      {1, Parity::kOdd, 0, Hermiticity::kAntiHermitian},
      {2, Parity::kEven, 1, Hermiticity::kHermitian},
      {3, Parity::kOdd, 0, Hermiticity::kHermitian},
      {4, Parity::kEven, 0, Hermiticity::kAntiHermitian},
      {5, Parity::kOdd, 1, Hermiticity::kHermitian}};
  return *kKinds;
}

class MSchemeTensorCommutatorTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(MSchemeTensorCommutatorTest, AllTermsAgainstMScheme) {
  const auto [space_index, kind_index] = GetParam();
  const MS ms = Space(space_index);
  const TensorKind kind = Kinds()[static_cast<std::size_t>(kind_index)];
  const imsrg::op::OperatorSpec yspec = imsrg::testing::SpecOf(kind);
  const imsrg::op::Operator y = imsrg::testing::RandomTensor(
      ms, yspec, 100 + static_cast<std::uint64_t>(kind_index));
  imsrg::commutator::TensorWorkspace ws(ms);
  // Component: mu = 0 for the scalar-like rank 0, mu = +1 otherwise, and
  // mu = -2 for the high ranks (two components for the ranks that matter
  // most).
  std::vector<int> two_mus = {kind.rank_J == 0 ? 0 : 2};
  if (kind.rank_J >= 3) two_mus.push_back(-4);
  std::uint64_t seed = 7;
  for (const Hermiticity hx :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian}) {
    const imsrg::op::Operator x =
        imsrg::testing::RandomOperator(ms, hx, seed++, 2);
    const auto t0 = std::chrono::steady_clock::now();
    const imsrg::op::Operator z_ref = imsrg::commutator::TensorCommutator(
        x, y, imsrg::commutator::TensorOptions{}, ws);
    z_ref.SyncToHost();
    const double scale = std::max(1.0, imsrg::op::MaxAbs(z_ref).max());
    for (const int two_mu : two_mus) {
      const auto t1 = std::chrono::steady_clock::now();
      const imsrg::op::Operator z = imsrg::testing::MSchemeTensorCommutator(
          x, y, z_ref.spec(), TwoM{two_mu});
      const auto d = imsrg::op::MaxAbsDifference(z, z_ref);
      std::printf(
          "[mscheme-tensor] %-12s rank %d parity %d dTz %d hY %d hX %d 2mu "
          "%+d: max|dZ| 1b %.2e 2b %.2e (max|Z| 1b %.3f 2b %.3f; J-scheme "
          "%.2f s, m-scheme %.2f s)\n",
          SpaceName(space_index), kind.rank_J, static_cast<int>(kind.parity),
          kind.delta_Tz, static_cast<int>(kind.hermiticity),
          static_cast<int>(hx), two_mu, d.one_body, d.two_body,
          imsrg::op::MaxAbs(z_ref).one_body, imsrg::op::MaxAbs(z_ref).two_body,
          std::chrono::duration<double>(t1 - t0).count(),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t1)
              .count());
      EXPECT_GT(imsrg::op::MaxAbs(z_ref).two_body, 1.0e-3)
          << "vacuous case: the commutator vanishes";
      EXPECT_LE(d.one_body, 1.0e-10 * scale);
      EXPECT_LE(d.two_body, 1.0e-10 * scale);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    EnsembleAndRanks, MSchemeTensorCommutatorTest,
    ::testing::Combine(::testing::Range(0, 3), ::testing::Range(0, 6)),
    [](const ::testing::TestParamInfo<std::tuple<int, int>>& p) {
      return std::string(SpaceName(std::get<0>(p.param))) + "_rank" +
             std::to_string(
                 Kinds()[static_cast<std::size_t>(std::get<1>(p.param))]
                     .rank_J);
    });

}  // namespace
