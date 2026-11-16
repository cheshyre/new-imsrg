// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// M12b review fix (D7): a commutator on a model space whose three-body rows
// are distributed is refused with NotImplementedError on EVERY rank before any
// collective (the phase-1 N7 path runs on rank 0 with whole operators and would
// hang the other ranks), until the distributed commutator of D10/D11. Run with
// 2 and 4 ranks; a hang is a failure (test timeout).

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_distribution.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "tests/support/mpi_test_main.h"

namespace imsrg::commutator {
namespace {

const comm::Communicator& World() { return test_support::TestWorld(); }

std::shared_ptr<const basis::ModelSpace> DistributedSpace() {
  const auto ms = basis::ModelSpace::FromNames(
                      basis::BasisTruncation{.emax = 2}, 16.0, "He4", "He4")
                      ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 4});
  const basis::ThreeBodySpace& s = *ms->three_body_space();
  basis::LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 1.0);
  w.memory = w.compute;
  auto part = std::make_shared<const basis::RowPartition>(
      basis::RowPartition::Balanced(s, w, World().size()));
  return ms->WithThreeBodyDistribution({part, World()});
}

// Number of ranks on which `f` threw NotImplementedError.
template <typename F>
std::int64_t CountNotImplemented(F&& f) {
  std::int64_t threw = 0;
  try {
    f();
  } catch (const NotImplementedError&) {
    threw = 1;
  }
  return World().AllreduceSum(threw);
}

TEST(DistributedThreeBodyCommutator, N7TermsThrowOnEveryRank) {
  const auto ms = DistributedSpace();
  Workspace ws(ms, WorkspaceOptions{World()});
  const op::OperatorSpec spec =
      op::OperatorSpec::Scalar(3, Hermiticity::kAntiHermitian);
  const op::Operator x(ms, spec);
  const op::Operator y(ms, spec);
  op::Operator z;
  CommutatorOptions opts;
  opts.terms = TermMask::Imsrg3N7();
  EXPECT_EQ(CountNotImplemented([&] { Commutator(x, y, z, opts, ws); }),
            World().size());
}

TEST(DistributedThreeBodyCommutator, Imsrg2TermsWithThreeBodyOperandThrow) {
  const auto ms = DistributedSpace();
  Workspace ws(ms, WorkspaceOptions{World()});
  const op::Operator x(
      ms, op::OperatorSpec::Scalar(3, Hermiticity::kAntiHermitian));
  const op::Operator y(ms,
                       op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  op::Operator z;
  CommutatorOptions opts;
  opts.terms = TermMask::Imsrg2();
  EXPECT_EQ(CountNotImplemented([&] { Commutator(x, y, z, opts, ws); }),
            World().size());
}

}  // namespace
}  // namespace imsrg::commutator
