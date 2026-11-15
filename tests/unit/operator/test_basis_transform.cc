// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of transform_basis and with_model_space: golden HF-basis data from
// imsrg-devel, identity, unitarity, round trip and error paths.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/random_operator.h"

namespace {

using namespace imsrg;
using namespace imsrg::op;
namespace fs = std::filesystem;

std::shared_ptr<const basis::ModelSpace> MakeMs(const char* ref = "O16") {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      ref, ref);
}

linalg::Matrix<double> Identity(std::int64_t n) {
  linalg::Matrix<double> C(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) C(i, j) = (i == j) ? 1.0 : 0.0;
  }
  return C;
}

/// Orthogonal matrix built from Givens rotations between orbits with equal (l,
/// 2j, tz2), so that the transformed operator keeps its channel structure.
linalg::Matrix<double> ClassRotation(const basis::ModelSpace& ms,
                                     double angle_scale) {
  const std::int64_t n = ms.num_orbits();
  linalg::Matrix<double> C = Identity(n);
  int k = 0;
  for (std::int64_t a = 0; a < n; ++a) {
    for (std::int64_t b = a + 1; b < n; ++b) {
      const auto& oa =
          ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(a)});
      const auto& ob =
          ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(b)});
      if (oa.l != ob.l || oa.j2.twice != ob.j2.twice ||
          oa.tz2.twice != ob.tz2.twice)
        continue;
      const double th = angle_scale * (0.3 + 0.1 * (k++));
      const double c = std::cos(th);
      const double s = std::sin(th);
      for (std::int64_t r = 0; r < n; ++r) {  // C <- C * G(a, b)
        const double x = C(r, a);
        const double y = C(r, b);
        C(r, a) = c * x - s * y;
        C(r, b) = s * x + c * y;
      }
    }
  }
  return C;
}

}  // namespace

TEST(OperatorBasisTransform, GoldenHartreeFockBasis) {
  const fs::path root = IMSRG_GOLDEN_DIR;
  const auto header =
      ReadOperatorDumpHeader(root / "hamiltonian/emax2_O16/H.dump");
  const auto ms = ModelSpaceForDump(header);
  const Operator H =
      ReadOperatorDump(root / "hamiltonian/emax2_O16/H.dump", ms);
  const Operator ref =
      ReadOperatorDump(root / "operator/emax2/H_HFbasis.dump", ms);

  const std::int64_t n = ms->num_orbits();
  std::ifstream in(root / "operator/emax2/C_HF_O16_2N.txt");
  ASSERT_TRUE(in.good());
  linalg::Matrix<double> C(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      ASSERT_TRUE(static_cast<bool>(in >> C(i, j)));
    }
  }

  const Operator out = TransformBasis(H, C.const_view());
  const PartMaxima dev = MaxAbsDifference(out, ref);
  RecordProperty("max_dev_zero", std::to_string(dev.zero_body));
  RecordProperty("max_dev_one", std::to_string(dev.one_body));
  RecordProperty("max_dev_two", std::to_string(dev.two_body));
  std::cout << "golden max deviation: 0B " << dev.zero_body << "  1B "
            << dev.one_body << "  2B " << dev.two_body << "\n";
  EXPECT_LT(dev.zero_body, 1e-12);
  EXPECT_LT(dev.one_body, 1e-10);
  EXPECT_LT(dev.two_body, 1e-10);
  // The input is untouched and unitary C preserves the norms.
  EXPECT_NEAR(OneBodyNorm(out), OneBodyNorm(H), 1e-10 * OneBodyNorm(H));
  EXPECT_NEAR(TwoBodyNorm(out), TwoBodyNorm(H), 1e-10 * TwoBodyNorm(H));
}

TEST(OperatorBasisTransform, IdentityIsNoOp) {
  const auto ms = MakeMs();
  const Operator A = RandomOperator(ms, OperatorSpec::Scalar(2),
                                    {.seed = 3, .zero_body_scale = 2.0});
  const Operator B = TransformBasis(A, Identity(ms->num_orbits()).const_view());
  const PartMaxima d = MaxAbsDifference(A, B);
  EXPECT_LT(d.max(), 1e-14);
}

TEST(OperatorBasisTransform, UnitaryPreservesNormsAndRoundTrips) {
  const auto ms = MakeMs();
  for (const Hermiticity h :
       {Hermiticity::kHermitian, Hermiticity::kAntiHermitian,
        Hermiticity::kNonHermitian}) {
    const Operator A =
        RandomOperator(ms, OperatorSpec::Scalar(2, h), {.seed = 11});
    const linalg::Matrix<double> C = ClassRotation(*ms, 1.0);
    const Operator B = TransformBasis(A, C.const_view());
    EXPECT_GT(MaxAbsDifference(A, B).two_body,
              1e-3);  // the rotation does change things
    EXPECT_NEAR(OneBodyNorm(B), OneBodyNorm(A), 1e-12 * OneBodyNorm(A));
    EXPECT_NEAR(TwoBodyNorm(B), TwoBodyNorm(A), 1e-12 * TwoBodyNorm(A));

    const std::int64_t n = ms->num_orbits();
    linalg::Matrix<double> Ct(n, n);
    for (std::int64_t i = 0; i < n; ++i) {
      for (std::int64_t j = 0; j < n; ++j) Ct(i, j) = C(j, i);
    }
    const Operator back = TransformBasis(B, Ct.const_view());
    EXPECT_LT(MaxAbsDifference(A, back).max(), 1e-13);
  }
}

TEST(OperatorBasisTransform, ComposesAsMatrixProduct) {
  const auto ms = MakeMs();
  const Operator A = RandomOperator(ms, OperatorSpec::Scalar(2), {.seed = 5});
  const linalg::Matrix<double> C1 = ClassRotation(*ms, 1.0);
  const linalg::Matrix<double> C2 = ClassRotation(*ms, -0.7);
  const std::int64_t n = ms->num_orbits();
  linalg::Matrix<double> C12(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t k = 0; k < n; ++k) s += C1(i, k) * C2(k, j);
      C12(i, j) = s;
    }
  }
  const Operator two_step =
      TransformBasis(TransformBasis(A, C1.const_view()), C2.const_view());
  const Operator one_step = TransformBasis(A, C12.const_view());
  EXPECT_LT(MaxAbsDifference(two_step, one_step).max(), 1e-13);
}

TEST(OperatorBasisTransform, RejectsWrongShape) {
  const auto ms = MakeMs();
  const Operator A = RandomOperator(ms, OperatorSpec::Scalar(2));
  linalg::Matrix<double> C(3, 3);
  EXPECT_THROW(TransformBasis(A, C.const_view()), InvariantError);
}

TEST(OperatorWithModelSpace, CopiesDataOntoSameOrbitList) {
  const auto ms = MakeMs("O16");
  const auto other = MakeMs("He4");
  const Operator A = RandomOperator(ms, OperatorSpec::Scalar(2),
                                    {.seed = 9, .zero_body_scale = 1.0});
  const Operator B = WithModelSpace(A, other);
  EXPECT_EQ(&B.model_space(), other.get());
  EXPECT_EQ(B.zero_body(), A.zero_body());
  const std::int32_t n = ms->num_orbits();
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < n; ++b) {
      EXPECT_EQ(B.one_body().get(OrbitIndex{a}, OrbitIndex{b}),
                A.one_body().get(OrbitIndex{a}, OrbitIndex{b}));
    }
  }
  const auto& chans = A.two_body().channels();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const std::int32_t nk = chans.channel(ChannelIndex{c}).num_kets;
    for (std::int32_t i = 0; i < nk; ++i) {
      for (std::int32_t j = 0; j < nk; ++j) {
        EXPECT_EQ(B.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                   LocalKetIndex{i}, LocalKetIndex{j}),
                  A.two_body().get(ChannelIndex{c}, ChannelIndex{c},
                                   LocalKetIndex{i}, LocalKetIndex{j}));
      }
    }
  }
}

TEST(OperatorWithModelSpace, RejectsDifferentOrbitList) {
  const Operator A = RandomOperator(MakeMs(), OperatorSpec::Scalar(2));
  const auto bigger = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 3}, 16.0, "O16", "O16");
  EXPECT_THROW(WithModelSpace(A, bigger), InvariantError);
  EXPECT_THROW(WithModelSpace(A, nullptr), InvariantError);
}
