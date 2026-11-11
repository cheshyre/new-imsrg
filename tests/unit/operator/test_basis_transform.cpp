// Tests of transform_basis and with_model_space: golden HF-basis data from imsrg-devel,
// identity, unitarity, round trip and error paths.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/linalg/matrix_view.hpp"
#include "imsrg/operator/basis_transform.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/operator/random_operator.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;
namespace fs = std::filesystem;

std::shared_ptr<const basis::ModelSpace> make_ms(const char* ref = "O16") {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 2}, 16.0, ref, ref);
}

linalg::Matrix<double> identity(std::int64_t n) {
  linalg::Matrix<double> C(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) C(i, j) = (i == j) ? 1.0 : 0.0;
  }
  return C;
}

/// Orthogonal matrix built from Givens rotations between orbits with equal (l, 2j, tz2), so that
/// the transformed operator keeps its channel structure.
linalg::Matrix<double> class_rotation(const basis::ModelSpace& ms, double angle_scale) {
  const std::int64_t n = ms.num_orbits();
  linalg::Matrix<double> C = identity(n);
  int k = 0;
  for (std::int64_t a = 0; a < n; ++a) {
    for (std::int64_t b = a + 1; b < n; ++b) {
      const auto& oa = ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(a)});
      const auto& ob = ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(b)});
      if (oa.l != ob.l || oa.j2.twice != ob.j2.twice || oa.tz2.twice != ob.tz2.twice) continue;
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
  const auto header = read_operator_dump_header(root / "hamiltonian/emax2_O16/H.dump");
  const auto ms = model_space_for_dump(header);
  const Operator H = read_operator_dump(root / "hamiltonian/emax2_O16/H.dump", ms);
  const Operator ref = read_operator_dump(root / "operator/emax2/H_HFbasis.dump", ms);

  const std::int64_t n = ms->num_orbits();
  std::ifstream in(root / "operator/emax2/C_HF_O16_2N.txt");
  ASSERT_TRUE(in.good());
  linalg::Matrix<double> C(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      ASSERT_TRUE(static_cast<bool>(in >> C(i, j)));
    }
  }

  const Operator out = transform_basis(H, C.const_view());
  const PartMaxima dev = max_abs_difference(out, ref);
  RecordProperty("max_dev_zero", std::to_string(dev.zero_body));
  RecordProperty("max_dev_one", std::to_string(dev.one_body));
  RecordProperty("max_dev_two", std::to_string(dev.two_body));
  std::cout << "golden max deviation: 0B " << dev.zero_body << "  1B " << dev.one_body << "  2B "
            << dev.two_body << "\n";
  EXPECT_LT(dev.zero_body, 1e-12);
  EXPECT_LT(dev.one_body, 1e-10);
  EXPECT_LT(dev.two_body, 1e-10);
  // The input is untouched and unitary C preserves the norms.
  EXPECT_NEAR(one_body_norm(out), one_body_norm(H), 1e-10 * one_body_norm(H));
  EXPECT_NEAR(two_body_norm(out), two_body_norm(H), 1e-10 * two_body_norm(H));
}

TEST(OperatorBasisTransform, IdentityIsNoOp) {
  const auto ms = make_ms();
  const Operator A =
      random_operator(ms, OperatorSpec::scalar(2), {.seed = 3, .zero_body_scale = 2.0});
  const Operator B = transform_basis(A, identity(ms->num_orbits()).const_view());
  const PartMaxima d = max_abs_difference(A, B);
  EXPECT_LT(d.max(), 1e-14);
}

TEST(OperatorBasisTransform, UnitaryPreservesNormsAndRoundTrips) {
  const auto ms = make_ms();
  for (const Hermiticity h :
       {Hermiticity::Hermitian, Hermiticity::AntiHermitian, Hermiticity::NonHermitian}) {
    const Operator A = random_operator(ms, OperatorSpec::scalar(2, h), {.seed = 11});
    const linalg::Matrix<double> C = class_rotation(*ms, 1.0);
    const Operator B = transform_basis(A, C.const_view());
    EXPECT_GT(max_abs_difference(A, B).two_body, 1e-3);  // the rotation does change things
    EXPECT_NEAR(one_body_norm(B), one_body_norm(A), 1e-12 * one_body_norm(A));
    EXPECT_NEAR(two_body_norm(B), two_body_norm(A), 1e-12 * two_body_norm(A));

    const std::int64_t n = ms->num_orbits();
    linalg::Matrix<double> Ct(n, n);
    for (std::int64_t i = 0; i < n; ++i) {
      for (std::int64_t j = 0; j < n; ++j) Ct(i, j) = C(j, i);
    }
    const Operator back = transform_basis(B, Ct.const_view());
    EXPECT_LT(max_abs_difference(A, back).max(), 1e-13);
  }
}

TEST(OperatorBasisTransform, ComposesAsMatrixProduct) {
  const auto ms = make_ms();
  const Operator A = random_operator(ms, OperatorSpec::scalar(2), {.seed = 5});
  const linalg::Matrix<double> C1 = class_rotation(*ms, 1.0);
  const linalg::Matrix<double> C2 = class_rotation(*ms, -0.7);
  const std::int64_t n = ms->num_orbits();
  linalg::Matrix<double> C12(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t k = 0; k < n; ++k) s += C1(i, k) * C2(k, j);
      C12(i, j) = s;
    }
  }
  const Operator two_step = transform_basis(transform_basis(A, C1.const_view()), C2.const_view());
  const Operator one_step = transform_basis(A, C12.const_view());
  EXPECT_LT(max_abs_difference(two_step, one_step).max(), 1e-13);
}

TEST(OperatorBasisTransform, RejectsWrongShape) {
  const auto ms = make_ms();
  const Operator A = random_operator(ms, OperatorSpec::scalar(2));
  linalg::Matrix<double> C(3, 3);
  EXPECT_THROW(transform_basis(A, C.const_view()), InvariantError);
}

TEST(OperatorWithModelSpace, CopiesDataOntoSameOrbitList) {
  const auto ms = make_ms("O16");
  const auto other = make_ms("He4");
  const Operator A =
      random_operator(ms, OperatorSpec::scalar(2), {.seed = 9, .zero_body_scale = 1.0});
  const Operator B = with_model_space(A, other);
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
        EXPECT_EQ(
            B.two_body().get(ChannelIndex{c}, ChannelIndex{c}, LocalKetIndex{i}, LocalKetIndex{j}),
            A.two_body().get(ChannelIndex{c}, ChannelIndex{c}, LocalKetIndex{i}, LocalKetIndex{j}));
      }
    }
  }
}

TEST(OperatorWithModelSpace, RejectsDifferentOrbitList) {
  const Operator A = random_operator(make_ms(), OperatorSpec::scalar(2));
  const auto bigger =
      basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 3}, 16.0, "O16", "O16");
  EXPECT_THROW(with_model_space(A, bigger), InvariantError);
  EXPECT_THROW(with_model_space(A, nullptr), InvariantError);
}
