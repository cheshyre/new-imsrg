// Tests of the operator arithmetic and of the flat part kernels it is built on.
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/exec/device_buffer.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/detail/part_kernels.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::shared_ptr<const basis::ModelSpace> make_ms(const char* ref = "O16") {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = 2}, 16.0, ref, ref);
}

/// Small deterministic generator, values in (-1, 1).
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : s_(seed) {}
  double next() {
    s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(static_cast<std::int64_t>(s_ >> 11)) / 4503599627370496.0 - 1.0;
  }

 private:
  std::uint64_t s_;
};

/// Fills all parts with pseudo-random values obeying the hermiticity of the operator
/// (including entries between orbits of different (l, j, tz): they are legal storage).
void fill(Operator& o, std::uint64_t seed) {
  Lcg rng(seed);
  const int h = hermiticity_sign(o.hermiticity());
  o.zero_body() = rng.next();
  const int n = o.model_space().num_orbits();
  for (int p = 0; p < n; ++p) {
    for (int q = p; q < n; ++q) {
      const double v = rng.next();
      if (p == q && h == -1) continue;
      o.one_body().set(OrbitIndex{p}, OrbitIndex{q}, v);
    }
  }
  if (!o.two_body().allocated()) return;
  const auto& chans = o.two_body().channels();
  for (int c = 0; c < chans.num_channels(); ++c) {
    const int nk = chans.channels()[static_cast<std::size_t>(c)].num_kets;
    const ChannelIndex ch{c};
    for (int i = 0; i < nk; ++i) {
      for (int j = i; j < nk; ++j) {
        const double v = rng.next();
        if (i == j && h == -1) continue;
        o.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
        if (i != j) {
          o.two_body().set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i}, h * v);
        }
      }
    }
  }
}

Operator make_op(const std::shared_ptr<const basis::ModelSpace>& ms, int rank, std::uint64_t seed,
                 Hermiticity h = Hermiticity::Hermitian) {
  Operator o(ms, OperatorSpec::scalar(rank, h));
  fill(o, seed);
  return o;
}

/// Largest violation of the (anti)symmetry of the 1B and 2B parts.
double symmetry_violation(const Operator& o) {
  const int h = hermiticity_sign(o.hermiticity());
  double worst = 0.0;
  const int n = o.model_space().num_orbits();
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      worst = std::max(worst, std::abs(o.one_body().get(OrbitIndex{p}, OrbitIndex{q}) -
                                       h * o.one_body().get(OrbitIndex{q}, OrbitIndex{p})));
    }
  }
  if (o.two_body().allocated()) {
    const auto& chans = o.two_body().channels();
    for (int c = 0; c < chans.num_channels(); ++c) {
      const auto b = o.two_body().host_block(ChannelIndex{c});
      for (std::int64_t i = 0; i < b.rows; ++i) {
        for (std::int64_t j = 0; j < b.cols; ++j) {
          worst = std::max(worst, std::abs(b(i, j) - h * b(j, i)));
        }
      }
    }
  }
  return worst;
}

// ---- kernels -------------------------------------------------------------------------------

exec::DeviceBuffer<double> make_buffer(const std::vector<double>& v) {
  exec::DeviceBuffer<double> b(v.size());
  if (!v.empty()) b.copy_from_host(v);
  return b;
}

TEST(PartKernels, AxpyAddsScaledBuffer) {
  const auto x = make_buffer({1.0, 2.0, 3.0, 4.0});
  auto y = make_buffer({10.0, 20.0, 30.0, 40.0});
  op::detail::axpy(0.5, x, y);
  const auto r = y.to_vector();
  EXPECT_DOUBLE_EQ(r[0], 10.5);
  EXPECT_DOUBLE_EQ(r[1], 21.0);
  EXPECT_DOUBLE_EQ(r[2], 31.5);
  EXPECT_DOUBLE_EQ(r[3], 42.0);
}

TEST(PartKernels, AxpySizeMismatchThrows) {
  const auto x = make_buffer({1.0, 2.0});
  auto y = make_buffer({1.0, 2.0, 3.0});
  EXPECT_THROW(op::detail::axpy(1.0, x, y), InvariantError);
}

TEST(PartKernels, ScaleMultipliesEveryElement) {
  auto x = make_buffer({1.0, -2.0, 4.0});
  op::detail::scale(-0.5, x);
  const auto r = x.to_vector();
  EXPECT_DOUBLE_EQ(r[0], -0.5);
  EXPECT_DOUBLE_EQ(r[1], 1.0);
  EXPECT_DOUBLE_EQ(r[2], -2.0);
}

TEST(PartKernels, SumOfSquaresUsesOffsetAndCount) {
  const auto x = make_buffer({1.0, 2.0, 3.0, 4.0, 5.0});
  EXPECT_DOUBLE_EQ(op::detail::sum_of_squares(x, 0, 5), 55.0);
  EXPECT_DOUBLE_EQ(op::detail::sum_of_squares(x, 1, 3), 4.0 + 9.0 + 16.0);
  EXPECT_DOUBLE_EQ(op::detail::sum_of_squares(x, 4, 1), 25.0);
  EXPECT_DOUBLE_EQ(op::detail::sum_of_squares(x, 2, 0), 0.0);
  EXPECT_THROW(op::detail::sum_of_squares(x, 3, 3), InvariantError);
  EXPECT_THROW(op::detail::sum_of_squares(x, -1, 2), InvariantError);
}

TEST(PartKernels, EmptyBuffersWork) {
  exec::DeviceBuffer<double> x;
  exec::DeviceBuffer<double> y;
  EXPECT_NO_THROW(op::detail::axpy(2.0, x, y));
  EXPECT_NO_THROW(op::detail::scale(2.0, x));
  EXPECT_DOUBLE_EQ(op::detail::sum_of_squares(x, 0, 0), 0.0);
}

// ---- arithmetic ----------------------------------------------------------------------------

TEST(OperatorArithmetic, AxpyEqualsElementwiseSum) {
  const auto ms = make_ms();
  const Operator x = make_op(ms, 2, 1);
  const Operator y0 = make_op(ms, 2, 2);
  Operator y = y0;
  axpy(-0.75, x, y);
  EXPECT_DOUBLE_EQ(y.zero_body(), y0.zero_body() - 0.75 * x.zero_body());
  const auto d = max_abs_difference(y, y0);
  EXPECT_GT(d.one_body, 0.1);
  Operator back = y;
  axpy(0.75, x, back);
  EXPECT_LE(max_abs_difference(back, y0).max(), 1e-14 * 2);
  // exact element check on a few entries
  EXPECT_DOUBLE_EQ(y.one_body().get(OrbitIndex{3}, OrbitIndex{5}),
                   y0.one_body().get(OrbitIndex{3}, OrbitIndex{5}) -
                       0.75 * x.one_body().get(OrbitIndex{3}, OrbitIndex{5}));
  EXPECT_DOUBLE_EQ(
      y.two_body().get(ChannelIndex{6}, ChannelIndex{6}, LocalKetIndex{0}, LocalKetIndex{1}),
      y0.two_body().get(ChannelIndex{6}, ChannelIndex{6}, LocalKetIndex{0}, LocalKetIndex{1}) -
          0.75 * x.two_body().get(ChannelIndex{6}, ChannelIndex{6}, LocalKetIndex{0},
                                  LocalKetIndex{1}));
}

TEST(OperatorArithmetic, ZeroCoefficientIsNoOp) {
  const auto ms = make_ms();
  const Operator x = make_op(ms, 2, 1, Hermiticity::AntiHermitian);
  Operator y = make_op(ms, 2, 2);
  const Operator y0 = y;
  axpy(0.0, x, y);
  EXPECT_EQ(max_abs_difference(y, y0).max(), 0.0);
  EXPECT_EQ(y.hermiticity(), Hermiticity::Hermitian);
}

TEST(OperatorArithmetic, SumThenDifferenceRecoversOperand) {
  const auto ms = make_ms();
  const Operator x = make_op(ms, 2, 3);
  const Operator y = make_op(ms, 2, 4);
  const Operator r = (x + y) - y;
  EXPECT_LE(max_abs_difference(r, x).max(), 1e-14 * 2);
}

TEST(OperatorArithmetic, SelfSumIsDouble) {
  const auto ms = make_ms();
  const Operator x = make_op(ms, 2, 5);
  EXPECT_EQ(max_abs_difference(x + x, 2.0 * x).max(), 0.0);
  EXPECT_LE(max_abs_difference(x + x, x * 2.0).max(), 0.0);
}

TEST(OperatorArithmetic, NegationAndDivision) {
  const auto ms = make_ms();
  const Operator x = make_op(ms, 2, 6);
  const Operator z = x + (-x);
  EXPECT_EQ(max_abs(z).max(), 0.0);
  const Operator q = x / 4.0;
  const Operator e = 0.25 * x;
  EXPECT_EQ(max_abs_difference(q, e).max(), 0.0);
  Operator w = x;
  w /= 2.0;
  w *= 2.0;
  EXPECT_LE(max_abs_difference(w, x).max(), 1e-15);
  EXPECT_THROW(x / 0.0, InvariantError);
  Operator v = x;
  EXPECT_THROW(v /= 0.0, InvariantError);
}

TEST(OperatorArithmetic, ScalarShiftsOnlyTouchZeroBody) {
  const auto ms = make_ms();
  Operator x = make_op(ms, 2, 7);
  const Operator x0 = x;
  x += 2.5;
  x -= 0.5;
  const auto d = max_abs_difference(x, x0);
  EXPECT_DOUBLE_EQ(d.zero_body, 2.0);
  EXPECT_EQ(d.one_body, 0.0);
  EXPECT_EQ(d.two_body, 0.0);
}

TEST(OperatorArithmetic, AliasedAxpyDoublesAndSubtractionZeroes) {
  const auto ms = make_ms();
  Operator x = make_op(ms, 2, 8);
  const Operator x0 = x;
  axpy(1.0, x, x);
  EXPECT_EQ(max_abs_difference(x, 2.0 * x0).max(), 0.0);
  Operator y = x0;
  y -= y;
  EXPECT_EQ(max_abs(y).max(), 0.0);
}

TEST(OperatorArithmetic, RankPromotion) {
  const auto ms = make_ms();
  const Operator t = make_op(ms, 1, 9);
  const Operator v = make_op(ms, 2, 10);
  const Operator s = t + v;
  EXPECT_EQ(s.spec().particle_rank, 2);
  const Operator s2 = v + t;
  EXPECT_EQ(s2.spec().particle_rank, 2);
  EXPECT_EQ(max_abs_difference(s, s2).max(), 0.0);
  // values: one-body parts add, two-body part is v's
  EXPECT_DOUBLE_EQ(s.zero_body(), t.zero_body() + v.zero_body());
  EXPECT_DOUBLE_EQ(s.one_body().get(OrbitIndex{2}, OrbitIndex{4}),
                   t.one_body().get(OrbitIndex{2}, OrbitIndex{4}) +
                       v.one_body().get(OrbitIndex{2}, OrbitIndex{4}));
  const auto dv = max_abs_difference(s, v);
  EXPECT_EQ(dv.two_body, 0.0);
  // t - v: copy v, negate, add t
  const Operator d = t - v;
  EXPECT_EQ(d.spec().particle_rank, 2);
  EXPECT_LE(max_abs_difference(d, -1.0 * (v - t)).max(), 1e-15);
  EXPECT_DOUBLE_EQ(d.one_body().get(OrbitIndex{2}, OrbitIndex{4}),
                   t.one_body().get(OrbitIndex{2}, OrbitIndex{4}) -
                       v.one_body().get(OrbitIndex{2}, OrbitIndex{4}));
  EXPECT_DOUBLE_EQ(
      d.two_body().get(ChannelIndex{3}, ChannelIndex{3}, LocalKetIndex{0}, LocalKetIndex{0}),
      -v.two_body().get(ChannelIndex{3}, ChannelIndex{3}, LocalKetIndex{0}, LocalKetIndex{0}));
  // in-place promotion into rank 2
  Operator y = v;
  y += t;
  EXPECT_LE(max_abs_difference(y, s).max(), 0.0);
}

TEST(OperatorArithmetic, RankTwoIntoRankOneThrows) {
  const auto ms = make_ms();
  const Operator v = make_op(ms, 2, 11);
  Operator t = make_op(ms, 1, 12);
  EXPECT_THROW(axpy(1.0, v, t), InvariantError);
  EXPECT_THROW(t += v, InvariantError);
}

TEST(OperatorArithmetic, DifferentModelSpacesThrow) {
  const Operator a = make_op(make_ms("O16"), 2, 13);
  Operator b = make_op(make_ms("He4"), 2, 14);
  EXPECT_THROW(b += a, InvariantError);
  EXPECT_THROW(a + b, InvariantError);
}

TEST(OperatorArithmetic, EqualButSeparatelyBuiltModelSpacesAreAccepted) {
  const auto ms1 = make_ms();
  const auto ms2 = make_ms();
  ASSERT_NE(ms1.get(), ms2.get());
  const Operator a = make_op(ms1, 2, 15);
  Operator b = make_op(ms2, 2, 16);
  const Operator b0 = b;
  EXPECT_NO_THROW(b += a);
  EXPECT_NEAR(b.zero_body(), a.zero_body() + b0.zero_body(), 1e-15);
}

TEST(OperatorArithmetic, EmptyOperatorsThrow) {
  const Operator e;
  Operator y = make_op(make_ms(), 2, 17);
  EXPECT_THROW(y += e, InvariantError);
}

TEST(OperatorArithmetic, HermiticityCombination) {
  const auto ms = make_ms();
  const Operator h = make_op(ms, 2, 18, Hermiticity::Hermitian);
  const Operator a = make_op(ms, 2, 19, Hermiticity::AntiHermitian);
  EXPECT_EQ((h + h).hermiticity(), Hermiticity::Hermitian);
  EXPECT_EQ((a + a).hermiticity(), Hermiticity::AntiHermitian);
  EXPECT_EQ((h + a).hermiticity(), Hermiticity::NonHermitian);
  EXPECT_EQ((a - h).hermiticity(), Hermiticity::NonHermitian);
  EXPECT_EQ((2.0 * a).hermiticity(), Hermiticity::AntiHermitian);
  EXPECT_EQ((-h).hermiticity(), Hermiticity::Hermitian);
  EXPECT_EQ((h / 3.0).hermiticity(), Hermiticity::Hermitian);
  // the flag reaches the parts
  const Operator n = h + a;
  EXPECT_EQ(n.one_body().hermiticity(), Hermiticity::NonHermitian);
  EXPECT_EQ(n.two_body().hermiticity(), Hermiticity::NonHermitian);
}

TEST(OperatorArithmetic, BlockSymmetryIsPreserved) {
  const auto ms = make_ms();
  const Operator h1 = make_op(ms, 2, 20, Hermiticity::Hermitian);
  const Operator h2 = make_op(ms, 2, 21, Hermiticity::Hermitian);
  const Operator a1 = make_op(ms, 2, 22, Hermiticity::AntiHermitian);
  const Operator a2 = make_op(ms, 2, 23, Hermiticity::AntiHermitian);
  ASSERT_EQ(symmetry_violation(h1), 0.0);
  ASSERT_EQ(symmetry_violation(a1), 0.0);
  EXPECT_EQ(symmetry_violation(h1 + 0.3 * h2), 0.0);
  EXPECT_EQ(symmetry_violation(h1 - h2 / 7.0), 0.0);
  EXPECT_EQ(symmetry_violation(a1 + 0.3 * a2), 0.0);
  EXPECT_EQ(symmetry_violation(-a1), 0.0);
}

}  // namespace
