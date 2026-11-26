// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// DeviceGemm / DeviceGroupedGemm (design/12 section 3, M14) against the host
// Gemm on host copies of the same data, on every backend (host BLAS, rocBLAS,
// Kokkos Kernels).

#include "imsrg/linalg/device_blas.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::linalg {
namespace {

// One GEMM shape: op(A) m x k, op(B) k x n, padded leading dimensions.
struct Shape {
  std::int64_t m, n, k;
  Op opA, opB;
  double alpha, beta;
};

struct Placed {
  std::int64_t a_off, b_off, c_off;
  std::int64_t a_rows, a_cols, a_ld;
  std::int64_t b_rows, b_cols, b_ld;
  std::int64_t c_ld;
};

// Lays out every problem in one buffer (A, B, C with ld = rows + 3), fills it
// with random numbers (C with NaN where beta == 0: BLAS must not read it).
std::vector<Placed> Layout(const std::vector<Shape>& shapes,
                           std::vector<double>& data) {
  std::vector<Placed> out;
  std::int64_t off = 0;
  for (const Shape& s : shapes) {
    Placed p{};
    p.a_rows = s.opA == Op::kTrans ? s.k : s.m;
    p.a_cols = s.opA == Op::kTrans ? s.m : s.k;
    p.b_rows = s.opB == Op::kTrans ? s.n : s.k;
    p.b_cols = s.opB == Op::kTrans ? s.k : s.n;
    p.a_ld = p.a_rows + 3;
    p.b_ld = p.b_rows + 3;
    p.c_ld = s.m + 3;
    p.a_off = off;
    off += p.a_ld * std::max<std::int64_t>(p.a_cols, 1);
    p.b_off = off;
    off += p.b_ld * std::max<std::int64_t>(p.b_cols, 1);
    p.c_off = off;
    off += p.c_ld * std::max<std::int64_t>(s.n, 1);
    out.push_back(p);
  }
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  data.assign(static_cast<std::size_t>(off), 0.0);
  for (double& x : data) x = d(rng);
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i].beta != 0.0) continue;
    for (std::int64_t e = 0; e < out[i].c_ld * shapes[i].n; ++e)
      data[static_cast<std::size_t>(out[i].c_off + e)] =
          std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

std::vector<GemmProblem> Problems(const std::vector<Shape>& shapes,
                                  const std::vector<Placed>& placed,
                                  double* base) {
  std::vector<GemmProblem> g;
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    const Shape& s = shapes[i];
    const Placed& p = placed[i];
    g.push_back(GemmProblem{
        s.alpha,
        MatrixView<const double>{base + p.a_off, p.a_rows, p.a_cols, p.a_ld},
        s.opA,
        MatrixView<const double>{base + p.b_off, p.b_rows, p.b_cols, p.b_ld},
        s.opB, s.beta, MatrixView<double>{base + p.c_off, s.m, s.n, p.c_ld}});
  }
  return g;
}

// Runs the shapes on the device (grouped or one by one) and compares every C
// element with the host Gemm on a host copy.
void Compare(const std::vector<Shape>& shapes, bool grouped) {
  std::vector<double> host;
  const auto placed = Layout(shapes, host);
  exec::DeviceBuffer<double> dev(host.size(), "device_blas_test");
  dev.CopyFromHost(host);
  const auto dprob = Problems(shapes, placed, dev.device_data());
  if (grouped) {
    DeviceGroupedGemm(dprob);
  } else {
    for (const GemmProblem& p : dprob)
      DeviceGemm(p.alpha, p.A, p.opA, p.B, p.opB, p.beta, p.C);
  }
  std::vector<double> got(host.size());
  dev.CopyToHost(got);  // fences
  const auto hprob = Problems(shapes, placed, host.data());
  GroupedGemm(hprob);
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    const Shape& s = shapes[i];
    const double scale = static_cast<double>(std::max<std::int64_t>(s.k, 1));
    for (std::int64_t c = 0; c < s.n; ++c)
      for (std::int64_t r = 0; r < s.m; ++r) {
        const auto e =
            static_cast<std::size_t>(placed[i].c_off + r + c * placed[i].c_ld);
        ASSERT_FALSE(std::isnan(got[e])) << "problem " << i;
        // Different BLAS libraries sum in different orders: |error| <=
        // k eps max|a||b| with |a|, |b| <= 1, |alpha| <= 2.
        EXPECT_NEAR(got[e], host[e], 4.0e-15 * scale + 1.0e-15)
            << "problem " << i << " (" << r << ", " << c << ")";
      }
  }
}

TEST(DeviceBlas, BackendName) {
  const auto name = DeviceBlasBackend();
  EXPECT_TRUE(name == "host" || name == "rocblas" || name == "kokkos-kernels")
      << name;
}

TEST(DeviceBlas, SingleGemmAllTransposes) {
  for (const Op a : {Op::kNoTrans, Op::kTrans})
    for (const Op b : {Op::kNoTrans, Op::kTrans})
      Compare({Shape{37, 29, 41, a, b, 1.5, 0.0},
               Shape{16, 64, 8, a, b, -0.5, 1.0}},
              /*grouped=*/false);
}

TEST(DeviceBlas, GroupedManySizes) {
  std::vector<Shape> shapes;
  std::mt19937_64 rng(7);
  std::uniform_int_distribution<int> dim(1, 120);
  for (int i = 0; i < 40; ++i) {
    const Op a = (i % 3 == 0) ? Op::kTrans : Op::kNoTrans;
    const Op b = (i % 5 == 0) ? Op::kTrans : Op::kNoTrans;
    shapes.push_back(Shape{dim(rng), dim(rng), dim(rng), a, b,
                           (i % 2 == 0) ? 1.0 : -2.0,
                           (i % 4 == 0) ? 0.0 : 1.0});
  }
  shapes.push_back(Shape{300, 280, 310, Op::kNoTrans, Op::kTrans, 1.0, 0.0});
  Compare(shapes, /*grouped=*/true);
}

TEST(DeviceBlas, EmptyAndZeroInnerDimension) {
  // k == 0: C = beta C (beta == 0: zero, NaN not read); m or n == 0: no-op.
  Compare({Shape{5, 7, 0, Op::kNoTrans, Op::kNoTrans, 1.0, 0.0},
           Shape{5, 7, 0, Op::kNoTrans, Op::kNoTrans, 1.0, 2.0},
           Shape{0, 7, 3, Op::kNoTrans, Op::kNoTrans, 1.0, 1.0},
           Shape{4, 0, 3, Op::kTrans, Op::kNoTrans, 1.0, 1.0}},
          /*grouped=*/true);
  DeviceGroupedGemm({});
}

// Two dependent grouped calls: the second reads the C of the first, so the
// side streams of the first call must be joined before the second starts.
TEST(DeviceBlas, DependentCallsAreOrdered) {
  const std::int64_t n = 200;
  std::vector<double> h(static_cast<std::size_t>(3 * n * n), 0.0);
  for (std::int64_t i = 0; i < n; ++i) {
    h[static_cast<std::size_t>(i + i * n)] = 2.0;          // A = 2 I
    h[static_cast<std::size_t>(n * n + i + i * n)] = 1.0;  // B = I
  }
  exec::DeviceBuffer<double> d(h.size(), "device_blas_dep");
  d.CopyFromHost(h);
  double* a = d.device_data();
  double* b = a + n * n;
  double* c = b + n * n;
  for (int rep = 0; rep < 20; ++rep) {
    // B <- A B (two problems on halves of the columns, i.e. two streams).
    std::vector<GemmProblem> g;
    for (int half = 0; half < 2; ++half)
      g.push_back(GemmProblem{
          1.0, MatrixView<const double>{a, n, n, n}, Op::kNoTrans,
          MatrixView<const double>{b + half * (n / 2) * n, n, n / 2, n},
          Op::kNoTrans, 0.0,
          MatrixView<double>{c + half * (n / 2) * n, n, n / 2, n}});
    DeviceGroupedGemm(g);
    // B <- 0.5 A C (= 2 B_prev), reading the C of the call above.
    std::vector<GemmProblem> back;
    back.push_back(
        GemmProblem{0.5, MatrixView<const double>{a, n, n, n}, Op::kNoTrans,
                    MatrixView<const double>{c, n, n, n}, Op::kNoTrans, 0.0,
                    MatrixView<double>{b, n, n, n}});
    DeviceGroupedGemm(back);
  }
  std::vector<double> got(h.size());
  d.CopyToHost(got);
  // Each round doubles B: B = 2^20 I, and the last C = 2 B_19 = 2^20 I.
  for (std::int64_t i = 0; i < n; ++i) {
    EXPECT_EQ(got[static_cast<std::size_t>(n * n + i + i * n)],
              std::ldexp(1.0, 20));
    EXPECT_EQ(got[static_cast<std::size_t>(2 * n * n + i + i * n)],
              std::ldexp(1.0, 20));
  }
}

TEST(DeviceBlas, ShapeMismatchThrows) {
  exec::DeviceBuffer<double> d(64, "device_blas_bad");
  double* p = d.device_data();
  EXPECT_THROW(DeviceGemm(1.0, MatrixView<const double>{p, 2, 3, 2},
                          Op::kNoTrans, MatrixView<const double>{p, 4, 2, 4},
                          Op::kNoTrans, 0.0, MatrixView<double>{p, 2, 2, 2}),
               InvariantError);
}

}  // namespace
}  // namespace imsrg::linalg
