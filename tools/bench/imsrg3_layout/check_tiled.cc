// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark (tiled kernels K1 and the
// perturbative triples); see imsrg3_layout_check.cc. The layouts A, B and C
// must agree with each other (through ToCanonical) and with the reference
// kernels K0 once those are implemented. Suites `Tiled*`.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/kernels/tile_engine.h"
#include "tools/bench/imsrg3_layout/layouts.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {
namespace {

constexpr double kNoCut = 1.0e10;
constexpr double kTol = 1.0e-10;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

std::shared_ptr<const basis::ModelSpace> MakeMs(const char* ref, int emax,
                                                int e3max) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, ref,
                                      ref)
      ->WithThreeBody(basis::ThreeBodyCuts{e3max});
}

/// Everything the tiled checks need for one space.
struct Fixture {
  Fixture(const char* ref, int emax, int e3max, double de3max)
      : space(MakeMs(ref, emax, e3max), de3max, -1.0),
        sixj(emax),
        pt(space, sixj),
        a(space, pt),
        b(space, pt, sixj),
        c(space, pt),
        ctx(space, pt, sixj) {}
  CanonicalSpace space;
  SixJ3Table sixj;
  PairThirdSpace pt;
  LayoutA a;
  LayoutB b;
  LayoutC c;
  KernelContext ctx;

  op::Operator Random(Hermiticity h, std::uint64_t seed) const {
    return op::RandomOperator(space.model_space_ptr(),
                              op::OperatorSpec::Scalar(2, h), {seed});
  }
};

/// Layout blocks of K1 (full, sentinel-initialized so that unwritten entries
/// show up) in the layout `l`.
template <class L>
std::vector<double> RunK1(const Fixture& f, const L& l, const op::Operator& x,
                          const op::Operator& y) {
  std::vector<double> z(U(l.total_size()), 1.0e300);
  Comm223Tiled(f.ctx, l.device_view(), x, y, z);
  return z;
}

/// The canonical image of layout blocks `z` (`ToCanonical` per channel).
template <class L>
std::vector<double> Canonical(const Fixture& f, const L& l,
                              const std::vector<double>& z) {
  const CanonicalView cv = f.space.host_view();
  std::vector<double> out(U(f.space.total_size()));
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t lo = l.BlockOffsets()[U(ch)];
    const std::int64_t n = l.BlockOffsets()[U(ch) + 1] - lo;
    const std::int64_t co = cv.ch_block_offset[ch];
    const std::int64_t cn = cv.ch_block_offset[ch + 1] - co;
    l.ToCanonical(ch, std::span<const double>(z).subspan(U(lo), U(n)),
                  std::span<double>(out).subspan(U(co), U(cn)));
  }
  return out;
}

/// max |x - y| / max(1, max |y|).
double RelDiff(std::span<const double> x, std::span<const double> y) {
  EXPECT_EQ(x.size(), y.size());
  double diff = 0.0;
  double scale = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    diff = std::max(diff, std::abs(x[i] - y[i]));
    scale = std::max(scale, std::abs(y[i]));
  }
  return diff / scale;
}

double RelScalar(double x, double y) {
  return std::abs(x - y) / std::max(1.0e-300, std::abs(y));
}

/// K0 result, or empty if the reference kernel is not implemented yet.
std::vector<double> TryReference(const Fixture& f, const op::Operator& x,
                                 const op::Operator& y) {
  std::vector<double> z(U(f.space.total_size()), 0.0);
  try {
    Comm223Reference(f.ctx, x, y, z, true);
  } catch (const NotImplementedError&) {
    return {};
  }
  return z;
}

/// The test spaces: reference nucleus, emax, E3max, de3max.
struct Case {
  const char* ref;
  int emax;
  int e3max;
  double de3max;
};
constexpr Case kCases[] = {{"He4", 2, 6, kNoCut},
                           {"O16", 2, 6, kNoCut},
                           {"O16", 3, 7, kNoCut},
                           {"O16", 3, 9, 3.0}};

/// epsilon_i = (n_i > 0.5 ? -30 : 10) + 2 e_i + 0.1 (i % 7).
std::vector<double> Epsilon(const KernelContext& ctx) {
  const basis::OrbitTableView ob = ctx.Orbits();
  std::vector<double> eps(static_cast<std::size_t>(ob.num_orbits));
  for (std::int32_t i = 0; i < ob.num_orbits; ++i) {
    const double e = 2 * ob.n[i] + ob.l[i];
    eps[U(i)] =
        (ob.occupation[i] > 0.5 ? -30.0 : 10.0) + 2.0 * e + 0.1 * (i % 7);
  }
  return eps;
}

void CheckAgreement(const Case& cs, Hermiticity hx, Hermiticity hy) {
  const Fixture f(cs.ref, cs.emax, cs.e3max, cs.de3max);
  const op::Operator x = f.Random(hx, 101);
  const op::Operator y = f.Random(hy, 202);
  const std::vector<double> zc_raw = RunK1(f, f.c, x, y);
  const std::vector<double> za = RunK1(f, f.a, x, y);
  const std::vector<double> zb_raw = RunK1(f, f.b, x, y);
  const std::vector<double> zb = Canonical(f, f.b, zb_raw);
  const std::vector<double> zc = Canonical(f, f.c, zc_raw);
  double scale = 1.0;
  for (const double v : za) scale = std::max(scale, std::abs(v));
  EXPECT_LT(scale, 1.0e200) << "unwritten entries in layout A";
  const double dab = RelDiff(zb, za);
  const double dac = RelDiff(zc, za);
  std::cout << "  K1 " << cs.ref << " emax " << cs.emax << " e3max " << cs.e3max
            << " de3max " << cs.de3max << ": max|z_A| " << scale << ", |B-A| "
            << dab << ", |C-A| " << dac;
  EXPECT_LE(dab, kTol);
  EXPECT_LE(dac, kTol);
  // C equals the C-image of the A result on full blocks.
  const CanonicalView cv = f.space.host_view();
  double dfull = 0.0;
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const std::int64_t co = cv.ch_block_offset[ch];
    const std::int64_t cn = cv.ch_block_offset[ch + 1] - co;
    const std::int64_t lo = f.c.BlockOffsets()[U(ch)];
    const std::int64_t ln = f.c.BlockOffsets()[U(ch) + 1] - lo;
    std::vector<double> img(U(ln));
    f.c.FromCanonical(ch, std::span<const double>(za).subspan(U(co), U(cn)),
                      img);
    dfull = std::max(
        dfull,
        RelDiff(std::span<const double>(zc_raw).subspan(U(lo), U(ln)), img));
  }
  std::cout << ", |C-FromCanonical(A)| " << dfull;
  EXPECT_LE(dfull, kTol);
  const std::vector<double> z0 = TryReference(f, x, y);
  if (z0.empty()) {
    std::cout << " (K0 not available)\n";
  } else {
    const double d0 = RelDiff(za, z0);
    std::cout << ", |A-K0| " << d0 << "\n";
    EXPECT_LE(d0, kTol);
    EXPECT_LE(RelDiff(zb, z0), kTol);
    EXPECT_LE(RelDiff(zc, z0), kTol);
  }
  // imsrg++ norm agrees across layouts.
  const double na = LayoutImsrgppNorm(f.a, za);
  EXPECT_LE(RelScalar(LayoutImsrgppNorm(f.b, zb_raw), na), 1.0e-12);
  EXPECT_LE(RelScalar(LayoutImsrgppNorm(f.c, zc_raw), na), 1.0e-12);
}

TEST(TiledK1, LayoutsAgreeHermitianAntiHermitian) {
  for (const Case& cs : kCases) {
    CheckAgreement(cs, Hermiticity::kAntiHermitian, Hermiticity::kHermitian);
  }
}

TEST(TiledK1, LayoutsAgreeBothAntiHermitian) {
  for (const Case& cs : {kCases[0], kCases[1], kCases[3]}) {
    CheckAgreement(cs, Hermiticity::kAntiHermitian,
                   Hermiticity::kAntiHermitian);
  }
}

TEST(TiledK1, WOnlyLeavesZUntouched) {
  const Fixture f("O16", 2, 6, kNoCut);
  const op::Operator x = f.Random(Hermiticity::kAntiHermitian, 5);
  const op::Operator y = f.Random(Hermiticity::kHermitian, 6);
  std::vector<double> z(U(f.c.total_size()), 0.25);
  Comm223Tiled(f.ctx, f.c.device_view(), x, y, z, TiledMode::kWOnly);
  for (const double v : z) ASSERT_EQ(v, 0.25);
  // And the full mode does change it.
  Comm223Tiled(f.ctx, f.c.device_view(), x, y, z, TiledMode::kFull);
  EXPECT_NE(z[0], 0.25);
}

TEST(TiledK1, RejectsShortOutput) {
  const Fixture f("He4", 2, 6, kNoCut);
  const op::Operator x = f.Random(Hermiticity::kAntiHermitian, 5);
  const op::Operator y = f.Random(Hermiticity::kHermitian, 6);
  std::vector<double> z(U(f.a.total_size() - 1));
  EXPECT_THROW(Comm223Tiled(f.ctx, f.a.device_view(), x, y, z), Error);
}

// The 6j strides of the tile engine address SixJ3Table like its operator().
TEST(TiledK1, SixJStridesMatchTheTable) {
  const Fixture f("He4", 3, 7, kNoCut);
  const op::Operator x = f.Random(Hermiticity::kAntiHermitian, 5);
  const detail::TileSizes sz = detail::MakeTileSizes(f.ctx, f.a.device_view());
  const detail::TileViews tv =
      detail::MakeTileViews(f.ctx, f.a.device_view(), x,
                            f.Random(Hermiticity::kHermitian, 6), sz, nullptr);
  const SixJ3View v = f.sixj.host_view();
  const int tmax = 2 * tv.sj.emax + 1;
  int checked = 0;
  for (int j1 = 1; j1 <= tmax; j1 += 2) {
    for (int j2 = 1; j2 <= tmax; j2 += 2) {
      for (int j3 = 1; j3 <= tmax; j3 += 2) {
        for (int J12 = 0; J12 <= tmax; ++J12) {
          for (int J23 = 0; J23 <= tmax; J23 += 3) {
            for (int J = 1; J <= 6 * tv.sj.emax + 3; J += 4) {
              const std::int64_t idx = tv.c1 * ((j1 - 1) / 2) +
                                       tv.c2 * ((j2 - 1) / 2) + tv.c3 * J12 +
                                       tv.c4 * ((j3 - 1) / 2) +
                                       tv.c5 * ((J - 1) / 2) + J23;
              ASSERT_EQ(v.data[idx], v(TwoJ{j1}, TwoJ{j2}, TwoJ{2 * J12},
                                       TwoJ{j3}, TwoJ{J}, TwoJ{2 * J23}));
              ++checked;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 1000);
}

template <class L>
double Triples(const Fixture& f, const L& l, const op::Operator& x,
               const op::Operator& y, const std::vector<double>& eps) {
  return TriplesTiled(f.ctx, l.device_view(), x, y, eps);
}

void CheckTriples(const Case& cs, Hermiticity hx, Hermiticity hy) {
  const Fixture f(cs.ref, cs.emax, cs.e3max, cs.de3max);
  const op::Operator x = f.Random(hx, 303);
  const op::Operator y = f.Random(hy, 404);
  const std::vector<double> eps = Epsilon(f.ctx);
  const double ea = Triples(f, f.a, x, y, eps);
  const double eb = Triples(f, f.b, x, y, eps);
  const double ec = Triples(f, f.c, x, y, eps);
  std::cout << "  triples " << cs.ref << " emax " << cs.emax << " e3max "
            << cs.e3max << " de3max " << cs.de3max << ": E_A " << ea
            << ", rel |B-A| " << RelScalar(eb, ea) << ", |C-A| "
            << RelScalar(ec, ea);
  EXPECT_NE(ea, 0.0);
  EXPECT_LE(RelScalar(eb, ea), kTol);
  EXPECT_LE(RelScalar(ec, ea), kTol);
  try {
    const double e0 = TriplesReference(f.ctx, x, y, eps);
    std::cout << ", |A-K0| " << RelScalar(ea, e0) << "\n";
    EXPECT_LE(RelScalar(ea, e0), kTol);
    EXPECT_LE(RelScalar(eb, e0), kTol);
    EXPECT_LE(RelScalar(ec, e0), kTol);
  } catch (const NotImplementedError&) {
    std::cout << " (K0 not available)\n";
  }
}

TEST(TiledTriples, LayoutsAgree) {
  for (const Case& cs : kCases) {
    CheckTriples(cs, Hermiticity::kAntiHermitian, Hermiticity::kHermitian);
  }
}

TEST(TiledTriples, LayoutsAgreeBothAntiHermitian) {
  CheckTriples(kCases[1], Hermiticity::kAntiHermitian,
               Hermiticity::kAntiHermitian);
}

/// min over `reps` runs of `f` in milliseconds.
template <class F>
double MinMs(int reps, const F& f) {
  double best = std::numeric_limits<double>::max();
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

/// Timing of K1 and the triples (O16 emax 3 E3max 9), only when the
/// environment variable IMSRG_LAYOUT_TIMING is set (use the release preset).
TEST(TiledTiming, K1AndTriples) {
  if (std::getenv("IMSRG_LAYOUT_TIMING") == nullptr) {
    GTEST_SKIP() << "set IMSRG_LAYOUT_TIMING=1 (release build) to time";
  }
  const Fixture f("O16", 3, 9, kNoCut);
  const op::Operator x = f.Random(Hermiticity::kAntiHermitian, 1);
  const op::Operator y = f.Random(Hermiticity::kHermitian, 2);
  const std::vector<double> eps = Epsilon(f.ctx);
  const auto time_layout = [&](const char* name, const auto& l) {
    std::vector<double> z(U(l.total_size()));
    const LayoutView lv = l.device_view();
    const detail::TileSizes sz = detail::MakeTileSizes(f.ctx, lv);
    const double k1 =
        MinMs(5, [&] { Comm223Tiled(f.ctx, lv, x, y, z, TiledMode::kFull); });
    const double w =
        MinMs(5, [&] { Comm223Tiled(f.ctx, lv, x, y, z, TiledMode::kWOnly); });
    const double tr = MinMs(5, [&] { TriplesTiled(f.ctx, lv, x, y, eps); });
    std::cout << "  layout " << name << ": K1 " << k1 << " ms, W-only " << w
              << " ms, triples " << tr << " ms; scratch "
              << 8 * detail::ScratchDoubles(sz) << " B/team (m_max " << sz.m_max
              << ", k_max " << sz.k_max << ", S " << sz.s_max << ")\n";
  };
  {
    const detail::TileSizes sz =
        detail::MakeTileSizes(f.ctx, f.a.device_view());
    const double pre = MinMs(5, [&] {
      detail::GatherTables tables;
      (void)detail::MakeTileViews(f.ctx, f.a.device_view(), x, y, sz, &tables);
    });
    std::cout << "  gather pre-pass (tables and fill): " << pre
              << " ms; sum N_C^2 / 2 (W entries upper bound) "
              << f.c.Counts().sum_n2 / 2 << "\n";
  }
  time_layout("A", f.a);
  time_layout("B", f.b);
  time_layout("C", f.c);
}

}  // namespace
}  // namespace imsrg::layout3
