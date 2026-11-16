// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The [2,2]->3 term `commutator::Comm223` and the perturbative triples
// `commutator::PerturbativeTriplesEnergy` (three_body.h):
// * against imsrg-devel `comm223ss` (tests/golden/imsrg3): the three-body
//   part element by element (O16 emax 2, E3max 3), the imsrg++ three-body
//   norm and the triples energy (O16 and He4 at emax 2, E3max 6);
// * the non-Hermitian (dense, all tiles) path against the packed path
//   (linearity), [Y, X] = -[X, Y], the triples energy against a host
//   recomputation from the blocks, also with a fractional reference;
// * kept spaces with a dE3max cut or emax_3body < emax against the
//   corresponding rows of the uncut result (each element depends only on the
//   two-body operands);
// * the context tables (6j strides, sizes), the error paths, and the
//   NotImplementedError of the IMSRG(2) entry point for three-body parts.
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/operator_spec.h"
#include "imsrg/operator/random_operator.h"
#include "imsrg/operator/three_body.h"
#include "tests/support/commutator_test_support.h"

namespace fs = std::filesystem;

namespace imsrg::commutator {
namespace {

using basis::KeptRow3Index;
using basis::ModelSpace;
using basis::ThreeBodyCuts;
using op::Operator;
using op::OperatorSpec;
using op::ThreeBody;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

std::shared_ptr<const ModelSpace> PlainSpace(const char* ref, int emax) {
  return ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0, ref, ref);
}

/// The benchmark operators of the goldens: X anti-Hermitian (seed 1), Y
/// Hermitian (seed 2), one- and two-body scale 1.
Operator Random(std::shared_ptr<const ModelSpace> ms, Hermiticity h,
                std::uint64_t seed) {
  op::RandomOperatorOptions o;
  o.seed = seed;
  o.one_body_scale = 1.0;
  o.two_body_scale = 1.0;
  return op::RandomOperator(std::move(ms), OperatorSpec::Scalar(2, h), o);
}

std::string Slurp(const fs::path& p) {
  std::ifstream f(p);
  EXPECT_TRUE(f.good()) << p;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/// One entry of a golden values.json.
struct GoldenEntry {
  int e3max = -1;
  double norm = 0.0;
  double triples = 0.0;
};

/// The entry of `case_dir`/values.json with E3max `e3max` (fails the test if
/// absent).
GoldenEntry Golden(const std::string& case_dir, int e3max) {
  const std::string json =
      Slurp(imsrg::testing::GoldenPath("imsrg3/" + case_dir + "/values.json"));
  static const auto* const kRe = new std::regex(
      R"re("e3max":\s*(\d+),\s*"three_body_norm":\s*[^,\s]+,\s*)re"
      R"re("three_body_norm_str":\s*"([^"]+)",\s*"triples_energy":\s*)re"
      R"re([^,\s]+,\s*"triples_energy_str":\s*"([^"]+)")re");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), *kRe);
       it != std::sregex_iterator(); ++it) {
    if (std::stoi((*it)[1].str()) == e3max) {
      return {e3max, std::stod((*it)[2].str()), std::stod((*it)[3].str())};
    }
  }
  ADD_FAILURE() << "no E3max " << e3max << " entry in " << case_dir;
  return {};
}

std::vector<double> ReadEpsilon(const std::string& case_dir) {
  std::ifstream f(
      imsrg::testing::GoldenPath("imsrg3/" + case_dir + "/epsilon.txt"));
  EXPECT_TRUE(f.good());
  std::vector<double> eps;
  double e = 0.0;
  while (f >> e) eps.push_back(e);
  return eps;
}

/// The bench epsilon_i = (n_i > 0.5 ? -30 : 10) + 2 e_i + 0.1 (i % 7).
std::vector<double> BenchEpsilon(const ModelSpace& ms) {
  const auto orbits = ms.basis().orbits();
  const auto occ = ms.reference().occupations();
  std::vector<double> eps(orbits.size());
  for (std::size_t i = 0; i < orbits.size(); ++i) {
    eps[i] = (occ[i] > 0.5 ? -30.0 : 10.0) + 2.0 * orbits[i].e() +
             0.1 * static_cast<double>(i % 7);
  }
  return eps;
}

double RelScalar(double x, double y) {
  return std::abs(x - y) / std::max(1.0e-300, std::abs(y));
}

/// Full matrix element (i, j) of channel `ch` of `z` (host mirror).
double At(const ThreeBody& z, std::int32_t ch, std::int32_t i, std::int32_t j) {
  return z.get(Channel3Index{ch}, KeptRow3Index{i}, KeptRow3Index{j});
}

/// max over all full-matrix elements of |a - s b| and max |b|; a and b on
/// the same kept space.
struct Deviation {
  double diff = 0.0;
  double scale = 0.0;
};

Deviation Compare(const ThreeBody& a, const ThreeBody& b, double s) {
  a.SyncToHost();
  b.SyncToHost();
  const basis::ThreeBodySpace& sp = a.space();
  Deviation d;
  for (std::int32_t ch = 0; ch < sp.num_channels(); ++ch) {
    const std::int32_t n = sp.dim(Channel3Index{ch});
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        const double vb = At(b, ch, i, j);
        d.diff = std::max(d.diff, std::abs(At(a, ch, i, j) - s * vb));
        d.scale = std::max(d.scale, std::abs(vb));
      }
    }
  }
  return d;
}

/// Host recomputation of the triples energy from the blocks of `z` (any
/// hermiticity): sum_ch (2J+1) sum_{i > j} (sym_i/6)(sym_j/6) z_ij^2
/// (n_i nbar_j - n_j nbar_i) / (e_i - e_j) with imsrg++'s skips.
double HostTriples(const ThreeBody& z, std::span<const double> eps) {
  z.SyncToHost();
  const basis::ThreeBodySpace& sp = z.space();
  const auto occ = z.model_space_ptr()->reference().occupations();
  const auto props = [&](std::int64_t flat, double& o, double& u, double& e,
                         double& w) {
    const basis::OrbitTriple& t = sp.triple(sp.row_triple(flat));
    const double na = occ[U(t.a.value())];
    const double nb = occ[U(t.b.value())];
    const double nc = occ[U(t.c.value())];
    o = na * nb * nc;
    u = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
    e = eps[U(t.a.value())] + eps[U(t.b.value())] + eps[U(t.c.value())];
    w = t.sym / 6.0;
  };
  double total = 0.0;
  for (std::int32_t ch = 0; ch < sp.num_channels(); ++ch) {
    const basis::KeptChannel3& kc = sp.channel(Channel3Index{ch});
    for (std::int32_t i = 0; i < kc.dim; ++i) {
      double oi = 0.0;
      double ui = 0.0;
      double ei = 0.0;
      double wi = 0.0;
      props(kc.row_offset + i, oi, ui, ei, wi);
      if (oi < 1.0e-8 && ui < 1.0e-8) continue;
      for (std::int32_t j = 0; j < i; ++j) {
        double oj = 0.0;
        double uj = 0.0;
        double ej = 0.0;
        double wj = 0.0;
        props(kc.row_offset + j, oj, uj, ej, wj);
        const double num = oi * uj - oj * ui;
        if (std::abs(num) < 1.0e-8) continue;
        const double zij = At(z, ch, i, j);
        total += (kc.two_j.twice + 1) * wi * wj * zij * zij * num / (ei - ej);
      }
    }
  }
  return total;
}

// ----------------------------------------------------------------- goldens

/// O16 emax 2, E3max 3: our three-body part element by element against the
/// imsrg++ comm223ss dump.
TEST(Comm223Golden, O16Emax2ElementWise) {
  const auto ms2 = PlainSpace("O16", 2);
  const auto ms3 = ms2->WithThreeBody(ThreeBodyCuts{.e3max = 3});
  const ThreeBodyContext ctx(ms3);
  const Operator X = Random(ms2, Hermiticity::kAntiHermitian, 1);
  const Operator Y = Random(ms2, Hermiticity::kHermitian, 2);
  const ThreeBody z = Comm223(X, Y, ctx);
  EXPECT_EQ(z.hermiticity(), Hermiticity::kHermitian);
  const Operator ref = op::ReadOperatorDump(
      imsrg::testing::GoldenPath("imsrg3/O16_emax2/Z3.dump"), ms3);
  ASSERT_TRUE(ref.three_body().has_value());
  const ThreeBody& zr = *ref.three_body();
  ASSERT_EQ(zr.size(), z.size());
  ASSERT_TRUE(zr.packed());
  z.SyncToHost();
  zr.SyncToHost();
  const auto a = z.host_span();
  const auto b = zr.host_span();
  double diff = 0.0;
  double scale = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    diff = std::max(diff, std::abs(a[k] - b[k]));
    scale = std::max(scale, std::abs(b[k]));
  }
  std::printf(
      "[comm223] O16 emax 2 E3max 3: %zu elements, max|Z| %.6g, "
      "max|diff| %.3e (rel %.3e)\n",
      a.size(), scale, diff, diff / scale);
  EXPECT_GT(scale, 1.0);
  EXPECT_LE(diff, 1.0e-10 * scale);
  const double golden = Golden("O16_emax2", 3).norm;
  EXPECT_LE(RelScalar(op::ImsrgppNorm(z), golden), 1.0e-10);
}

/// The imsrg++ three-body norm and the perturbative-triples energy of
/// comm223ss for `case_dir` at (emax, E3max).
void CheckNormAndTriples(const char* case_dir, const char* ref, int emax,
                         int e3max) {
  SCOPED_TRACE(case_dir);
  const auto ms2 = PlainSpace(ref, emax);
  const auto ms3 = ms2->WithThreeBody(ThreeBodyCuts{.e3max = e3max});
  const ThreeBodyContext ctx(ms3);
  const Operator X = Random(ms2, Hermiticity::kAntiHermitian, 1);
  const Operator Y = Random(ms2, Hermiticity::kHermitian, 2);
  const std::vector<double> eps = ReadEpsilon(case_dir);
  ASSERT_EQ(eps.size(), U(ms3->num_orbits()));
  const GoldenEntry g = Golden(case_dir, e3max);
  const ThreeBody z = Comm223(X, Y, ctx);
  const double norm = op::ImsrgppNorm(z);
  const double et = PerturbativeTriplesEnergy(X, Y, eps, ctx);
  std::printf(
      "[comm223] %s E3max %d: norm %.17g (imsrg++ %.17g, rel %.2e), "
      "triples %.17g (imsrg++ %.17g, rel %.2e)\n",
      case_dir, e3max, norm, g.norm, RelScalar(norm, g.norm), et, g.triples,
      RelScalar(et, g.triples));
  EXPECT_LE(RelScalar(norm, g.norm), 1.0e-10);
  EXPECT_LE(RelScalar(et, g.triples), 1.0e-10);
  // The streamed energy equals the energy of the stored blocks.
  EXPECT_LE(RelScalar(et, HostTriples(z, eps)), 1.0e-12);
}

TEST(Comm223Golden, O16Emax2NormAndTriples) {
  CheckNormAndTriples("O16_emax2", "O16", 2, 6);
}

TEST(Comm223Golden, He4Emax2NormAndTriples) {
  CheckNormAndTriples("He4_emax2", "He4", 2, 6);
}

TEST(Comm223Golden, O16Emax3NormAndTriples) {
  CheckNormAndTriples("O16_emax3", "O16", 3, 9);
}

// --------------------------------------------------------- symmetry paths

/// Operands on a shared space for the property tests.
struct PropertyFixture {
  explicit PropertyFixture(std::shared_ptr<const ModelSpace> plain,
                           int e3max = 6)
      : ms2(std::move(plain)),
        ms3(ms2->WithThreeBody(ThreeBodyCuts{.e3max = e3max})),
        ctx(ms3) {}
  std::shared_ptr<const ModelSpace> ms2;
  std::shared_ptr<const ModelSpace> ms3;
  ThreeBodyContext ctx;
};

/// Comm223(X_A + X_H, Y) (non-Hermitian X: dense, all tiles) equals
/// Comm223(X_A, Y) + Comm223(X_H, Y) (two packed results of opposite
/// hermiticity) element by element, and [Y, X] = -[X, Y].
TEST(Comm223NonHermitian, LinearityAndAntisymmetry) {
  const PropertyFixture f(PlainSpace("O16", 2));
  const Operator xa = Random(f.ms2, Hermiticity::kAntiHermitian, 11);
  const Operator xh = Random(f.ms2, Hermiticity::kHermitian, 12);
  const Operator y = Random(f.ms2, Hermiticity::kHermitian, 13);
  const Operator xn = xa + xh;
  ASSERT_EQ(xn.hermiticity(), Hermiticity::kNonHermitian);
  const ThreeBody zn = Comm223(xn, y, f.ctx);
  const ThreeBody za = Comm223(xa, y, f.ctx);
  const ThreeBody zh = Comm223(xh, y, f.ctx);
  EXPECT_EQ(zn.hermiticity(), Hermiticity::kNonHermitian);
  EXPECT_FALSE(zn.packed());
  EXPECT_EQ(za.hermiticity(), Hermiticity::kHermitian);
  EXPECT_EQ(zh.hermiticity(), Hermiticity::kAntiHermitian);
  ThreeBody sum = za;
  sum.set_hermiticity(Hermiticity::kNonHermitian);
  {
    ThreeBody zh_dense = zh;
    zh_dense.set_hermiticity(Hermiticity::kNonHermitian);
    sum.SyncToHost();
    zh_dense.SyncToHost();
    auto s = sum.host_span();
    const auto h = std::as_const(zh_dense).host_span();
    for (std::size_t k = 0; k < s.size(); ++k) s[k] += h[k];
    sum.SyncToDevice();
  }
  const Deviation lin = Compare(zn, sum, 1.0);
  std::printf("[comm223] linearity: max|Z| %.6g, max|diff| %.3e\n", lin.scale,
              lin.diff);
  EXPECT_GT(lin.scale, 1.0);
  EXPECT_LE(lin.diff, 1.0e-12 * lin.scale);
  // Antisymmetry, non-Hermitian path.
  const ThreeBody zyx = Comm223(y, xn, f.ctx);
  const Deviation anti = Compare(zyx, zn, -1.0);
  EXPECT_LE(anti.diff, 1.0e-12 * anti.scale);
  // Antisymmetry, packed paths.
  const Deviation anti_h = Compare(Comm223(y, xa, f.ctx), za, -1.0);
  EXPECT_LE(anti_h.diff, 1.0e-12 * anti_h.scale);
  // The upper triangle of the dense result is the h_Z image of the lower one
  // for operands of definite hermiticity: Comm223(X_A as non-Hermitian, Y).
  Operator xa_n = xa;
  xa_n.set_hermiticity(Hermiticity::kNonHermitian);
  const Deviation dense = Compare(Comm223(xa_n, y, f.ctx), za, 1.0);
  EXPECT_LE(dense.diff, 1.0e-12 * dense.scale);
}

/// Two anti-Hermitian operands give an anti-Hermitian Z (packed, zero
/// diagonal), equal to the dense result of the same operands.
TEST(Comm223NonHermitian, BothAntiHermitian) {
  const PropertyFixture f(PlainSpace("He4", 2));
  const Operator x = Random(f.ms2, Hermiticity::kAntiHermitian, 21);
  const Operator y = Random(f.ms2, Hermiticity::kAntiHermitian, 22);
  const ThreeBody z = Comm223(x, y, f.ctx);
  EXPECT_EQ(z.hermiticity(), Hermiticity::kAntiHermitian);
  Operator yn = y;
  yn.set_hermiticity(Hermiticity::kNonHermitian);
  const ThreeBody zn = Comm223(x, yn, f.ctx);
  const Deviation d = Compare(zn, z, 1.0);
  EXPECT_GT(d.scale, 1.0);
  EXPECT_LE(d.diff, 1.0e-12 * d.scale);
}

/// The streamed triples energy of non-Hermitian operands against the host
/// recomputation from the dense blocks; and of definite-hermiticity
/// operands on a fractional (ensemble) reference.
TEST(Comm223NonHermitian, TriplesMatchHostRecomputation) {
  {
    const PropertyFixture f(PlainSpace("O16", 2));
    const Operator xn = Random(f.ms2, Hermiticity::kAntiHermitian, 31) +
                        Random(f.ms2, Hermiticity::kHermitian, 32);
    const Operator y = Random(f.ms2, Hermiticity::kHermitian, 33);
    const std::vector<double> eps = BenchEpsilon(*f.ms3);
    const double e = PerturbativeTriplesEnergy(xn, y, eps, f.ctx);
    const double host = HostTriples(Comm223(xn, y, f.ctx), eps);
    std::printf("[comm223] triples non-Hermitian: %.17g (host %.17g)\n", e,
                host);
    EXPECT_NE(e, 0.0);
    EXPECT_LE(RelScalar(e, host), 1.0e-12);
  }
  {
    const PropertyFixture f(imsrg::testing::FractionalModelSpace(2, "O16"));
    const Operator x = Random(f.ms2, Hermiticity::kAntiHermitian, 34);
    const Operator y = Random(f.ms2, Hermiticity::kHermitian, 35);
    const std::vector<double> eps = BenchEpsilon(*f.ms3);
    const double e = PerturbativeTriplesEnergy(x, y, eps, f.ctx);
    const double host = HostTriples(Comm223(x, y, f.ctx), eps);
    std::printf("[comm223] triples fractional reference: %.17g (host %.17g)\n",
                e, host);
    EXPECT_NE(e, 0.0);
    EXPECT_LE(RelScalar(e, host), 1.0e-12);
  }
}

// ---------------------------------------------------- kept-space variants

/// Every element of `cut` (a smaller kept space) equals the element of the
/// same kets in `full`.
void CheckRestriction(const ThreeBodyCuts& cut_cuts, const char* label) {
  SCOPED_TRACE(label);
  const auto ms2 = PlainSpace("O16", 2);
  const auto full = ms2->WithThreeBody(ThreeBodyCuts{.e3max = 6});
  const auto cut = ms2->WithThreeBody(cut_cuts);
  const Operator x = Random(ms2, Hermiticity::kAntiHermitian, 41);
  const Operator y = Random(ms2, Hermiticity::kHermitian, 42);
  const ThreeBody zf = Comm223(x, y, ThreeBodyContext(full));
  const ThreeBody zc = Comm223(x, y, ThreeBodyContext(cut));
  zf.SyncToHost();
  zc.SyncToHost();
  const basis::ThreeBodySpace& sc = *cut->three_body_space();
  const basis::ThreeBodySpace& sf = *full->three_body_space();
  const basis::ThreeBodyChannels& kets = *cut->three_body();
  std::int64_t compared = 0;
  double diff = 0.0;
  double scale = 0.0;
  for (std::int32_t ch = 0; ch < sc.num_channels(); ++ch) {
    const basis::KeptChannel3& kc = sc.channel(Channel3Index{ch});
    std::vector<std::int32_t> map(U(kc.dim));
    for (std::int32_t i = 0; i < kc.dim; ++i) {
      const basis::Ket3& k = kets.ket(sc.row_ket(kc.row_offset + i));
      const auto row = sf.find_row(Channel3Index{ch}, k.p, k.q, k.r, k.two_jpq);
      ASSERT_TRUE(row.has_value());
      map[U(i)] = row->value();
    }
    for (std::int32_t j = 0; j < kc.dim; ++j) {
      for (std::int32_t i = 0; i < kc.dim; ++i) {
        const double vf = At(zf, ch, map[U(i)], map[U(j)]);
        diff = std::max(diff, std::abs(At(zc, ch, i, j) - vf));
        scale = std::max(scale, std::abs(vf));
        ++compared;
      }
    }
  }
  std::printf("[comm223] %s: %" PRId64
              " elements, max|Z| %.6g, max|diff| "
              "%.3e\n",
              label, compared, scale, diff);
  EXPECT_GT(compared, 0);
  EXPECT_LT(sc.NumRows(), sf.NumRows());
  EXPECT_LE(diff, 1.0e-12 * scale);
}

TEST(Comm223KeptSpace, DE3maxCutMatchesUncutRows) {
  CheckRestriction(ThreeBodyCuts{.e3max = 6, .de3max = 2.0}, "dE3max 2");
}

TEST(Comm223KeptSpace, SmallerEmax3BodyMatchesUncutRows) {
  // The internal orbit still runs over the whole basis (e <= 2).
  CheckRestriction(ThreeBodyCuts{.e3max = 3, .emax_3body = 1}, "emax_3body 1");
}

/// Operands on the model space with the three-body space give the same
/// result as operands on the plain space.
TEST(Comm223KeptSpace, OperandsOnEitherModelSpace) {
  const PropertyFixture f(PlainSpace("He4", 2), 4);
  const Operator x2 = Random(f.ms2, Hermiticity::kAntiHermitian, 51);
  const Operator y2 = Random(f.ms2, Hermiticity::kHermitian, 52);
  const Operator x3 = Random(f.ms3, Hermiticity::kAntiHermitian, 51);
  const Operator y3 = Random(f.ms3, Hermiticity::kHermitian, 52);
  const Deviation d =
      Compare(Comm223(x3, y3, f.ctx), Comm223(x2, y2, f.ctx), 1.0);
  EXPECT_EQ(d.diff, 0.0);
}

// ------------------------------------------------- context and error paths

/// The 6j strides of the kernel views address SixJ3Table like its operator().
TEST(Comm223Context, SixJStridesMatchTheTable) {
  const ThreeBodyContext ctx(
      PlainSpace("He4", 3)->WithThreeBody(ThreeBodyCuts{.e3max = 3}));
  const detail::TileViews v = ctx.impl().Views();
  const angmom::SixJ3View t = ctx.impl().sixj.host_view();
  const int tmax = 2 * t.emax() + 1;
  int checked = 0;
  for (int j1 = 1; j1 <= tmax; j1 += 2) {
    for (int j2 = 1; j2 <= tmax; j2 += 2) {
      for (int j3 = 1; j3 <= tmax; j3 += 2) {
        for (int J12 = 0; J12 <= tmax; ++J12) {
          for (int J23 = 0; J23 <= tmax; J23 += 3) {
            for (int J = 1; J <= 6 * t.emax() + 3; J += 4) {
              const std::int64_t idx =
                  v.c1 * ((j1 - 1) / 2) + v.c2 * ((j2 - 1) / 2) + v.c3 * J12 +
                  v.c4 * ((j3 - 1) / 2) + v.c5 * ((J - 1) / 2) + J23;
              ASSERT_EQ(t.data()[idx], t(TwoJ{j1}, TwoJ{j2}, TwoJ{2 * J12},
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

/// Orbit groups: every orbit once, sorted by 2j inside a group, j blocks
/// consistent; extents and sizes positive; moves keep the tables. (E3max 6
/// is the smallest that couples hhh and ppp triples of O16.)
TEST(Comm223Context, TablesAndSizes) {
  const auto ms3 =
      PlainSpace("O16", 3)->WithThreeBody(ThreeBodyCuts{.e3max = 6});
  ThreeBodyContext ctx(ms3);
  const ThreeBodyContext::Impl& impl = ctx.impl();
  const std::vector<std::int32_t> off = impl.group_offset.ToVector();
  const std::vector<std::int32_t> orb = impl.group_orbits.ToVector();
  const std::vector<std::int32_t> jpos = impl.group_j_pos.ToVector();
  const auto orbits = ms3->basis().orbits();
  ASSERT_EQ(orb.size(), orbits.size());
  std::vector<int> seen(orbits.size(), 0);
  for (std::int32_t g = 0; g < 4; ++g) {
    const std::int32_t n = off[U(g) + 1] - off[U(g)];
    EXPECT_EQ(jpos[U(g * (impl.nj + 1) + impl.nj)], n);
    for (std::int32_t k = 0; k < n; ++k) {
      const std::int32_t a = orb[U(off[U(g)] + k)];
      ++seen[U(a)];
      const std::int32_t ij = (orbits[U(a)].j2.twice - 1) / 2;
      EXPECT_GE(k, jpos[U(g * (impl.nj + 1) + ij)]);
      EXPECT_LT(k, jpos[U(g * (impl.nj + 1) + ij + 1)]);
      EXPECT_EQ(detail::OrbitGroupsView::GroupOf(orbits[U(a)].l % 2,
                                                 orbits[U(a)].tz2.twice),
                g);
    }
  }
  for (const int s : seen) EXPECT_EQ(s, 1);
  EXPECT_GT(impl.ext.m_max, 0);
  EXPECT_GT(impl.ext.k_max, 0);
  EXPECT_EQ(impl.ext.s_max, 6);  // odd parity at emax 3: p and f shells
  EXPECT_GT(impl.triples_league, 0);
  EXPECT_LE(impl.triples_league, impl.num_triples);
  const std::int64_t bytes = ctx.TableBytes();
  const std::int64_t scratch = ctx.ScratchBytesPerTeam();
  EXPECT_GT(bytes, impl.sixj.size_bytes());
  EXPECT_GE(scratch, 8 * detail::ScratchCounts(impl.ext).doubles);
  std::printf("[comm223] O16 emax 3 E3max 6: tables %" PRId64
              " B, scratch %" PRId64 " B/team (m_max %d, k_max %d, S %d)\n",
              bytes, scratch, impl.ext.m_max, impl.ext.k_max, impl.ext.s_max);
  const ThreeBodyContext moved = std::move(ctx);
  EXPECT_EQ(moved.TableBytes(), bytes);
  EXPECT_EQ(ctx.TableBytes(), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_THROW((void)ctx.impl(), InvariantError);
}

TEST(Comm223Context, ErrorPaths) {
  EXPECT_THROW(ThreeBodyContext(nullptr), InputError);
  const auto ms2 = PlainSpace("He4", 2);
  EXPECT_THROW(ThreeBodyContext{ms2}, InputError);
  const auto ms3 = ms2->WithThreeBody(ThreeBodyCuts{.e3max = 3});
  const ThreeBodyContext ctx(ms3);
  const Operator x = Random(ms2, Hermiticity::kAntiHermitian, 61);
  const Operator y = Random(ms2, Hermiticity::kHermitian, 62);
  // Particle rank 1: no two-body part.
  const Operator x1 = op::RandomOperator(
      ms2, OperatorSpec::Scalar(1, Hermiticity::kAntiHermitian), {.seed = 3});
  EXPECT_THROW((void)Comm223(x1, y, ctx), InputError);
  EXPECT_THROW((void)Comm223(Operator{}, y, ctx), InputError);
  // Another reference.
  const Operator xo = Random(PlainSpace("O16", 2), Hermiticity::kHermitian, 63);
  EXPECT_THROW((void)Comm223(xo, y, ctx), InvariantError);
  // Epsilon size.
  const std::vector<double> eps(U(ms3->num_orbits()) - 1, 1.0);
  EXPECT_THROW((void)PerturbativeTriplesEnergy(x, y, eps, ctx), InputError);
  // E3max 3 couples no hhh-ppp pair of He4 kets: the energy is zero.
  const std::vector<double> eps_ok = BenchEpsilon(*ms3);
  EXPECT_EQ(PerturbativeTriplesEnergy(x, y, eps_ok, ctx), 0.0);
}

/// The IMSRG(2) entry point refuses three-body parts until the N7 terms
/// exist (operands of particle rank 3, or an output Z with a three-body
/// part).
TEST(Comm223Context, Imsrg2CommutatorRefusesThreeBodyParts) {
  const auto ms3 =
      PlainSpace("He4", 1)->WithThreeBody(ThreeBodyCuts{.e3max = 3});
  Workspace ws(ms3);
  const Operator x = Random(ms3, Hermiticity::kAntiHermitian, 71);
  const Operator y = Random(ms3, Hermiticity::kHermitian, 72);
  const Operator y3(ms3, OperatorSpec::Scalar(3, Hermiticity::kHermitian));
  ASSERT_TRUE(y3.three_body().has_value());
  EXPECT_THROW((void)Commutator(x, y3, CommutatorOptions{}, ws),
               NotImplementedError);
  EXPECT_THROW((void)Commutator(y3, x, CommutatorOptions{}, ws),
               NotImplementedError);
  // An IMSRG(2) term set replaces a rank-3 Z by a rank-2 result.
  Operator z(ms3, OperatorSpec::Scalar(3, Hermiticity::kHermitian));
  Commutator(x, y, z, CommutatorOptions{}, ws);
  EXPECT_FALSE(z.three_body().has_value());
  // Without three-body parts the entry point works on this model space.
  Operator z2;
  Commutator(x, y, z2, CommutatorOptions{}, ws);
  EXPECT_FALSE(z2.three_body().has_value());
  // The N8/N9 terms of the full IMSRG(3) are not implemented.
  CommutatorOptions full;
  full.terms = TermMask::Imsrg3Full();
  EXPECT_THROW((void)Commutator(x, y, full, ws), NotImplementedError);
  // IMSRG(3) terms need a three-body space.
  Workspace ws2(PlainSpace("He4", 1));
  CommutatorOptions n7;
  n7.terms = TermMask::Imsrg3N7();
  const Operator x2 = Random(PlainSpace("He4", 1), Hermiticity::kHermitian, 3);
  EXPECT_THROW((void)Commutator(x2, x2, n7, ws2), InputError);
}

}  // namespace
}  // namespace imsrg::commutator
