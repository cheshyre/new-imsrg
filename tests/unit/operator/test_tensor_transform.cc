// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Basis transformation of tensor operators (M15): brute-force check in the
// m-scheme (Wigner-Eckart expansion of the reduced elements, rotation of the
// single-particle states, comparison with the expansion of TransformBasis),
// hermiticity of the result and the identity transformation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::op {
namespace {

using ModelSpacePtr = std::shared_ptr<const basis::ModelSpace>;

ModelSpacePtr MakeSpace(int emax) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, "O16", "O16");
}

int Phase(int x) { return (x % 2 == 0) ? 1 : -1; }

/// Deterministic generator (splitmix64) with uniform doubles in [-1, 1).
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed) {}
  std::uint64_t Next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double Uniform() {
    return 2.0 * (static_cast<double>(Next() >> 11) * 0x1.0p-53) - 1.0;
  }
  /// Integer in [0, n).
  int Index(int n) {
    return static_cast<int>(Next() % static_cast<unsigned>(n));
  }

 private:
  std::uint64_t state_;
};

/// A random tensor operator of kind `spec`: every one-body element allowed by
/// `ConnectsOrbits` with its partner from the hermiticity rule O(b,a) = h
/// (-1)^((2j_a-2j_b)/2) O(a,b), every stored two-body block random, diagonal
/// blocks (anti)symmetric with h.
Operator RandomTensor(const ModelSpacePtr& ms, const OperatorSpec& spec,
                      std::uint64_t seed) {
  Operator o(ms, spec);
  Rng rng(seed);
  const int h = HermiticitySign(spec.hermiticity);
  const auto& basis = ms->basis();
  const int n = ms->num_orbits();
  for (int a = 0; a < n; ++a) {
    for (int b = a; b < n; ++b) {
      const auto& oa = basis.orbit(OrbitIndex{a});
      const auto& ob = basis.orbit(OrbitIndex{b});
      if (!ConnectsOrbits(spec, oa, ob)) continue;
      if (a == b && h < 0) continue;
      const double v = rng.Uniform();
      o.one_body().set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      if (a != b) {
        o.one_body().set_element(
            OrbitIndex{b}, OrbitIndex{a},
            h * Phase((oa.j2.twice - ob.j2.twice) / 2) * v);
      }
    }
  }
  auto& m = o.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto info = m.block_info(k);
    auto blk = m.host_block(k);
    const bool diagonal = info.row_block == info.col_block;
    for (std::int64_t i = 0; i < info.rows; ++i) {
      for (std::int64_t j = 0; j < info.cols; ++j) {
        if (diagonal && j < i) {
          blk(i, j) = h * blk(j, i);
        } else if (diagonal && j == i && h < 0) {
          blk(i, j) = 0.0;
        } else {
          blk(i, j) = rng.Uniform();
        }
      }
    }
  }
  o.SyncToDevice();
  return o;
}

/// Max deviation from the hermiticity relations of the stored reduced
/// elements (one-body rule and diagonal two-body blocks).
double HermiticityDefect(const Operator& op) {
  op.SyncToHost();
  const auto& basis = op.model_space().basis();
  const int h = HermiticitySign(op.hermiticity());
  double d = 0.0;
  for (int a = 0; a < basis.num_orbits(); ++a) {
    for (int b = 0; b < basis.num_orbits(); ++b) {
      const int ja = basis.orbit(OrbitIndex{a}).j2.twice;
      const int jb = basis.orbit(OrbitIndex{b}).j2.twice;
      const double ab = op.one_body().get(OrbitIndex{a}, OrbitIndex{b});
      const double ba = op.one_body().get(OrbitIndex{b}, OrbitIndex{a});
      d = std::max(d, std::abs(ba - h * Phase((ja - jb) / 2) * ab));
    }
  }
  const auto& m = op.two_body().matrix();
  for (std::size_t k = 0; k < m.num_blocks(); ++k) {
    const auto& info = m.block_info(k);
    if (info.row_block != info.col_block) continue;
    const auto blk = m.host_block(k);
    for (std::int64_t i = 0; i < info.rows; ++i) {
      for (std::int64_t j = 0; j < info.cols; ++j)
        d = std::max(d, std::abs(blk(j, i) - h * blk(i, j)));
    }
  }
  return d;
}

/// Orbits grouped by (l, 2j, tz2): the classes C may mix.
std::vector<std::vector<int>> OrbitClasses(const basis::ModelSpace& ms) {
  std::vector<std::vector<int>> classes;
  const int n = ms.num_orbits();
  for (int a = 0; a < n; ++a) {
    const auto& oa = ms.basis().orbit(OrbitIndex{a});
    bool placed = false;
    for (auto& c : classes) {
      const auto& oc = ms.basis().orbit(OrbitIndex{c.front()});
      if (oc.l == oa.l && oc.j2.twice == oa.j2.twice &&
          oc.tz2.twice == oa.tz2.twice) {
        c.push_back(a);
        placed = true;
        break;
      }
    }
    if (!placed) classes.push_back({a});
  }
  return classes;
}

/// Orthogonal C mixing only orbits of one class: Givens rotations of every
/// pair with random angles, then random column signs (`signed_permutation`:
/// a random signed permutation of every class instead).
linalg::Matrix<double> RandomClassRotation(const basis::ModelSpace& ms,
                                           std::uint64_t seed,
                                           bool signed_permutation) {
  const std::int64_t n = ms.num_orbits();
  linalg::Matrix<double> C(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) C(i, j) = (i == j) ? 1.0 : 0.0;
  }
  Rng rng(seed);
  for (const auto& cls : OrbitClasses(ms)) {
    if (signed_permutation) {
      std::vector<int> perm = cls;
      for (int i = static_cast<int>(perm.size()) - 1; i > 0; --i)
        std::swap(perm[static_cast<std::size_t>(i)],
                  perm[static_cast<std::size_t>(rng.Index(i + 1))]);
      for (std::size_t k = 0; k < cls.size(); ++k) {
        for (const int r : cls) C(r, cls[k]) = 0.0;
        C(perm[k], cls[k]) = (rng.Index(2) == 0) ? 1.0 : -1.0;
      }
      continue;
    }
    for (std::size_t i = 0; i < cls.size(); ++i) {
      for (std::size_t j = i + 1; j < cls.size(); ++j) {
        const double th = 3.0 * rng.Uniform();
        const double c = std::cos(th);
        const double s = std::sin(th);
        for (std::int64_t r = 0; r < n; ++r) {  // C <- C G(i, j)
          const double x = C(r, cls[i]);
          const double y = C(r, cls[j]);
          C(r, cls[i]) = c * x - s * y;
          C(r, cls[j]) = s * x + c * y;
        }
      }
    }
    for (const int a : cls) {
      if (rng.Index(2) == 0) {
        for (std::int64_t r = 0; r < n; ++r) C(r, a) = -C(r, a);
      }
    }
  }
  return C;
}

/// Wigner-Eckart factor <j_b m_b|T^L_mu|j_k m_k> / <j_b||T||j_k> =
/// (-1)^(j_b-m_b) (j_b L j_k; -m_b mu m_k) (all arguments doubled).
double WignerEckart(int jb, int mb, int L2, int mu, int jk, int mk) {
  if (mb != mk + mu) return 0.0;
  return Phase((jb - mb) / 2) * angmom::Wigner3j(TwoJ{jb}, TwoJ{L2}, TwoJ{jk},
                                                 TwoM{-mb}, TwoM{mu}, TwoM{mk});
}

/// m-scheme one-body element <a m_a|O_mu|b m_b> (doubled m, mu).
double OneBodyM(const Operator& op, int a, int ma, int b, int mb, int mu) {
  const auto& basis = op.model_space().basis();
  const int ja = basis.orbit(OrbitIndex{a}).j2.twice;
  const int jb = basis.orbit(OrbitIndex{b}).j2.twice;
  const double w = WignerEckart(ja, ma, 2 * op.spec().rank_J, mu, jb, mb);
  return (w == 0.0) ? 0.0 : w * op.one_body().get(OrbitIndex{a}, OrbitIndex{b});
}

/// m-scheme antisymmetrized element <p m_p q m_q|O_mu|r m_r s m_s> with the
/// kets (|pq> - |qp>)/sqrt(2) (the "unnormalized" convention of
/// `get_unnormalized`), by coupling both pairs and applying the Wigner-Eckart
/// theorem with the pair spins.
double TwoBodyM(const Operator& op, int p, int mp, int q, int mq, int r, int mr,
                int s, int ms, int mu) {
  const auto& basis = op.model_space().basis();
  const int L2 = 2 * op.spec().rank_J;
  const int jp = basis.orbit(OrbitIndex{p}).j2.twice;
  const int jq = basis.orbit(OrbitIndex{q}).j2.twice;
  const int jr = basis.orbit(OrbitIndex{r}).j2.twice;
  const int js = basis.orbit(OrbitIndex{s}).j2.twice;
  const int Mb = mp + mq;
  const int Mk = mr + ms;
  if (Mb != Mk + mu) return 0.0;
  double sum = 0.0;
  for (int Jb = std::abs(jp - jq); Jb <= jp + jq; Jb += 2) {
    if (std::abs(Mb) > Jb) continue;
    const double cb = angmom::ClebschGordan(TwoJ{jp}, TwoM{mp}, TwoJ{jq},
                                            TwoM{mq}, TwoJ{Jb}, TwoM{Mb});
    if (cb == 0.0) continue;
    for (int Jk = std::abs(jr - js); Jk <= jr + js; Jk += 2) {
      if (std::abs(Mk) > Jk) continue;
      const double ck = angmom::ClebschGordan(TwoJ{jr}, TwoM{mr}, TwoJ{js},
                                              TwoM{ms}, TwoJ{Jk}, TwoM{Mk});
      if (ck == 0.0) continue;
      const double w = WignerEckart(Jb, Mb, L2, mu, Jk, Mk);
      if (w == 0.0) continue;
      sum += cb * ck * w *
             op.two_body().get_unnormalized(OrbitIndex{p}, OrbitIndex{q},
                                            OrbitIndex{r}, OrbitIndex{s},
                                            TwoJ{Jb}, TwoJ{Jk});
    }
  }
  return sum;
}

struct Deviation {
  double max_diff_one = 0.0;
  double max_diff_two = 0.0;
  double max_ref = 0.0;
  int num_one = 0;
  int num_two = 0;
};

/// Compares the m-scheme elements of `out` with those of `in` rotated with C.
/// One-body: every (a, m_a, b, m_b) with the constraint; two-body: `samples`
/// random quadruples of m-states (or all of them if `exhaustive`).
Deviation CompareMscheme(const Operator& in, const Operator& out,
                         const linalg::Matrix<double>& C, int mu, int samples,
                         std::uint64_t seed) {
  const auto& ms = in.model_space();
  const auto& basis = ms.basis();
  const int n = ms.num_orbits();
  const auto classes = OrbitClasses(ms);
  std::vector<const std::vector<int>*> class_of(static_cast<std::size_t>(n));
  for (const auto& c : classes) {
    for (const int a : c) class_of[static_cast<std::size_t>(a)] = &c;
  }
  const OperatorSpec& spec = in.spec();
  Deviation dev;
  // One body.
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      const int ja = basis.orbit(OrbitIndex{a}).j2.twice;
      const int jb = basis.orbit(OrbitIndex{b}).j2.twice;
      for (int ma = -ja; ma <= ja; ma += 2) {
        const int mb = ma - mu;
        if (std::abs(mb) > jb) continue;
        double rot = 0.0;
        for (const int p : *class_of[static_cast<std::size_t>(a)]) {
          for (const int q : *class_of[static_cast<std::size_t>(b)])
            rot += C(p, a) * C(q, b) * OneBodyM(in, p, ma, q, mb, mu);
        }
        const double ours = OneBodyM(out, a, ma, b, mb, mu);
        dev.max_diff_one = std::max(dev.max_diff_one, std::abs(ours - rot));
        dev.max_ref = std::max(dev.max_ref, std::abs(rot));
        if (rot != 0.0) ++dev.num_one;
      }
    }
  }
  // Two body: random quadruples consistent with the selection rules.
  Rng rng(seed);
  for (int it = 0; it < samples; ++it) {
    const int P = rng.Index(n);
    const int Q = rng.Index(n);
    const int R = rng.Index(n);
    const int S = rng.Index(n);
    const auto& oP = basis.orbit(OrbitIndex{P});
    const auto& oQ = basis.orbit(OrbitIndex{Q});
    const auto& oR = basis.orbit(OrbitIndex{R});
    const auto& oS = basis.orbit(OrbitIndex{S});
    if ((oP.l + oQ.l + oR.l + oS.l + static_cast<int>(spec.parity)) % 2 != 0)
      continue;
    const int dtz =
        std::abs(oP.tz2.twice + oQ.tz2.twice - oR.tz2.twice - oS.tz2.twice) / 2;
    if (dtz != spec.delta_Tz) continue;
    const int mP = -oP.j2.twice + 2 * rng.Index(oP.j2.twice + 1);
    const int mQ = -oQ.j2.twice + 2 * rng.Index(oQ.j2.twice + 1);
    const int mR = -oR.j2.twice + 2 * rng.Index(oR.j2.twice + 1);
    const int mS = mP + mQ - mR - mu;
    if (std::abs(mS) > oS.j2.twice) continue;
    double rot = 0.0;
    for (const int p : *class_of[static_cast<std::size_t>(P)]) {
      for (const int q : *class_of[static_cast<std::size_t>(Q)]) {
        for (const int r : *class_of[static_cast<std::size_t>(R)]) {
          for (const int s : *class_of[static_cast<std::size_t>(S)]) {
            rot += C(p, P) * C(q, Q) * C(r, R) * C(s, S) *
                   TwoBodyM(in, p, mP, q, mQ, r, mR, s, mS, mu);
          }
        }
      }
    }
    const double ours = TwoBodyM(out, P, mP, Q, mQ, R, mR, S, mS, mu);
    dev.max_diff_two = std::max(dev.max_diff_two, std::abs(ours - rot));
    dev.max_ref = std::max(dev.max_ref, std::abs(rot));
    if (rot != 0.0) ++dev.num_two;
  }
  return dev;
}

struct Case {
  const char* name;
  int rank_J;
  int delta_Tz;
  Parity parity;
  Hermiticity hermiticity;
};

constexpr Case kCases[] = {
    {"J0_dTz1_H", 0, 1, Parity::kEven, Hermiticity::kHermitian},
    {"J0_dTz1_A", 0, 1, Parity::kEven, Hermiticity::kAntiHermitian},
    {"J1_odd_H", 1, 0, Parity::kOdd, Hermiticity::kHermitian},
    {"J1_odd_A", 1, 0, Parity::kOdd, Hermiticity::kAntiHermitian},
    {"J2_even_H", 2, 0, Parity::kEven, Hermiticity::kHermitian},
    {"J2_even_A", 2, 0, Parity::kEven, Hermiticity::kAntiHermitian},
    {"J3_odd_dTz1_H", 3, 1, Parity::kOdd, Hermiticity::kHermitian},
    {"J5_odd_H", 5, 0, Parity::kOdd, Hermiticity::kHermitian},
    {"J5_even_dTz2_A", 5, 2, Parity::kEven, Hermiticity::kAntiHermitian},
    {"J0_dTz2_H", 0, 2, Parity::kEven, Hermiticity::kHermitian},
};

OperatorSpec SpecOf(const Case& c) {
  OperatorSpec s;
  s.particle_rank = 2;
  s.rank_J = c.rank_J;
  s.parity = c.parity;
  s.delta_Tz = c.delta_Tz;
  s.hermiticity = c.hermiticity;
  return s;
}

/// Runs one case on model space `ms` with the rotation `C` for the
/// projections mu = 0 and mu = rank_J (doubled: 0 and 2 rank_J).
void CheckCase(const ModelSpacePtr& ms, const Case& c,
               const linalg::Matrix<double>& C, int samples,
               std::uint64_t seed) {
  SCOPED_TRACE(std::string(c.name) + " with " +
               std::to_string(ms->num_orbits()) + " orbits");
  const Operator in = RandomTensor(ms, SpecOf(c), seed);
  ASSERT_LE(HermiticityDefect(in), 1.0e-15);
  Operator out = TransformBasis(in, C.const_view());
  out.SyncToHost();
  EXPECT_EQ(out.spec(), in.spec());
  const double scale = std::max(1.0, MaxAbs(in).max());
  EXPECT_LE(HermiticityDefect(out), 1.0e-13 * scale);
  for (const int mu : {0, 2 * c.rank_J}) {
    const Deviation d = CompareMscheme(in, out, C, mu, samples, seed + 17);
    std::cout << "[tensor-transform] " << c.name << " mu2=" << mu
              << ": checked " << d.num_one << " 1B / " << d.num_two
              << " 2B nonzero elements, max|ref| " << d.max_ref << "  dev 1B "
              << d.max_diff_one << " 2B " << d.max_diff_two << "\n";
    EXPECT_LE(d.max_diff_one, 1.0e-12 * std::max(1.0, d.max_ref));
    EXPECT_LE(d.max_diff_two, 1.0e-12 * std::max(1.0, d.max_ref));
    if (MaxAbs(in).two_body > 0.0)
      EXPECT_GT(d.num_two, 0) << "no nonzero two-body sample";
  }
}

TEST(TensorTransform, MschemeRotationEmax1) {
  const auto ms = MakeSpace(1);
  std::uint64_t seed = 100;
  for (const Case& c : kCases) {
    // At emax 1 every class has one orbit: C is a diagonal sign matrix.
    CheckCase(ms, c, RandomClassRotation(*ms, seed, false), 100000, seed);
    CheckCase(ms, c, RandomClassRotation(*ms, seed + 1, true), 100000, seed);
    ++seed;
  }
}

TEST(TensorTransform, MschemeRotationEmax2Givens) {
  const auto ms = MakeSpace(2);
  std::uint64_t seed = 200;
  for (const Case& c : kCases) {
    CheckCase(ms, c, RandomClassRotation(*ms, seed, false), 6000, seed);
    ++seed;
  }
}

TEST(TensorTransform, MschemeRotationEmax2SignedPermutation) {
  const auto ms = MakeSpace(2);
  std::uint64_t seed = 300;
  for (const Case& c : kCases) {
    CheckCase(ms, c, RandomClassRotation(*ms, seed, true), 3000, seed);
    ++seed;
  }
}

TEST(TensorTransform, MschemeRotationEmax3) {
  const auto ms = MakeSpace(3);
  std::uint64_t seed = 400;
  for (const Case& c : {kCases[4], kCases[6], kCases[2]}) {
    CheckCase(ms, c, RandomClassRotation(*ms, seed, false), 3000, seed);
    ++seed;
  }
}

/// The mixing of C is real: the transformed operator differs from the input.
TEST(TensorTransform, RotationChangesTheOperator) {
  const auto ms = MakeSpace(2);
  const Operator in = RandomTensor(ms, SpecOf(kCases[4]), 7);
  const Operator out =
      TransformBasis(in, RandomClassRotation(*ms, 8, false).const_view());
  const PartMaxima d = MaxAbsDifference(in, out);
  EXPECT_GT(d.one_body, 1.0e-2);
  EXPECT_GT(d.two_body, 1.0e-2);
}

/// Exact equality of every stored value of two operators.
void ExpectBitwiseEqual(const Operator& a, const Operator& b) {
  a.SyncToHost();
  b.SyncToHost();
  EXPECT_EQ(a.zero_body(), b.zero_body());
  const int n = a.model_space().num_orbits();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      ASSERT_EQ(a.one_body().get(OrbitIndex{i}, OrbitIndex{j}),
                b.one_body().get(OrbitIndex{i}, OrbitIndex{j}));
  }
  const auto& ma = a.two_body().matrix();
  const auto& mb = b.two_body().matrix();
  ASSERT_EQ(ma.num_blocks(), mb.num_blocks());
  for (std::size_t k = 0; k < ma.num_blocks(); ++k) {
    const auto va = ma.host_block(k);
    const auto vb = mb.host_block(k);
    ASSERT_EQ(ma.block_info(k).rows, mb.block_info(k).rows);
    for (std::int64_t i = 0; i < ma.block_info(k).rows; ++i) {
      for (std::int64_t j = 0; j < ma.block_info(k).cols; ++j)
        ASSERT_EQ(va(i, j), vb(i, j));
    }
  }
}

TEST(TensorTransform, IdentityReturnsTheOperatorBitwise) {
  const auto ms = MakeSpace(2);
  const std::int64_t n = ms->num_orbits();
  linalg::Matrix<double> I(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) I(i, j) = (i == j) ? 1.0 : 0.0;
  }
  std::uint64_t seed = 500;
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    const Operator in = RandomTensor(ms, SpecOf(c), seed++);
    ExpectBitwiseEqual(in, TransformBasis(in, I.const_view()));
  }
}

}  // namespace
}  // namespace imsrg::op
