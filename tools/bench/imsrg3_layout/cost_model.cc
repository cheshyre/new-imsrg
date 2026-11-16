// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/cost_model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "imsrg/basis/orbit.h"
#include "imsrg/basis/single_particle_basis.h"
#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {

namespace {

constexpr int kPairClasses = 6;   // parity (2) x tz2 sum in {-2, 0, 2} (3)
constexpr int kThirdClasses = 4;  // parity (2) x tz2 in {-1, 1} (2)

using PairVec = std::array<double, kPairClasses>;
using ThirdVec = std::array<double, kThirdClasses>;
using GroupTable = std::array<PairVec, kThirdClasses>;

constexpr std::size_t U(int i) { return static_cast<std::size_t>(i); }

int PairClass(const basis::Orbit& p, const basis::Orbit& q) {
  return 3 * ((p.l + q.l) % 2) + (p.tz2.twice + q.tz2.twice + 2) / 2;
}

int ThirdClass(const basis::Orbit& w) {
  return 2 * (w.l % 2) + (w.tz2.twice + 1) / 2;
}

/// g[j][i] = n_g(pair class i, third class j): the size of the orbit group
/// with parity (pair parity + third parity) % 2 and 2tz = pair tz2 sum - third
/// tz2 (0 if that is not +-1).
GroupTable GroupSizes(std::span<const basis::Orbit> orbits) {
  std::array<double, 4> count{};
  for (const basis::Orbit& o : orbits) {
    count[U(2 * (o.l % 2) + (o.tz2.twice + 1) / 2)] += 1.0;
  }
  GroupTable g{};
  for (int i = 0; i < kPairClasses; ++i) {
    const int ppar = i / 3;
    const int tzsum = 2 * (i % 3) - 2;
    for (int j = 0; j < kThirdClasses; ++j) {
      const int wpar = j / 2;
      const int wtz = 2 * (j % 2) - 1;
      const int tz = tzsum - wtz;
      if (tz != 1 && tz != -1) continue;
      g[U(j)][U(i)] = count[U(2 * ((ppar + wpar) % 2) + (tz + 1) / 2)];
    }
  }
  return g;
}

/// One splitting key (pair p <= q, third r) of a triple.
struct Key {
  int p;
  int q;
  int r;
};

}  // namespace

CostModels ComputeCostModels(const CanonicalSpace& space) {
  const CanonicalView cv = space.host_view();
  const std::span<const basis::Orbit> orbits =
      space.model_space().basis().orbits();
  const GroupTable groups = GroupSizes(orbits);
  const auto orb = [&](int i) -> const basis::Orbit& { return orbits[U(i)]; };
  constexpr Perm kPerms[3] = {Perm::kABC, Perm::kCBA, Perm::kACB};

  CostModels cost;
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const int two_j = cv.ch_two_j[ch];
    ThirdVec w_third_sum{};   // running sums over t' <= t
    ThirdVec k0_third_sum{};  //
    double a_m_nl = 0.0;      // sum of n_t' m_t'  (layouts A and B)
    double a_nl = 0.0;        // sum of n_t'
    double c_m_nl = 0.0;      // sum of m_t'^2     (layout C)
    double c_nl = 0.0;        // sum of m_t'
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      const int a = cv.tr_a[t];
      const int b = cv.tr_b[t];
      const int c = cv.tr_c[t];
      const double n_t = cv.tr_row_end[t] - cv.tr_row_begin[t];

      PairVec w_pair_t{};
      ThirdVec w_third_t{};
      PairVec k0_pair_t{};
      ThirdVec k0_third_t{};
      double m_t = 0.0;
      Key keys[3];
      int num_keys = 0;
      for (const Perm perm : kPerms) {
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        Permute(perm, a, b, c, i1, i2, i3);
        const basis::Orbit& o1 = orb(i1);
        const basis::Orbit& o2 = orb(i2);
        const basis::Orbit& o3 = orb(i3);
        const int j1 = o1.j2.twice;
        const int j2 = o2.j2.twice;
        const int j3 = o3.j2.twice;

        // K0: the J' range of the permuted pair, as in the K0 loop.
        double nj = 1.0;
        if (perm != Perm::kABC) {
          const int lo = std::max(std::abs(j1 - j2), std::abs(two_j - j3)) / 2;
          const int hi = std::min(j1 + j2, two_j + j3) / 2;
          nj = hi >= lo ? hi - lo + 1 : 0.0;
        }
        k0_pair_t[U(PairClass(o1, o2))] += n_t * nj;
        k0_third_t[U(ThirdClass(o3))] += n_t * nj;

        // K1: the distinct (pair, third) keys and n_J of each.
        const Key key{std::min(i1, i2), std::max(i1, i2), i3};
        bool seen = false;
        for (int k = 0; k < num_keys; ++k) {
          seen = seen || (keys[k].p == key.p && keys[k].q == key.q &&
                          keys[k].r == key.r);
        }
        if (seen) continue;
        keys[num_keys++] = key;
        const int jp = orb(key.p).j2.twice;
        const int jq = orb(key.q).j2.twice;
        const int jr = orb(key.r).j2.twice;
        double n_jpq = 0.0;
        for (int jpq = std::abs(jp - jq) / 2; jpq <= (jp + jq) / 2; ++jpq) {
          if (key.p == key.q && (jpq & 1) == 1) continue;
          if (!Triangle2(2 * jpq, jr, two_j)) continue;
          n_jpq += 1.0;
        }
        w_pair_t[U(PairClass(orb(key.p), orb(key.q)))] += n_jpq;
        w_third_t[U(ThirdClass(orb(key.r)))] += n_jpq;
        m_t += n_jpq;
      }

      // Sums over t' <= t (t' = t included): add t to the running sums first.
      for (int j = 0; j < kThirdClasses; ++j) {
        w_third_sum[U(j)] += w_third_t[U(j)];
        k0_third_sum[U(j)] += k0_third_t[U(j)];
        for (int i = 0; i < kPairClasses; ++i) {
          const double g = groups[U(j)][U(i)];
          cost.w_cost += w_pair_t[U(i)] * g * w_third_sum[U(j)];
          cost.k0_cost += k0_pair_t[U(i)] * g * k0_third_sum[U(j)];
        }
      }
      a_m_nl += n_t * m_t;
      a_nl += n_t;
      c_m_nl += m_t * m_t;
      c_nl += m_t;
      cost.epi_cost_a += m_t * a_m_nl + n_t * m_t * a_nl;
      cost.epi_cost_c += m_t * c_m_nl + m_t * m_t * c_nl;
    }
  }
  cost.epi_cost_b = cost.epi_cost_a;
  return cost;
}

}  // namespace imsrg::layout3
