// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/n7_cost.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <span>
#include <tuple>
#include <vector>

#include "imsrg/angmom/three_body_recoupling.h"
#include "imsrg/base/assert.h"
#include "imsrg/basis/orbit.h"

namespace imsrg::commutator {
namespace {

using angmom::Triangle2;

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

constexpr std::size_t Idx(N7Term t) { return static_cast<std::size_t>(t); }

/// A pair-third key: pair (p <= q) and third orbit r.
struct Key {
  int p;
  int q;
  int r;
  friend bool operator==(const Key&, const Key&) = default;
};

/// Counts of one triple obtained from its keys.
struct TripleCounts {
  std::int32_t states = 0;
  std::int32_t active = 0;
  std::int32_t row_entries = 0;  // entries of one R row
};

/// True iff the occupation class of the pair is not 3 (`PairClass` of
/// kernels/comm232.cc: n_p n_q != 0 or nbar_p nbar_q != 0).
bool IsActivePair(double np, double nq) noexcept {
  const bool hh = np != 0.0 && nq != 0.0;
  const bool pp = np != 1.0 && nq != 1.0;
  return hh || pp;
}

/// The key of the permutation (i1, i2, i3) of (a, b, c): pair sorted.
Key MakeKey(int i1, int i2, int i3) {
  return {std::min(i1, i2), std::max(i1, i2), i3};
}

/// Number of Jpq of the key in a channel of doubled J `two_j`: Pauli (Jpq
/// even if p == q) and triangle(2 Jpq, 2 jr, two_j), as in
/// `op::PairThirdSpace`. O(J range).
std::int32_t CountKeyStates(std::span<const basis::Orbit> orbits, const Key& k,
                            int two_j) {
  const int jp = orbits[U(k.p)].j2.twice;
  const int jq = orbits[U(k.q)].j2.twice;
  const int jr = orbits[U(k.r)].j2.twice;
  std::int32_t n = 0;
  for (int jpq = std::abs(jp - jq) / 2; jpq <= (jp + jq) / 2; ++jpq) {
    if (k.p == k.q && (jpq & 1) == 1) continue;
    if (Triangle2(2 * jpq, jr, two_j)) ++n;
  }
  return n;
}

/// States, active states and R-row entries of the triple (a <= b <= c).
/// The distinct keys are those of the permutations ABC, CBA, ACB (the
/// splittings of `op::PairThirdSpace`). An R row (J1) holds the state (ABC
/// key, J1) and every state of the CBA and ACB keys (all J1'), merged where
/// keys coincide. O(J range).
TripleCounts CountTriple(std::span<const basis::Orbit> orbits,
                         std::span<const double> occ, int a, int b, int c,
                         int two_j) {
  const std::array<Key, 3> perm = {MakeKey(a, b, c), MakeKey(b, c, a),
                                   MakeKey(a, c, b)};
  TripleCounts out;
  std::array<Key, 3> distinct{};
  int num_distinct = 0;
  for (const Key& k : perm) {
    bool seen = false;
    for (int i = 0; i < num_distinct; ++i) seen = seen || distinct[U(i)] == k;
    if (seen) continue;
    distinct[U(num_distinct++)] = k;
    const std::int32_t n = CountKeyStates(orbits, k, two_j);
    out.states += n;
    if (IsActivePair(occ[U(k.p)], occ[U(k.q)])) out.active += n;
  }
  const bool abc_covered = perm[0] == perm[1] || perm[0] == perm[2];
  out.row_entries = abc_covered ? 0 : 1;
  out.row_entries += CountKeyStates(orbits, perm[1], two_j);
  if (!(perm[2] == perm[1])) {
    out.row_entries += CountKeyStates(orbits, perm[2], two_j);
  }
  return out;
}

/// Mean over orbits of the number of orbits with the same (l, 2j, 2tz).
double MeanOneBodyChannelSize(std::span<const basis::Orbit> orbits) {
  std::map<std::tuple<int, int, int>, std::int64_t> cnt;
  for (const basis::Orbit& o : orbits) {
    ++cnt[{o.l, o.j2.twice, o.tz2.twice}];
  }
  double sum = 0.0;
  for (const auto& [key, n] : cnt) sum += static_cast<double>(n * n);
  return orbits.empty() ? 0.0 : sum / static_cast<double>(orbits.size());
}

/// Fills `ma` of the triples [t0, t1) of `v` (one channel `ch`). The suffix
/// sums run over the triples T >= t of the channel (223 W sums and
/// epilogue). O(t1 - t0).
void FillChannelCosts(std::vector<N7TripleCost>& v, std::size_t t0,
                      std::size_t t1, const N7ChannelCost& ch, double n1,
                      double n_orb, const N7CostParams& p) {
  const double S = static_cast<double>(ch.states);
  const double A = static_cast<double>(ch.active);
  double skm = 0.0;
  for (std::size_t t = t0; t < t1; ++t) {
    skm += static_cast<double>(v[t].rows) * v[t].states;
  }
  const double n_a = p.fa * n_orb / 4.0;
  const double n_p = p.fp * n_orb / 4.0;
  double suffix_m = 0.0;
  double suffix_nnz = 0.0;
  for (std::size_t t = t1; t-- > t0;) {
    N7TripleCost& c = v[t];
    const double k = c.rows;
    const double m = c.states;
    const double nnz = static_cast<double>(c.nnz_r);
    suffix_m += m;
    suffix_nnz += nnz;
    c.epilogue_223 = nnz * suffix_m + suffix_nnz * k;
    c.ma[Idx(N7Term::k223)] = m * suffix_m * n_a + c.epilogue_223;
    c.ma[Idx(N7Term::kH)] = 2.0 * k * skm;
    c.ma[Idx(N7Term::kW)] = 2.0 * S * k * m;
    c.ma[Idx(N7Term::k232)] = 4.0 * A * m * n_p;
    c.ma[Idx(N7Term::k331)] = 2.0 * k * S * n1;
    c.ma[Idx(N7Term::k133)] = 2.0 * k * (S * n1 + skm);
    c.ma[Idx(N7Term::k330)] = 2.0 * static_cast<double>(c.mem);
    c.ma[Idx(N7Term::k231And132)] = 4.0 * m * S * n1 / n_orb;
  }
}

/// The triple of `space` with index `t`, as a counted `N7TripleCost` (no
/// costs yet).
N7TripleCost CountedTriple(const basis::ThreeBodySpace& space,
                           std::span<const basis::Orbit> orbits,
                           std::span<const double> occ,
                           const basis::KeptChannel3& kc, std::int32_t t) {
  const basis::OrbitTriple& ot = space.triples()[U(t)];
  const TripleCounts tc = CountTriple(orbits, occ, ot.a.value(), ot.b.value(),
                                      ot.c.value(), kc.two_j.twice);
  N7TripleCost c;
  c.rows = ot.row_end.value() - ot.row_begin.value();
  c.states = tc.states;
  c.active = tc.active;
  c.first_row = ot.row_begin.value();
  c.nnz_r = static_cast<std::int64_t>(c.rows) * tc.row_entries;
  c.mem = static_cast<std::int64_t>(c.rows) * kc.dim -
          (static_cast<std::int64_t>(c.rows) * c.first_row +
           static_cast<std::int64_t>(c.rows) * (c.rows - 1) / 2);
  return c;
}

}  // namespace

const char* N7TermName(N7Term t) noexcept {
  switch (t) {
    case N7Term::k223:
      return "223";
    case N7Term::kH:
      return "H";
    case N7Term::kW:
      return "W";
    case N7Term::k232:
      return "232";
    case N7Term::k331:
      return "331";
    case N7Term::k133:
      return "133";
    case N7Term::k330:
      return "330";
    case N7Term::k231And132:
      return "231+132";
  }
  return "?";
}

double N7TripleCost::Total() const noexcept {
  double s = 0.0;
  for (const double x : ma) s += x;
  return s;
}

N7CostModel::N7CostModel(const basis::ModelSpace& ms3,
                         const N7CostParams& params) {
  const basis::ThreeBodySpace* space = ms3.three_body_space();
  IMSRG_CHECK(space != nullptr, "N7CostModel: no three-body space");
  const auto orbits = ms3.basis().orbits();
  const auto occ = ms3.reference().occupations();
  num_orbits_ = ms3.num_orbits();
  n1_ = MeanOneBodyChannelSize(orbits);
  for (const basis::TwoBodyChannel& tc : ms3.two_body().channels()) {
    two_body_ += static_cast<std::int64_t>(tc.num_kets) * tc.num_kets;
  }
  triples_.reserve(U(space->num_triples()));
  channels_.resize(U(space->num_channels()));
  for (std::int32_t ch = 0; ch < space->num_channels(); ++ch) {
    const basis::KeptChannel3& kc = space->channels()[U(ch)];
    N7ChannelCost& cc = channels_[U(ch)];
    cc.dim = kc.dim;
    cc.packed = static_cast<std::int64_t>(kc.dim) * (kc.dim + 1) / 2;
    cc.num_triples = kc.triple_end.value() - kc.triple_begin.value();
    const std::size_t t0 = triples_.size();
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      triples_.push_back(CountedTriple(*space, orbits, occ, kc, t));
      cc.states += triples_.back().states;
      cc.active += triples_.back().active;
    }
    FillChannelCosts(triples_, t0, triples_.size(), cc, n1_,
                     static_cast<double>(num_orbits_), params);
    num_rows_ += kc.dim;
    num_states_ += cc.states;
    num_active_ += cc.active;
    packed_ += cc.packed;
    max_dim_ = std::max(max_dim_, kc.dim);
    max_states_ = std::max(max_states_, cc.states);
  }
}

std::array<double, kNumCostTerms> N7CostModel::TermTotals() const {
  std::array<double, kNumCostTerms> tot{};
  for (const N7TripleCost& c : triples_) {
    for (std::size_t i = 0; i < tot.size(); ++i) tot[i] += c.ma[i];
  }
  return tot;
}

double N7CostModel::TotalMultiplyAdds() const {
  double s = 0.0;
  for (const double x : TermTotals()) s += x;
  return s;
}

std::array<double, 2> N7CostModel::Comm223Parts() const {
  double epilogue = 0.0;
  for (const N7TripleCost& c : triples_) epilogue += c.epilogue_223;
  return {TermTotals()[Idx(N7Term::k223)] - epilogue, epilogue};
}

N7BatchScratch N7CostModel::BatchScratch(std::int64_t lines_per_batch) const {
  IMSRG_CHECK(lines_per_batch >= 1, "BatchScratch: lines_per_batch < 1");
  N7BatchScratch best;
  std::size_t t = 0;
  for (std::size_t ch = 0; ch < channels_.size(); ++ch) {
    const std::int64_t S = channels_[ch].states;
    const std::size_t t_end = t + U(channels_[ch].num_triples);
    while (t < t_end) {
      const std::int64_t first = triples_[t].first_row;
      std::int64_t lines = 0;
      std::int64_t states = 0;
      while (t < t_end &&
             (lines == 0 || lines + triples_[t].rows <= lines_per_batch)) {
        lines += triples_[t].rows;
        states += triples_[t].states;
        ++t;
      }
      const std::int64_t doubles =
          S * lines + S * states + lines * first + lines * (lines + 1) / 2;
      if (doubles > best.max_doubles) {
        best = {doubles, lines, states, static_cast<std::int32_t>(ch)};
      }
    }
  }
  return best;
}

basis::LineWeights N7LineWeights(const basis::ModelSpace& ms3) {
  const N7CostModel model(ms3);
  basis::LineWeights w;
  w.compute.reserve(model.triples().size());
  w.memory.reserve(model.triples().size());
  for (const N7TripleCost& c : model.triples()) {
    w.compute.push_back(c.Total());
    w.memory.push_back(static_cast<double>(c.mem));
  }
  return w;
}

}  // namespace imsrg::commutator
