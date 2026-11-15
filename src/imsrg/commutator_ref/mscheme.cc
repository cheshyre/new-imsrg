// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Brute-force m-scheme commutator reference (see mscheme.h).
#include "imsrg/commutator_ref/mscheme.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"

namespace imsrg::commutator_ref::mscheme {
namespace {

using Labels = std::array<int, 3>;

/// Sorts the first `r` (<= 3) entries of `v` ascending; returns the sign of
/// the sorting permutation, or 0 if two entries are equal.
int SortSign(int r, int* v) {
  if (r < 2) return 1;
  if (r == 2) {
    if (v[0] == v[1]) return 0;
    if (v[0] > v[1]) {
      std::swap(v[0], v[1]);
      return -1;
    }
    return 1;
  }
  int sign = 1;
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    sign = -sign;
  }
  if (v[1] > v[2]) {
    std::swap(v[1], v[2]);
    sign = -sign;
  }
  if (v[0] > v[1]) {
    std::swap(v[0], v[1]);
    sign = -sign;
  }
  if (v[0] == v[1] || v[1] == v[2]) return 0;
  return sign;
}

/// Parity of the permutation that sorts `seq` ascending (distinct values):
/// the parity of its inversion count; returns +1 or -1.
int PermutationSign(const std::vector<int>& seq) {
  int inversions = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    for (std::size_t j = i + 1; j < seq.size(); ++j) {
      if (seq[i] > seq[j]) ++inversions;
    }
  }
  return inversions % 2 == 0 ? 1 : -1;
}

/// Sign of the Wick pattern (nA, nB, k1, k2); see `CommutatorPart` and the
/// design note in the file mscheme.h. The operator string of A B,
///   a+P1..a+PnA a_QnA..a_Q1 a+R1..a+RnB a_SnB..a_S1,
/// is reordered to: each contracted pair adjacent (A operator first), then the
/// uncontracted operators in the canonical output order
///   a+out1..a+outnC a_innC..a_in1,
/// with out = (P1..P_{nA-k2}, R_{k1+1}..R_nB) and in = (Q1..Q_{nA-k1},
/// S_{k2+1}..S_nB). Slot ids are string positions; the sign is the parity of
/// the permutation.
int PatternSign(int nA, int nB, int k1, int k2) {
  const int nC = nA + nB - k1 - k2;
  auto pos_p = [](int i) { return i - 1; };
  auto pos_q = [&](int i) { return 2 * nA - i; };
  auto pos_r = [&](int i) { return 2 * nA + i - 1; };
  auto pos_s = [&](int i) { return 2 * nA + nB + (nB - i); };
  std::vector<int> target;
  const int n_q_free = nA - k1;
  const int n_p_free = nA - k2;
  for (int i = 1; i <= k1; ++i) {
    target.push_back(pos_q(n_q_free + i));
    target.push_back(pos_r(i));
  }
  for (int j = 1; j <= k2; ++j) {
    target.push_back(pos_p(n_p_free + j));
    target.push_back(pos_s(j));
  }
  for (int i = 1; i <= n_p_free; ++i) target.push_back(pos_p(i));
  for (int i = k1 + 1; i <= nB; ++i) target.push_back(pos_r(i));
  std::vector<int> in_pos;
  for (int i = 1; i <= n_q_free; ++i) in_pos.push_back(pos_q(i));
  for (int i = k2 + 1; i <= nB; ++i) in_pos.push_back(pos_s(i));
  IMSRG_CHECK(static_cast<int>(in_pos.size()) == nC, "PatternSign: rank");
  for (int t = nC - 1; t >= 0; --t) {
    target.push_back(in_pos[static_cast<std::size_t>(t)]);
  }
  IMSRG_CHECK(static_cast<int>(target.size()) == 2 * (nA + nB),
              "PatternSign: size");
  return PermutationSign(target);
}

/// A way of splitting a sorted list of n labels into two groups: bit i of
/// `mask` set means label i goes to the first group; `sign` is the parity of
/// the permutation (group 1, group 2) relative to the sorted list.
struct Shuffle {
  unsigned mask;
  int sign;
};

/// All shuffles of n labels with g1 labels in the first group.
std::vector<Shuffle> Shuffles(int n, int g1) {
  std::vector<Shuffle> out;
  for (unsigned mask = 0; mask < (1u << n); ++mask) {
    int count = 0;
    for (int i = 0; i < n; ++i) count += (mask >> i) & 1u;
    if (count != g1) continue;
    int inversions = 0;
    for (int p = 0; p < n; ++p) {
      if (((mask >> p) & 1u) == 0u) continue;
      for (int q = 0; q < p; ++q) {
        if (((mask >> q) & 1u) == 0u) ++inversions;
      }
    }
    out.push_back({mask, inversions % 2 == 0 ? 1 : -1});
  }
  return out;
}

/// Splits v[0..n) by `mask` into g1 (bits set) and g2 (bits clear), both in
/// increasing position order.
void Split(const int* v, int n, unsigned mask, int* g1, int* g2) {
  int a = 0;
  int b = 0;
  for (int i = 0; i < n; ++i) {
    if (((mask >> i) & 1u) != 0u) {
      g1[a++] = v[i];
    } else {
      g2[b++] = v[i];
    }
  }
}

/// A strictly increasing label set of size k with its weight prod n or
/// prod nbar.
struct WeightedSet {
  Labels v{};
  double w = 1.0;
};

/// All strictly increasing sets of k m-states with nonzero weight, the weight
/// being prod n_s (`use_n`) or prod nbar_s.
std::vector<WeightedSet> WeightedSubsets(const MBasis& b, int k, bool use_n) {
  const int n = b.size();
  auto weight = [&](int s) {
    return use_n ? b.StateAt(s).n : b.StateAt(s).nbar;
  };
  std::vector<WeightedSet> out;
  WeightedSet t;
  if (k == 0) {
    out.push_back(t);
  } else if (k == 1) {
    for (int a = 0; a < n; ++a) {
      t.v = {a, 0, 0};
      t.w = weight(a);
      if (t.w != 0.0) out.push_back(t);
    }
  } else if (k == 2) {
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) {
        t.v = {a, c, 0};
        t.w = weight(a) * weight(c);
        if (t.w != 0.0) out.push_back(t);
      }
    }
  } else {
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) {
        for (int d = c + 1; d < n; ++d) {
          t.v = {a, c, d};
          t.w = weight(a) * weight(c) * weight(d);
          if (t.w != 0.0) out.push_back(t);
        }
      }
    }
  }
  return out;
}

/// One Wick pattern of engine(X, Y) (X the first factor of the product).
struct Pattern {
  bool swapped = false;  // true: X = B, Y = A and the result is subtracted
  int nX = 0;
  int nY = 0;
  int k1 = 0;
  int k2 = 0;
  int sign = 1;
  std::vector<Shuffle> shuffle_out;
  std::vector<Shuffle> shuffle_in;
  std::vector<WeightedSet> sets_d;  // weight prod nbar
  std::vector<WeightedSet> sets_c;  // weight prod n
};

/// C_{out; in} of one pattern (without the commutator sign of `swapped`).
double EvaluatePattern(const MOperator& X, const MOperator& Y, const Pattern& p,
                       int nC, const int* out, const int* in) {
  const int g_xa = p.nX - p.k2;
  const int g_xb = p.nY - p.k1;
  const int g_ya = p.nX - p.k1;
  const int g_yb = p.nY - p.k2;
  double acc = 0.0;
  for (const Shuffle& so : p.shuffle_out) {
    int xa[3] = {0, 0, 0};
    int xb[3] = {0, 0, 0};
    Split(out, nC, so.mask, xa, xb);
    for (const Shuffle& si : p.shuffle_in) {
      int ya[3] = {0, 0, 0};
      int yb[3] = {0, 0, 0};
      Split(in, nC, si.mask, ya, yb);
      const double s = static_cast<double>(so.sign * si.sign);
      for (const WeightedSet& c : p.sets_c) {
        int P[3] = {0, 0, 0};
        int S[3] = {0, 0, 0};
        for (int i = 0; i < g_xa; ++i) P[i] = xa[i];
        for (int i = 0; i < p.k2; ++i)
          P[g_xa + i] = c.v[static_cast<size_t>(i)];
        for (int i = 0; i < p.k2; ++i) S[i] = c.v[static_cast<size_t>(i)];
        for (int i = 0; i < g_yb; ++i) S[p.k2 + i] = yb[i];
        for (const WeightedSet& d : p.sets_d) {
          int Q[3] = {0, 0, 0};
          int R[3] = {0, 0, 0};
          for (int i = 0; i < g_ya; ++i) Q[i] = ya[i];
          for (int i = 0; i < p.k1; ++i) {
            Q[g_ya + i] = d.v[static_cast<size_t>(i)];
          }
          for (int i = 0; i < p.k1; ++i) R[i] = d.v[static_cast<size_t>(i)];
          for (int i = 0; i < g_xb; ++i) R[p.k1 + i] = xb[i];
          const double a = X.Get(p.nX, std::span<const int>(P, 3),
                                 std::span<const int>(Q, 3));
          if (a == 0.0) continue;
          const double b = Y.Get(p.nY, std::span<const int>(R, 3),
                                 std::span<const int>(S, 3));
          acc += s * c.w * d.w * a * b;
        }
      }
    }
  }
  return static_cast<double>(p.sign) * acc;
}

/// Sorted label tuples of size r over N states, grouped by the conserved
/// quantities (sum 2m, parity, sum 2tz) and filtered by the kept three-body
/// triples for r = 3.
std::map<std::tuple<int, int, int>, std::vector<Labels>> GroupedTuples(
    const MBasis& b, int r) {
  std::map<std::tuple<int, int, int>, std::vector<Labels>> groups;
  const int n = b.size();
  auto add = [&](Labels t) {
    int m = 0;
    int l = 0;
    int tz = 0;
    for (int i = 0; i < r; ++i) {
      const MState& s = b.StateAt(t[static_cast<size_t>(i)]);
      m += s.two_m;
      l += s.l;
      tz += s.two_tz;
    }
    if (r == 3 && !b.IsKeptTriple(t[0], t[1], t[2])) return;
    groups[{m, l % 2, tz}].push_back(t);
  };
  if (r == 0) {
    add({0, 0, 0});
  } else if (r == 1) {
    for (int a = 0; a < n; ++a) add({a, 0, 0});
  } else if (r == 2) {
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) add({a, c, 0});
    }
  } else {
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) {
        for (int d = c + 1; d < n; ++d) add({a, c, d});
      }
    }
  }
  return groups;
}

}  // namespace

// ------------------------------------------------------------------ MBasis

MBasis::MBasis(const basis::ModelSpace& ms) {
  const int no = ms.num_orbits();
  first_of_orbit_.assign(static_cast<std::size_t>(no), 0);
  for (int o = 0; o < no; ++o) {
    const basis::Orbit& orb = ms.basis().orbit(OrbitIndex{o});
    const double n = ms.reference().occupation(OrbitIndex{o});
    first_of_orbit_[static_cast<std::size_t>(o)] = size();
    for (int m = -orb.j2.twice; m <= orb.j2.twice; m += 2) {
      MState s;
      s.orbit = o;
      s.two_m = m;
      s.l = orb.l;
      s.two_j = orb.j2.twice;
      s.two_tz = orb.tz2.twice;
      s.n = n;
      s.nbar = 1.0 - n;
      states_.push_back(s);
    }
  }
  if (const basis::ThreeBodySpace* sp = ms.three_body_space()) {
    const std::size_t nn = static_cast<std::size_t>(no);
    kept_orbit_triple_.assign(nn * nn * nn, 0);
    for (const basis::OrbitTriple& t : sp->triples()) {
      kept_orbit_triple_[static_cast<std::size_t>(t.a.value()) * nn * nn +
                         static_cast<std::size_t>(t.b.value()) * nn +
                         static_cast<std::size_t>(t.c.value())] = 1;
    }
  }
}

int MBasis::Index(int orbit, int two_m) const {
  const int first = first_of_orbit_[static_cast<std::size_t>(orbit)];
  const int two_j = states_[static_cast<std::size_t>(first)].two_j;
  IMSRG_CHECK(std::abs(two_m) <= two_j && (two_m + two_j) % 2 == 0,
              "MBasis::Index: invalid m");
  return first + (two_m + two_j) / 2;
}

bool MBasis::IsKeptTriple(int a, int b, int c) const {
  if (kept_orbit_triple_.empty()) return false;
  int v[3] = {a, b, c};
  if (SortSign(3, v) == 0) return false;
  const std::size_t nn = first_of_orbit_.size();
  return kept_orbit_triple_[static_cast<std::size_t>(
                                states_[static_cast<std::size_t>(v[0])].orbit) *
                                nn * nn +
                            static_cast<std::size_t>(
                                states_[static_cast<std::size_t>(v[1])].orbit) *
                                nn +
                            static_cast<std::size_t>(
                                states_[static_cast<std::size_t>(v[2])]
                                    .orbit)] != 0;
}

// --------------------------------------------------------------- MOperator

MOperator::MOperator(int num_states) : n_(num_states) {
  const auto n = static_cast<std::size_t>(n_);
  n2_ = n_ * (n_ - 1) / 2;
  n3_ = n_ * (n_ - 1) * (n_ - 2) / 6;
  one_.assign(n * n, 0.0);
  two_.assign(static_cast<std::size_t>(n2_) * static_cast<std::size_t>(n2_),
              0.0);
  three_.assign(static_cast<std::size_t>(n3_) * static_cast<std::size_t>(n3_),
                0.0);
  pair_index_.assign(n * n, -1);
  triple_index_.assign(n * n * n, -1);
  std::int32_t k = 0;
  for (int p = 0; p < n_; ++p) {
    for (int q = p + 1; q < n_; ++q) {
      pair_index_[static_cast<std::size_t>(p * n_ + q)] = k++;
    }
  }
  k = 0;
  for (int p = 0; p < n_; ++p) {
    for (int q = p + 1; q < n_; ++q) {
      for (int r = q + 1; r < n_; ++r) {
        triple_index_[static_cast<std::size_t>((p * n_ + q) * n_ + r)] = k++;
      }
    }
  }
}

double MOperator::Get(int rank, std::span<const int> out,
                      std::span<const int> in) const {
  if (rank == 0) return zero_;
  int o[3] = {0, 0, 0};
  int i[3] = {0, 0, 0};
  for (int t = 0; t < rank; ++t) {
    o[t] = out[static_cast<std::size_t>(t)];
    i[t] = in[static_cast<std::size_t>(t)];
  }
  const int so = SortSign(rank, o);
  const int si = SortSign(rank, i);
  if (so == 0 || si == 0) return 0.0;
  const double sign = static_cast<double>(so * si);
  if (rank == 1) {
    return one_[static_cast<std::size_t>(o[0] * n_ + i[0])];
  }
  if (rank == 2) {
    const auto row = static_cast<std::size_t>(
        pair_index_[static_cast<std::size_t>(o[0] * n_ + o[1])]);
    const auto col = static_cast<std::size_t>(
        pair_index_[static_cast<std::size_t>(i[0] * n_ + i[1])]);
    return sign * two_[row * static_cast<std::size_t>(n2_) + col];
  }
  const auto row = static_cast<std::size_t>(
      triple_index_[static_cast<std::size_t>((o[0] * n_ + o[1]) * n_ + o[2])]);
  const auto col = static_cast<std::size_t>(
      triple_index_[static_cast<std::size_t>((i[0] * n_ + i[1]) * n_ + i[2])]);
  return sign * three_[row * static_cast<std::size_t>(n3_) + col];
}

void MOperator::Add(int rank, std::span<const int> out, std::span<const int> in,
                    double v) {
  if (rank == 0) {
    zero_ += v;
    return;
  }
  int o[3] = {0, 0, 0};
  int i[3] = {0, 0, 0};
  for (int t = 0; t < rank; ++t) {
    o[t] = out[static_cast<std::size_t>(t)];
    i[t] = in[static_cast<std::size_t>(t)];
  }
  const int so = SortSign(rank, o);
  const int si = SortSign(rank, i);
  if (so == 0 || si == 0) return;
  const double x = static_cast<double>(so * si) * v;
  if (rank == 1) {
    one_[static_cast<std::size_t>(o[0] * n_ + i[0])] += x;
  } else if (rank == 2) {
    const auto row = static_cast<std::size_t>(
        pair_index_[static_cast<std::size_t>(o[0] * n_ + o[1])]);
    const auto col = static_cast<std::size_t>(
        pair_index_[static_cast<std::size_t>(i[0] * n_ + i[1])]);
    two_[row * static_cast<std::size_t>(n2_) + col] += x;
  } else {
    const auto row =
        static_cast<std::size_t>(triple_index_[static_cast<std::size_t>(
            (o[0] * n_ + o[1]) * n_ + o[2])]);
    const auto col =
        static_cast<std::size_t>(triple_index_[static_cast<std::size_t>(
            (i[0] * n_ + i[1]) * n_ + i[2])]);
    three_[row * static_cast<std::size_t>(n3_) + col] += x;
  }
}

void MOperator::Set(int rank, std::span<const int> out, std::span<const int> in,
                    double v) {
  const double old = Get(rank, out, in);
  Add(rank, out, in, v - old);
}

std::int64_t MOperator::NumLabelSets(int rank) const {
  switch (rank) {
    case 0:
      return 1;
    case 1:
      return n_;
    case 2:
      return n2_;
    default:
      return n3_;
  }
}

double MOperator::MaxAbs(int rank) const {
  double m = 0.0;
  if (rank == 0) return std::abs(zero_);
  const std::vector<double>& v = rank == 1 ? one_ : (rank == 2 ? two_ : three_);
  for (const double x : v) m = std::max(m, std::abs(x));
  return m;
}

double MOperator::MaxAbsDifference(const MOperator& other, int rank) const {
  IMSRG_CHECK(n_ == other.n_, "MOperator::MaxAbsDifference: sizes differ");
  if (rank == 0) return std::abs(zero_ - other.zero_);
  const std::vector<double>& a = rank == 1 ? one_ : (rank == 2 ? two_ : three_);
  const std::vector<double>& b =
      rank == 1 ? other.one_ : (rank == 2 ? other.two_ : other.three_);
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void MOperator::Axpy(double f, const MOperator& other) {
  IMSRG_CHECK(n_ == other.n_, "MOperator::Axpy: sizes differ");
  zero_ += f * other.zero_;
  for (std::size_t i = 0; i < one_.size(); ++i) one_[i] += f * other.one_[i];
  for (std::size_t i = 0; i < two_.size(); ++i) two_[i] += f * other.two_[i];
  for (std::size_t i = 0; i < three_.size(); ++i) {
    three_[i] += f * other.three_[i];
  }
}

// ------------------------------------------------------------------ Expand

namespace {

double Cg(int two_j1, int two_m1, int two_j2, int two_m2, int two_j,
          int two_m) {
  return angmom::ClebschGordan(TwoJ{two_j1}, TwoM{two_m1}, TwoJ{two_j2},
                               TwoM{two_m2}, TwoJ{two_j}, TwoM{two_m});
}

/// One term of the J -> m expansion of a three-body triple: channel, row of the
/// canonical ket and the product of two Clebsch-Gordan coefficients.
struct TripleEntry {
  std::int32_t channel;
  std::int32_t row;
  double coef;
};

/// Entries of the m-triple (a, b, c) (any order of orbits is allowed, the
/// canonical row requires orbits a <= b <= c) at fixed 2M = two_m_total.
std::vector<TripleEntry> ExpandEntries(const op::Operator& X, const MBasis& mb,
                                       int a, int b, int c) {
  std::vector<TripleEntry> entries;
  const basis::ThreeBodySpace& space = *X.model_space().three_body_space();
  const MState& sa = mb.StateAt(a);
  const MState& sb = mb.StateAt(b);
  const MState& sc = mb.StateAt(c);
  const int two_m = sa.two_m + sb.two_m + sc.two_m;
  const int two_tz = sa.two_tz + sb.two_tz + sc.two_tz;
  const Parity parity = ParityOfL(sa.l + sb.l + sc.l);
  const int jab_lo = std::abs(sa.two_j - sb.two_j);
  const int jab_hi = sa.two_j + sb.two_j;
  const basis::ThreeBodyChannels& ch_table = *X.model_space().three_body();
  for (int two_jab = jab_lo; two_jab <= jab_hi; two_jab += 2) {
    const int two_mab = sa.two_m + sb.two_m;
    if (std::abs(two_mab) > two_jab) continue;
    const double cg1 =
        Cg(sa.two_j, sa.two_m, sb.two_j, sb.two_m, two_jab, two_mab);
    if (cg1 == 0.0) continue;
    const int j_lo = std::abs(two_jab - sc.two_j);
    const int j_hi = two_jab + sc.two_j;
    for (int two_j = j_lo; two_j <= j_hi; two_j += 2) {
      if (std::abs(two_m) > two_j) continue;
      const auto ch = ch_table.find_channel(TwoJ{two_j}, parity, TwoTz{two_tz});
      if (!ch.has_value()) continue;
      const auto row =
          space.find_row(*ch, OrbitIndex{sa.orbit}, OrbitIndex{sb.orbit},
                         OrbitIndex{sc.orbit}, TwoJ{two_jab});
      if (!row.has_value()) continue;
      const double cg2 = Cg(two_jab, two_mab, sc.two_j, sc.two_m, two_j, two_m);
      if (cg2 == 0.0) continue;
      entries.push_back({ch->value(), row->value(), cg1 * cg2});
    }
  }
  return entries;
}

}  // namespace

MOperator Expand(const op::Operator& X, const MBasis& mb) {
  X.SyncToHost();
  const int n = mb.size();
  MOperator A(n);
  A.zero_body() = X.zero_body();
  // One body.
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      if (mb.StateAt(a).two_m != mb.StateAt(b).two_m) continue;
      const int ia[1] = {a};
      const int ib[1] = {b};
      A.Set(1, ia, ib,
            X.one_body().get(OrbitIndex{mb.StateAt(a).orbit},
                             OrbitIndex{mb.StateAt(b).orbit}));
    }
  }
  // Two body.
  if (X.two_body().allocated()) {
    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) {
        const MState& sa = mb.StateAt(a);
        const MState& sb = mb.StateAt(b);
        const int two_mab = sa.two_m + sb.two_m;
        for (int c = 0; c < n; ++c) {
          for (int d = c + 1; d < n; ++d) {
            const MState& sc = mb.StateAt(c);
            const MState& sd = mb.StateAt(d);
            if (sc.two_m + sd.two_m != two_mab) continue;
            if ((sa.l + sb.l + sc.l + sd.l) % 2 != 0) continue;
            if (sa.two_tz + sb.two_tz != sc.two_tz + sd.two_tz) continue;
            const int j_lo =
                std::max({std::abs(sa.two_j - sb.two_j),
                          std::abs(sc.two_j - sd.two_j), std::abs(two_mab)});
            const int j_hi = std::min(sa.two_j + sb.two_j, sc.two_j + sd.two_j);
            double v = 0.0;
            for (int two_j = j_lo; two_j <= j_hi; two_j += 2) {
              const double cg1 =
                  Cg(sa.two_j, sa.two_m, sb.two_j, sb.two_m, two_j, two_mab);
              const double cg2 =
                  Cg(sc.two_j, sc.two_m, sd.two_j, sd.two_m, two_j, two_mab);
              if (cg1 == 0.0 || cg2 == 0.0) continue;
              v += cg1 * cg2 *
                   X.two_body().get_unnormalized(
                       OrbitIndex{sa.orbit}, OrbitIndex{sb.orbit},
                       OrbitIndex{sc.orbit}, OrbitIndex{sd.orbit}, TwoJ{two_j});
            }
            const int o[2] = {a, b};
            const int i[2] = {c, d};
            A.Set(2, o, i, v);
          }
        }
      }
    }
  }
  // Three body.
  if (X.three_body().has_value() && !X.three_body()->empty()) {
    const op::ThreeBody& x3 = *X.three_body();
    std::vector<Labels> triples;
    std::vector<std::vector<TripleEntry>> entries;
    std::vector<std::array<int, 3>> key;  // (2M, 2Tz, parity)
    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) {
        for (int c = b + 1; c < n; ++c) {
          if (!mb.IsKeptTriple(a, b, c)) continue;
          triples.push_back({a, b, c});
          entries.push_back(ExpandEntries(X, mb, a, b, c));
          key.push_back(
              {mb.StateAt(a).two_m + mb.StateAt(b).two_m + mb.StateAt(c).two_m,
               mb.StateAt(a).two_tz + mb.StateAt(b).two_tz +
                   mb.StateAt(c).two_tz,
               (mb.StateAt(a).l + mb.StateAt(b).l + mb.StateAt(c).l) % 2});
        }
      }
    }
    for (std::size_t t = 0; t < triples.size(); ++t) {
      for (std::size_t u = 0; u < triples.size(); ++u) {
        if (key[t] != key[u]) continue;
        double v = 0.0;
        for (const TripleEntry& e : entries[t]) {
          for (const TripleEntry& f : entries[u]) {
            if (e.channel != f.channel) continue;
            v += e.coef * f.coef *
                 x3.get(Channel3Index{e.channel}, basis::KeptRow3Index{e.row},
                        basis::KeptRow3Index{f.row});
          }
        }
        if (v != 0.0) A.Set(3, triples[t], triples[u], v);
      }
    }
  }
  return A;
}

// ---------------------------------------------------------------- Contract

namespace {

/// One term of a bra or ket in the m -> J contraction: the (ordered) m-states
/// of the coupled orbits and the coupling coefficient.
struct CoupledTerm {
  Labels m;
  double coef;
};

/// Terms of the two-orbit state |(p q) J M = J>.
std::vector<CoupledTerm> CoupleTwo(const MBasis& mb, int orb_p, int orb_q,
                                   int two_j) {
  std::vector<CoupledTerm> out;
  const int fp = mb.FirstOfOrbit(orb_p);
  const int jp = mb.StateAt(fp).two_j;
  for (int mp = -jp; mp <= jp; mp += 2) {
    const int a = mb.Index(orb_p, mp);
    const int fq = mb.FirstOfOrbit(orb_q);
    const int jq = mb.StateAt(fq).two_j;
    const int mq = two_j - mp;
    if (std::abs(mq) > jq) continue;
    const int b = mb.Index(orb_q, mq);
    const double c = Cg(jp, mp, jq, mq, two_j, two_j);
    if (c != 0.0) out.push_back({{a, b, 0}, c});
  }
  return out;
}

/// Terms of the three-orbit state |((a b) Jab c) J, 2M = 1>.
std::vector<CoupledTerm> CoupleThree(const MBasis& mb, int oa, int ob, int oc,
                                     int two_jab, int two_j) {
  std::vector<CoupledTerm> out;
  const int ja = mb.StateAt(mb.FirstOfOrbit(oa)).two_j;
  const int jb = mb.StateAt(mb.FirstOfOrbit(ob)).two_j;
  const int jc = mb.StateAt(mb.FirstOfOrbit(oc)).two_j;
  for (int ma = -ja; ma <= ja; ma += 2) {
    for (int mb2 = -jb; mb2 <= jb; mb2 += 2) {
      const int mab = ma + mb2;
      if (std::abs(mab) > two_jab) continue;
      const int mc = 1 - mab;
      if (std::abs(mc) > jc) continue;
      const double c1 = Cg(ja, ma, jb, mb2, two_jab, mab);
      if (c1 == 0.0) continue;
      const double c2 = Cg(two_jab, mab, jc, mc, two_j, 1);
      if (c2 == 0.0) continue;
      out.push_back(
          {{mb.Index(oa, ma), mb.Index(ob, mb2), mb.Index(oc, mc)}, c1 * c2});
    }
  }
  return out;
}

}  // namespace

op::Operator Contract(const MOperator& A,
                      std::shared_ptr<const basis::ModelSpace> ms,
                      const op::OperatorSpec& spec) {
  IMSRG_CHECK(ms != nullptr, "mscheme::Contract: null model space");
  if (spec.particle_rank == 3 && ms->three_body_space() == nullptr) {
    throw InputError("mscheme::Contract: rank 3 needs a three-body space");
  }
  const MBasis mb(*ms);
  IMSRG_CHECK(mb.size() == A.num_states(),
              "mscheme::Contract: operator and model space disagree");
  op::Operator Z(ms, spec);
  Z.zero_body() = A.zero_body();
  // One body.
  const int no = ms->num_orbits();
  for (int a = 0; a < no; ++a) {
    for (const OrbitIndex bo : ms->basis().SameLjtz(OrbitIndex{a})) {
      const int sa = mb.Index(a, mb.StateAt(mb.FirstOfOrbit(a)).two_j);
      const int sb = mb.Index(bo.value(), mb.StateAt(mb.FirstOfOrbit(a)).two_j);
      const int ia[1] = {sa};
      const int ib[1] = {sb};
      Z.one_body().set_element(OrbitIndex{a}, bo, A.Get(1, ia, ib));
    }
  }
  // Two body.
  if (spec.particle_rank >= 2) {
    const basis::TwoBodyChannels& chans = ms->two_body();
    for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
      const ChannelIndex ch{c};
      const int nk = chans.channel(ch).num_kets;
      const int two_j = chans.channel(ch).two_j.twice;
      std::vector<std::vector<CoupledTerm>> terms(static_cast<std::size_t>(nk));
      for (int i = 0; i < nk; ++i) {
        const auto& k = chans.ket(chans.global_ket(ch, LocalKetIndex{i}));
        terms[static_cast<std::size_t>(i)] =
            CoupleTwo(mb, k.p.value(), k.q.value(), two_j);
      }
      for (int i = 0; i < nk; ++i) {
        const auto& bra = chans.ket(chans.global_ket(ch, LocalKetIndex{i}));
        for (int j = 0; j < nk; ++j) {
          const auto& ket = chans.ket(chans.global_ket(ch, LocalKetIndex{j}));
          double v = 0.0;
          for (const CoupledTerm& tb : terms[static_cast<std::size_t>(i)]) {
            for (const CoupledTerm& tk : terms[static_cast<std::size_t>(j)]) {
              v += tb.coef * tk.coef *
                   A.Get(2, std::span<const int>(tb.m.data(), 2),
                         std::span<const int>(tk.m.data(), 2));
            }
          }
          v /= std::sqrt((bra.p == bra.q ? 2.0 : 1.0) *
                         (ket.p == ket.q ? 2.0 : 1.0));
          Z.two_body().set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
        }
      }
    }
  }
  // Three body.
  if (spec.particle_rank >= 3) {
    const basis::ThreeBodySpace& space = *ms->three_body_space();
    op::ThreeBody& z3 = *Z.three_body();
    const bool packed = z3.packed();
    const bool anti = spec.hermiticity == Hermiticity::kAntiHermitian;
    for (std::int32_t c = 0; c < space.num_channels(); ++c) {
      const Channel3Index ch{c};
      const int dim = space.dim(ch);
      if (dim == 0) continue;
      const int two_j = space.channel(ch).two_j.twice;
      std::vector<std::vector<CoupledTerm>> terms(
          static_cast<std::size_t>(dim));
      for (int i = 0; i < dim; ++i) {
        const std::int64_t flat = space.FlatRow(ch, basis::KeptRow3Index{i});
        const basis::OrbitTriple& t = space.triple(space.row_triple(flat));
        terms[static_cast<std::size_t>(i)] =
            CoupleThree(mb, t.a.value(), t.b.value(), t.c.value(),
                        space.row_two_jab(flat).twice, two_j);
      }
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          if (packed && j > i) continue;
          if (anti && i == j) continue;
          double v = 0.0;
          for (const CoupledTerm& tb : terms[static_cast<std::size_t>(i)]) {
            for (const CoupledTerm& tk : terms[static_cast<std::size_t>(j)]) {
              v += tb.coef * tk.coef *
                   A.Get(3, std::span<const int>(tb.m.data(), 3),
                         std::span<const int>(tk.m.data(), 3));
            }
          }
          z3.set(ch, basis::KeptRow3Index{i}, basis::KeptRow3Index{j}, v);
        }
      }
    }
  }
  Z.SyncToDevice();
  return Z;
}

// ------------------------------------------------------------- Commutator

MOperator CommutatorPart(const MOperator& A, int nA, const MOperator& B, int nB,
                         int nC, const MBasis& mb) {
  IMSRG_CHECK(nA >= 0 && nA <= 3 && nB >= 0 && nB <= 3 && nC >= 0 && nC <= 3,
              "CommutatorPart: ranks must be in 0..3");
  IMSRG_CHECK(A.num_states() == mb.size() && B.num_states() == mb.size(),
              "CommutatorPart: operator and basis disagree");
  MOperator Z(mb.size());
  // The patterns of engine(A, B) and engine(B, A).
  std::vector<Pattern> patterns;
  for (int swapped = 0; swapped < 2; ++swapped) {
    const int nX = swapped != 0 ? nB : nA;
    const int nY = swapped != 0 ? nA : nB;
    const MOperator& X = swapped != 0 ? B : A;
    const MOperator& Y = swapped != 0 ? A : B;
    if (X.MaxAbs(nX) == 0.0 || Y.MaxAbs(nY) == 0.0) continue;
    for (int k1 = 0; k1 <= std::min(nX, nY); ++k1) {
      for (int k2 = 0; k2 <= std::min(nX, nY); ++k2) {
        if (nX + nY - k1 - k2 != nC) continue;
        Pattern p;
        p.swapped = swapped != 0;
        p.nX = nX;
        p.nY = nY;
        p.k1 = k1;
        p.k2 = k2;
        p.sign = PatternSign(nX, nY, k1, k2);
        p.shuffle_out = Shuffles(nC, nX - k2);
        p.shuffle_in = Shuffles(nC, nX - k1);
        p.sets_d = WeightedSubsets(mb, k1, false);
        p.sets_c = WeightedSubsets(mb, k2, true);
        patterns.push_back(std::move(p));
      }
    }
  }
  if (patterns.empty()) return Z;
  const auto groups = GroupedTuples(mb, nC);
  for (const auto& [key, tuples] : groups) {
    for (const Labels& out : tuples) {
      for (const Labels& in : tuples) {
        double c = 0.0;
        for (const Pattern& p : patterns) {
          const MOperator& X = p.swapped ? B : A;
          const MOperator& Y = p.swapped ? A : B;
          const double v = EvaluatePattern(X, Y, p, nC, out.data(), in.data());
          c += p.swapped ? -v : v;
        }
        if (c == 0.0) continue;
        if (nC == 0) {
          Z.zero_body() += c;
        } else {
          Z.Add(nC, std::span<const int>(out.data(), static_cast<size_t>(nC)),
                std::span<const int>(in.data(), static_cast<size_t>(nC)), c);
        }
      }
    }
  }
  return Z;
}

MOperator Commutator(const MOperator& A, const MOperator& B, const MBasis& mb,
                     int max_rank) {
  MOperator Z(mb.size());
  for (int nC = 0; nC <= max_rank; ++nC) {
    for (int nA = 0; nA <= 3; ++nA) {
      for (int nB = 0; nB <= 3; ++nB) {
        if (nA == 0 || nB == 0) continue;  // commutes with a number
        Z.Axpy(1.0, CommutatorPart(A, nA, B, nB, nC, mb));
      }
    }
  }
  return Z;
}

}  // namespace imsrg::commutator_ref::mscheme
