// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/pair_third.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/base/constants.h"
#include "imsrg/base/types.h"
#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {
namespace {

constexpr double kDropBelow = 1.0e-13;

/// Index conversion for std containers.
constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

/// One pair-third state of a triple (channel-local order = vector order).
struct State {
  std::int32_t p;
  std::int32_t q;
  std::int32_t r;
  std::int32_t two_jpq;
};

/// Appends the pair-third states of the triple (a <= b <= c) in a channel of
/// doubled total J `two_j`: distinct splittings in order of first appearance
/// among ABC, CBA, ACB; Jpq ascending; Pauli and triangle(Jpq, jr, J) kept.
/// `orbits` gives j. O(J range).
void AppendTripleStates(std::span<const basis::Orbit> orbits, int a, int b,
                        int c, int two_j, std::vector<State>* out) {
  constexpr Perm kPerms[3] = {Perm::kABC, Perm::kCBA, Perm::kACB};
  int keys[3][3];
  int num_keys = 0;
  for (const Perm perm : kPerms) {
    int i1 = 0;
    int i2 = 0;
    int i3 = 0;
    Permute(perm, a, b, c, i1, i2, i3);
    const int key[3] = {std::min(i1, i2), std::max(i1, i2), i3};
    bool seen = false;
    for (int k = 0; k < num_keys; ++k) {
      seen = seen || (keys[k][0] == key[0] && keys[k][1] == key[1] &&
                      keys[k][2] == key[2]);
    }
    if (seen) continue;
    for (int d = 0; d < 3; ++d) keys[num_keys][d] = key[d];
    ++num_keys;
    const int jp = orbits[U(key[0])].j2.twice;
    const int jq = orbits[U(key[1])].j2.twice;
    const int jr = orbits[U(key[2])].j2.twice;
    for (int jpq = std::abs(jp - jq) / 2; jpq <= (jp + jq) / 2; ++jpq) {
      if (key[0] == key[1] && (jpq & 1) == 1) continue;
      if (!Triangle2(2 * jpq, jr, two_j)) continue;
      out->push_back({key[0], key[1], key[2], 2 * jpq});
    }
  }
}

SpaceCounts Summarize(const std::vector<std::int64_t>& dims,
                      std::int64_t num_triples) {
  SpaceCounts s;
  for (const std::int64_t n : dims) {
    s.sum_n += n;
    s.sum_n2 += n * n;
    s.sum_packed += n * (n + 1) / 2;
    s.max_n = std::max(s.max_n, static_cast<std::int32_t>(n));
  }
  s.num_channels = static_cast<std::int32_t>(dims.size());
  s.num_triples = num_triples;
  return s;
}

template <class T>
exec::DeviceBuffer<T> ToBuffer(const std::vector<T>& v) {
  exec::DeviceBuffer<T> buf(v.size());
  buf.CopyFromHost(std::span<const T>(v));
  return buf;
}

/// Sorts (col, val) pairs by column, sums equal columns and drops tiny sums,
/// appending the result to `col` / `val`; clears `row`.
void AppendRow(std::vector<std::pair<std::int32_t, double>>* row,
               std::vector<std::int32_t>* col, std::vector<double>* val) {
  std::sort(row->begin(), row->end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });
  std::size_t i = 0;
  while (i < row->size()) {
    const std::int32_t c = (*row)[i].first;
    double sum = 0.0;
    for (; i < row->size() && (*row)[i].first == c; ++i) {
      sum += (*row)[i].second;
    }
    if (std::abs(sum) >= kDropBelow) {
      col->push_back(c);
      val->push_back(sum);
    }
  }
  row->clear();
}

std::int64_t SizeBytes(const exec::DeviceBuffer<std::int32_t>& b) {
  return static_cast<std::int64_t>(b.size() * sizeof(std::int32_t));
}

}  // namespace

CsrView Csr::host_view() const noexcept {
  return {ptr.host_span().data(), col.host_span().data(),
          val.host_span().data()};
}

CsrView Csr::device_view() const noexcept {
  return {ptr.device_data(), col.device_data(), val.device_data()};
}

std::int64_t Csr::Bytes() const noexcept {
  return static_cast<std::int64_t>(ptr.size() * sizeof(std::int64_t) +
                                   col.size() * sizeof(std::int32_t) +
                                   val.size() * sizeof(double));
}

Csr Csr::FromHost(const std::vector<std::int64_t>& ptr,
                  const std::vector<std::int32_t>& col,
                  const std::vector<double>& val) {
  Csr m;
  m.ptr = ToBuffer(ptr);
  m.col = ToBuffer(col);
  m.val = ToBuffer(val);
  return m;
}

PairThirdSpace::PairThirdSpace(const CanonicalSpace& space,
                               const SixJ3Table& sixj) {
  const CanonicalView cv = space.host_view();
  const SixJ3View sj = sixj.host_view();
  const auto orbits = space.model_space().basis().orbits();
  num_channels_ = cv.num_channels;

  std::vector<std::int32_t> ch_off(U(cv.num_channels));
  std::vector<std::int32_t> ch_num(U(cv.num_channels));
  std::vector<std::int32_t> tr_begin(U(cv.num_triples));
  std::vector<std::int32_t> tr_end(U(cv.num_triples));
  std::vector<std::int32_t> pp;
  std::vector<std::int32_t> pq;
  std::vector<std::int32_t> pr;
  std::vector<std::int32_t> pj;
  std::vector<std::int64_t> r_ptr{0};
  std::vector<std::int32_t> r_col;
  std::vector<double> r_val;
  std::vector<std::int64_t> e_ptr{0};
  std::vector<std::int32_t> e_col;
  std::vector<double> e_val;
  std::vector<std::pair<std::int32_t, double>> row;
  std::vector<State> states;
  std::int64_t flat = 0;

  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    const int two_j = cv.ch_two_j[ch];
    const TwoJ tj{two_j};
    ch_off[U(ch)] = static_cast<std::int32_t>(flat);
    std::int32_t local = 0;
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      const int a = cv.tr_a[t];
      const int b = cv.tr_b[t];
      const int c = cv.tr_c[t];
      const TwoJ ja = orbits[U(a)].j2;
      const TwoJ jb = orbits[U(b)].j2;
      const TwoJ jc = orbits[U(c)].j2;
      states.clear();
      AppendTripleStates(orbits, a, b, c, two_j, &states);
      const std::int32_t base = local;
      tr_begin[U(t)] = base;
      for (const State& s : states) {
        pp.push_back(s.p);
        pq.push_back(s.q);
        pr.push_back(s.r);
        pj.push_back(s.two_jpq);
      }
      local += static_cast<std::int32_t>(states.size());
      tr_end[U(t)] = local;
      const auto find_state = [&](int p, int q, int r, int two_jpq) {
        for (std::size_t k = 0; k < states.size(); ++k) {
          if (states[k].p == p && states[k].q == q && states[k].r == r &&
              states[k].two_jpq == two_jpq) {
            return base + static_cast<std::int32_t>(k);
          }
        }
        return static_cast<std::int32_t>(-1);
      };

      // R: one row per canonical row of the triple.
      for (std::int32_t lr = cv.tr_row_begin[t]; lr < cv.tr_row_end[t]; ++lr) {
        const int j1 = cv.row_two_jab[cv.ch_row_offset[ch] + lr] / 2;
        for (const Perm perm : {Perm::kABC, Perm::kCBA, Perm::kACB}) {
          int i1 = 0;
          int i2 = 0;
          int i3 = 0;
          Permute(perm, a, b, c, i1, i2, i3);
          const int jo1 = orbits[U(i1)].j2.twice;
          const int jo2 = orbits[U(i2)].j2.twice;
          const int lo = perm == Perm::kABC ? j1 : std::abs(jo1 - jo2) / 2;
          const int hi = perm == Perm::kABC ? j1 : (jo1 + jo2) / 2;
          for (int j1p = lo; j1p <= hi; ++j1p) {
            const std::int32_t col =
                find_state(std::min(i1, i2), std::max(i1, i2), i3, 2 * j1p);
            if (col < 0) continue;
            double v = PermutationPhase(perm) *
                       RecouplingCoefficient(perm, ja, jb, jc, j1p, j1, tj, sj);
            if (i1 > i2) v *= -Phase((jo1 + jo2) / 2 - j1p);
            if (i1 == i2) v *= constants::kSqrt2;
            row.emplace_back(col, v);
          }
        }
        AppendRow(&row, &r_col, &r_val);
        r_ptr.push_back(static_cast<std::int64_t>(r_col.size()));
      }

      // E: one row per state.
      for (const State& s : states) {
        int sa = 0;
        int sb = 0;
        int sc = 0;
        const Perm perm = SortOrbits(s.p, s.q, s.r, sa, sb, sc);
        const int jpq = s.two_jpq / 2;
        for (std::int32_t lr = cv.tr_row_begin[t]; lr < cv.tr_row_end[t];
             ++lr) {
          const int jab = cv.row_two_jab[cv.ch_row_offset[ch] + lr] / 2;
          const double v =
              PermutationPhase(perm) *
              RecouplingCoefficient(perm, ja, jb, jc, jpq, jab, tj, sj);
          row.emplace_back(lr, v);
        }
        AppendRow(&row, &e_col, &e_val);
        e_ptr.push_back(static_cast<std::int64_t>(e_col.size()));
      }
    }
    ch_num[U(ch)] = local;
    flat += local;
  }
  num_states_ = flat;
  ch_pt_offset_ = ToBuffer(ch_off);
  ch_num_pt_ = ToBuffer(ch_num);
  tr_pt_begin_ = ToBuffer(tr_begin);
  tr_pt_end_ = ToBuffer(tr_end);
  pt_p_ = ToBuffer(pp);
  pt_q_ = ToBuffer(pq);
  pt_r_ = ToBuffer(pr);
  pt_two_jpq_ = ToBuffer(pj);
  r_ = Csr::FromHost(r_ptr, r_col, r_val);
  e_ = Csr::FromHost(e_ptr, e_col, e_val);
}

SpaceCounts PairThirdSpace::Counts() const {
  const auto num = ch_num_pt_.host_span();
  std::vector<std::int64_t> dims(num.begin(), num.end());
  return Summarize(dims, static_cast<std::int64_t>(tr_pt_begin_.size()));
}

std::int64_t PairThirdSpace::StateBytes() const noexcept {
  return SizeBytes(ch_pt_offset_) + SizeBytes(ch_num_pt_) +
         SizeBytes(tr_pt_begin_) + SizeBytes(tr_pt_end_) + SizeBytes(pt_p_) +
         SizeBytes(pt_q_) + SizeBytes(pt_r_) + SizeBytes(pt_two_jpq_);
}

PairThirdView PairThirdSpace::host_view() const noexcept {
  PairThirdView v;
  v.ch_pt_offset = ch_pt_offset_.host_span().data();
  v.ch_num_pt = ch_num_pt_.host_span().data();
  v.tr_pt_begin = tr_pt_begin_.host_span().data();
  v.tr_pt_end = tr_pt_end_.host_span().data();
  v.pt_p = pt_p_.host_span().data();
  v.pt_q = pt_q_.host_span().data();
  v.pt_r = pt_r_.host_span().data();
  v.pt_two_jpq = pt_two_jpq_.host_span().data();
  v.r = r_.host_view();
  v.num_channels = num_channels_;
  v.num_states = num_states_;
  return v;
}

PairThirdView PairThirdSpace::device_view() const noexcept {
  PairThirdView v;
  v.ch_pt_offset = ch_pt_offset_.device_data();
  v.ch_num_pt = ch_num_pt_.device_data();
  v.tr_pt_begin = tr_pt_begin_.device_data();
  v.tr_pt_end = tr_pt_end_.device_data();
  v.pt_p = pt_p_.device_data();
  v.pt_q = pt_q_.device_data();
  v.pt_r = pt_r_.device_data();
  v.pt_two_jpq = pt_two_jpq_.device_data();
  v.r = r_.device_view();
  v.num_channels = num_channels_;
  v.num_states = num_states_;
  return v;
}

SpaceCounts PairThirdSpace::CountOnly(const CanonicalSpace& space) {
  const CanonicalView cv = space.host_view();
  const auto orbits = space.model_space().basis().orbits();
  std::vector<std::int64_t> dims(U(cv.num_channels), 0);
  std::vector<State> states;
  for (std::int32_t ch = 0; ch < cv.num_channels; ++ch) {
    for (std::int32_t t = cv.ch_triple_begin[ch]; t < cv.ch_triple_end[ch];
         ++t) {
      states.clear();
      AppendTripleStates(orbits, cv.tr_a[t], cv.tr_b[t], cv.tr_c[t],
                         cv.ch_two_j[ch], &states);
      dims[U(ch)] += static_cast<std::int64_t>(states.size());
    }
  }
  return Summarize(dims, cv.num_triples);
}

}  // namespace imsrg::layout3
