// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/pair_third.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/angmom/three_body_recoupling.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/types.h"

namespace imsrg::op {
namespace {

using angmom::PermutationPhase;
using angmom::Permute;
using angmom::Phase;
using angmom::RecouplingCoefficient;
using angmom::SortOrbits;
using angmom::ThreeBodyPerm;
using angmom::Triangle2;

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
/// O(J range).
void AppendTripleStates(std::span<const basis::Orbit> orbits, int a, int b,
                        int c, int two_j, std::vector<State>* out) {
  constexpr ThreeBodyPerm kPerms[3] = {ThreeBodyPerm::kABC, ThreeBodyPerm::kCBA,
                                       ThreeBodyPerm::kACB};
  int keys[3][3];
  int num_keys = 0;
  for (const ThreeBodyPerm perm : kPerms) {
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

PairThirdCsr PairThirdCsr::FromHost(const std::vector<std::int64_t>& ptr,
                                    const std::vector<std::int32_t>& col,
                                    const std::vector<double>& val) {
  IMSRG_CHECK(!ptr.empty() && col.size() == val.size() &&
                  static_cast<std::size_t>(ptr.back()) == val.size(),
              "inconsistent CSR arrays");
  PairThirdCsr m;
  m.ptr_ = ToBuffer(ptr);
  m.col_ = ToBuffer(col);
  m.val_ = ToBuffer(val);
  return m;
}

CsrView PairThirdCsr::host_view() const noexcept {
  return {ptr_.host_span().data(), col_.host_span().data(),
          val_.host_span().data()};
}

CsrView PairThirdCsr::device_view() const noexcept {
  return {ptr_.device_data(), col_.device_data(), val_.device_data()};
}

std::int64_t PairThirdCsr::Bytes() const noexcept {
  return static_cast<std::int64_t>(ptr_.size() * sizeof(std::int64_t) +
                                   col_.size() * sizeof(std::int32_t) +
                                   val_.size() * sizeof(double));
}

PairThirdSpace::PairThirdSpace(const basis::ThreeBodySpace& space,
                               const basis::SingleParticleBasis& basis,
                               const angmom::SixJ3Table& sixj) {
  IMSRG_CHECK(sixj.emax() >= basis.emax(),
              "SixJ3Table does not cover the emax of the basis");
  const angmom::SixJ3View sj = sixj.host_view();
  const auto orbits = basis.orbits();
  num_channels_ = space.num_channels();
  num_triples_ = space.num_triples();

  std::vector<std::int32_t> ch_off(U(num_channels_));
  std::vector<std::int32_t> ch_num(U(num_channels_));
  std::vector<std::int32_t> tr_begin(U(num_triples_));
  std::vector<std::int32_t> tr_end(U(num_triples_));
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

  for (std::int32_t ch = 0; ch < num_channels_; ++ch) {
    const basis::KeptChannel3& kc = space.channel(Channel3Index{ch});
    const int two_j = kc.two_j.twice;
    const TwoJ tj{two_j};
    ch_off[U(ch)] = static_cast<std::int32_t>(flat);
    std::int32_t local = 0;
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      const basis::OrbitTriple& tr = space.triple(basis::Triple3Index{t});
      const int a = tr.a.value();
      const int b = tr.b.value();
      const int c = tr.c.value();
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
      const std::int32_t row0 = tr.row_begin.value();
      const std::int32_t row1 = tr.row_end.value();

      // R: one row per kept row of the triple (flat row order).
      for (std::int32_t lr = row0; lr < row1; ++lr) {
        const int j1 = space.row_two_jab(kc.row_offset + lr).twice / 2;
        for (const ThreeBodyPerm perm :
             {ThreeBodyPerm::kABC, ThreeBodyPerm::kCBA, ThreeBodyPerm::kACB}) {
          int i1 = 0;
          int i2 = 0;
          int i3 = 0;
          Permute(perm, a, b, c, i1, i2, i3);
          const int jo1 = orbits[U(i1)].j2.twice;
          const int jo2 = orbits[U(i2)].j2.twice;
          const bool abc = perm == ThreeBodyPerm::kABC;
          const int lo = abc ? j1 : std::abs(jo1 - jo2) / 2;
          const int hi = abc ? j1 : (jo1 + jo2) / 2;
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
        const ThreeBodyPerm perm = SortOrbits(s.p, s.q, s.r, sa, sb, sc);
        const int jpq = s.two_jpq / 2;
        for (std::int32_t lr = row0; lr < row1; ++lr) {
          const int jab = space.row_two_jab(kc.row_offset + lr).twice / 2;
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
    max_states_ = std::max(max_states_, local);
    sum_states2_ += static_cast<std::int64_t>(local) * local;
  }
  num_states_ = flat;
  ch_state_offset_ = ToBuffer(ch_off);
  ch_num_states_ = ToBuffer(ch_num);
  tr_state_begin_ = ToBuffer(tr_begin);
  tr_state_end_ = ToBuffer(tr_end);
  st_p_ = ToBuffer(pp);
  st_q_ = ToBuffer(pq);
  st_r_ = ToBuffer(pr);
  st_two_jpq_ = ToBuffer(pj);
  r_ = PairThirdCsr::FromHost(r_ptr, r_col, r_val);
  e_ = PairThirdCsr::FromHost(e_ptr, e_col, e_val);
}

std::int32_t PairThirdSpace::num_states(Channel3Index ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels_,
               "three-body channel index out of range");
  return ch_num_states_.host_span()[ch.ToSize()];
}

std::int32_t PairThirdSpace::state_offset(Channel3Index ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels_,
               "three-body channel index out of range");
  return ch_state_offset_.host_span()[ch.ToSize()];
}

std::int32_t PairThirdSpace::state_begin(basis::Triple3Index t) const {
  IMSRG_ASSERT(t.value() >= 0 && t.value() < num_triples_,
               "triple index out of range");
  return tr_state_begin_.host_span()[t.ToSize()];
}

std::int32_t PairThirdSpace::state_end(basis::Triple3Index t) const {
  IMSRG_ASSERT(t.value() >= 0 && t.value() < num_triples_,
               "triple index out of range");
  return tr_state_end_.host_span()[t.ToSize()];
}

std::int64_t PairThirdSpace::StateBytes() const noexcept {
  return SizeBytes(ch_state_offset_) + SizeBytes(ch_num_states_) +
         SizeBytes(tr_state_begin_) + SizeBytes(tr_state_end_) +
         SizeBytes(st_p_) + SizeBytes(st_q_) + SizeBytes(st_r_) +
         SizeBytes(st_two_jpq_);
}

std::int64_t PairThirdSpace::Bytes() const noexcept {
  return StateBytes() + r_.Bytes() + e_.Bytes();
}

PairThirdView PairThirdSpace::host_view() const noexcept {
  PairThirdView v;
  v.ch_state_offset = ch_state_offset_.host_span().data();
  v.ch_num_states = ch_num_states_.host_span().data();
  v.tr_state_begin = tr_state_begin_.host_span().data();
  v.tr_state_end = tr_state_end_.host_span().data();
  v.st_p = st_p_.host_span().data();
  v.st_q = st_q_.host_span().data();
  v.st_r = st_r_.host_span().data();
  v.st_two_jpq = st_two_jpq_.host_span().data();
  v.r = r_.host_view();
  v.e = e_.host_view();
  v.num_channels = num_channels_;
  v.num_triples = num_triples_;
  v.num_states = num_states_;
  return v;
}

PairThirdView PairThirdSpace::device_view() const noexcept {
  PairThirdView v;
  v.ch_state_offset = ch_state_offset_.device_data();
  v.ch_num_states = ch_num_states_.device_data();
  v.tr_state_begin = tr_state_begin_.device_data();
  v.tr_state_end = tr_state_end_.device_data();
  v.st_p = st_p_.device_data();
  v.st_q = st_q_.device_data();
  v.st_r = st_r_.device_data();
  v.st_two_jpq = st_two_jpq_.device_data();
  v.r = r_.device_view();
  v.e = e_.device_view();
  v.num_channels = num_channels_;
  v.num_triples = num_triples_;
  v.num_states = num_states_;
  return v;
}

}  // namespace imsrg::op
