// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/three_body.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator {

namespace {

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> buf(v.size(), label);
  if (!v.empty()) buf.CopyFromHost(std::span<const T>(v));
  return buf;
}

template <class T>
std::int64_t BufferBytes(const exec::DeviceBuffer<T>& b) {
  return static_cast<std::int64_t>(b.size() * sizeof(T));
}

/// Bytes `basis::DeviceTables(ms)` uploads (orbit, ket, channel and subset
/// tables; device_tables.cc). O(1).
std::int64_t DeviceTablesBytes(const basis::ModelSpace& ms) {
  const auto& ch = ms.two_body();
  const auto& sub = ms.two_body_subsets();
  const std::int64_t i4 = sizeof(std::int32_t);
  const std::int64_t f8 = sizeof(double);
  const std::int64_t norb = ms.num_orbits();
  return norb * (5 * i4 + f8) + 2 * i4 * ch.num_kets() +
         5 * i4 * ch.num_channels() +
         i4 * static_cast<std::int64_t>(
                  ch.channel_ket_table().size() + ch.ket_J_offsets().size() +
                  ch.local_index_table().size() + sub.subset_offsets().size() +
                  sub.subset_table().size()) +
         f8 * static_cast<std::int64_t>(sub.occupation_product_table().size() +
                                        sub.vacancy_product_table().size());
}

/// Indices sorted by `cost`, largest first (stable: ties in index order).
std::vector<std::int32_t> ByCost(const std::vector<double>& cost) {
  std::vector<std::int32_t> order(cost.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&cost](std::int32_t x, std::int32_t y) {
                     return cost[U(x)] > cost[U(y)];
                   });
  return order;
}

/// Group of the orbit `o` (see `detail::OrbitGroupsView`).
std::int32_t GroupOfOrbit(const basis::Orbit& o) {
  return detail::OrbitGroupsView::GroupOf(o.l % 2, o.tz2.twice);
}

/// Hermiticity of [X, Y] for the three-body part (CommutatorSpec rules, but
/// for any particle rank of the operands).
Hermiticity CommutatorHermiticity(const op::Operator& X,
                                  const op::Operator& Y) {
  if (X.hermiticity() == Hermiticity::kNonHermitian ||
      Y.hermiticity() == Hermiticity::kNonHermitian) {
    return Hermiticity::kNonHermitian;
  }
  return X.hermiticity() == Y.hermiticity() ? Hermiticity::kAntiHermitian
                                            : Hermiticity::kHermitian;
}

/// The operand checks of `Comm223` and `PerturbativeTriplesEnergy`.
void CheckOperand(const op::Operator& op, const char* name,
                  const ThreeBodyContext& ctx) {
  const std::string who = std::string("Comm223: ") + name;
  if (op.empty()) throw InputError(who + " is an empty operator");
  if (!op.spec().is_scalar() || !op.spec().is_number_conserving()) {
    throw NotImplementedError(who +
                              " must be a scalar, number-conserving operator");
  }
  if (!op.two_body().allocated()) {
    throw InputError(who + " has no two-body part (particle rank >= 2 needed)");
  }
  const basis::ModelSpace& ms = op.model_space();
  const basis::ModelSpace& ms3 = *ctx.model_space_ptr();
  IMSRG_CHECK(&ms == &ms3 || (ms.basis() == ms3.basis() &&
                              ms.reference() == ms3.reference() &&
                              ms.partition() == ms3.partition()),
              who +
                  " lives on a model space whose basis, reference or "
                  "partition differs from the context's");
  const auto& offsets = ctx.impl().two_body_offset_host;
  const auto& m = op.two_body().matrix();
  IMSRG_CHECK(
      m.num_blocks() + 1 == offsets.size() && m.total_size() == offsets.back(),
      who + ": two-body layout differs from the context's");
}

}  // namespace

ThreeBodyContext::Impl::Impl(
    const std::shared_ptr<const basis::ModelSpace>& ms3)
    : tables(*ms3),
      sixj(ms3->basis().emax()),
      pair_third(*ms3->three_body_space(), ms3->basis(), sixj) {
  const basis::ThreeBodySpace& space = *ms3->three_body_space();
  const auto orbits = ms3->basis().orbits();
  const op::PairThirdView pv = pair_third.host_view();

  // Two-body block offsets as op::TwoBody stores a scalar operator.
  const auto& tbc = ms3->two_body();
  two_body_offset_host.assign(U(tbc.num_channels()) + 1, 0);
  for (std::int32_t ch = 0; ch < tbc.num_channels(); ++ch) {
    const std::int64_t n = tbc.channel(ChannelIndex{ch}).num_kets;
    two_body_offset_host[U(ch) + 1] = two_body_offset_host[U(ch)] + n * n;
  }
  two_body_offset =
      Upload(two_body_offset_host, "commutator::comm223_two_body_offset");

  // Kept canonical space.
  const auto channels = space.channels();
  const std::size_t nc = channels.size();
  std::vector<std::int32_t> c_two_j(nc), c_dim(nc), c_tb(nc), c_te(nc);
  std::vector<std::int64_t> c_row(nc);
  packed_offset_host.assign(nc + 1, 0);
  dense_offset_host.assign(nc + 1, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    const basis::KeptChannel3& kc = channels[c];
    c_two_j[c] = kc.two_j.twice;
    c_dim[c] = kc.dim;
    c_tb[c] = kc.triple_begin.value();
    c_te[c] = kc.triple_end.value();
    c_row[c] = kc.row_offset;
    packed_offset_host[c + 1] =
        packed_offset_host[c] + op::ThreeBody::BlockSize(kc.dim, true);
    dense_offset_host[c + 1] =
        dense_offset_host[c] + op::ThreeBody::BlockSize(kc.dim, false);
  }
  ch_two_j = Upload(c_two_j, "commutator::comm223_ch_two_j");
  ch_dim = Upload(c_dim, "commutator::comm223_ch_dim");
  ch_triple_begin = Upload(c_tb, "commutator::comm223_ch_triple_begin");
  ch_triple_end = Upload(c_te, "commutator::comm223_ch_triple_end");
  ch_row_offset = Upload(c_row, "commutator::comm223_ch_row_offset");
  ch_packed_offset =
      Upload(packed_offset_host, "commutator::comm223_ch_packed_offset");
  ch_dense_offset =
      Upload(dense_offset_host, "commutator::comm223_ch_dense_offset");

  const auto triples = space.triples();
  num_triples = static_cast<std::int32_t>(triples.size());
  const std::size_t nt = triples.size();
  std::vector<std::int32_t> t_a(nt), t_b(nt), t_c(nt), t_ch(nt), t_rb(nt),
      t_re(nt), t_sym(nt);
  for (std::size_t t = 0; t < nt; ++t) {
    const basis::OrbitTriple& tr = triples[t];
    t_a[t] = tr.a.value();
    t_b[t] = tr.b.value();
    t_c[t] = tr.c.value();
    t_ch[t] = tr.channel.value();
    t_rb[t] = tr.row_begin.value();
    t_re[t] = tr.row_end.value();
    t_sym[t] = tr.sym;
  }
  tr_a = Upload(t_a, "commutator::comm223_tr_a");
  tr_b = Upload(t_b, "commutator::comm223_tr_b");
  tr_c = Upload(t_c, "commutator::comm223_tr_c");
  tr_channel = Upload(t_ch, "commutator::comm223_tr_channel");
  tr_row_begin = Upload(t_rb, "commutator::comm223_tr_row_begin");
  tr_row_end = Upload(t_re, "commutator::comm223_tr_row_end");
  tr_sym = Upload(t_sym, "commutator::comm223_tr_sym");

  // Orbit groups by (parity, 2tz), sorted by (2j, index), with the j blocks.
  nj = ms3->basis().emax() + 1;
  std::vector<std::int32_t> g_off(5, 0);
  std::vector<std::int32_t> g_orb;
  std::vector<std::int32_t> g_jpos(U(4 * (nj + 1)), 0);
  for (std::int32_t g = 0; g < 4; ++g) {
    std::vector<std::int32_t> members;
    for (std::size_t i = 0; i < orbits.size(); ++i) {
      if (GroupOfOrbit(orbits[i]) == g) {
        members.push_back(static_cast<std::int32_t>(i));
      }
    }
    std::stable_sort(members.begin(), members.end(),
                     [&orbits](std::int32_t x, std::int32_t y) {
                       return orbits[U(x)].j2.twice < orbits[U(y)].j2.twice;
                     });
    for (std::int32_t ij = 0; ij <= nj; ++ij) {
      std::int32_t pos = 0;
      while (pos < static_cast<std::int32_t>(members.size()) &&
             (orbits[U(members[U(pos)])].j2.twice - 1) / 2 < ij) {
        ++pos;
      }
      g_jpos[U(g * (nj + 1) + ij)] = pos;
    }
    g_orb.insert(g_orb.end(), members.begin(), members.end());
    g_off[U(g) + 1] = static_cast<std::int32_t>(g_orb.size());
    ext.s_max = std::max(ext.s_max, static_cast<std::int32_t>(members.size()));
  }
  group_offset = Upload(g_off, "commutator::comm223_group_offset");
  group_orbits = Upload(g_orb, "commutator::comm223_group_orbits");
  group_j_pos = Upload(g_jpos, "commutator::comm223_group_j_pos");

  // Tile extents and work orders. Estimated work of bra triple t: m_t times
  // the summed m of its ket triples (m = pair-third states).
  std::vector<double> cost_lower(nt, 0.0);
  std::vector<double> cost_full(nt, 0.0);
  std::vector<double> cost_triples(nt, 0.0);
  const auto occ = ms3->reference().occupations();
  // Host copies of the occupation tests of the triples kernel, with half the
  // threshold: a triple listed here without work is skipped by the kernel,
  // which applies imsrg++'s exact test.
  const auto occupation = [&](std::size_t t, double& o, double& u) {
    const double na = occ[U(t_a[t])];
    const double nb = occ[U(t_b[t])];
    const double nc3 = occ[U(t_c[t])];
    o = na * nb * nc3;
    u = (1.0 - na) * (1.0 - nb) * (1.0 - nc3);
  };
  const double loose = 0.5 * detail::kTriplesSkip;
  for (std::size_t c = 0; c < nc; ++c) {
    double sum_all = 0.0;
    for (std::int32_t t = c_tb[c]; t < c_te[c]; ++t) {
      sum_all += pv.tr_state_end[t] - pv.tr_state_begin[t];
    }
    double running = 0.0;
    for (std::int32_t t = c_tb[c]; t < c_te[c]; ++t) {
      const double m = pv.tr_state_end[t] - pv.tr_state_begin[t];
      running += m;
      cost_lower[U(t)] = m * running;
      cost_full[U(t)] = m * sum_all;
      ext.m_max = std::max(ext.m_max, static_cast<std::int32_t>(m));
      ext.k_max = std::max(ext.k_max, t_re[U(t)] - t_rb[U(t)]);
      double o_t = 0.0;
      double u_t = 0.0;
      occupation(U(t), o_t, u_t);
      if (o_t < loose && u_t < loose) continue;
      double work = 0.0;
      for (std::int32_t tp = c_tb[c]; tp < t; ++tp) {
        double o_p = 0.0;
        double u_p = 0.0;
        occupation(U(tp), o_p, u_p);
        if (std::abs(o_t * u_p - o_p * u_t) < loose) continue;
        work += pv.tr_state_end[tp] - pv.tr_state_begin[tp];
      }
      cost_triples[U(t)] = m * work;
    }
  }
  order_lower = Upload(ByCost(cost_lower), "commutator::comm223_order_lower");
  order_full = Upload(ByCost(cost_full), "commutator::comm223_order_full");
  const std::vector<std::int32_t> tri = ByCost(cost_triples);
  triples_league = static_cast<std::int32_t>(
      std::count_if(cost_triples.begin(), cost_triples.end(),
                    [](double x) { return x > 0.0; }));
  order_triples = Upload(tri, "commutator::comm223_order_triples");

  scratch_bytes = num_triples > 0 ? detail::TeamScratchBytes(ext) : 0;
  table_bytes = DeviceTablesBytes(*ms3) + sixj.size_bytes() +
                pair_third.Bytes() + BufferBytes(two_body_offset) +
                BufferBytes(ch_two_j) + BufferBytes(ch_dim) +
                BufferBytes(ch_triple_begin) + BufferBytes(ch_triple_end) +
                BufferBytes(ch_row_offset) + BufferBytes(ch_packed_offset) +
                BufferBytes(ch_dense_offset) + BufferBytes(tr_a) +
                BufferBytes(tr_b) + BufferBytes(tr_c) +
                BufferBytes(tr_channel) + BufferBytes(tr_row_begin) +
                BufferBytes(tr_row_end) + BufferBytes(tr_sym) +
                BufferBytes(group_offset) + BufferBytes(group_orbits) +
                BufferBytes(group_j_pos) + BufferBytes(order_lower) +
                BufferBytes(order_full) + BufferBytes(order_triples);
}

detail::TileViews ThreeBodyContext::Impl::Views() const {
  detail::TileViews v;
  v.sp.ch_two_j = ch_two_j.device_data();
  v.sp.ch_dim = ch_dim.device_data();
  v.sp.ch_row_offset = ch_row_offset.device_data();
  v.sp.ch_triple_begin = ch_triple_begin.device_data();
  v.sp.ch_triple_end = ch_triple_end.device_data();
  v.sp.ch_packed_offset = ch_packed_offset.device_data();
  v.sp.ch_dense_offset = ch_dense_offset.device_data();
  v.sp.tr_a = tr_a.device_data();
  v.sp.tr_b = tr_b.device_data();
  v.sp.tr_c = tr_c.device_data();
  v.sp.tr_channel = tr_channel.device_data();
  v.sp.tr_row_begin = tr_row_begin.device_data();
  v.sp.tr_row_end = tr_row_end.device_data();
  v.sp.tr_sym = tr_sym.device_data();
  v.sp.num_channels = static_cast<std::int32_t>(ch_dim.size());
  v.sp.num_triples = num_triples;
  v.pt = pair_third.device_view();
  v.sj = sixj.device_view();
  v.ob = tables.orbits();
  v.tb = tables.two_body();
  v.groups.offset = group_offset.device_data();
  v.groups.orbits = group_orbits.device_data();
  v.groups.j_pos = group_j_pos.device_data();
  v.groups.nj = nj;
  v.two_body_offset = two_body_offset.device_data();
  // Strides of SixJ3View's layout ((((i1 nj + i2) nJ + J12) nj + i3) nJ3 + iJ)
  // nJ + J23.
  const std::int64_t e = sixj.emax();
  const std::int64_t n_j = e + 1;
  const std::int64_t n_J = 2 * e + 2;
  const std::int64_t n_J3 = 3 * e + 2;
  v.c5 = n_J;
  v.c4 = n_J3 * n_J;
  v.c3 = n_j * v.c4;
  v.c2 = n_J * v.c3;
  v.c1 = n_j * v.c2;
  v.ext = ext;
  return v;
}

ThreeBodyContext::ThreeBodyContext(std::shared_ptr<const basis::ModelSpace> ms3)
    : ms_(std::move(ms3)) {
  if (ms_ == nullptr) throw InputError("ThreeBodyContext: null model space");
  if (ms_->three_body_space() == nullptr) {
    throw InputError(
        "ThreeBodyContext: the model space has no three-body space (build it "
        "with ModelSpace::WithThreeBody)");
  }
  impl_ = std::make_unique<Impl>(ms_);
}

ThreeBodyContext::ThreeBodyContext(ThreeBodyContext&&) noexcept = default;
ThreeBodyContext& ThreeBodyContext::operator=(ThreeBodyContext&&) noexcept =
    default;
ThreeBodyContext::~ThreeBodyContext() = default;

std::int64_t ThreeBodyContext::TableBytes() const noexcept {
  return impl_ != nullptr ? impl_->table_bytes : 0;
}

std::int64_t ThreeBodyContext::ScratchBytesPerTeam() const noexcept {
  return impl_ != nullptr ? impl_->scratch_bytes : 0;
}

const ThreeBodyContext::Impl& ThreeBodyContext::impl() const {
  IMSRG_CHECK(impl_ != nullptr, "ThreeBodyContext: moved-from context");
  return *impl_;
}

op::ThreeBody Comm223(const op::Operator& X, const op::Operator& Y,
                      const ThreeBodyContext& ctx) {
  const ThreeBodyContext::Impl& impl = ctx.impl();
  CheckOperand(X, "X", ctx);
  CheckOperand(Y, "Y", ctx);
  const Hermiticity h = CommutatorHermiticity(X, Y);
  op::ThreeBody z(ctx.model_space_ptr(), h);
  const std::vector<std::int64_t>& expected =
      z.packed() ? impl.packed_offset_host : impl.dense_offset_host;
  IMSRG_CHECK(std::equal(z.block_offsets().begin(), z.block_offsets().end(),
                         expected.begin(), expected.end()),
              "Comm223: three-body block layout differs from the context's");
  detail::TileViews v = impl.Views();
  v.x2 = X.two_body().matrix().buffer().device_data();
  v.y2 = Y.two_body().matrix().buffer().device_data();
  v.z = z.device_data();
  v.zero_diagonal = h == Hermiticity::kAntiHermitian ? 1 : 0;
  if (z.packed()) {
    detail::RunComm223Tiles(v, detail::TileMode::kLowerPacked,
                            impl.order_lower.device_data(), impl.num_triples);
  } else {
    detail::RunComm223Tiles(v, detail::TileMode::kFullDense,
                            impl.order_full.device_data(), impl.num_triples);
  }
  return z;
}

double PerturbativeTriplesEnergy(const op::Operator& X, const op::Operator& Y,
                                 std::span<const double> epsilon,
                                 const ThreeBodyContext& ctx) {
  const ThreeBodyContext::Impl& impl = ctx.impl();
  CheckOperand(X, "X", ctx);
  CheckOperand(Y, "Y", ctx);
  const std::int32_t norb = ctx.model_space_ptr()->num_orbits();
  if (static_cast<std::int64_t>(epsilon.size()) != norb) {
    throw InputError("PerturbativeTriplesEnergy: epsilon has " +
                     std::to_string(epsilon.size()) + " values, expected " +
                     std::to_string(norb) + " (one per orbit)");
  }
  if (impl.triples_league == 0) return 0.0;
  exec::DeviceBuffer<double> eps(epsilon.size(), "commutator::triples_eps");
  eps.CopyFromHost(epsilon);
  detail::TileViews v = impl.Views();
  v.x2 = X.two_body().matrix().buffer().device_data();
  v.y2 = Y.two_body().matrix().buffer().device_data();
  v.epsilon = eps.device_data();
  return detail::RunTriplesTiles(v, impl.order_triples.device_data(),
                                 impl.triples_league);
}

}  // namespace imsrg::commutator
