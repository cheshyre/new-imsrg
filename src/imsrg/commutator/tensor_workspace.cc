// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Host construction of the scalar-tensor commutator tables
// (`detail::TensorContext`), the spec rule and the driver
// `TensorCommutator`.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/commutator/detail/tensor_impl.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::commutator {

namespace detail {

Csr MakeCsr(const std::vector<std::vector<std::int32_t>>& rows) {
  std::vector<std::int32_t> off(rows.size() + 1, 0);
  std::vector<std::int32_t> items;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    off[i + 1] = off[i] + static_cast<std::int32_t>(rows[i].size());
    items.insert(items.end(), rows[i].begin(), rows[i].end());
  }
  Csr c;
  c.offsets = exec::DeviceBuffer<std::int32_t>(off.size(), "imsrg::Csr::off");
  c.offsets.CopyFromHost(off);
  c.items = exec::DeviceBuffer<std::int32_t>(items.size(), "imsrg::Csr::items");
  if (!items.empty()) c.items.CopyFromHost(items);
  return c;
}

namespace {

exec::DeviceBuffer<std::int32_t> ToDevice(const std::vector<std::int32_t>& v,
                                          const char* label) {
  exec::DeviceBuffer<std::int32_t> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(v);
  return b;
}

}  // namespace

TensorContext::TensorContext(std::shared_ptr<const basis::ModelSpace> ms_in)
    : ms(std::move(ms_in)), tables(*ms), sixj(ms->basis().emax()) {
  const auto& basis = ms->basis();
  const std::int32_t n = basis.num_orbits();
  std::vector<std::vector<std::int32_t>> rows(static_cast<std::size_t>(n));
  std::vector<std::int32_t> h;
  std::vector<std::int32_t> p;
  for (std::int32_t a = 0; a < n; ++a) {
    for (const OrbitIndex b : basis.SameLjtz(OrbitIndex{a}))
      rows[static_cast<std::size_t>(a)].push_back(b.value());
    if (ms->reference().occupation(OrbitIndex{a}) >= basis::kOccupationCut)
      h.push_back(a);
    else
      p.push_back(a);
  }
  scalar_ch1 = MakeCsr(rows);
  holes = ToDevice(h, "imsrg::TensorContext::holes");
  particles = ToDevice(p, "imsrg::TensorContext::particles");
  num_holes = static_cast<std::int32_t>(h.size());
  num_particles = static_cast<std::int32_t>(p.size());
}

TensorRankData& TensorContext::rank(const op::OperatorSpec& spec) {
  const auto key = std::make_tuple(spec.rank_J, static_cast<int>(spec.parity),
                                   spec.delta_Tz);
  auto it = ranks.find(key);
  if (it != ranks.end()) return *it->second;
  const auto& basis = ms->basis();
  const std::int32_t n = basis.num_orbits();
  std::vector<std::vector<std::int32_t>> rows(static_cast<std::size_t>(n));
  for (std::int32_t a = 0; a < n; ++a)
    for (std::int32_t b = 0; b < n; ++b)
      if (op::ConnectsOrbits(spec, basis.orbit(OrbitIndex{a}),
                             basis.orbit(OrbitIndex{b})))
        rows[static_cast<std::size_t>(a)].push_back(b);
  auto data = std::make_unique<TensorRankData>(
      TensorRankData{angmom::LambdaSixJTable(basis.emax(), spec.rank_J),
                     nullptr, MakeCsr(rows)});
  auto& ref = *data;
  ranks.emplace(key, std::move(data));
  return ref;
}

const angmom::NineJTable& TensorContext::ninej(const op::OperatorSpec& spec,
                                               double threshold) {
  TensorRankData& r = rank(spec);
  if (!r.ninej || r.ninej->skip_threshold() != threshold) {
    r.ninej = std::make_unique<angmom::NineJTable>(ms->basis().emax(),
                                                   spec.rank_J, threshold);
  }
  return *r.ninej;
}

const CrossCoupled& TensorContext::cross_coupled() {
  if (cc_built) return cc;
  const auto& basis = ms->basis();
  const std::int32_t n = basis.num_orbits();
  const std::int32_t jmax = basis.max_two_body_J();
  cc.max_J = jmax;
  cc.num_channels = 4 * (jmax + 1);
  std::vector<std::vector<std::pair<int, int>>> pairs(
      static_cast<std::size_t>(cc.num_channels));
  for (std::int32_t a = 0; a < n; ++a)
    for (std::int32_t b = 0; b < n; ++b) {
      const auto& oa = basis.orbit(OrbitIndex{a});
      const auto& ob = basis.orbit(OrbitIndex{b});
      const int par = (oa.l + ob.l) % 2;
      const int t = oa.tz2 == ob.tz2 ? 0 : 1;
      for (int J = std::abs(oa.j2.twice - ob.j2.twice) / 2;
           J <= (oa.j2.twice + ob.j2.twice) / 2; ++J)
        pairs[static_cast<std::size_t>(4 * J + 2 * t + par)].push_back({a, b});
    }
  cc.J.resize(static_cast<std::size_t>(cc.num_channels));
  cc.parity.resize(cc.J.size());
  cc.tz.resize(cc.J.size());
  cc.offset.assign(cc.J.size() + 1, 0);
  cc.local.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) *
                      static_cast<std::size_t>(jmax + 1),
                  -1);
  for (std::int32_t c = 0; c < cc.num_channels; ++c) {
    const auto& pc = pairs[static_cast<std::size_t>(c)];
    cc.J[static_cast<std::size_t>(c)] = c / 4;
    cc.tz[static_cast<std::size_t>(c)] = (c / 2) % 2;
    cc.parity[static_cast<std::size_t>(c)] = c % 2;
    cc.offset[static_cast<std::size_t>(c) + 1] =
        cc.offset[static_cast<std::size_t>(c)] +
        static_cast<std::int32_t>(pc.size());
    for (std::size_t k = 0; k < pc.size(); ++k) {
      cc.pair_a.push_back(pc[k].first);
      cc.pair_b.push_back(pc[k].second);
      cc.local[(static_cast<std::size_t>(pc[k].first) *
                    static_cast<std::size_t>(n) +
                static_cast<std::size_t>(pc[k].second)) *
                   static_cast<std::size_t>(jmax + 1) +
               static_cast<std::size_t>(c / 4)] = static_cast<std::int32_t>(k);
    }
  }
  cc.d_offset = ToDevice(cc.offset, "imsrg::CrossCoupled::offset");
  cc.d_pair_a = ToDevice(cc.pair_a, "imsrg::CrossCoupled::a");
  cc.d_pair_b = ToDevice(cc.pair_b, "imsrg::CrossCoupled::b");
  cc.d_local = ToDevice(cc.local, "imsrg::CrossCoupled::local");
  cc_built = true;
  return cc;
}

exec::DeviceBuffer<std::int64_t> BlockOffsetTable(const op::TwoBody& part,
                                                  std::int32_t num_channels) {
  std::vector<std::int64_t> t(static_cast<std::size_t>(num_channels) *
                                  static_cast<std::size_t>(num_channels),
                              -1);
  if (part.allocated()) {
    for (const auto& info : part.matrix().blocks()) {
      if (info.rows == 0 || info.cols == 0) continue;
      t[static_cast<std::size_t>(info.row_block.value()) *
            static_cast<std::size_t>(num_channels) +
        static_cast<std::size_t>(info.col_block.value())] = info.offset;
    }
  }
  exec::DeviceBuffer<std::int64_t> b(t.size(), "imsrg::BlockOffsetTable");
  b.CopyFromHost(t);
  return b;
}

TensorTwoBodyView MakeTensorView(const op::TwoBody& part,
                                 const basis::TwoBodyTableView& tb,
                                 const exec::DeviceBuffer<std::int64_t>& offs) {
  TensorTwoBodyView v;
  v.tb = tb;
  v.block_offset = offs.device_data();
  v.data = part.allocated() ? part.matrix().buffer().device_data() : nullptr;
  v.herm = op::HermiticitySign(part.hermiticity()) >= 0 ? 1 : -1;
  return v;
}

}  // namespace detail

TensorWorkspace::TensorWorkspace(std::shared_ptr<const basis::ModelSpace> ms)
    : ms_(std::move(ms)) {
  IMSRG_CHECK(ms_ != nullptr, "TensorWorkspace: null model space");
  ctx_ = std::make_unique<detail::TensorContext>(ms_);
}
TensorWorkspace::TensorWorkspace(TensorWorkspace&&) noexcept = default;
TensorWorkspace& TensorWorkspace::operator=(TensorWorkspace&&) noexcept =
    default;
TensorWorkspace::~TensorWorkspace() = default;

bool TensorWorkspace::Compatible(const op::Operator& op) const noexcept {
  if (op.empty()) return false;
  return op.model_space_ptr() == ms_ || op.model_space() == *ms_;
}

op::OperatorSpec TensorCommutatorSpec(const op::OperatorSpec& x,
                                      const op::OperatorSpec& y) {
  if (!x.is_scalar() || !x.is_number_conserving() || x.particle_rank > 2) {
    throw NotImplementedError(
        "TensorCommutator: X must be a scalar, number-conserving operator of "
        "particle rank <= 2");
  }
  if (y.is_scalar() || !y.is_number_conserving()) {
    throw NotImplementedError(
        "TensorCommutator: Y must be a number-conserving tensor operator");
  }
  if (x.hermiticity == Hermiticity::kNonHermitian ||
      y.hermiticity == Hermiticity::kNonHermitian) {
    throw NotImplementedError(
        "TensorCommutator: X and Y must be Hermitian or anti-Hermitian");
  }
  op::OperatorSpec z = y;
  z.particle_rank = std::max(x.particle_rank, y.particle_rank);
  z.hermiticity = x.hermiticity == y.hermiticity ? Hermiticity::kAntiHermitian
                                                 : Hermiticity::kHermitian;
  return z;
}

void TensorCommutator(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, const TensorOptions& opts,
                      TensorWorkspace& ws) {
  IMSRG_CHECK(ws.Compatible(X) && ws.Compatible(Y),
              "TensorCommutator: X and Y must be on the workspace's model "
              "space");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "TensorCommutator: Z aliases an input");
  const op::OperatorSpec zs = TensorCommutatorSpec(X.spec(), Y.spec());
  if (!Z.empty() && Z.spec() == zs && ws.Compatible(Z)) {
    Z.set_zero();
  } else {
    Z = op::Operator(X.model_space_ptr(), zs);
  }
  using tensor_terms::Comm111st;
  using tensor_terms::Comm121st;
  using tensor_terms::Comm122st;
  using tensor_terms::Comm222Phst;
  using tensor_terms::Comm222PpHh221st;
  if (opts.comm111) Comm111st(X, Y, Z, ws);
  if (opts.comm121) Comm121st(X, Y, Z, ws);
  if (opts.comm122) Comm122st(X, Y, Z, ws);
  if (opts.comm221 || opts.comm222_pp_hh)
    Comm222PpHh221st(X, Y, Z, ws, opts.comm221, opts.comm222_pp_hh);
  if (opts.comm222_ph) Comm222Phst(X, Y, Z, ws, opts.ninej_threshold);
}

op::Operator TensorCommutator(const op::Operator& X, const op::Operator& Y,
                              const TensorOptions& opts, TensorWorkspace& ws) {
  op::Operator Z;
  TensorCommutator(X, Y, Z, opts, ws);
  return Z;
}

}  // namespace imsrg::commutator
