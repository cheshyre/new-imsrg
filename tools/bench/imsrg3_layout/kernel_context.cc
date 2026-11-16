// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/kernel_context.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/detail/workspace_impl.h"

namespace imsrg::layout3 {

namespace {

template <class T>
exec::DeviceBuffer<T> ToBuffer(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> buf(std::max<std::size_t>(v.size(), 1), label);
  if (!v.empty()) {
    std::copy(v.begin(), v.end(), buf.host_span().begin());
    buf.SyncToDevice();
  }
  return buf;
}

}  // namespace

KernelContext::KernelContext(const CanonicalSpace& space,
                             const PairThirdSpace& pt, const SixJ3Table& sixj)
    : space_(&space), pt_(&pt), sixj_(&sixj) {
  IMSRG_CHECK(sixj.emax() >= space.model_space().basis().emax(),
              "KernelContext: the 6j table does not cover the basis emax");
  ws_ = std::make_unique<commutator::Workspace>(space.model_space_ptr());

  // Orbit groups by (parity, tz), ascending orbit index inside a group.
  const auto orbits = space.model_space().basis().orbits();
  std::vector<std::int32_t> offset(5, 0);
  std::vector<std::int32_t> list;
  for (std::int32_t g = 0; g < 4; ++g) {
    for (std::size_t i = 0; i < orbits.size(); ++i) {
      if (OrbitGroupView::Group(orbits[i].l % 2, orbits[i].tz2.twice) == g) {
        list.push_back(static_cast<std::int32_t>(i));
      }
    }
    offset[static_cast<std::size_t>(g) + 1] =
        static_cast<std::int32_t>(list.size());
  }
  group_offset_ = ToBuffer(offset, "imsrg3_layout::group_offset");
  group_orbits_ = ToBuffer(list, "imsrg3_layout::group_orbits");

  // K1 work order: the tile work of bra triple t is about m_t times the sum
  // of m_t' over the ket triples t' <= t of its channel (m = pair-third
  // states); largest first, ties in triple order.
  const CanonicalView cv = space.host_view();
  const PairThirdView pv = pt.host_view();
  std::vector<double> cost(static_cast<std::size_t>(cv.num_triples), 0.0);
  for (std::int32_t c = 0; c < cv.num_channels; ++c) {
    double running = 0.0;
    for (std::int32_t t = cv.ch_triple_begin[c]; t < cv.ch_triple_end[c]; ++t) {
      const double m = pv.tr_pt_end[t] - pv.tr_pt_begin[t];
      running += m;
      cost[static_cast<std::size_t>(t)] = m * running;
    }
  }
  std::vector<std::int32_t> order(static_cast<std::size_t>(cv.num_triples));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::int32_t x, std::int32_t y) {
                     return cost[static_cast<std::size_t>(x)] >
                            cost[static_cast<std::size_t>(y)];
                   });
  work_order_ = ToBuffer(order, "imsrg3_layout::work_order");
}

KernelContext::~KernelContext() = default;

commutator::detail::TwoBodyView KernelContext::TwoBody(
    const op::Operator& op) const {
  IMSRG_CHECK(ws_->Compatible(op),
              "KernelContext::TwoBody: operator on another model space");
  return ws_->impl().two_body_view(op);
}

basis::OrbitTableView KernelContext::Orbits() const noexcept {
  return ws_->impl().tables.orbits();
}

OrbitGroupView KernelContext::Groups() const noexcept {
  return OrbitGroupView{group_offset_.device_data(),
                        group_orbits_.device_data()};
}

const std::int32_t* KernelContext::WorkOrder() const noexcept {
  return work_order_.device_data();
}

int CommutatorHermiticity(const op::Operator& X, const op::Operator& Y) {
  const auto sign = [](Hermiticity h) {
    IMSRG_CHECK(h != Hermiticity::kNonHermitian,
                "CommutatorHermiticity: operands need a definite hermiticity");
    return h == Hermiticity::kHermitian ? 1 : -1;
  };
  return -sign(X.spec().hermiticity) * sign(Y.spec().hermiticity);
}

}  // namespace imsrg::layout3
