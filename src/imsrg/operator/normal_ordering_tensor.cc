// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Normal ordering of tensor operators (reduced elements, M15): imsrg++
// Operator::DoNormalOrdering2, the reduced branch (Operator.cc:526-560).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/angmom/tensor_tables.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::op {

Operator NormalOrderTensor(const Operator& op,
                           std::span<const double> occupations,
                           NormalOrderDirection direction,
                           const TensorNormalOrderOptions& options) {
  IMSRG_CHECK(!op.empty(), "normal_order_tensor: empty operator");
  if (op.spec().is_scalar()) {
    throw InvariantError(
        "normal_order_tensor: scalar operator (use NormalOrder)");
  }
  const basis::ModelSpace& ms = op.model_space();
  const basis::SingleParticleBasis& sp = ms.basis();
  const std::int32_t norb = sp.num_orbits();
  IMSRG_CHECK(static_cast<std::int32_t>(occupations.size()) == norb,
              "normal_order_tensor: one occupation per orbit required");
  Operator result = op;
  result.SyncToHost();
  if (!result.two_body().allocated()) return result;
  const double s = direction == NormalOrderDirection::kToReference ? 1.0 : -1.0;
  const int L = op.spec().rank_J;
  const angmom::LambdaSixJTable sixj_table(sp.emax(), L);
  const angmom::LambdaSixJView sixj = sixj_table.host_view();
  const TwoBody& v = result.two_body();
  const auto& tb = ms.two_body();
  std::vector<std::int32_t> holes;
  for (std::int32_t h = 0; h < norb; ++h)
    if (occupations[static_cast<std::size_t>(h)] >= basis::kOccupationCut)
      holes.push_back(h);
  std::vector<double> delta(
      static_cast<std::size_t>(norb) * static_cast<std::size_t>(norb), 0.0);
  for (std::int32_t a = 0; a < norb; ++a) {
    const basis::Orbit& oa = sp.orbit(OrbitIndex{a});
    const int ja = oa.j2.twice;
    for (std::int32_t b = 0; b < norb; ++b) {
      const basis::Orbit& ob = sp.orbit(OrbitIndex{b});
      if (!ConnectsOrbits(op.spec(), oa, ob)) continue;
      const int jb = ob.j2.twice;
      double sum = 0.0;
      for (const std::int32_t h : holes) {
        const basis::Orbit& oh = sp.orbit(OrbitIndex{h});
        const int jh = oh.j2.twice;
        const double nh = occupations[static_cast<std::size_t>(h)];
        for (int Jb = std::abs(ja - jh) / 2; Jb <= (ja + jh) / 2; ++Jb) {
          for (int Jk = std::abs(jb - jh) / 2; Jk <= (jb + jh) / 2; ++Jk) {
            if (L < std::abs(Jb - Jk) || L > Jb + Jk) continue;
            if (options.fork79e9330_diagonal_bug && a == b && Jb != Jk) {
              // The fork sums only stored blocks (ch_bra <= ch_ket) and
              // recovers the reversed ones through the (b, a) mirror, which
              // does not exist for a == b (reference/06 section 4.1).
              const ChannelIndex cb =
                  tb.channel_of(OrbitIndex{a}, OrbitIndex{h}, TwoJ{2 * Jb});
              const ChannelIndex ck =
                  tb.channel_of(OrbitIndex{b}, OrbitIndex{h}, TwoJ{2 * Jk});
              if (cb > ck) continue;
            }
            const double me =
                v.get_unnormalized(OrbitIndex{a}, OrbitIndex{h}, OrbitIndex{b},
                                   OrbitIndex{h}, TwoJ{2 * Jb}, TwoJ{2 * Jk});
            if (me == 0.0) continue;
            const int ph = ((ja + jh) / 2 - Jk - L) % 2 == 0 ? 1 : -1;
            sum +=
                std::sqrt((2.0 * Jb + 1.0) * (2.0 * Jk + 1.0)) * nh * ph *
                sixj(TwoJ{2 * Jb}, TwoJ{2 * Jk}, TwoJ{jb}, TwoJ{ja}, TwoJ{jh}) *
                me;
          }
        }
      }
      delta[static_cast<std::size_t>(a) +
            static_cast<std::size_t>(b) * static_cast<std::size_t>(norb)] =
          s * sum;
    }
  }
  for (std::int32_t a = 0; a < norb; ++a)
    for (std::int32_t b = 0; b < norb; ++b) {
      const double d =
          delta[static_cast<std::size_t>(a) +
                static_cast<std::size_t>(b) * static_cast<std::size_t>(norb)];
      if (d == 0.0) continue;
      result.one_body().set_element(
          OrbitIndex{a}, OrbitIndex{b},
          result.one_body().get(OrbitIndex{a}, OrbitIndex{b}) + d);
    }
  result.SyncToDevice();
  return result;
}

Operator NormalOrderTensor(const Operator& op,
                           const TensorNormalOrderOptions& options) {
  IMSRG_CHECK(!op.empty(), "normal_order_tensor: empty operator");
  return NormalOrderTensor(op, op.model_space().reference().occupations(),
                           NormalOrderDirection::kToReference, options);
}

Operator UndoNormalOrderTensor(const Operator& op,
                               const TensorNormalOrderOptions& options) {
  IMSRG_CHECK(!op.empty(), "undo_normal_order_tensor: empty operator");
  return NormalOrderTensor(op, op.model_space().reference().occupations(),
                           NormalOrderDirection::kToVacuum, options);
}

Operator ReNormalOrder(const Operator& op,
                       std::shared_ptr<const basis::ModelSpace> target,
                       const TensorNormalOrderOptions& options) {
  IMSRG_CHECK(!op.empty() && target != nullptr,
              "renormal_order: empty operator or null target");
  const auto from = op.model_space().reference().occupations();
  const auto to = target->reference().occupations();
  Operator vac = op.spec().is_scalar()
                     ? NormalOrder(op, from, NormalOrderDirection::kToVacuum)
                     : NormalOrderTensor(
                           op, from, NormalOrderDirection::kToVacuum, options);
  Operator moved = WithModelSpace(vac, std::move(target));
  return moved.spec().is_scalar()
             ? NormalOrder(moved, to, NormalOrderDirection::kToReference)
             : NormalOrderTensor(moved, to, NormalOrderDirection::kToReference,
                                 options);
}

}  // namespace imsrg::op
