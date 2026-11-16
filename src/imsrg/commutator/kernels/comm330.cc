// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm330ss, [X3, Y3] -> Z0 (imsrg3.h). Kernel TU.

#include <cstdint>
#include <type_traits>

#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator::terms {

namespace {

/// Occupation products n_abc, nbar_abc and the multiplicity sym/6 of the
/// triple of flat row `r`.
struct RowOcc {
  double n = 0.0;
  double nbar = 0.0;
  double sym6 = 0.0;
};
static_assert(std::is_trivially_copyable_v<RowOcc>);

IMSRG_INLINE_FUNCTION RowOcc RowOccupation(const detail::N7Views& v,
                                           std::int64_t r) noexcept {
  const int t = v.row_triple[r];
  const double na = v.ob.occupation[v.sp.tr_a[t]];
  const double nb = v.ob.occupation[v.sp.tr_b[t]];
  const double nc = v.ob.occupation[v.sp.tr_c[t]];
  RowOcc o;
  o.n = na * nb * nc;
  o.nbar = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
  o.sym6 = v.sp.tr_sym[t] / 6.0;
  return o;
}

}  // namespace

void Comm330ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  const detail::N7Views v = detail::PrepareN7("Comm330ss", X, Y, Z, ctx);
  if (v.x3.data == nullptr || v.y3.data == nullptr) return;
  // The scalar part of an anti-Hermitian Z vanishes (the sum below cancels
  // pairwise); skip it to keep Z0 exactly zero.
  if (Z.hermiticity() == Hermiticity::kAntiHermitian) return;
  const exec::Index nrows = static_cast<exec::Index>(
      ctx.model_space_ptr()->three_body_space()->NumRows());
  double sum = 0.0;
  exec::ParallelReduce(
      "commutator::comm330ss", nrows,
      IMSRG_LAMBDA(const exec::Index r, double& partial) {
        const int t = v.row_triple[r];
        const int ch = v.sp.tr_channel[t];
        const std::int64_t base = v.sp.ch_row_offset[ch];
        const int i = static_cast<int>(r - base);
        const int n = v.sp.ch_dim[ch];
        const RowOcc oi = RowOccupation(v, r);
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
          if (j == i) continue;
          const RowOcc oj = RowOccupation(v, base + j);
          const double occ = oi.n * oj.nbar - oi.nbar * oj.n;
          if (occ == 0.0) continue;
          acc += oj.sym6 * occ * detail::ThreeBodyLocal(v.x3, v.sp, ch, i, j) *
                 detail::ThreeBodyLocal(v.y3, v.sp, ch, j, i);
        }
        partial += (v.sp.ch_two_j[ch] + 1) * oi.sym6 * acc;
      },
      sum);
  Z.zero_body() += sum;
}

}  // namespace imsrg::commutator::terms
