// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The element-wise reference implementation of comm122st: the kernel as it was
// before the lifted-list rewrite (every element through
// `TensorTwoBodyView::get_unnormalized`), kept to check the fast kernel.
#include "tests/unit/commutator/tensor_comm122_ref.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/constants.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/commutator/detail/tensor_impl.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::commutator::test {

namespace {

using detail::CsrView;
using detail::Phase;
using detail::TensorTwoBodyView;
using linalg::MatrixView;

/// Everything the one-body-output kernels read.
struct OneBodyArgs {
  MatrixView<const double> x1;
  MatrixView<const double> y1;
  MatrixView<double> z1;
  TensorTwoBodyView x2;
  TensorTwoBodyView y2;
  const double* occ = nullptr;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* holes = nullptr;
  std::int32_t num_holes = 0;
  const std::int32_t* particles = nullptr;
  std::int32_t num_particles = 0;
  CsrView sc1;  ///< Scalar one-body channels (equal l, 2j, tz2).
  CsrView tc1;  ///< Tensor one-body connections.
  angmom::SixJView sixj;
  angmom::LambdaSixJView sixjL;
  int L = 0;
  int hZ = 1;
};
static_assert(std::is_trivially_copyable_v<OneBodyArgs>);

OneBodyArgs MakeOneBodyArgs(const op::Operator& X, const op::Operator& Y,
                            op::Operator& Z, detail::TensorContext& ctx,
                            detail::TensorRankData& rd,
                            const exec::DeviceBuffer<std::int64_t>& xoff,
                            const exec::DeviceBuffer<std::int64_t>& yoff,
                            int hZ) {
  OneBodyArgs a;
  a.x1 = std::as_const(X).one_body().device_view();
  a.y1 = std::as_const(Y).one_body().device_view();
  a.z1 = Z.one_body().device_view();
  const auto& tb = ctx.tables.two_body();
  a.x2 = detail::MakeTensorView(X.two_body(), tb, xoff);
  a.y2 = detail::MakeTensorView(Y.two_body(), tb, yoff);
  a.occ = ctx.tables.orbits().occupation;
  a.j2 = ctx.tables.orbits().j2;
  a.holes = ctx.holes.device_data();
  a.num_holes = ctx.num_holes;
  a.particles = ctx.particles.device_data();
  a.num_particles = ctx.num_particles;
  a.sc1 = ctx.scalar_ch1.view();
  a.tc1 = rd.tensor_ch1.view();
  a.sixj = ctx.sixj.device_view();
  a.sixjL = rd.sixj_lambda.device_view();
  a.L = Y.spec().rank_J;
  a.hZ = hZ;
  return a;
}

IMSRG_INLINE_FUNCTION int Imin(int a, int b) { return a < b ? a : b; }
IMSRG_INLINE_FUNCTION int Imax(int a, int b) { return a > b ? a : b; }
IMSRG_INLINE_FUNCTION int Iabs(int a) { return a < 0 ? -a : a; }

/// Flat description of the stored blocks of a tensor part and of their rows.
struct BlockRows {
  exec::DeviceBuffer<std::int32_t> ch_bra, ch_ket, nbra, nket;
  exec::DeviceBuffer<std::int64_t> offset;
  exec::DeviceBuffer<std::int32_t> row_offset;  ///< Per block + total.
  std::int32_t num_blocks = 0;
  std::int32_t num_rows = 0;
};

BlockRows MakeBlockRows(const op::TwoBody& part) {
  std::vector<std::int32_t> cb, ck, nb, nk, ro{0};
  std::vector<std::int64_t> off;
  for (const auto& info : part.matrix().blocks()) {
    cb.push_back(info.row_block.value());
    ck.push_back(info.col_block.value());
    nb.push_back(static_cast<std::int32_t>(info.rows));
    nk.push_back(static_cast<std::int32_t>(info.cols));
    off.push_back(info.offset);
    ro.push_back(ro.back() + static_cast<std::int32_t>(info.rows));
  }
  BlockRows r;
  const auto dev32 = [](const std::vector<std::int32_t>& v) {
    exec::DeviceBuffer<std::int32_t> b(v.size(), "imsrg::BlockRows");
    if (!v.empty()) b.CopyFromHost(v);
    return b;
  };
  r.ch_bra = dev32(cb);
  r.ch_ket = dev32(ck);
  r.nbra = dev32(nb);
  r.nket = dev32(nk);
  r.row_offset = dev32(ro);
  r.offset = exec::DeviceBuffer<std::int64_t>(off.size(), "imsrg::BlockRows");
  if (!off.empty()) r.offset.CopyFromHost(off);
  r.num_blocks = static_cast<std::int32_t>(cb.size());
  r.num_rows = ro.back();
  return r;
}

struct BlockRowsView {
  const std::int32_t* ch_bra = nullptr;
  const std::int32_t* ch_ket = nullptr;
  const std::int32_t* nbra = nullptr;
  const std::int32_t* nket = nullptr;
  const std::int64_t* offset = nullptr;
  const std::int32_t* row_offset = nullptr;
  std::int32_t num_blocks = 0;
};
static_assert(std::is_trivially_copyable_v<BlockRowsView>);

BlockRowsView ViewOf(const BlockRows& r) {
  return BlockRowsView{r.ch_bra.device_data(),
                       r.ch_ket.device_data(),
                       r.nbra.device_data(),
                       r.nket.device_data(),
                       r.offset.device_data(),
                       r.row_offset.device_data(),
                       r.num_blocks};
}
}  // namespace

void ReferenceComm122st(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, TensorWorkspace& ws) {
  const int hZ = op::HermiticitySign(Z.hermiticity());
  exec::Fence("test::comm122st_ref");
  if (!Z.two_body().allocated()) return;
  auto& ctx = ws.context();
  auto& rd = ctx.rank(Y.spec());
  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const auto yoff = detail::BlockOffsetTable(Y.two_body(), nch);
  const OneBodyArgs a = MakeOneBodyArgs(X, Y, Z, ctx, rd, xoff, yoff, hZ);
  const BlockRows rows = MakeBlockRows(Z.two_body());
  const BlockRowsView br = ViewOf(rows);
  const basis::TwoBodyTableView tb = ctx.tables.two_body();
  double* zd = Z.two_body().matrix().buffer().device_data();
  exec::ParallelForDynamic(
      "test::comm122st_ref", rows.num_rows, IMSRG_LAMBDA(const exec::Index rr) {
        const int k = detail::FindBlock(br.row_offset, br.num_blocks, rr);
        const int ibra = static_cast<int>(rr) - br.row_offset[k];
        const int cb = br.ch_bra[k];
        const int ck = br.ch_ket[k];
        const int J1 = tb.channel_two_j[cb] / 2;
        const int J2 = tb.channel_two_j[ck] / 2;
        const int nb = br.nbra[k];
        const int nk = br.nket[k];
        const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1));
        const KetIndex gb =
            tb.global_ket(ChannelIndex{cb}, LocalKetIndex{ibra});
        const int i = tb.ket_p[gb.value()];
        const int j = tb.ket_q[gb.value()];
        const int ji = a.j2[i];
        const int jj = a.j2[j];
        const int iket_end = (cb == ck) ? ibra + 1 : nk;
        for (int iket = 0; iket < iket_end; ++iket) {
          const KetIndex gk =
              tb.global_ket(ChannelIndex{ck}, LocalKetIndex{iket});
          const int kk = tb.ket_p[gk.value()];
          const int l = tb.ket_q[gk.value()];
          const int jk = a.j2[kk];
          const int jl = a.j2[l];
          double c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0;
          for (int e = a.sc1.offsets[i]; e < a.sc1.offsets[i + 1]; ++e) {
            const int p = a.sc1.items[e];
            c1 +=
                a.x1(i, p) * a.y2.get_unnormalized(p, j, kk, l, 2 * J1, 2 * J2);
          }
          if (i == j) {
            c2 = c1;
          } else {
            for (int e = a.sc1.offsets[j]; e < a.sc1.offsets[j + 1]; ++e) {
              const int p = a.sc1.items[e];
              c2 += a.x1(j, p) *
                    a.y2.get_unnormalized(i, p, kk, l, 2 * J1, 2 * J2);
            }
          }
          for (int e = a.sc1.offsets[kk]; e < a.sc1.offsets[kk + 1]; ++e) {
            const int p = a.sc1.items[e];
            c3 +=
                a.x1(p, kk) * a.y2.get_unnormalized(i, j, p, l, 2 * J1, 2 * J2);
          }
          if (kk == l) {
            c4 = c3;
          } else {
            for (int e = a.sc1.offsets[l]; e < a.sc1.offsets[l + 1]; ++e) {
              const int p = a.sc1.items[e];
              c4 += a.x1(p, l) *
                    a.y2.get_unnormalized(i, j, kk, p, 2 * J1, 2 * J2);
            }
          }
          double cijkl = c1 + c2 - c3 - c4;
          c1 = c2 = c3 = c4 = 0.0;
          const int ph1 = Phase((ji + jj) / 2 + J2 + a.L);
          const int ph23 = Phase(J1 - J2 + a.L);
          const int ph4 = Phase((jk + jl) / 2 - J1 + a.L);
          for (int e = a.tc1.offsets[i]; e < a.tc1.offsets[i + 1]; ++e) {
            const int p = a.tc1.items[e];
            const int ja = a.j2[p];
            if (!angmom::Triangle(TwoJ{2 * J2}, TwoJ{ja}, TwoJ{jj})) continue;
            c1 -= a.sixjL(TwoJ{2 * J2}, TwoJ{2 * J1}, TwoJ{ji}, TwoJ{ja},
                          TwoJ{jj}) *
                  a.y1(i, p) *
                  a.x2.get_unnormalized(p, j, kk, l, 2 * J2, 2 * J2);
          }
          for (int e = a.tc1.offsets[j]; e < a.tc1.offsets[j + 1]; ++e) {
            const int p = a.tc1.items[e];
            const int ja = a.j2[p];
            if (!angmom::Triangle(TwoJ{2 * J2}, TwoJ{ja}, TwoJ{ji})) continue;
            c2 += a.sixjL(TwoJ{2 * J2}, TwoJ{2 * J1}, TwoJ{jj}, TwoJ{ja},
                          TwoJ{ji}) *
                  a.y1(j, p) *
                  a.x2.get_unnormalized(p, i, kk, l, 2 * J2, 2 * J2);
          }
          for (int e = a.tc1.offsets[kk]; e < a.tc1.offsets[kk + 1]; ++e) {
            const int p = a.tc1.items[e];
            const int ja = a.j2[p];
            if (!angmom::Triangle(TwoJ{2 * J1}, TwoJ{ja}, TwoJ{jl})) continue;
            c3 -= a.sixjL(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{jk}, TwoJ{ja},
                          TwoJ{jl}) *
                  a.y1(p, kk) *
                  a.x2.get_unnormalized(i, j, l, p, 2 * J1, 2 * J1);
          }
          if (kk == l) {
            c4 = -c3;
          } else {
            for (int e = a.tc1.offsets[l]; e < a.tc1.offsets[l + 1]; ++e) {
              const int p = a.tc1.items[e];
              const int ja = a.j2[p];
              if (!angmom::Triangle(TwoJ{2 * J1}, TwoJ{ja}, TwoJ{jk})) continue;
              c4 += a.sixjL(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{jl}, TwoJ{ja},
                            TwoJ{jk}) *
                    a.y1(p, l) *
                    a.x2.get_unnormalized(i, j, kk, p, 2 * J1, 2 * J1);
            }
          }
          cijkl += hat * (ph1 * c1 + ph23 * c2 + ph23 * c3 + ph4 * c4);
          const bool dij = (i == j);
          const bool dkl = (kk == l);
          const double norm =
              (dij == dkl) ? (dij ? 2.0 : 1.0) : constants::kSqrt2;
          const double v = cijkl / norm;
          double* blk = zd + br.offset[k];
          blk[ibra + static_cast<std::int64_t>(iket) * nb] += v;
          if (cb == ck && iket < ibra)
            blk[iket + static_cast<std::int64_t>(ibra) * nb] += a.hZ * v;
        }
      });
  exec::Fence("test::comm122st_ref");
}

}  // namespace imsrg::commutator::test
