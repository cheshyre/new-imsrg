// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Scalar-tensor IMSRG(2) terms comm111st, comm121st, comm122st and
// comm222_pp_hh_221st (imsrg-devel Commutator/TensorCommutators.cc:25-504).
// Kernel TU: element-wise kernels with value captures; the ladder products are
// host BLAS on host-accessible backends (as the scalar terms).

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/commutator/detail/tensor_impl.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::commutator::tensor_terms {

namespace {

using detail::CsrView;
using detail::Phase;
using detail::TensorTwoBodyView;
using linalg::MatrixView;

/// Common checks; returns hZ.
int CheckInputs(const op::Operator& X, const op::Operator& Y,
                const op::Operator& Z, TensorWorkspace& ws) {
  IMSRG_CHECK(ws.Compatible(X) && ws.Compatible(Y) && ws.Compatible(Z),
              "tensor commutator: operators not on the workspace's space");
  const op::OperatorSpec zs = TensorCommutatorSpec(X.spec(), Y.spec());
  IMSRG_CHECK(Z.spec().SameSymmetry(zs) && Z.hermiticity() == zs.hermiticity,
              "tensor commutator: Z must have the spec of [X, Y]");
  IMSRG_CHECK(&Z != &X && &Z != &Y, "tensor commutator: Z aliases an input");
  const int emax = ws.model_space().basis().emax();
  if (Y.spec().rank_J > 2 * emax + 1) {
    throw NotImplementedError(
        "tensor commutator: rank_J > 2 emax + 1 exceeds the 6j tables");
  }
  IMSRG_CHECK(Z.one_body().buffer().host_is_device(),
              "tensor commutator: host BLAS on device memory (GPU backends "
              "need a device gemm)");
  exec::Fence("commutator::tensor");
  return op::HermiticitySign(zs.hermiticity);
}

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

}  // namespace

void Comm111st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws) {
  CheckInputs(X, Y, Z, ws);
  const auto x1 = std::as_const(X).one_body().device_view();
  const auto y1 = std::as_const(Y).one_body().device_view();
  const auto z1 = Z.one_body().device_view();
  linalg::Gemm(1.0, x1, linalg::Op::kNoTrans, y1, linalg::Op::kNoTrans, 1.0,
               z1);
  linalg::Gemm(-1.0, y1, linalg::Op::kNoTrans, x1, linalg::Op::kNoTrans, 1.0,
               z1);
}

void Comm121st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws) {
  const int hZ = CheckInputs(X, Y, Z, ws);
  auto& ctx = ws.context();
  auto& rd = ctx.rank(Y.spec());
  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const auto yoff = detail::BlockOffsetTable(Y.two_body(), nch);
  const OneBodyArgs a = MakeOneBodyArgs(X, Y, Z, ctx, rd, xoff, yoff, hZ);
  const exec::Index norb = ctx.tables.orbits().num_orbits;
  exec::ParallelForDynamic(
      "commutator::comm121st", norb, IMSRG_LAMBDA(const exec::Index ii) {
        const int i = static_cast<int>(ii);
        const int ji = a.j2[i];
        for (int ej = a.tc1.offsets[i]; ej < a.tc1.offsets[i + 1]; ++ej) {
          const int j = a.tc1.items[ej];
          if (j < i) continue;
          const int jj = a.j2[j];
          double zij = 0.0;
          for (int ih = 0; ih < a.num_holes; ++ih) {
            const int p = a.holes[ih];  // imsrg++ a
            const int ja = a.j2[p];
            // X one-body, Y two-body.
            for (int eb = a.sc1.offsets[p]; eb < a.sc1.offsets[p + 1]; ++eb) {
              const int b = a.sc1.items[eb];
              const double nanb = a.occ[p] * (1.0 - a.occ[b]);
              const double xab = a.x1(p, b);
              const double xba = a.x1(b, p);
              for (int J1 = Iabs(ji - ja) / 2; J1 <= (ji + ja) / 2; ++J1) {
                const int pf = Phase((jj + ja) / 2 + J1 + a.L);
                const int J2min = Imax(Iabs(a.L - J1), Iabs(ja - jj) / 2);
                const int J2max = Imin(a.L + J1, (ja + jj) / 2);
                for (int J2 = J2min; J2 <= J2max; ++J2) {
                  const double pref =
                      nanb * pf *
                      Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1)) *
                      a.sixjL(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{jj}, TwoJ{ji},
                              TwoJ{ja});
                  zij +=
                      pref *
                      (xab * a.y2.get_unnormalized(b, i, p, j, 2 * J1, 2 * J2) -
                       xba * a.y2.get_unnormalized(p, i, b, j, 2 * J1, 2 * J2));
                }
              }
            }
            // X two-body, Y one-body.
            for (int eb = a.tc1.offsets[p]; eb < a.tc1.offsets[p + 1]; ++eb) {
              const int b = a.tc1.items[eb];
              const int jb = a.j2[b];
              const double ob = a.occ[b];
              if (Kokkos::fabs(ob - 1.0) < basis::kOccupationCut) continue;
              const double nanb = a.occ[p] * (1.0 - ob);
              double xbar_ijab = 0.0;
              for (int J1 = Imax(Iabs(ji - jb), Iabs(jj - ja)) / 2;
                   J1 <= Imin(ji + jb, jj + ja) / 2; ++J1) {
                xbar_ijab -= (2 * J1 + 1) *
                             a.sixj(TwoJ{ji}, TwoJ{jj}, TwoJ{2 * a.L}, TwoJ{ja},
                                    TwoJ{jb}, TwoJ{2 * J1}) *
                             a.x2.get_unnormalized(i, b, p, j, 2 * J1, 2 * J1);
              }
              zij -= nanb * a.y1(p, b) * xbar_ijab;
              double xbar_ijba = 0.0;
              for (int J1 = Imax(Iabs(ji - ja), Iabs(jj - jb)) / 2;
                   J1 <= Imin(ji + ja, jj + jb) / 2; ++J1) {
                xbar_ijba -= (2 * J1 + 1) *
                             a.sixj(TwoJ{ji}, TwoJ{jj}, TwoJ{2 * a.L}, TwoJ{jb},
                                    TwoJ{ja}, TwoJ{2 * J1}) *
                             a.x2.get_unnormalized(i, p, b, j, 2 * J1, 2 * J1);
              }
              zij += nanb * a.y1(b, p) * xbar_ijba;
            }
          }
          a.z1(i, j) += zij;
          if (i != j) a.z1(j, i) += a.hZ * Phase((ji - jj) / 2) * zij;
        }
      });
  exec::Fence("commutator::comm121st");
}

namespace {

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

void Comm122st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws) {
  const int hZ = CheckInputs(X, Y, Z, ws);
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
      "commutator::comm122st", rows.num_rows,
      IMSRG_LAMBDA(const exec::Index rr) {
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
  exec::Fence("commutator::comm122st");
}

void Comm222PpHh221st(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, TensorWorkspace& ws, bool one_body,
                      bool two_body) {
  const int hZ = CheckInputs(X, Y, Z, ws);
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  if (!one_body && !two_body) return;
  auto& ctx = ws.context();
  auto& rd = ctx.rank(Y.spec());
  const auto& tbh = ctx.ms->two_body();
  const auto& ref = ctx.ms->reference();
  // Ket weights per channel: Pbar = 1 for pp kets, (1-n_a)(1-n_b) otherwise;
  // P = n_a n_b for hh kets, 0 otherwise (imsrg++ Ket_unocc_*, Ket_occ_hh).
  std::vector<std::vector<double>> wpp(
      static_cast<std::size_t>(tbh.num_channels()));
  std::vector<std::vector<double>> whh(wpp.size());
  for (std::int32_t c = 0; c < tbh.num_channels(); ++c) {
    for (const KetIndex kidx : tbh.channel_kets(ChannelIndex{c})) {
      const auto& ket = tbh.ket(kidx);
      const double na = ref.occupation(ket.p);
      const double nb = ref.occupation(ket.q);
      const bool ha = na >= basis::kOccupationCut;
      const bool hb = nb >= basis::kOccupationCut;
      wpp[static_cast<std::size_t>(c)].push_back(
          (!ha && !hb) ? 1.0 : (1.0 - na) * (1.0 - nb));
      whh[static_cast<std::size_t>(c)].push_back((ha && hb) ? na * nb : 0.0);
    }
  }
  // Mpp and Mhh with Z's block layout (host-accessible memory, host BLAS).
  op::TwoBody mpp(Z.two_body());
  op::TwoBody mhh(Z.two_body());
  mpp.set_zero();
  mhh.set_zero();
  mpp.SyncToHost();
  mhh.SyncToHost();
  X.SyncToHost();
  Y.SyncToHost();
  Z.SyncToHost();
  const op::TwoBody& x2 = X.two_body();
  const op::TwoBody& y2 = Y.two_body();
  for (const auto& info : Z.two_body().matrix().blocks()) {
    if (info.rows == 0 || info.cols == 0) continue;
    const ChannelIndex cb{info.row_block.value()};
    const ChannelIndex ck{info.col_block.value()};
    const auto yblk = y2.host_block(cb, ck);
    const auto xb = x2.host_block(cb);
    const auto xk = x2.host_block(ck);
    linalg::Matrix<double> ys(info.rows, info.cols);
    for (const int which : {0, 1}) {
      const auto& w = (which == 0) ? wpp : whh;
      const auto out =
          (which == 0) ? mpp.host_block(cb, ck) : mhh.host_block(cb, ck);
      const auto& wb = w[cb.ToSize()];
      const auto& wk = w[ck.ToSize()];
      for (std::int64_t c = 0; c < info.cols; ++c)
        for (std::int64_t r = 0; r < info.rows; ++r)
          ys(r, c) = wb[static_cast<std::size_t>(r)] * yblk(r, c);
      linalg::Gemm(1.0, xb, linalg::Op::kNoTrans, ys.const_view(),
                   linalg::Op::kNoTrans, 0.0, out);
      for (std::int64_t c = 0; c < info.cols; ++c)
        for (std::int64_t r = 0; r < info.rows; ++r)
          ys(r, c) = yblk(r, c) * wk[static_cast<std::size_t>(c)];
      linalg::Gemm(-1.0, ys.const_view(), linalg::Op::kNoTrans, xk,
                   linalg::Op::kNoTrans, 1.0, out);
    }
    if (two_body) {
      const auto z = Z.two_body().host_block(cb, ck);
      const auto p = std::as_const(mpp).host_block(cb, ck);
      const auto h = std::as_const(mhh).host_block(cb, ck);
      for (std::int64_t c = 0; c < info.cols; ++c)
        for (std::int64_t r = 0; r < info.rows; ++r)
          z(r, c) += p(r, c) - h(r, c);
    }
  }
  mpp.set_hermiticity(Z.hermiticity());
  mhh.set_hermiticity(Z.hermiticity());
  mpp.SyncToDevice();
  mhh.SyncToDevice();
  Z.SyncToDevice();
  if (!one_body) return;
  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto zoff = detail::BlockOffsetTable(Z.two_body(), nch);
  const auto& tb = ctx.tables.two_body();
  const TensorTwoBodyView vpp = detail::MakeTensorView(mpp, tb, zoff);
  const TensorTwoBodyView vhh = detail::MakeTensorView(mhh, tb, zoff);
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const OneBodyArgs a = MakeOneBodyArgs(X, Y, Z, ctx, rd, xoff, zoff, hZ);
  const exec::Index norb = ctx.tables.orbits().num_orbits;
  exec::ParallelForDynamic(
      "commutator::comm221st", norb, IMSRG_LAMBDA(const exec::Index ii) {
        const int i = static_cast<int>(ii);
        const int ji = a.j2[i];
        for (int ej = a.tc1.offsets[i]; ej < a.tc1.offsets[i + 1]; ++ej) {
          const int j = a.tc1.items[ej];
          if (j < i) continue;
          const int jj = a.j2[j];
          double cij = 0.0;
          for (int pass = 0; pass < 2; ++pass) {
            const int n = pass == 0 ? a.num_holes : a.num_particles;
            const std::int32_t* list = pass == 0 ? a.holes : a.particles;
            for (int ic = 0; ic < n; ++ic) {
              const int c = list[ic];
              const int jc = a.j2[c];
              const double oc = a.occ[c];
              for (int J1 = Iabs(jc - ji) / 2; J1 <= (jc + ji) / 2; ++J1) {
                const int J2min = Imax(Iabs(jc - jj) / 2, Iabs(a.L - J1));
                const int J2max = Imin((jc + jj) / 2, J1 + a.L);
                for (int J2 = J2min; J2 <= J2max; ++J2) {
                  const double hat =
                      Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1));
                  const double sj = a.sixjL(TwoJ{2 * J1}, TwoJ{2 * J2},
                                            TwoJ{jj}, TwoJ{ji}, TwoJ{jc});
                  const int ph = Phase((jj + jc) / 2 + J1 + a.L);
                  const double mh =
                      vhh.get_unnormalized(c, i, c, j, 2 * J1, 2 * J2);
                  const double m = pass == 0
                                       ? oc * vpp.get_unnormalized(
                                                  c, i, c, j, 2 * J1, 2 * J2) +
                                             (1.0 - oc) * mh
                                       : mh;
                  cij += hat * sj * ph * m;
                }
              }
            }
          }
          a.z1(i, j) += cij;
          if (i != j) a.z1(j, i) += a.hZ * Phase((ji - jj) / 2) * cij;
        }
      });
  exec::Fence("commutator::comm221st");
}

}  // namespace imsrg::commutator::tensor_terms
