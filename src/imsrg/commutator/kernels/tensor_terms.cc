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
#include "imsrg/linalg/device_blas.h"

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

template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(v);
  return b;
}

IMSRG_INLINE_FUNCTION int Imin(int a, int b) { return a < b ? a : b; }
IMSRG_INLINE_FUNCTION int Imax(int a, int b) { return a > b ? a : b; }
IMSRG_INLINE_FUNCTION int Iabs(int a) { return a < 0 ? -a : a; }

/// Adds the contribution of orbit c (pass 0: a hole, pass 1: a particle) to
/// Z1(i, j) of comm221st (the J1, J2 sums), in the order of the host loop.
IMSRG_FORCEINLINE_FUNCTION void Add221Orbit(const OneBodyArgs& a,
                                            const TensorTwoBodyView& vpp,
                                            const TensorTwoBodyView& vhh, int i,
                                            int j, int pass, int c,
                                            double& cij) {
  const int ji = a.j2[i];
  const int jj = a.j2[j];
  const int jc = a.j2[c];
  const double oc = a.occ[c];
  for (int J1 = Iabs(jc - ji) / 2; J1 <= (jc + ji) / 2; ++J1) {
    const int J2min = Imax(Iabs(jc - jj) / 2, Iabs(a.L - J1));
    const int J2max = Imin((jc + jj) / 2, J1 + a.L);
    for (int J2 = J2min; J2 <= J2max; ++J2) {
      const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1));
      const double sj =
          a.sixjL(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{jj}, TwoJ{ji}, TwoJ{jc});
      const int ph = Phase((jj + jc) / 2 + J1 + a.L);
      const double mh = vhh.get_unnormalized(c, i, c, j, 2 * J1, 2 * J2);
      const double m =
          pass == 0 ? oc * vpp.get_unnormalized(c, i, c, j, 2 * J1, 2 * J2) +
                          (1.0 - oc) * mh
                    : mh;
      cij += hat * sj * ph * m;
    }
  }
}

}  // namespace

void Comm111st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws) {
  CheckInputs(X, Y, Z, ws);
  const auto x1 = std::as_const(X).one_body().device_view();
  const auto y1 = std::as_const(Y).one_body().device_view();
  const auto z1 = Z.one_body().device_view();
  linalg::DeviceGemm(1.0, x1, linalg::Op::kNoTrans, y1, linalg::Op::kNoTrans,
                     1.0, z1);
  linalg::DeviceGemm(-1.0, y1, linalg::Op::kNoTrans, x1, linalg::Op::kNoTrans,
                     1.0, z1);
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
  exec::DeviceBuffer<std::int32_t> row_offset;         ///< Per block + total.
  std::vector<std::int32_t> host_ch_bra, host_ch_ket;  ///< Host copies.
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
  r.host_ch_bra = cb;
  r.host_ch_ket = ck;
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

/// Everything the comm122st kernel reads beyond `OneBodyArgs`.
struct Comm122Args {
  BlockRowsView br;
  const std::int32_t* ket_offset = nullptr;  ///< Flat ket row per channel.
  detail::EntryListView lifted;              ///< Part 1 (two lists per row).
  detail::EntryListView bra1, bra2, ket3, ket4;  ///< Part 2.
  const std::int32_t* col_offset = nullptr;
  std::int64_t num_channels = 0;
  double* zd = nullptr;
};
static_assert(std::is_trivially_copyable_v<Comm122Args>);

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
  const auto& lifted = ctx.lifted_kets();
  const auto& lists =
      ctx.comm122_lists(Y.spec(), rows.host_ch_bra, rows.host_ch_ket);
  Comm122Args k;
  k.br = ViewOf(rows);
  k.ket_offset = ctx.tables.two_body().channel_ket_offset;
  k.lifted = lifted.list.view();
  k.bra1 = lists.bra1.view();
  k.bra2 = lists.bra2.view();
  k.ket3 = lists.ket3.view();
  k.ket4 = lists.ket4.view();
  k.col_offset = lists.col_offset.device_data();
  k.num_channels = nch;
  k.zd = Z.two_body().matrix().buffer().device_data();
  const basis::TwoBodyTableView tb = ctx.tables.two_body();
  exec::ParallelForDynamic(
      "commutator::comm122st", rows.num_rows,
      IMSRG_LAMBDA(const exec::Index rr) {
        const int blk = detail::FindBlock(k.br.row_offset, k.br.num_blocks, rr);
        const int ibra = static_cast<int>(rr) - k.br.row_offset[blk];
        const int cb = k.br.ch_bra[blk];
        const int ck = k.br.ch_ket[blk];
        const int J1 = tb.channel_two_j[cb] / 2;
        const int J2 = tb.channel_two_j[ck] / 2;
        const int nb = k.br.nbra[blk];
        const int nk = k.br.nket[blk];
        const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1));
        const KetIndex gb =
            tb.global_ket(ChannelIndex{cb}, LocalKetIndex{ibra});
        const int i = tb.ket_p[gb.value()];
        const int j = tb.ket_q[gb.value()];
        const int ji = a.j2[i];
        const int jj = a.j2[j];
        // Stored blocks of Y (part 1) and of the scalar X (part 2), if any.
        const std::int64_t yo =
            (a.y2.data == nullptr)
                ? -1
                : a.y2.block_offset[cb * k.num_channels + ck];
        const double* yb = (yo >= 0) ? a.y2.data + yo : nullptr;
        const std::int64_t xbo =
            (a.x2.data == nullptr)
                ? -1
                : a.x2.block_offset[cb * k.num_channels + cb];
        const std::int64_t xko =
            (a.x2.data == nullptr)
                ? -1
                : a.x2.block_offset[ck * k.num_channels + ck];
        const double* xbb = (xbo >= 0) ? a.x2.data + xbo : nullptr;  // (cb, cb)
        const double* xkk = (xko >= 0) ? a.x2.data + xko : nullptr;  // (ck, ck)
        const bool dij = (i == j);
        const int row_b = k.ket_offset[cb] + ibra;
        const int a1b = k.lifted.offsets[2 * row_b];
        const int a1e = k.lifted.offsets[2 * row_b + 1];
        const int a2e = k.lifted.offsets[2 * row_b + 2];
        const int b1b = k.bra1.offsets[k.br.row_offset[blk] + ibra];
        const int b1e = k.bra1.offsets[k.br.row_offset[blk] + ibra + 1];
        const int b2b = k.bra2.offsets[k.br.row_offset[blk] + ibra];
        const int b2e = k.bra2.offsets[k.br.row_offset[blk] + ibra + 1];
        const int iket_end = (cb == ck) ? ibra + 1 : nk;
        for (int iket = 0; iket < iket_end; ++iket) {
          const KetIndex gk =
              tb.global_ket(ChannelIndex{ck}, LocalKetIndex{iket});
          const int kk = tb.ket_p[gk.value()];
          const int l = tb.ket_q[gk.value()];
          const int jk = a.j2[kk];
          const int jl = a.j2[l];
          const bool dkl = (kk == l);
          const int row_k = k.ket_offset[ck] + iket;
          const std::int64_t col = k.col_offset[blk] + iket;
          // Part 1: scalar one-body X1 with the tensor two-body Y2.
          double c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0;
          if (yb != nullptr) {
            const double* ycol = yb + static_cast<std::int64_t>(iket) * nb;
            for (int e = a1b; e < a1e; ++e)
              c1 += a.x1(k.lifted.from[e], k.lifted.to[e]) * k.lifted.coef[e] *
                    ycol[k.lifted.m[e]];
            if (dij) {
              c2 = c1;
            } else {
              for (int e = a1e; e < a2e; ++e)
                c2 += a.x1(k.lifted.from[e], k.lifted.to[e]) *
                      k.lifted.coef[e] * ycol[k.lifted.m[e]];
            }
            const int ka = k.lifted.offsets[2 * row_k];
            const int kb = k.lifted.offsets[2 * row_k + 1];
            const int ke = k.lifted.offsets[2 * row_k + 2];
            for (int e = ka; e < kb; ++e)
              c3 += a.x1(k.lifted.to[e], k.lifted.from[e]) * k.lifted.coef[e] *
                    yb[ibra + static_cast<std::int64_t>(k.lifted.m[e]) * nb];
            if (dkl) {
              c4 = c3;
            } else {
              for (int e = kb; e < ke; ++e)
                c4 += a.x1(k.lifted.to[e], k.lifted.from[e]) *
                      k.lifted.coef[e] *
                      yb[ibra + static_cast<std::int64_t>(k.lifted.m[e]) * nb];
            }
            // The unnormalized elements carry sqrt2 for k == l (bra lists) or
            // i == j (ket lists).
            const double sbra = dkl ? constants::kSqrt2 : 1.0;
            const double sket = dij ? constants::kSqrt2 : 1.0;
            c1 *= sbra;
            c2 *= sbra;
            c3 *= sket;
            c4 *= sket;
          }
          double cijkl = c1 + c2 - c3 - c4;
          // Part 2: tensor one-body Y1 with the scalar two-body X2.
          c1 = c2 = c3 = c4 = 0.0;
          const int ph1 = Phase((ji + jj) / 2 + J2 + a.L);
          const int ph23 = Phase(J1 - J2 + a.L);
          const int ph4 = Phase((jk + jl) / 2 - J1 + a.L);
          if (xkk != nullptr) {
            const double* xcol = xkk + static_cast<std::int64_t>(iket) * nk;
            for (int e = b1b; e < b1e; ++e)
              c1 += a.y1(k.bra1.from[e], k.bra1.to[e]) * k.bra1.coef[e] *
                    xcol[k.bra1.m[e]];
            for (int e = b2b; e < b2e; ++e)
              c2 += a.y1(k.bra2.from[e], k.bra2.to[e]) * k.bra2.coef[e] *
                    xcol[k.bra2.m[e]];
            if (dkl) {
              c1 *= constants::kSqrt2;
              c2 *= constants::kSqrt2;
            }
          }
          if (xbb != nullptr) {
            const int c3b = k.ket3.offsets[col];
            const int c3e = k.ket3.offsets[col + 1];
            for (int e = c3b; e < c3e; ++e)
              c3 += a.y1(k.ket3.from[e], k.ket3.to[e]) * k.ket3.coef[e] *
                    xbb[ibra + static_cast<std::int64_t>(k.ket3.m[e]) * nb];
            if (dkl) {
              c4 = -c3;
            } else {
              const int c4b = k.ket4.offsets[col];
              const int c4e = k.ket4.offsets[col + 1];
              for (int e = c4b; e < c4e; ++e)
                c4 += a.y1(k.ket4.from[e], k.ket4.to[e]) * k.ket4.coef[e] *
                      xbb[ibra + static_cast<std::int64_t>(k.ket4.m[e]) * nb];
            }
            if (dij) {
              c3 *= constants::kSqrt2;
              c4 *= constants::kSqrt2;
            }
          }
          cijkl += hat * (ph1 * c1 + ph23 * c2 + ph23 * c3 + ph4 * c4);
          const double norm =
              (dij == dkl) ? (dij ? 2.0 : 1.0) : constants::kSqrt2;
          const double v = cijkl / norm;
          double* zb = k.zd + k.br.offset[blk];
          zb[ibra + static_cast<std::int64_t>(iket) * nb] += v;
          if (cb == ck && iket < ibra)
            zb[iket + static_cast<std::int64_t>(ibra) * nb] += a.hZ * v;
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
  // Mpp and Mhh with Z's block layout, on the device (M14):
  //   M_c(b, k) = X_b diag(w_b) Y_bk - Y_bk diag(w_k) X_k
  // for w = Pbar (pp) and P (hh). The weights go into scaled copies of the
  // scalar X (XD = X diag(w), DX = diag(w) X: four copies of the small scalar
  // two-body part) instead of a weighted copy of every tensor block of Y.
  // Two grouped GEMM calls (the beta = 0 products, then the subtractions into
  // the same C), then Z += Mpp - Mhh element-wise.
  op::TwoBody mpp(Z.two_body());
  op::TwoBody mhh(Z.two_body());
  mpp.set_zero();
  mhh.set_zero();
  const auto nchan = static_cast<std::size_t>(tbh.num_channels());
  std::vector<std::int64_t> ket_off(nchan + 1, 0);
  for (std::size_t c = 0; c < nchan; ++c)
    ket_off[c + 1] = ket_off[c] + static_cast<std::int64_t>(wpp[c].size());
  const std::int64_t nk_all = ket_off.back();
  std::vector<double> w_host(2 * static_cast<std::size_t>(nk_all));
  for (std::size_t c = 0; c < nchan; ++c) {
    for (std::size_t k = 0; k < wpp[c].size(); ++k) {
      w_host[static_cast<std::size_t>(ket_off[c]) + k] = wpp[c][k];
      w_host[static_cast<std::size_t>(nk_all + ket_off[c]) + k] = whh[c][k];
    }
  }
  // X's diagonal blocks: offset (-1: absent) and dimension per channel, and
  // the compact offsets of the scaled copies.
  std::vector<std::int64_t> x_block(nchan, -1);
  std::vector<std::int64_t> x_dim(nchan, 0);
  for (const auto& info : X.two_body().matrix().blocks()) {
    if (info.row_block != info.col_block || info.rows == 0) continue;
    const auto c = static_cast<std::size_t>(info.row_block.value());
    x_block[c] = info.offset;
    x_dim[c] = info.rows;
  }
  std::vector<std::int64_t> x_off(nchan + 1, 0);
  for (std::size_t c = 0; c < nchan; ++c)
    x_off[c + 1] = x_off[c] + x_dim[c] * x_dim[c];
  const std::int64_t xsize = x_off.back();
  const auto d_w = ToDevice(w_host, "imsrg::tensor_ladder::w");
  const auto d_xblock = ToDevice(x_block, "imsrg::tensor_ladder::x_block");
  const auto d_xdim = ToDevice(x_dim, "imsrg::tensor_ladder::x_dim");
  const auto d_xoff = ToDevice(x_off, "imsrg::tensor_ladder::x_off");
  const auto d_kofs = ToDevice(ket_off, "imsrg::tensor_ladder::ket_off");
  // xs = [XD_pp | DX_pp | XD_hh | DX_hh], each compact by channel.
  exec::DeviceBuffer<double> xs(static_cast<std::size_t>(4 * xsize),
                                "imsrg::tensor_ladder::xs");
  if (xsize > 0) {
    const double* xd = X.two_body().matrix().buffer().device_data();
    const double* w = d_w.device_data();
    const std::int64_t* xb = d_xblock.device_data();
    const std::int64_t* xn = d_xdim.device_data();
    const std::int64_t* xo = d_xoff.device_data();
    const std::int64_t* ko = d_kofs.device_data();
    const int nc = static_cast<int>(nchan);
    double* out = xs.device_data();
    exec::ParallelFor(
        "commutator::comm222_pp_hh_221st::scale_x", xsize,
        IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::FindBlock(xo, nc, e);
          const std::int64_t dim = xn[c];
          const std::int64_t local = e - xo[c];
          const std::int64_t col = local / dim;
          const std::int64_t row = local - col * dim;
          const double x = xd[xb[c] + local];
          const double* wpp_c = w + ko[c];
          const double* whh_c = w + nk_all + ko[c];
          out[e] = x * wpp_c[col];
          out[xsize + e] = wpp_c[row] * x;
          out[2 * xsize + e] = x * whh_c[col];
          out[3 * xsize + e] = whh_c[row] * x;
        });
  }
  std::vector<linalg::GemmProblem> first;
  std::vector<linalg::GemmProblem> second;
  const auto& ymat = Y.two_body().matrix();
  const double* yd = ymat.buffer().device_data();
  const double* xsd = xs.device_data();
  for (const auto& info : Z.two_body().matrix().blocks()) {
    if (info.rows == 0 || info.cols == 0) continue;
    const auto cb = static_cast<std::size_t>(info.row_block.value());
    const auto ck = static_cast<std::size_t>(info.col_block.value());
    const auto yk = ymat.find_block(info.row_block, info.col_block);
    if (!yk.has_value()) continue;  // Y block absent: M = 0
    const linalg::MatrixView<const double> yblk{
        yd + ymat.block_info(*yk).offset, info.rows, info.cols, info.rows};
    for (int which = 0; which < 2; ++which) {
      double* out = (which == 0 ? mpp : mhh).matrix().buffer().device_data() +
                    info.offset;
      const linalg::MatrixView<double> cv{out, info.rows, info.cols, info.rows};
      // An absent X block is zero: its product term vanishes.
      if (x_block[cb] >= 0) {
        const linalg::MatrixView<const double> xdb{
            xsd + 2 * which * xsize + x_off[cb], info.rows, info.rows,
            info.rows};
        first.push_back(linalg::GemmProblem{1.0, xdb, linalg::Op::kNoTrans,
                                            yblk, linalg::Op::kNoTrans, 0.0,
                                            cv});
      }
      if (x_block[ck] >= 0) {
        const linalg::MatrixView<const double> dxk{
            xsd + (2 * which + 1) * xsize + x_off[ck], info.cols, info.cols,
            info.cols};
        second.push_back(linalg::GemmProblem{-1.0, yblk, linalg::Op::kNoTrans,
                                             dxk, linalg::Op::kNoTrans, 1.0,
                                             cv});
      }
    }
  }
  linalg::DeviceGroupedGemm(first);
  linalg::DeviceGroupedGemm(second);
  if (two_body) {
    double* zd = Z.two_body().matrix().buffer().device_data();
    const double* pd = mpp.matrix().buffer().device_data();
    const double* hd = mhh.matrix().buffer().device_data();
    exec::ParallelFor(
        "commutator::comm222_pp_hh_221st::z2",
        static_cast<exec::Index>(Z.two_body().matrix().buffer().size()),
        IMSRG_LAMBDA(const exec::Index e) { zd[e] += pd[e] - hd[e]; });
  }
  mpp.set_hermiticity(Z.hermiticity());
  mhh.set_hermiticity(Z.hermiticity());
  // The temporaries above (xs, offset tables) must outlive the queued work.
  exec::Fence("commutator::comm222_pp_hh_221st::gemm");
  if (!one_body) return;
  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto zoff = detail::BlockOffsetTable(Z.two_body(), nch);
  const auto& tb = ctx.tables.two_body();
  const TensorTwoBodyView vpp = detail::MakeTensorView(mpp, tb, zoff);
  const TensorTwoBodyView vhh = detail::MakeTensorView(mhh, tb, zoff);
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const auto yoff = detail::BlockOffsetTable(Y.two_body(), nch);
  const OneBodyArgs a = MakeOneBodyArgs(X, Y, Z, ctx, rd, xoff, yoff, hZ);
  const exec::Index norb = ctx.tables.orbits().num_orbits;
  if constexpr (exec::kDeviceKernels) {
    // Device shape (M14): one team per (i, j) entry of the tensor one-body
    // list, its threads reducing over the orbits c (a thread per orbit i
    // gave the GPU ~1e2 items). The reduction order differs from the host
    // loop (not bitwise equal to the host shape).
    const auto entries = static_cast<int>(rd.tensor_ch1.items.size());
    const int nh = a.num_holes;
    const int nc = a.num_holes + a.num_particles;
    exec::TeamOptions opts;
    opts.team_size = 64;
    opts.vector_length = 1;
    exec::TeamFor(
        "commutator::comm221st_team", entries, opts,
        IMSRG_LAMBDA(const exec::TeamMember& member) {
          const int ej = member.league_rank();
          const int i =
              detail::FindBlock(a.tc1.offsets, static_cast<int>(norb), ej);
          const int j = a.tc1.items[ej];
          if (j < i) return;
          double cij = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(member, nc),
              [&](const int t, double& part) {
                const int pass = t < nh ? 0 : 1;
                const int c = pass == 0 ? a.holes[t] : a.particles[t - nh];
                Add221Orbit(a, vpp, vhh, i, j, pass, c, part);
              },
              cij);
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            const int ji = a.j2[i];
            const int jj = a.j2[j];
            a.z1(i, j) += cij;
            if (i != j) a.z1(j, i) += a.hZ * Phase((ji - jj) / 2) * cij;
          });
        });
  } else {
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
              for (int ic = 0; ic < n; ++ic)
                Add221Orbit(a, vpp, vhh, i, j, pass, list[ic], cij);
            }
            a.z1(i, j) += cij;
            if (i != j) a.z1(j, i) += a.hZ * Phase((ji - jj) / 2) * cij;
          }
        });
  }
  exec::Fence("commutator::comm221st");
}

}  // namespace imsrg::commutator::tensor_terms
