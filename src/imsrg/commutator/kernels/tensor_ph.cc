// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The scalar-tensor particle-hole term comm222_phst (imsrg-devel
// Commutator/TensorCommutators.cc:536-1413), evaluated over ALL ordered
// cross-coupled pairs: Pandya transforms of X (scalar) and Y (rank lambda,
// 9j recoupling), one pair of gemms per cross-coupled channel pair, and the
// inverse transform as a gather kernel. Kernel TU.

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/commutator/detail/tensor_impl.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/device_blas.h"

namespace imsrg::commutator::tensor_terms {

namespace {

using detail::CrossCoupledChannel;
using detail::Phase;
using detail::TensorTwoBodyView;

IMSRG_INLINE_FUNCTION int Imin(int a, int b) { return a < b ? a : b; }
IMSRG_INLINE_FUNCTION int Imax(int a, int b) { return a > b ? a : b; }
IMSRG_INLINE_FUNCTION int Iabs(int a) { return a < 0 ? -a : a; }

template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(v);
  return b;
}

/// Device arguments of the Pandya kernels.
struct PandyaArgs {
  detail::CrossCoupledView cc;
  const std::int32_t* ph_offset = nullptr;  ///< Per channel + total.
  const std::int32_t* ph_local = nullptr;   ///< Local pair indices.
  const double* ph_w = nullptr;             ///< n_a - n_b per ph pair.
  const std::int32_t* j2 = nullptr;
  const std::int32_t* l = nullptr;
  const std::int32_t* tz2 = nullptr;
  angmom::SixJView sixj;
  angmom::NineJView ninej;
  TensorTwoBodyView x2;
  TensorTwoBodyView y2;
  int L = 0;
};
static_assert(std::is_trivially_copyable_v<PandyaArgs>);

/// Xbar^{J3}_{(il),(ab)} = - sum_J' (2J'+1) {j_i j_l J3; j_a j_b J'}
/// X^{J'}_{ib,al} (unnormalized). O(J range).
IMSRG_INLINE_FUNCTION double XbarElement(const PandyaArgs& a, int i, int l,
                                         int p, int b, int J3) {
  const int ji = a.j2[i], jl = a.j2[l], ja = a.j2[p], jb = a.j2[b];
  const int jmin = Imax(Iabs(ji - jb), Iabs(ja - jl)) / 2;
  const int jmax = Imin(ji + jb, ja + jl) / 2;
  double s = 0.0;
  for (int Jp = jmin; Jp <= jmax; ++Jp) {
    s -= (2 * Jp + 1) *
         a.sixj(TwoJ{ji}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{ja}, TwoJ{jb},
                TwoJ{2 * Jp}) *
         a.x2.get_unnormalized(i, b, p, l, 2 * Jp, 2 * Jp);
  }
  return s;
}

/// Ybar^{J3 J4}_{(ab),(cd)} = - sum_{J1 J2} hat(J1 J2 J3 J4) (-1)^(j_b + j_d
/// + J4 + J2) {j_a j_d J1; j_b j_c J2; J3 J4 L} Y^{J1 J2}_{ad,cb}
/// (imsrg++ DoTensorPandyaTransformation_SingleChannel). O(J range^2).
IMSRG_INLINE_FUNCTION double YbarElement(const PandyaArgs& a, int p, int b,
                                         int c, int d, int J3, int J4) {
  const int ja = a.j2[p], jb = a.j2[b], jc = a.j2[c], jd = a.j2[d];
  double s = 0.0;
  for (int J1 = Iabs(ja - jd) / 2; J1 <= (ja + jd) / 2; ++J1) {
    const int J2min = Imax(Iabs(jc - jb) / 2, Iabs(J1 - a.L));
    const int J2max = Imin((jc + jb) / 2, J1 + a.L);
    for (int J2 = J2min; J2 <= J2max; ++J2) {
      const double nj =
          a.ninej(TwoJ{ja}, TwoJ{jd}, TwoJ{2 * J1}, TwoJ{jb}, TwoJ{jc},
                  TwoJ{2 * J2}, TwoJ{2 * J3}, TwoJ{2 * J4});
      if (nj == 0.0) continue;
      const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                      (2.0 * J3 + 1) * (2.0 * J4 + 1));
      s -= hat * Phase((jb + jd) / 2 + J4 + J2) * nj *
           a.y2.get_unnormalized(p, d, c, b, 2 * J1, 2 * J2);
    }
  }
  return s;
}

/// Arguments of the inverse transform.
struct InverseArgs {
  detail::CrossCoupledView cc;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* l = nullptr;
  const std::int32_t* tz2 = nullptr;
  angmom::NineJView ninej;
  basis::TwoBodyTableView tb;
  const std::int32_t* pair_of = nullptr;  ///< ncc^2 -> pair index or -1.
  const std::int64_t* zbar_off = nullptr;
  const std::int32_t* cc_size = nullptr;  ///< Pairs per cc channel.
  const double* zbar = nullptr;
  std::int32_t ncc = 0;
  int L = 0;
  int hZ = 1;
  // Z blocks
  const std::int32_t* ch_bra = nullptr;
  const std::int32_t* ch_ket = nullptr;
  const std::int32_t* nbra = nullptr;
  const std::int32_t* nket = nullptr;
  const std::int64_t* offset = nullptr;
  const std::int32_t* row_offset = nullptr;
  std::int32_t num_blocks = 0;
  double* zd = nullptr;

  /// Zbar^{J3 J4}_{(il),(kj)}, 0 if the channel pair was not computed.
  IMSRG_INLINE_FUNCTION double Zbar(int i, int ll, int k, int j, int J3,
                                    int J4) const {
    const int c3 = CrossCoupledChannel(l, tz2, i, ll, J3);
    const int c4 = CrossCoupledChannel(l, tz2, k, j, J4);
    const int r = cc.local_index(i, ll, J3);
    const int c = cc.local_index(k, j, J4);
    if (c3 <= c4) {
      const int P = pair_of[c3 * ncc + c4];
      if (P < 0) return 0.0;
      return zbar[zbar_off[P] + r + static_cast<std::int64_t>(c) * cc_size[c3]];
    }
    // Only c3 <= c4 is stored: Zbar^{c3 c4}_{x,y} = hZ (-1)^(J3-J4)
    // Zbar^{c4 c3}_{y,x} (hermiticity of the tensor Pandya transform,
    // imsrg++ AddInverseTensorPandyaTransformation).
    const int P = pair_of[c4 * ncc + c3];
    if (P < 0) return 0.0;
    return hZ * Phase(J3 - J4) *
           zbar[zbar_off[P] + c + static_cast<std::int64_t>(r) * cc_size[c4]];
  }
};
static_assert(std::is_trivially_copyable_v<InverseArgs>);

}  // namespace

void Comm222Phst(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 TensorWorkspace& ws, double ninej_threshold) {
  IMSRG_CHECK(ws.Compatible(X) && ws.Compatible(Y) && ws.Compatible(Z),
              "comm222_phst: operators not on the workspace's space");
  const op::OperatorSpec zs = TensorCommutatorSpec(X.spec(), Y.spec());
  IMSRG_CHECK(Z.spec().SameSymmetry(zs) && Z.hermiticity() == zs.hermiticity,
              "comm222_phst: Z must have the spec of [X, Y]");
  if (!X.two_body().allocated() || !Y.two_body().allocated() ||
      !Z.two_body().allocated())
    return;
  exec::Fence("commutator::comm222_phst");
  auto& ctx = ws.context();
  const auto& cc = ctx.cross_coupled();
  const auto& ninej = ctx.ninej(Y.spec(), ninej_threshold);
  const auto& ref = ctx.ms->reference();
  const int L = Y.spec().rank_J;
  const int piY = static_cast<int>(Y.spec().parity);
  const int dTz = Y.spec().delta_Tz;
  const int hX = op::HermiticitySign(X.hermiticity());
  const int hZ = op::HermiticitySign(zs.hermiticity);
  const std::int32_t ncc = cc.num_channels;

  // ph pairs (n_a != n_b) per cross-coupled channel.
  std::vector<std::int32_t> ph_off{0};
  std::vector<std::int32_t> ph_local;
  std::vector<double> ph_w;
  std::vector<std::int32_t> size(static_cast<std::size_t>(ncc));
  for (std::int32_t c = 0; c < ncc; ++c) {
    const std::int32_t b = cc.offset[static_cast<std::size_t>(c)];
    const std::int32_t e = cc.offset[static_cast<std::size_t>(c) + 1];
    size[static_cast<std::size_t>(c)] = e - b;
    for (std::int32_t k = b; k < e; ++k) {
      const double na =
          ref.occupation(OrbitIndex{cc.pair_a[static_cast<std::size_t>(k)]});
      const double nb =
          ref.occupation(OrbitIndex{cc.pair_b[static_cast<std::size_t>(k)]});
      if (na != nb) {
        ph_local.push_back(k - b);
        ph_w.push_back(na - nb);
      }
    }
    ph_off.push_back(static_cast<std::int32_t>(ph_local.size()));
  }
  const auto nph = [&](std::int32_t c) {
    return ph_off[static_cast<std::size_t>(c) + 1] -
           ph_off[static_cast<std::size_t>(c)];
  };
  // Channel pairs (c3, c4).
  std::vector<std::int32_t> p3, p4;
  std::vector<std::int32_t> pair_of(
      static_cast<std::size_t>(ncc) * static_cast<std::size_t>(ncc), -1);
  std::vector<std::int64_t> ya_off{0}, yb_off{0}, zb_off{0};
  std::vector<std::int64_t> e_off{0};
  std::vector<char> xc_needed(static_cast<std::size_t>(ncc), 0);
  for (std::int32_t c3 = 0; c3 < ncc; ++c3) {
    const std::int32_t n3 = size[static_cast<std::size_t>(c3)];
    if (n3 == 0) continue;
    // c4 >= c3 only: the other half follows from the hermiticity relation
    // used in `InverseArgs::Zbar`.
    for (std::int32_t c4 = c3; c4 < ncc; ++c4) {
      const std::int32_t n4 = size[static_cast<std::size_t>(c4)];
      if (n4 == 0) continue;
      const int J3 = cc.J[static_cast<std::size_t>(c3)];
      const int J4 = cc.J[static_cast<std::size_t>(c4)];
      if (L < std::abs(J3 - J4) || L > J3 + J4) continue;
      if ((cc.parity[static_cast<std::size_t>(c3)] +
           cc.parity[static_cast<std::size_t>(c4)] + piY) %
              2 !=
          0)
        continue;
      const int t3 = cc.tz[static_cast<std::size_t>(c3)];
      const int t4 = cc.tz[static_cast<std::size_t>(c4)];
      if (!(t3 + t4 == dTz || std::abs(t3 - t4) == dTz)) continue;
      pair_of[static_cast<std::size_t>(c3) * static_cast<std::size_t>(ncc) +
              static_cast<std::size_t>(c4)] =
          static_cast<std::int32_t>(p3.size());
      p3.push_back(c3);
      p4.push_back(c4);
      ya_off.push_back(ya_off.back() + static_cast<std::int64_t>(nph(c3)) * n4);
      yb_off.push_back(yb_off.back() + static_cast<std::int64_t>(n3) * nph(c4));
      zb_off.push_back(zb_off.back() + static_cast<std::int64_t>(n3) * n4);
      e_off.push_back(e_off.back() + static_cast<std::int64_t>(nph(c3)) * n4 +
                      static_cast<std::int64_t>(n3) * nph(c4));
      xc_needed[static_cast<std::size_t>(c3)] = 1;
      xc_needed[static_cast<std::size_t>(c4)] = 1;
    }
  }
  const std::int32_t npairs = static_cast<std::int32_t>(p3.size());
  if (npairs == 0) return;
  // XbarCols per channel: N_c x Nph_c.
  std::vector<std::int64_t> xc_off(static_cast<std::size_t>(ncc) + 1, 0);
  for (std::int32_t c = 0; c < ncc; ++c)
    xc_off[static_cast<std::size_t>(c) + 1] =
        xc_off[static_cast<std::size_t>(c)] +
        (xc_needed[static_cast<std::size_t>(c)]
             ? static_cast<std::int64_t>(size[static_cast<std::size_t>(c)]) *
                   nph(c)
             : 0);

  const auto d_ph_off = ToDevice(ph_off, "imsrg::ph::ph_off");
  const auto d_ph_local = ToDevice(ph_local, "imsrg::ph::ph_local");
  const auto d_ph_w = ToDevice(ph_w, "imsrg::ph::ph_w");
  const auto d_xc_off = ToDevice(xc_off, "imsrg::ph::xc_off");
  const auto d_p3 = ToDevice(p3, "imsrg::ph::p3");
  const auto d_p4 = ToDevice(p4, "imsrg::ph::p4");
  const auto d_ya_off = ToDevice(ya_off, "imsrg::ph::ya_off");
  const auto d_yb_off = ToDevice(yb_off, "imsrg::ph::yb_off");
  const auto d_e_off = ToDevice(e_off, "imsrg::ph::e_off");
  const auto d_size = ToDevice(size, "imsrg::ph::size");
  exec::DeviceBuffer<double> xc(static_cast<std::size_t>(xc_off.back()),
                                "imsrg::ph::xbar");
  exec::DeviceBuffer<double> ya(static_cast<std::size_t>(ya_off.back()),
                                "imsrg::ph::ybarA");
  exec::DeviceBuffer<double> yb(static_cast<std::size_t>(yb_off.back()),
                                "imsrg::ph::ybarB");
  exec::DeviceBuffer<double> zb(static_cast<std::size_t>(zb_off.back()),
                                "imsrg::ph::zbar");

  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const auto yoff = detail::BlockOffsetTable(Y.two_body(), nch);
  PandyaArgs a;
  a.cc = detail::CrossCoupledView{
      cc.d_offset.device_data(),      cc.d_pair_a.device_data(),
      cc.d_pair_b.device_data(),      cc.d_local.device_data(),
      ctx.tables.orbits().num_orbits, cc.max_J};
  a.ph_offset = d_ph_off.device_data();
  a.ph_local = d_ph_local.device_data();
  a.ph_w = d_ph_w.device_data();
  a.j2 = ctx.tables.orbits().j2;
  a.l = ctx.tables.orbits().l;
  a.tz2 = ctx.tables.orbits().tz2;
  a.sixj = ctx.sixj.device_view();
  a.ninej = ninej.device_view();
  a.x2 = detail::MakeTensorView(X.two_body(), ctx.tables.two_body(), xoff);
  a.y2 = detail::MakeTensorView(Y.two_body(), ctx.tables.two_body(), yoff);
  a.L = L;

  // 1. Xbar columns of the ph pairs, all needed channels.
  {
    const std::int64_t* xo = d_xc_off.device_data();
    const std::int32_t* sz = d_size.device_data();
    double* out = xc.device_data();
    const std::int32_t ncc_ = ncc;
    exec::ParallelFor(
        "commutator::comm222_phst::xbar", xc_off.back(),
        IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::FindBlock(xo, ncc_, e);
          const std::int64_t loc = e - xo[c];
          const int n = sz[c];
          const int r = static_cast<int>(loc % n);
          const int q = static_cast<int>(loc / n);
          const int base = a.cc.offset[c];
          const int i = a.cc.pair_a[base + r];
          const int ll = a.cc.pair_b[base + r];
          const int s = base + a.ph_local[a.ph_offset[c] + q];
          out[e] = XbarElement(a, i, ll, a.cc.pair_a[s], a.cc.pair_b[s], c / 4);
        });
  }
  // 2. Ybar pieces, scaled by n_a - n_b on the ph index: YA = W3 Ybar[ph3, :],
  //    YB = Ybar[:, ph4] W4.
  {
    const std::int64_t* eo = d_e_off.device_data();
    const std::int64_t* yao = d_ya_off.device_data();
    const std::int64_t* ybo = d_yb_off.device_data();
    const std::int32_t* q3 = d_p3.device_data();
    const std::int32_t* q4 = d_p4.device_data();
    const std::int32_t* sz = d_size.device_data();
    double* yad = ya.device_data();
    double* ybd = yb.device_data();
    const std::int32_t np = npairs;
    exec::ParallelFor(
        "commutator::comm222_phst::ybar", e_off.back(),
        IMSRG_LAMBDA(const exec::Index e) {
          const int P = detail::FindBlock(eo, np, e);
          const std::int64_t loc = e - eo[P];
          const int c3 = q3[P];
          const int c4 = q4[P];
          const int n3 = sz[c3];
          const int n4 = sz[c4];
          const int nph3 = a.ph_offset[c3 + 1] - a.ph_offset[c3];
          const int b3 = a.cc.offset[c3];
          const int b4 = a.cc.offset[c4];
          const std::int64_t na = static_cast<std::int64_t>(nph3) * n4;
          if (loc < na) {
            const int q = static_cast<int>(loc % nph3);
            const int col = static_cast<int>(loc / nph3);
            const int s = b3 + a.ph_local[a.ph_offset[c3] + q];
            const int t = b4 + col;
            yad[yao[P] + loc] =
                a.ph_w[a.ph_offset[c3] + q] *
                YbarElement(a, a.cc.pair_a[s], a.cc.pair_b[s], a.cc.pair_a[t],
                            a.cc.pair_b[t], c3 / 4, c4 / 4);
          } else {
            const std::int64_t lb = loc - na;
            const int r = static_cast<int>(lb % n3);
            const int q = static_cast<int>(lb / n3);
            const int s = b3 + r;
            const int t = b4 + a.ph_local[a.ph_offset[c4] + q];
            ybd[ybo[P] + lb] =
                a.ph_w[a.ph_offset[c4] + q] *
                YbarElement(a, a.cc.pair_a[s], a.cc.pair_b[s], a.cc.pair_a[t],
                            a.cc.pair_b[t], c3 / 4, c4 / 4);
          }
        });
  }
  exec::Fence("commutator::comm222_phst::pandya");
  // 3. Zbar^{c3 c4} = Xbar3[:, ph3] YA - hX YB Xbar4[:, ph4]^T (device BLAS:
  //    the first products in one grouped call, then the subtractions).
  {
    const double* xcd = xc.device_data();
    const double* yad = ya.device_data();
    const double* ybd = yb.device_data();
    double* zbd = zb.device_data();
    std::vector<linalg::GemmProblem> first;
    std::vector<linalg::GemmProblem> second;
    for (std::int32_t P = 0; P < npairs; ++P) {
      const std::int32_t c3 = p3[static_cast<std::size_t>(P)];
      const std::int32_t c4 = p4[static_cast<std::size_t>(P)];
      const std::int64_t n3 = size[static_cast<std::size_t>(c3)];
      const std::int64_t n4 = size[static_cast<std::size_t>(c4)];
      const std::int64_t m3 = nph(c3);
      const std::int64_t m4 = nph(c4);
      linalg::MatrixView<double> Zm{zbd + zb_off[static_cast<std::size_t>(P)],
                                    n3, n4, n3};
      linalg::MatrixView<const double> X3{
          xcd + xc_off[static_cast<std::size_t>(c3)], n3, m3, n3};
      linalg::MatrixView<const double> YA{
          yad + ya_off[static_cast<std::size_t>(P)], m3, n4, m3 > 0 ? m3 : 1};
      linalg::MatrixView<const double> YB{
          ybd + yb_off[static_cast<std::size_t>(P)], n3, m4, n3};
      linalg::MatrixView<const double> X4{
          xcd + xc_off[static_cast<std::size_t>(c4)], n4, m4, n4};
      first.push_back(linalg::GemmProblem{1.0, X3, linalg::Op::kNoTrans, YA,
                                          linalg::Op::kNoTrans, 0.0, Zm});
      second.push_back(linalg::GemmProblem{-1.0 * hX, YB, linalg::Op::kNoTrans,
                                           X4, linalg::Op::kTrans, 1.0, Zm});
    }
    linalg::DeviceGroupedGemm(first);
    linalg::DeviceGroupedGemm(second);
  }
  // 4. Inverse transform into the stored blocks of Z.
  std::vector<std::int32_t> zcb, zck, znb, znk, zro{0};
  std::vector<std::int64_t> zoffv;
  for (const auto& info : Z.two_body().matrix().blocks()) {
    zcb.push_back(info.row_block.value());
    zck.push_back(info.col_block.value());
    znb.push_back(static_cast<std::int32_t>(info.rows));
    znk.push_back(static_cast<std::int32_t>(info.cols));
    zoffv.push_back(info.offset);
    zro.push_back(zro.back() + static_cast<std::int32_t>(info.rows));
  }
  const auto d_zcb = ToDevice(zcb, "imsrg::ph::zcb");
  const auto d_zck = ToDevice(zck, "imsrg::ph::zck");
  const auto d_znb = ToDevice(znb, "imsrg::ph::znb");
  const auto d_znk = ToDevice(znk, "imsrg::ph::znk");
  const auto d_zro = ToDevice(zro, "imsrg::ph::zro");
  const auto d_zoff = ToDevice(zoffv, "imsrg::ph::zoff");
  const auto d_pair_of = ToDevice(pair_of, "imsrg::ph::pair_of");
  const auto d_zb_off = ToDevice(zb_off, "imsrg::ph::zb_off");
  InverseArgs v;
  v.cc = a.cc;
  v.j2 = a.j2;
  v.l = a.l;
  v.tz2 = a.tz2;
  v.ninej = a.ninej;
  v.tb = ctx.tables.two_body();
  v.pair_of = d_pair_of.device_data();
  v.zbar_off = d_zb_off.device_data();
  v.cc_size = d_size.device_data();
  v.zbar = zb.device_data();
  v.ncc = ncc;
  v.L = L;
  v.hZ = hZ;
  v.ch_bra = d_zcb.device_data();
  v.ch_ket = d_zck.device_data();
  v.nbra = d_znb.device_data();
  v.nket = d_znk.device_data();
  v.offset = d_zoff.device_data();
  v.row_offset = d_zro.device_data();
  v.num_blocks = static_cast<std::int32_t>(zcb.size());
  v.zd = Z.two_body().matrix().buffer().device_data();
  exec::ParallelForDynamic(
      "commutator::comm222_phst::inverse", zro.back(),
      IMSRG_LAMBDA(const exec::Index rr) {
        const int k = detail::FindBlock(v.row_offset, v.num_blocks, rr);
        const int ibra = static_cast<int>(rr) - v.row_offset[k];
        const int cb = v.ch_bra[k];
        const int ck = v.ch_ket[k];
        const int J1 = v.tb.channel_two_j[cb] / 2;
        const int J2 = v.tb.channel_two_j[ck] / 2;
        const int nb = v.nbra[k];
        const int nk = v.nket[k];
        const KetIndex gb =
            v.tb.global_ket(ChannelIndex{cb}, LocalKetIndex{ibra});
        const int i = v.tb.ket_p[gb.value()];
        const int j = v.tb.ket_q[gb.value()];
        const int ji = v.j2[i];
        const int jj = v.j2[j];
        double* blk = v.zd + v.offset[k];
        for (int iket = (cb == ck ? ibra : 0); iket < nk; ++iket) {
          const KetIndex gk =
              v.tb.global_ket(ChannelIndex{ck}, LocalKetIndex{iket});
          const int kk = v.tb.ket_p[gk.value()];
          const int ll = v.tb.ket_q[gk.value()];
          const int jk = v.j2[kk];
          const int jl = v.j2[ll];
          double commij = 0.0;
          for (int J3 = Iabs(ji - jl) / 2; J3 <= (ji + jl) / 2; ++J3) {
            const int J4min = Imax(Iabs(jk - jj) / 2, Iabs(J3 - v.L));
            const int J4max = Imin((jk + jj) / 2, J3 + v.L);
            for (int J4 = J4min; J4 <= J4max; ++J4) {
              const double nj =
                  v.ninej(TwoJ{ji}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{jj}, TwoJ{jk},
                          TwoJ{2 * J4}, TwoJ{2 * J1}, TwoJ{2 * J2});
              if (nj == 0.0) continue;
              const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                              (2.0 * J3 + 1) * (2.0 * J4 + 1));
              commij += hat * Phase((jj + jl) / 2 + J2 + J4) * nj *
                        v.Zbar(i, ll, kk, j, J3, J4);
            }
          }
          double commji = commij;
          if (i != j) {
            commji = 0.0;
            for (int J3 = Iabs(jj - jl) / 2; J3 <= (jj + jl) / 2; ++J3) {
              const int J4min = Imax(Iabs(jk - ji) / 2, Iabs(J3 - v.L));
              const int J4max = Imin((jk + ji) / 2, J3 + v.L);
              for (int J4 = J4min; J4 <= J4max; ++J4) {
                const double nj =
                    v.ninej(TwoJ{jj}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{ji},
                            TwoJ{jk}, TwoJ{2 * J4}, TwoJ{2 * J1}, TwoJ{2 * J2});
                if (nj == 0.0) continue;
                const double hat =
                    Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                 (2.0 * J3 + 1) * (2.0 * J4 + 1));
                commji += hat * Phase((ji + jl) / 2 + J2 + J4) * nj *
                          v.Zbar(j, ll, kk, i, J3, J4);
              }
            }
          }
          const bool dij = (i == j);
          const bool dkl = (kk == ll);
          const double norm =
              (dij == dkl) ? (dij ? 2.0 : 1.0) : constants::kSqrt2;
          const double val =
              (commij - Phase((ji + jj) / 2 - J1) * commji) / norm;
          blk[ibra + static_cast<std::int64_t>(iket) * nb] += val;
          if (cb == ck && iket > ibra)
            blk[iket + static_cast<std::int64_t>(ibra) * nb] += v.hZ * val;
        }
      });
  exec::Fence("commutator::comm222_phst");
}

}  // namespace imsrg::commutator::tensor_terms
