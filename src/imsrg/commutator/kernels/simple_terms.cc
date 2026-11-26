// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The one-, two-body-reduced and mixed IMSRG(2) terms: comm110ss, comm220ss,
// comm111ss, comm121ss, comm122ss (design/12 section 4.1). Kernel TU: the only
// place of this module (with ladder.cc and particle_hole.cc) that includes the
// Kokkos vocabulary.

#include <cmath>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/device_blas.h"

namespace imsrg::commutator::terms {

namespace {

using detail::OneBodyChannelView;
using detail::TwoBodyView;
using linalg::MatrixView;

/// POD bundle of everything the one-body-driven kernels read.
struct OneBodyArgs {
  MatrixView<const double> x;
  MatrixView<const double> y;
  const double* occ = nullptr;
  const std::int32_t* j2 = nullptr;
  OneBodyChannelView ch1;
};
static_assert(std::is_trivially_copyable_v<OneBodyArgs>);

/// POD bundle of the general (non-Hermitian) comm122ss kernels.
struct Comm122GeneralArgs {
  MatrixView<const double> x1;
  MatrixView<const double> y1;
  const double* xd = nullptr;  ///< Two-body data of X (null: unallocated).
  const double* yd = nullptr;  ///< Two-body data of Y (null: unallocated).
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* num_kets = nullptr;
  const std::int32_t* row_base = nullptr;
  const std::int32_t* entry_offset = nullptr;
  const std::int32_t* lm = nullptr;
  const std::int32_t* lfrom = nullptr;
  const std::int32_t* lto = nullptr;
  const double* lcoef = nullptr;

  /// The element (r, c) of block `ch` of [X, Y]_122 without symmetry: the
  /// bra part B(r, c) is the lifted one-body list of row r acting on column c
  /// of X, Y; the ket part K(r, c) = sum_a (Y_ij,al X_ak + Y_ij,ka X_al) -
  /// (X and Y exchanged) is the list of row c acting on row r of X, Y with the
  /// one-body matrix transposed (coef is shared: it depends only on the lifted
  /// ket and on the normalization of the row or column). Returns B - K.
  /// O(lifted entries of rows r and c).
  IMSRG_INLINE_FUNCTION double Element(int ch, int r, int c) const {
    const std::int64_t off = block_offset[ch];
    const int n = num_kets[ch];
    const int rb = row_base[ch];
    double bra = 0.0;
    for (int t = entry_offset[rb + r]; t < entry_offset[rb + r + 1]; ++t) {
      const std::int64_t idx = off + lm[t] + static_cast<std::int64_t>(c) * n;
      double v = 0.0;
      if (yd != nullptr) v += x1(lfrom[t], lto[t]) * yd[idx];
      if (xd != nullptr) v -= y1(lfrom[t], lto[t]) * xd[idx];
      bra += lcoef[t] * v;
    }
    double ket = 0.0;
    for (int t = entry_offset[rb + c]; t < entry_offset[rb + c + 1]; ++t) {
      const std::int64_t idx = off + r + static_cast<std::int64_t>(lm[t]) * n;
      double v = 0.0;
      if (yd != nullptr) v += x1(lto[t], lfrom[t]) * yd[idx];
      if (xd != nullptr) v -= y1(lto[t], lfrom[t]) * xd[idx];
      ket += lcoef[t] * v;
    }
    return bra - ket;
  }
};
static_assert(std::is_trivially_copyable_v<Comm122GeneralArgs>);

}  // namespace

void Comm110ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  // [X,Y] is anti-Hermitian (hZ = -hX hY = -1) -> the scalar part vanishes.
  // (A general Z may have a scalar part; the sums below need no symmetry.)
  if (!detail::IsGeneral(X, Y) && detail::CommutatorHermiticitySign(X, Y) < 0)
    return;
  const auto& impl = ws.impl();
  OneBodyArgs a;
  a.x = std::as_const(X).one_body().device_view();
  a.y = std::as_const(Y).one_body().device_view();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.ch1 = impl.one_body_channels();
  const exec::Index norb = impl.tables.orbits().num_orbits;
  double sum = 0.0;
  exec::ParallelReduce(
      "commutator::comm110ss", norb,
      IMSRG_LAMBDA(const exec::Index ia, double& partial) {
        const int p = static_cast<int>(ia);
        const double na = a.occ[p];
        if (na == 0.0) return;
        double s = 0.0;
        for (int k = a.ch1.offsets[p]; k < a.ch1.offsets[p + 1]; ++k) {
          const int q = a.ch1.orbits[k];
          const double nbbar = 1.0 - a.occ[q];
          s += nbbar * (a.x(p, q) * a.y(q, p) - a.y(p, q) * a.x(q, p));
        }
        partial += (a.j2[p] + 1) * na * s;
      },
      sum);
  Z.zero_body() += sum;
}

void Comm220ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  // [X,Y] is anti-Hermitian (hZ = -hX hY = -1) -> the scalar part vanishes.
  // (A general Z may have a scalar part; the sums below need no symmetry.)
  if (!detail::IsGeneral(X, Y) && detail::CommutatorHermiticitySign(X, Y) < 0)
    return;
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const auto& impl = ws.impl();
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  const double* occ = impl.tables.orbits().occupation;
  const std::int32_t* entry_channel = impl.entry_channel.device_data();
  // The kets k with (1 - n_p)(1 - n_q) != 0 of each channel, ascending: the
  // ladder's pp list (the same exact test), so the loop below visits the
  // non-zero terms of the full loop over k in the same order (M12a).
  const std::int32_t* pp_list = impl.ladder.pp.list.device_data();
  const std::int64_t* pp_offset = impl.ladder.pp.list_offset.device_data();
  const exec::Index nentries =
      static_cast<exec::Index>(impl.entry_channel.size());
  double sum = 0.0;
  exec::ParallelReduce(
      "commutator::comm220ss", nentries,
      IMSRG_LAMBDA(const exec::Index e, double& partial) {
        const int ch = entry_channel[e];
        const std::int64_t off = xv.tb.channel_ket_offset[ch];
        const int i = static_cast<int>(e - off);
        const int n = xv.tb.channel_num_kets[ch];
        const double* xb = xv.data + xv.block_offset[ch];
        const double* yb = yv.data + yv.block_offset[ch];
        const int* kets = xv.tb.channel_kets + off;
        const int pi = xv.tb.ket_p[kets[i]];
        const int qi = xv.tb.ket_q[kets[i]];
        const double nn_i = occ[pi] * occ[qi];
        if (nn_i == 0.0) return;  // every term of the row vanishes
        double s = 0.0;
        for (std::int64_t m = pp_offset[ch]; m < pp_offset[ch + 1]; ++m) {
          const int k = pp_list[m];
          const int pk = xv.tb.ket_p[kets[k]];
          const int qk = xv.tb.ket_q[kets[k]];
          const double nb_k = (1.0 - occ[pk]) * (1.0 - occ[qk]);
          const double w = nn_i * nb_k;
          if (w == 0.0) continue;
          s += w * (xb[i + static_cast<std::int64_t>(k) * n] *
                        yb[k + static_cast<std::int64_t>(i) * n] -
                    yb[i + static_cast<std::int64_t>(k) * n] *
                        xb[k + static_cast<std::int64_t>(i) * n]);
        }
        partial += (xv.tb.channel_two_j[ch] + 1) * s;
      },
      sum);
  Z.zero_body() += sum;
}

void Comm111ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  // Device BLAS on the one-body matrices (tiny: norb^3), ordered after the
  // pending kernels like a kernel (design/12 section 3).
  const auto x1 = std::as_const(X).one_body().device_view();
  const auto y1 = std::as_const(Y).one_body().device_view();
  const auto z1 = Z.one_body().device_view();
  linalg::DeviceGemm(1.0, x1, linalg::Op::kNoTrans, y1, linalg::Op::kNoTrans,
                     1.0, z1);
  linalg::DeviceGemm(-1.0, y1, linalg::Op::kNoTrans, x1, linalg::Op::kNoTrans,
                     1.0, z1);
}

void Comm121ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  // [X1,Y2] and [X2,Y1] each need only one two-body operand (a missing part
  // reads as zero).
  if (!X.two_body().allocated() && !Y.two_body().allocated()) return;
  const auto& impl = ws.impl();
  OneBodyArgs a;
  a.x = std::as_const(X).one_body().device_view();
  a.y = std::as_const(Y).one_body().device_view();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.ch1 = impl.one_body_channels();
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  const MatrixView<double> z1 = Z.one_body().device_view();
  const std::int32_t* owner = impl.ob_owner.device_data();
  const std::int32_t* orbits = impl.ob_orbits.device_data();
  const int norb = impl.tables.orbits().num_orbits;
  const exec::Index nentries = static_cast<exec::Index>(impl.ob_orbits.size());
  // The contribution of orbit p (its q loop) to Z1(i, j), in host order.
  const auto add_orbit =
      IMSRG_LAMBDA(int i, int j, int ji, int jj, int p, double& z) {
    const int jp = a.j2[p];
    for (int k = a.ch1.offsets[p]; k < a.ch1.offsets[p + 1]; ++k) {
      const int q = a.ch1.orbits[k];
      const double dn = a.occ[p] - a.occ[q];
      if (dn == 0.0) continue;
      const double xab = a.x(p, q);
      const double yab = a.y(p, q);
      if (xab == 0.0 && yab == 0.0) continue;
      const int jq = a.j2[q];
      const int lo1 = jq > ji ? jq - ji : ji - jq;
      const int lo2 = jp > jj ? jp - jj : jj - jp;
      const int lo = lo1 > lo2 ? lo1 : lo2;
      const int hi1 = jq + ji;
      const int hi2 = jp + jj;
      const int hi = hi1 < hi2 ? hi1 : hi2;
      double s = 0.0;
      for (int two_J = lo; two_J <= hi; two_J += 2) {
        const double ybar = yv.get_unnormalized(q, i, p, j, two_J);
        const double xbar = xv.get_unnormalized(q, i, p, j, two_J);
        s += (two_J + 1) * (xab * ybar - yab * xbar);
      }
      z += dn * s;
    }
  };
  if constexpr (exec::kDeviceKernels) {
    // Device shape (M14): one team per (i, j), threads reducing over p (a
    // thread per (i, j) gave a few hundred items). Order differs from the host.
    exec::TeamOptions opts;
    opts.team_size = 64;
    opts.vector_length = 1;
    exec::TeamFor(
        "commutator::comm121ss_team", static_cast<int>(nentries), opts,
        IMSRG_LAMBDA(const exec::TeamMember& member) {
          const exec::Index e = member.league_rank();
          const int i = owner[e];
          const int j = orbits[e];
          const int ji = a.j2[i];
          const int jj = a.j2[j];
          double z = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(member, norb),
              [&](const int p, double& part) {
                add_orbit(i, j, ji, jj, p, part);
              },
              z);
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { z1(i, j) += z / (ji + 1); });
        });
  } else {
    exec::ParallelFor(
        "commutator::comm121ss", nentries, IMSRG_LAMBDA(const exec::Index e) {
          const int i = owner[e];
          const int j = orbits[e];
          const int ji = a.j2[i];
          const int jj = a.j2[j];
          double z = 0.0;
          for (int p = 0; p < norb; ++p) add_orbit(i, j, ji, jj, p, z);
          z1(i, j) += z / (ji + 1);
        });
  }
  exec::Fence("commutator::comm121ss");
}

void Comm122ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  if (!Z.two_body().allocated()) return;
  if (!X.two_body().allocated() && !Y.two_body().allocated()) return;
  const auto& impl = ws.impl();
  const auto x1 = std::as_const(X).one_body().device_view();
  const auto y1 = std::as_const(Y).one_body().device_view();
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  const double* xd = xv.data;
  const double* yd = yv.data;
  double* z2 = Z.two_body().matrix().buffer().device_data();
  const std::int64_t* block_offset = impl.block_offsets.device_data();
  const std::int32_t* num_kets = xv.tb.channel_num_kets;
  const std::int32_t* row_base = impl.lifted.row_base.device_data();
  const std::int32_t* entry_offset = impl.lifted.entry_offset.device_data();
  const std::int32_t* lm = impl.lifted.m.device_data();
  const std::int32_t* lfrom = impl.lifted.from.device_data();
  const std::int32_t* lto = impl.lifted.to.device_data();
  const double* lcoef = impl.lifted.coef.device_data();
  const int nch = xv.tb.num_channels;
  const exec::Index total = static_cast<exec::Index>(impl.two_body_size);
  const bool owned = impl.work.distributed();
  // Distributed: only the columns of the owned comm122 channels (the blocks of
  // the other channels are not written; the scratch of the ket pass is
  // overwritten in the owned blocks before it is read).
  const std::int32_t* owned_columns =
      owned ? impl.work_device.comm122_columns.device_data() : nullptr;
  if (owned && impl.work_device.comm122_columns.size() == 0) return;
  if (detail::IsGeneral(X, Y)) {
    // General X or Y: no symmetry is used. One thread per element (r, c) of
    // every block: the bra part B(r, c) is the lifted one-body list of row r
    // acting on column c of X, Y; the ket part K(r, c) = sum_a (Y_ij,al X_ak +
    // Y_ij,ka X_al) - (X <-> Y) is the list of row c acting on row r of X, Y
    // with the one-body matrix transposed (coef is shared: it depends only on
    // the lifted ket and on the normalization of the row or column).
    Comm122GeneralArgs g;
    g.x1 = x1;
    g.y1 = y1;
    g.xd = xd;
    g.yd = yd;
    g.block_offset = block_offset;
    g.num_kets = num_kets;
    g.row_base = row_base;
    g.entry_offset = entry_offset;
    g.lm = lm;
    g.lfrom = lfrom;
    g.lto = lto;
    g.lcoef = lcoef;
    if (owned) {
      // One item per owned column; the rows are looped inside.
      const std::int32_t* entry_channel = impl.entry_channel.device_data();
      const std::int32_t* ket_offset = xv.tb.channel_ket_offset;
      exec::ParallelForDynamic(
          "commutator::comm122ss_general_owned",
          static_cast<exec::Index>(impl.work_device.comm122_columns.size()),
          IMSRG_LAMBDA(const exec::Index idx) {
            const int e = owned_columns[idx];
            const int ch = entry_channel[e];
            const int c = e - ket_offset[ch];
            const int n = num_kets[ch];
            const std::int64_t off = block_offset[ch];
            for (int r = 0; r < n; ++r)
              z2[off + r + static_cast<std::int64_t>(c) * n] +=
                  g.Element(ch, r, c);
          });
    } else {
      exec::ParallelFor(
          "commutator::comm122ss_general", total,
          IMSRG_LAMBDA(const exec::Index e) {
            const int ch = detail::FindBlock(block_offset, nch, e);
            const std::int64_t local = e - block_offset[ch];
            const int n = num_kets[ch];
            z2[e] += g.Element(ch, static_cast<int>(local % n),
                               static_cast<int>(local / n));
          });
    }
    exec::Fence("commutator::comm122ss");
    return;
  }
  const double hz =
      static_cast<double>(detail::CommutatorHermiticitySign(X, Y));
  // Two passes over the columns of every block (M12a):
  //  1. B(r, c) for all r, c: the bra part, i.e. the lifted one-body list of
  //     row r acting on column c of X and Y (gathers inside one contiguous
  //     column), stored in a scratch with the layout of a two-body operator;
  //  2. Z(r, c) += B(r, c) + hZ B(c, r) for r <= c and Z(c, r) = hZ times
  //     that (the ket part of Z(r, c) is hZ B(c, r) by hermiticity of X, Y).
  // Reading B(c, r) from the scratch replaces the strided reads of rows of X
  // and Y that a single pass needs. The per-element arithmetic is unchanged.
  // The scratch is the ladder's pp product buffer, which no other term needs
  // at this point; pass 2 sets every element it reads back to zero, so the
  // buffer is all zero afterwards (the ladder terms rely on the blocks they
  // do not write being zero, detail/ladder.h).
  double* bsc = ws.impl().ladder.pp.product.device_data();
  const std::int32_t* entry_channel = impl.entry_channel.device_data();
  const std::int32_t* channel_ket_offset = xv.tb.channel_ket_offset;
  const auto columns =
      static_cast<exec::Index>(owned ? impl.work_device.comm122_columns.size()
                                     : impl.entry_channel.size());
  if (exec::kDeviceKernels && !owned) {
    // Device shape (M14): the bra part as GEMMs. B = Lx Y - Ly X per channel,
    // with the lifted one-body operators Lx(r, m) = sum over the lifted
    // entries t of row r with lm[t] = m of lcoef[t] x1(from[t], to[t]) (Ly
    // with y1) stored densely in two-body layout: one thread per row builds
    // its row, then two grouped device GEMMs (the lifted lists are sparse,
    // ~10-20 entries per row, but dense GEMMs on the matrix cores beat the
    // gathers). The summation order differs from the host loops (not
    // bitwise equal to the host shape).
    exec::DeviceBuffer<double> lx(static_cast<std::size_t>(total),
                                  "imsrg::comm122ss::lx");
    exec::DeviceBuffer<double> ly(static_cast<std::size_t>(total),
                                  "imsrg::comm122ss::ly");
    double* lxd = lx.device_data();
    double* lyd = ly.device_data();
    exec::ParallelFor(
        "commutator::comm122ss::lift", columns,
        IMSRG_LAMBDA(const exec::Index e) {
          const int ch = entry_channel[e];
          const std::int64_t off = block_offset[ch];
          const std::int64_t n = num_kets[ch];
          const std::int64_t r = e - channel_ket_offset[ch];
          const std::int64_t rb = row_base[ch];
          for (int t = entry_offset[rb + r]; t < entry_offset[rb + r + 1];
               ++t) {
            const std::int64_t idx = off + r + lm[t] * n;
            lxd[idx] += lcoef[t] * x1(lfrom[t], lto[t]);
            lyd[idx] += lcoef[t] * y1(lfrom[t], lto[t]);
          }
        });
    std::vector<linalg::GemmProblem> first;
    std::vector<linalg::GemmProblem> second;
    for (const auto& info : Z.two_body().matrix().blocks()) {
      const std::int64_t n = info.rows;
      if (n == 0 || info.row_block != info.col_block) continue;
      const std::int64_t off = info.offset;
      const linalg::MatrixView<double> bv{bsc + off, n, n, n};
      if (yd != nullptr) {
        first.push_back(linalg::GemmProblem{
            1.0, linalg::MatrixView<const double>{lxd + off, n, n, n},
            linalg::Op::kNoTrans,
            linalg::MatrixView<const double>{yd + off, n, n, n},
            linalg::Op::kNoTrans, 0.0, bv});
      }
      if (xd != nullptr) {
        second.push_back(linalg::GemmProblem{
            -1.0, linalg::MatrixView<const double>{lyd + off, n, n, n},
            linalg::Op::kNoTrans,
            linalg::MatrixView<const double>{xd + off, n, n, n},
            linalg::Op::kNoTrans, yd != nullptr ? 1.0 : 0.0, bv});
      }
    }
    linalg::DeviceGroupedGemm(first);
    linalg::DeviceGroupedGemm(second);
    exec::ParallelFor(
        "commutator::comm122ss_flat", total, IMSRG_LAMBDA(const exec::Index e) {
          const int ch = detail::FindBlock(block_offset, nch, e);
          const std::int64_t off = block_offset[ch];
          const std::int64_t n = num_kets[ch];
          const std::int64_t local = e - off;
          const std::int64_t c = local / n;
          const std::int64_t r = local - c * n;
          if (r > c) return;  // the item (c, r) writes this element
          double* zb = z2 + off;
          double* bb = bsc + off;
          const double b_rc = bb[r + c * n];
          const double b_cr = bb[c + r * n];
          const double z = b_rc + hz * b_cr;
          zb[r + c * n] += z;
          if (r < c) zb[c + r * n] += hz * z;
        });
    // Pass 2 of the column shape zeroes the scratch as it goes; here no
    // element may be zeroed before its mirror item has read it.
    ws.impl().ladder.pp.product.fill(0.0);
    exec::Fence("commutator::comm122ss");
    return;
  }
  exec::ParallelForDynamic(
      "commutator::comm122ss::bra", columns,
      IMSRG_LAMBDA(const exec::Index col) {
        const exec::Index e =
            owned_columns != nullptr ? owned_columns[col] : col;
        const int ch = entry_channel[e];
        const int c = static_cast<int>(e - channel_ket_offset[ch]);
        const std::int64_t off = block_offset[ch];
        const std::int64_t n = num_kets[ch];
        const int rb = row_base[ch];
        double* bcol = bsc + off + c * n;
        for (int r = 0; r < n; ++r) {
          double b = 0.0;
          for (int t = entry_offset[rb + r]; t < entry_offset[rb + r + 1];
               ++t) {
            const std::int64_t idx = off + lm[t] + c * n;
            double v = 0.0;
            if (yd != nullptr) v += x1(lfrom[t], lto[t]) * yd[idx];
            if (xd != nullptr) v -= y1(lfrom[t], lto[t]) * xd[idx];
            b += lcoef[t] * v;
          }
          bcol[r] = b;
        }
      });
  exec::Fence("commutator::comm122ss::bra");
  exec::ParallelForDynamic(
      "commutator::comm122ss", columns, IMSRG_LAMBDA(const exec::Index col) {
        const exec::Index e =
            owned_columns != nullptr ? owned_columns[col] : col;
        const int ch = entry_channel[e];
        const std::int64_t c = e - channel_ket_offset[ch];
        const std::int64_t off = block_offset[ch];
        const std::int64_t n = num_kets[ch];
        double* zb = z2 + off;
        double* bb = bsc + off;
        for (std::int64_t r = 0; r <= c; ++r) {
          const double b_rc = bb[r + c * n];
          const double b_cr = bb[c + r * n];
          const double z = b_rc + hz * b_cr;
          zb[r + c * n] += z;
          bb[r + c * n] = 0.0;
          if (r < c) {
            zb[c + r * n] += hz * z;
            bb[c + r * n] = 0.0;
          }
        }
      });
  exec::Fence("commutator::comm122ss");
}

}  // namespace imsrg::commutator::terms
