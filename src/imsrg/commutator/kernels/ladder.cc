// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Ladder (pp/hh) terms: comm222_pp_hhss, comm221ss and their fusion (design/12
// section 4.1). Kernel TU (see simple_terms.cc). Both terms are built from the
// same per-channel intermediates
//   Mpp = X P_pp Y - Y P_pp X,   Mhh = X P_hh Y - Y P_hh X
// (P_pp = diag((1 - n_p)(1 - n_q)), P_hh = diag(n_p n_q) over normalized kets),
// computed by
//   1. gathering the columns/rows of X and Y on the kets with non-zero weight,
//   2. one grouped gemm W = Xg Yg per channel and projector,
//   3. the in-place antisymmetrization M = W + h_Z W^T, valid because X^T = h_X
//      X, Y^T = h_Y Y give Y P X = h_X h_Y (X P Y)^T and h_Z = -h_X h_Y.
// For general X or Y (no such symmetry) step 2 also computes W' = Y P X
// (`detail::LadderProductsGeneral`) and step 3 is M = W - W'.

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/device_blas.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator {

namespace {

using detail::LadderProjector;
using detail::TwoBodyView;
using linalg::MatrixView;

/// POD bundle of the gather kernel of one projector.
struct GatherArgs {
  const std::int64_t* gather_offset = nullptr;
  const std::int64_t* list_offset = nullptr;
  const std::int32_t* list = nullptr;
  const std::int32_t* direct_first = nullptr;
  const double* weight = nullptr;
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* ket_offset = nullptr;
  const std::int32_t* entry_channel = nullptr;
  const std::int32_t* num_kets = nullptr;
  const std::int32_t* columns = nullptr;  ///< Active columns (null: all).
  const double* x = nullptr;
  const double* y = nullptr;
  double* xg = nullptr;
  double* yg = nullptr;
};
static_assert(std::is_trivially_copyable_v<GatherArgs>);

/// Gathers the operands of `p` (Xg(i, m) = w_m X(i, K_m), Yg(m, j) = Y(K_m, j))
/// on the device. One work item per two-body column j of every channel (the
/// flat channel-ket table, so no per-element block search or division): it
/// writes column j of Yg (k rows, a gather from the contiguous column j of Y)
/// and, if j < k, column j of Xg (N rows, a weighted copy of the contiguous
/// column K_j of X). Every element of Xg and Yg is written by exactly one item;
/// direct channels (`direct_first >= 0`) are skipped (the gemm reads X and Y in
/// place). Dynamic scheduling: the
/// column costs differ by channel. If `columns` is not null only those global
/// columns (ascending, `num_columns` of them) are processed, the others of Xg
/// and Yg are left as they are. Complexity O(sum_c N_c (N_c + k_c)) over the
/// processed channels.
void Gather(const char* name, LadderProjector& p, const Workspace::Impl& impl,
            const TwoBodyView& xv, const TwoBodyView& yv,
            const std::int32_t* columns, std::int64_t num_columns) {
  const std::int64_t total = p.gather_offset_host.back();
  if (total == 0 || num_columns == 0) return;
  GatherArgs a;
  a.gather_offset = p.gather_offset.device_data();
  a.list_offset = p.list_offset.device_data();
  a.list = p.list.device_data();
  a.direct_first = p.direct_first.device_data();
  a.weight = p.weight.device_data();
  a.block_offset = impl.block_offsets.device_data();
  a.ket_offset = xv.tb.channel_ket_offset;
  a.entry_channel = impl.entry_channel.device_data();
  a.num_kets = xv.tb.channel_num_kets;
  a.columns = columns;
  a.x = xv.data;
  a.y = yv.data;
  a.xg = p.xg.device_data();
  a.yg = p.yg.device_data();
  exec::ParallelForDynamic(
      name, static_cast<exec::Index>(num_columns),
      IMSRG_LAMBDA(const exec::Index idx) {
        const exec::Index e = a.columns != nullptr ? a.columns[idx] : idx;
        const int ch = a.entry_channel[e];
        const std::int64_t k = a.list_offset[ch + 1] - a.list_offset[ch];
        if (k == 0 || a.direct_first[ch] >= 0) return;  // direct: in place
        const std::int64_t j = e - a.ket_offset[ch];
        const std::int64_t n = a.num_kets[ch];
        const std::int32_t* kets = a.list + a.list_offset[ch];
        const double* xb = a.x + a.block_offset[ch];
        const double* yb = a.y + a.block_offset[ch];
        const std::int64_t goff = a.gather_offset[ch];
        // Yg is k x n: column j.
        const double* ycol = yb + j * n;
        double* yg = a.yg + goff + j * k;
        for (std::int64_t m = 0; m < k; ++m)
          yg[m] = ycol[static_cast<std::int64_t>(kets[m])];
        // Xg is n x k: column j (only the first k columns exist).
        if (j < k) {
          const double w = a.weight[a.list_offset[ch] + j];
          const double* xcol = xb + static_cast<std::int64_t>(kets[j]) * n;
          double* xg = a.xg + goff + j * n;
          for (std::int64_t i = 0; i < n; ++i) xg[i] = w * xcol[i];
        }
      });
}

/// POD bundle of the antisymmetrization kernel.
struct SymArgs {
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* ket_offset = nullptr;
  const std::int32_t* entry_channel = nullptr;
  const std::int32_t* num_kets = nullptr;
  double* mpp = nullptr;
  double* mhh = nullptr;
  double* z2 = nullptr;  ///< Optional: Z2 += Mpp - Mhh (nullptr: skip).
  const std::int32_t* columns = nullptr;  ///< Active columns (null: all).
  double h_z = 1.0;
};
static_assert(std::is_trivially_copyable_v<SymArgs>);

/// POD bundle of the zeroing kernel.
struct ZeroArgs {
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* ket_offset = nullptr;
  const std::int32_t* entry_channel = nullptr;
  const std::int32_t* num_kets = nullptr;
  const std::int32_t* columns = nullptr;
  double* mpp = nullptr;
  double* mhh = nullptr;
};
static_assert(std::is_trivially_copyable_v<ZeroArgs>);

/// True if the distributed work assignment of `impl` applies to a call with
/// `scope`.
bool UseOwned(const Workspace::Impl& impl, detail::LadderScope scope) {
  return scope == detail::LadderScope::kOwned && impl.work.distributed();
}

}  // namespace

namespace detail {

namespace {

/// Zeroes the full blocks of the active ladder channels in both product
/// buffers (one work item per active column). Needed on every distributed call
/// because the antisymmetrization writes transposes (rows of other parts) and
/// the rows outside the owned blocks are never written by the gemms.
/// Complexity O(sum over active channels of N_c^2).
void ZeroActiveBlocks(Workspace::Impl& impl, const TwoBodyView& xv) {
  const auto& cols = impl.work_device.ladder_columns;
  if (cols.size() == 0) return;
  ZeroArgs z;
  z.block_offset = impl.block_offsets.device_data();
  z.ket_offset = xv.tb.channel_ket_offset;
  z.entry_channel = impl.entry_channel.device_data();
  z.num_kets = xv.tb.channel_num_kets;
  z.columns = cols.device_data();
  z.mpp = impl.ladder.pp.product.device_data();
  z.mhh = impl.ladder.hh.product.device_data();
  exec::ParallelFor(
      "commutator::ladder_zero_active", static_cast<exec::Index>(cols.size()),
      IMSRG_LAMBDA(const exec::Index idx) {
        const int ch = z.entry_channel[z.columns[idx]];
        const std::int64_t n = z.num_kets[ch];
        const std::int64_t j = z.columns[idx] - z.ket_offset[ch];
        const std::int64_t base = z.block_offset[ch] + j * n;
        for (std::int64_t i = 0; i < n; ++i) {
          z.mpp[base + i] = 0.0;
          z.mhh[base + i] = 0.0;
        }
      });
  exec::Fence("commutator::ladder_zero_active");
}

/// Number of kets N_c of channel `ch` from the block size N_c^2 (0 if empty).
std::int64_t BlockDim(const Workspace::Impl& impl, int ch) {
  const auto c = static_cast<std::size_t>(ch);
  const std::int64_t n =
      impl.block_offsets_host[c + 1] - impl.block_offsets_host[c];
  std::int64_t dim = 0;
  while (dim * dim < n) ++dim;
  return dim;
}

/// Appends the gemm C(r0:r1, :) = alpha Xg(r0:r1, :) Yg + beta C(r0:r1, :) of
/// channel `ch` and projector `p` to `lad.problems`; skips channels without
/// listed kets (their blocks stay zero) and empty row ranges. `dim` = N_c;
/// rows [0, dim) give the whole block. Direct channels read X and Y in place.
void AppendGemm(detail::LadderData& lad, const Workspace::Impl& impl,
                LadderProjector& p, const TwoBodyView& xv,
                const TwoBodyView& yv, int ch, std::int64_t r0, std::int64_t r1,
                double alpha, double beta) {
  const auto c = static_cast<std::size_t>(ch);
  const std::int64_t k = p.count_host[c];
  if (k == 0 || r1 <= r0) return;
  const std::int64_t off = impl.block_offsets_host[c];
  const std::int64_t dim = BlockDim(impl, ch);
  linalg::GemmProblem g;
  g.alpha = alpha;
  const std::int32_t k0 = p.direct_first_host[c];
  const std::int64_t rows = r1 - r0;
  if (k0 >= 0) {
    // Direct channel: Xg = X(:, K0 .. K0 + k - 1), Yg = Y(K0 .. , :).
    g.A = MatrixView<const double>{xv.data + off + k0 * dim + r0, rows, k, dim};
    g.B = MatrixView<const double>{yv.data + off + k0, k, dim, dim};
  } else {
    const std::int64_t goff = p.gather_offset_host[c];
    g.A =
        MatrixView<const double>{p.xg.device_data() + goff + r0, rows, k, dim};
    g.B = MatrixView<const double>{p.yg.device_data() + goff, k, dim, k};
  }
  g.beta = beta;
  g.C = MatrixView<double>{p.product.device_data() + off + r0, rows, dim, dim};
  lad.problems.push_back(g);
}

/// product = alpha X P Y + beta product per channel and projector for the
/// gathered operands (beta = 0: overwrite, beta = 1: accumulate). With `owned`
/// only the active channels are gathered and only the owned row blocks are
/// multiplied. See `LadderProducts`.
void ProductsInto(const op::Operator& X, const op::Operator& Y,
                  Workspace::Impl& impl, double alpha, double beta,
                  bool owned) {
  auto& lad = impl.ladder;
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  const std::int32_t* cols =
      owned ? impl.work_device.ladder_columns.device_data() : nullptr;
  const auto ncols =
      static_cast<std::int64_t>(owned ? impl.work_device.ladder_columns.size()
                                      : impl.entry_channel.size());
  Gather("commutator::ladder_gather_pp", lad.pp, impl, xv, yv, cols, ncols);
  Gather("commutator::ladder_gather_hh", lad.hh, impl, xv, yv, cols, ncols);
  exec::Fence("commutator::ladder_gather");

  // W = Xg Yg per channel and projector. A channel without listed kets (k == 0)
  // is skipped: its product block stays zero (zero-initialized at construction,
  // written only by these gemms and the symmetrization, which maps 0 to 0).
  lad.problems.clear();
  for (LadderProjector* p : {&lad.pp, &lad.hh}) {
    if (owned) {
      for (const detail::RowBlock& b : impl.work.ladder_blocks())
        AppendGemm(lad, impl, *p, xv, yv, b.channel, b.row_begin, b.row_end,
                   alpha, beta);
    } else {
      for (int ch = 0; ch < xv.tb.num_channels; ++ch)
        AppendGemm(lad, impl, *p, xv, yv, ch, 0, BlockDim(impl, ch), alpha,
                   beta);
    }
  }
  linalg::DeviceGroupedGemm(lad.problems);
}

}  // namespace

void LadderProducts(const op::Operator& X, const op::Operator& Y,
                    Workspace::Impl& impl, LadderScope scope) {
  const bool owned = UseOwned(impl, scope);
  if (owned) ZeroActiveBlocks(impl, impl.two_body_view(X));
  ProductsInto(X, Y, impl, 1.0, 0.0, owned);
}

void LadderProductsGeneral(const op::Operator& X, const op::Operator& Y,
                           Workspace::Impl& impl, LadderScope scope) {
  // M = X P Y - Y P X: the second grouped gemm accumulates with alpha = -1.
  // Blocks without listed kets are skipped by both: they stay zero.
  const bool owned = UseOwned(impl, scope);
  if (owned) ZeroActiveBlocks(impl, impl.two_body_view(X));
  ProductsInto(X, Y, impl, 1.0, 0.0, owned);
  ProductsInto(Y, X, impl, -1.0, 1.0, owned);
}

}  // namespace detail

namespace terms {

namespace {

/// Z2 += Mpp - Mhh.
void AddLadderTwoBody(double* z2, Workspace::Impl& impl) {
  const double* mpp = impl.ladder.pp.product.device_data();
  const double* mhh = impl.ladder.hh.product.device_data();
  exec::ParallelFor(
      "commutator::comm222_pp_hhss",
      static_cast<exec::Index>(impl.two_body_size),
      IMSRG_LAMBDA(const exec::Index e) { z2[e] += mpp[e] - mhh[e]; });
  exec::Fence("commutator::comm222_pp_hhss");
}

/// Distributed form of `AddLadderTwoBody`: only the columns of the active
/// channels (`work_device.ladder_columns`); the other blocks of the product
/// buffers may hold stale data. One work item per column,
/// O(sum over active channels of N_c^2).
void AddLadderTwoBodyOwned(double* z2, Workspace::Impl& impl) {
  const double* mpp = impl.ladder.pp.product.device_data();
  const double* mhh = impl.ladder.hh.product.device_data();
  const std::int64_t* block_offset = impl.block_offsets.device_data();
  const std::int32_t* entry_channel = impl.entry_channel.device_data();
  const auto tb = impl.tables.two_body();
  const std::int32_t* ket_offset = tb.channel_ket_offset;
  const std::int32_t* num_kets = tb.channel_num_kets;
  const std::int32_t* columns = impl.work_device.ladder_columns.device_data();
  exec::ParallelForDynamic(
      "commutator::comm222_pp_hhss_owned",
      static_cast<exec::Index>(impl.work_device.ladder_columns.size()),
      IMSRG_LAMBDA(const exec::Index idx) {
        const int e = columns[idx];
        const int ch = entry_channel[e];
        const std::int64_t n = num_kets[ch];
        const std::int64_t base = block_offset[ch] + (e - ket_offset[ch]) * n;
        for (std::int64_t i = 0; i < n; ++i)
          z2[base + i] += mpp[base + i] - mhh[base + i];
      });
  exec::Fence("commutator::comm222_pp_hhss_owned");
}

/// Computes Mpp and Mhh for X and Y into `impl.ladder.{pp,hh}.product`. X and Y
/// must have an allocated two-body part. If `z2` is not null, also adds
/// Z2 += Mpp - Mhh there (fused into the antisymmetrization on the Hermitian
/// path). (Anti-)Hermitian X, Y: one product and
/// the antisymmetrization M = W + hZ W^T; general X or Y: both products
/// (`detail::LadderProductsGeneral`). Leaves the device idle (fenced).
/// Complexity: two grouped gemms of sum_c N_c^2 (|Kp| + |Kh|) flops.
void ComputeIntermediates(const op::Operator& X, const op::Operator& Y,
                          Workspace::Impl& impl, double* z2) {
  const bool owned = impl.work.distributed();
  if (detail::IsGeneral(X, Y)) {
    detail::LadderProductsGeneral(X, Y, impl, detail::LadderScope::kOwned);
    if (z2 == nullptr) return;
    if (owned) {
      AddLadderTwoBodyOwned(z2, impl);
    } else {
      AddLadderTwoBody(z2, impl);
    }
    return;
  }
  const int h_z = detail::CommutatorHermiticitySign(X, Y);
  // W = X P Y for P = P_pp, P_hh
  detail::LadderProducts(X, Y, impl, detail::LadderScope::kOwned);

  auto& lad = impl.ladder;
  const TwoBodyView xv = impl.two_body_view(X);
  SymArgs s;
  s.block_offset = impl.block_offsets.device_data();
  s.ket_offset = xv.tb.channel_ket_offset;
  s.entry_channel = impl.entry_channel.device_data();
  s.num_kets = xv.tb.channel_num_kets;
  s.mpp = lad.pp.product.device_data();
  s.mhh = lad.hh.product.device_data();
  s.z2 = z2;
  s.columns = owned ? impl.work_device.ladder_columns.device_data() : nullptr;
  s.h_z = static_cast<double>(h_z);
  // One item per column j of every block; it owns the pairs (i, j) and (j, i)
  // for i <= j, so every element is read and written by exactly one item. The
  // optional Z2 update uses the final values of both elements.
  if (exec::kDeviceKernels && !owned) {
    // Device shape (M14): one thread per element (i, j); the items i <= j do
    // the work of the column loop below (same arithmetic).
    const int nch = xv.tb.num_channels;
    exec::ParallelFor(
        "commutator::ladder_antisymmetrize_flat",
        static_cast<exec::Index>(impl.two_body_size),
        IMSRG_LAMBDA(const exec::Index e) {
          const int ch = detail::FindBlock(s.block_offset, nch, e);
          const std::int64_t base = s.block_offset[ch];
          const std::int64_t n = s.num_kets[ch];
          const std::int64_t local = e - base;
          const std::int64_t j = local / n;
          const std::int64_t i = local - j * n;
          if (i > j) return;
          const std::int64_t ij = base + i + j * n;
          const std::int64_t ji = base + j + i * n;
          const double a_pp = s.mpp[ij];
          const double b_pp = s.mpp[ji];
          const double a_hh = s.mhh[ij];
          const double b_hh = s.mhh[ji];
          const double c_pp = a_pp + s.h_z * b_pp;
          const double c_hh = a_hh + s.h_z * b_hh;
          s.mpp[ij] = c_pp;
          s.mhh[ij] = c_hh;
          if (s.z2 != nullptr) s.z2[ij] += c_pp - c_hh;
          if (i != j) {
            const double d_pp = b_pp + s.h_z * a_pp;
            const double d_hh = b_hh + s.h_z * a_hh;
            s.mpp[ji] = d_pp;
            s.mhh[ji] = d_hh;
            if (s.z2 != nullptr) s.z2[ji] += d_pp - d_hh;
          }
        });
    exec::Fence("commutator::ladder_antisymmetrize");
    return;
  }
  exec::ParallelForDynamic(
      "commutator::ladder_antisymmetrize",
      static_cast<exec::Index>(owned ? impl.work_device.ladder_columns.size()
                                     : impl.entry_channel.size()),
      IMSRG_LAMBDA(const exec::Index idx) {
        const exec::Index e = s.columns != nullptr ? s.columns[idx] : idx;
        const int ch = s.entry_channel[e];
        const std::int64_t n = s.num_kets[ch];
        const std::int64_t j = e - s.ket_offset[ch];
        const std::int64_t base = s.block_offset[ch];
        for (std::int64_t i = 0; i <= j; ++i) {
          const std::int64_t ij = base + i + j * n;
          const std::int64_t ji = base + j + i * n;
          const double a_pp = s.mpp[ij];
          const double b_pp = s.mpp[ji];
          const double a_hh = s.mhh[ij];
          const double b_hh = s.mhh[ji];
          const double c_pp = a_pp + s.h_z * b_pp;
          const double c_hh = a_hh + s.h_z * b_hh;
          s.mpp[ij] = c_pp;
          s.mhh[ij] = c_hh;
          if (s.z2 != nullptr) s.z2[ij] += c_pp - c_hh;
          if (i != j) {
            const double d_pp = b_pp + s.h_z * a_pp;
            const double d_hh = b_hh + s.h_z * a_hh;
            s.mpp[ji] = d_pp;
            s.mhh[ji] = d_hh;
            if (s.z2 != nullptr) s.z2[ji] += d_pp - d_hh;
          }
        }
      });
  exec::Fence("commutator::ladder_antisymmetrize");
}

/// POD bundle of the 221 kernel.
struct Ladder221Args {
  TwoBodyView mpp;
  TwoBodyView mhh;
  const double* occ = nullptr;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* owner = nullptr;
  const std::int32_t* orbits = nullptr;
  const std::int32_t* active = nullptr;  ///< Active channels (null: all).
  MatrixView<double> z1;
  int num_orbits = 0;
};
static_assert(std::is_trivially_copyable_v<Ladder221Args>);

/// Channel of the bra pair (c, i) coupled to 2J = `two_J` (the pair in either
/// order), or -1 if that ket does not exist (the element is then zero). O(1).
IMSRG_INLINE_FUNCTION int BraChannel(const TwoBodyView& v, int c, int i,
                                     int two_J) {
  const int a = c < i ? c : i;
  const int b = c < i ? i : c;
  const std::int32_t ib = v.tb.local_index(
      v.tb.ket_index(OrbitIndex{a}, OrbitIndex{b}), TwoJ{two_J});
  if (ib < 0) return -1;
  return v.tb.channel_of(OrbitIndex{a}, OrbitIndex{b}, TwoJ{two_J}).value();
}

/// Z1 += sum_c sum_J (2J + 1) / (2 j_i + 1) (n_c Mpp~(ci,cj;J) + (1 - n_c)
/// Mhh~(ci,cj;J)). Distributed: only the (c, J) of active channels.
/// Adds orbit c's contribution to Z1(i, j) of comm221ss (the J sum), in
/// the order of the host loop.
IMSRG_FORCEINLINE_FUNCTION void Add221Orbit(const Ladder221Args& a, int i,
                                            int j, int ji, int jj, int c,
                                            double& z) {
  const double wp = a.occ[c];
  const double wh = 1.0 - a.occ[c];
  const int jc = a.j2[c];
  const int lo1 = jc > ji ? jc - ji : ji - jc;
  const int lo2 = jc > jj ? jc - jj : jj - jc;
  const int lo = lo1 > lo2 ? lo1 : lo2;
  const int hi1 = jc + ji;
  const int hi2 = jc + jj;
  const int hi = hi1 < hi2 ? hi1 : hi2;
  for (int two_J = lo; two_J <= hi; two_J += 2) {
    if (a.active != nullptr) {
      // Distributed: channels without owned blocks hold stale data.
      const int ch = BraChannel(a.mpp, c, i, two_J);
      if (ch < 0 || a.active[ch] == 0) continue;
    }
    double s = 0.0;
    if (wp != 0.0) s += wp * a.mpp.get_unnormalized(c, i, c, j, two_J);
    if (wh != 0.0) s += wh * a.mhh.get_unnormalized(c, i, c, j, two_J);
    z += (two_J + 1) * s;
  }
}

void Add221(op::Operator& Z, Workspace::Impl& impl, const op::Operator& X) {
  Ladder221Args a;
  a.mpp = impl.two_body_view(X);
  a.mpp.data = impl.ladder.pp.product.device_data();
  a.mhh = a.mpp;
  a.mhh.data = impl.ladder.hh.product.device_data();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.owner = impl.ob_owner.device_data();
  a.orbits = impl.ob_orbits.device_data();
  if (impl.work.distributed())
    a.active = impl.work_device.ladder_active.device_data();
  a.z1 = Z.one_body().device_view();
  a.num_orbits = impl.tables.orbits().num_orbits;
  const auto items = static_cast<exec::Index>(impl.ob_orbits.size());
  if constexpr (exec::kDeviceKernels) {
    // Device shape (M14): one team per (i, j), its threads reducing over the
    // orbits c (a thread per (i, j) gave the GPU a few hundred items). The
    // reduction order differs from the host loop.
    exec::TeamOptions opts;
    opts.team_size = 64;
    opts.vector_length = 1;
    exec::TeamFor(
        "commutator::comm221ss_team", static_cast<int>(items), opts,
        IMSRG_LAMBDA(const exec::TeamMember& member) {
          const exec::Index e = member.league_rank();
          const int i = a.owner[e];
          const int j = a.orbits[e];
          const int ji = a.j2[i];
          const int jj = a.j2[j];
          double z = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(member, a.num_orbits),
              [&](const int c, double& part) {
                Add221Orbit(a, i, j, ji, jj, c, part);
              },
              z);
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { a.z1(i, j) += z / (ji + 1); });
        });
  } else {
    exec::ParallelFor(
        "commutator::comm221ss", items, IMSRG_LAMBDA(const exec::Index e) {
          const int i = a.owner[e];
          const int j = a.orbits[e];
          const int ji = a.j2[i];
          const int jj = a.j2[j];
          double z = 0.0;
          for (int c = 0; c < a.num_orbits; ++c)
            Add221Orbit(a, i, j, ji, jj, c, z);
          a.z1(i, j) += z / (ji + 1);
        });
  }
  exec::Fence("commutator::comm221ss");
}

bool HasTwoBody(const op::Operator& X, const op::Operator& Y) {
  return X.two_body().allocated() && Y.two_body().allocated();
}

}  // namespace

void Comm222PpHhss(const op::Operator& X, const op::Operator& Y,
                   op::Operator& Z, Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  if (!HasTwoBody(X, Y) || !Z.two_body().allocated()) return;
  auto& impl = ws.impl();
  ComputeIntermediates(X, Y, impl,
                       Z.two_body().matrix().buffer().device_data());
}

void Comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  if (!HasTwoBody(X, Y)) return;
  auto& impl = ws.impl();
  ComputeIntermediates(X, Y, impl, nullptr);
  Add221(Z, impl, X);
}

void Comm222PpHh221ss(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, Workspace& ws) {
  detail::CheckTermInputs(X, Y, Z, ws);
  if (!HasTwoBody(X, Y)) return;
  auto& impl = ws.impl();
  ComputeIntermediates(X, Y, impl,
                       Z.two_body().allocated()
                           ? Z.two_body().matrix().buffer().device_data()
                           : nullptr);
  Add221(Z, impl, X);
}

}  // namespace terms

}  // namespace imsrg::commutator
