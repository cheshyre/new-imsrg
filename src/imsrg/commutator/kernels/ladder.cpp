// Ladder (pp/hh) terms: comm222_pp_hhss, comm221ss and their fusion (design/12 section 4.1).
// Kernel TU (see simple_terms.cpp). Both terms are built from the same per-channel intermediates
//   Mpp = X P_pp Y - Y P_pp X,   Mhh = X P_hh Y - Y P_hh X
// (P_pp = diag((1 - n_p)(1 - n_q)), P_hh = diag(n_p n_q) over normalized kets), computed by
//   1. gathering the columns/rows of X and Y on the kets with non-zero weight,
//   2. one grouped gemm W = Xg Yg per channel and projector,
//   3. the in-place antisymmetrization M = W + h_Z W^T, valid because X^T = h_X X, Y^T = h_Y Y
//      give Y P X = h_X h_Y (X P Y)^T and h_Z = -h_X h_Y.

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "imsrg/base/assert.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/device_views.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"
#include "imsrg/exec/kokkos_backend.hpp"
#include "imsrg/linalg/blas.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::commutator::terms {

namespace {

using detail::find_block;
using detail::LadderProjector;
using detail::TwoBodyView;
using linalg::MatrixView;

/// POD bundle of the gather kernel of one projector.
struct GatherArgs {
  const std::int64_t* gather_offset = nullptr;
  const std::int64_t* list_offset = nullptr;
  const std::int32_t* list = nullptr;
  const double* weight = nullptr;
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* num_kets = nullptr;
  const double* x = nullptr;
  const double* y = nullptr;
  double* xg = nullptr;
  double* yg = nullptr;
  int num_channels = 0;
};
static_assert(std::is_trivially_copyable_v<GatherArgs>);

/// Gathers the operands of `p` (Xg(i, m) = w_m X(i, K_m), Yg(m, j) = Y(K_m, j)) on the device.
void gather(const char* name, LadderProjector& p, const Workspace::Impl& impl,
            const TwoBodyView& xv, const TwoBodyView& yv) {
  const std::int64_t total = p.gather_offset_host.back();
  if (total == 0) return;
  GatherArgs a;
  a.gather_offset = p.gather_offset.device_data();
  a.list_offset = p.list_offset.device_data();
  a.list = p.list.device_data();
  a.weight = p.weight.device_data();
  a.block_offset = impl.block_offsets.device_data();
  a.num_kets = xv.tb.channel_num_kets;
  a.x = xv.data;
  a.y = yv.data;
  a.xg = p.xg.device_data();
  a.yg = p.yg.device_data();
  a.num_channels = xv.tb.num_channels;
  exec::parallel_for(
      name, static_cast<exec::Index>(total), IMSRG_LAMBDA(const exec::Index e) {
        const int ch = find_block(a.gather_offset, a.num_channels, e);
        const std::int64_t local = e - a.gather_offset[ch];
        const std::int64_t n = a.num_kets[ch];
        const std::int64_t k = (a.list_offset[ch + 1] - a.list_offset[ch]);
        const std::int32_t* kets = a.list + a.list_offset[ch];
        const double* w = a.weight + a.list_offset[ch];
        const double* xb = a.x + a.block_offset[ch];
        const double* yb = a.y + a.block_offset[ch];
        // Xg is n x k: element (i, m).
        {
          const std::int64_t i = local % n;
          const std::int64_t m = local / n;
          a.xg[e] = w[m] * xb[i + static_cast<std::int64_t>(kets[m]) * n];
        }
        // Yg is k x n: element (m, j).
        {
          const std::int64_t m = local % k;
          const std::int64_t j = local / k;
          a.yg[e] = yb[static_cast<std::int64_t>(kets[m]) + j * n];
        }
      });
}

/// POD bundle of the antisymmetrization kernel.
struct SymArgs {
  const std::int64_t* block_offset = nullptr;
  const std::int32_t* num_kets = nullptr;
  double* mpp = nullptr;
  double* mhh = nullptr;
  double h_z = 1.0;
  int num_channels = 0;
};
static_assert(std::is_trivially_copyable_v<SymArgs>);

/// Computes Mpp and Mhh for X and Y into `impl.ladder.{pp,hh}.product`. X and Y must have an
/// allocated two-body part; `h_z` is the hermiticity sign of the commutator. Leaves the device
/// idle (fenced). Complexity: two grouped gemms of sum_c N_c^2 (|Kp| + |Kh|) flops.
void compute_intermediates(const op::Operator& X, const op::Operator& Y, int h_z,
                           Workspace::Impl& impl) {
  auto& lad = impl.ladder;
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  gather("commutator::ladder_gather_pp", lad.pp, impl, xv, yv);
  gather("commutator::ladder_gather_hh", lad.hh, impl, xv, yv);
  exec::fence("commutator::ladder_gather");

  // W = Xg Yg per channel and projector. A channel without listed kets (k == 0) is skipped: its
  // product block stays zero (zero-initialized at construction, written only by these gemms and
  // the symmetrization, which maps 0 to 0).
  lad.problems.clear();
  const int nch = xv.tb.num_channels;
  for (LadderProjector* p : {&lad.pp, &lad.hh}) {
    double* prod = p->product.device_data();
    const double* xg = p->xg.device_data();
    const double* yg = p->yg.device_data();
    for (int ch = 0; ch < nch; ++ch) {
      const std::int64_t n = impl.block_offsets_host[static_cast<std::size_t>(ch) + 1] -
                             impl.block_offsets_host[static_cast<std::size_t>(ch)];
      if (n == 0) continue;
      std::int64_t dim = 1;
      while (dim * dim < n) ++dim;  // block size n = N^2
      const std::int64_t k = p->count_host[static_cast<std::size_t>(ch)];
      if (k == 0) continue;
      const std::int64_t goff = p->gather_offset_host[static_cast<std::size_t>(ch)];
      linalg::GemmProblem g;
      g.alpha = 1.0;
      g.A = MatrixView<const double>{xg + goff, dim, k, dim};
      g.B = MatrixView<const double>{yg + goff, k, dim, k};
      g.beta = 0.0;
      g.C = MatrixView<double>{prod + impl.block_offsets_host[static_cast<std::size_t>(ch)], dim,
                               dim, dim};
      lad.problems.push_back(g);
    }
  }
  // Host BLAS only; device backends need a device gemm (design/12 section 3).
  IMSRG_CHECK(lad.pp.product.host_is_device(),
              "ladder: host BLAS on device memory; needs a device gemm (GPU backends)");
  linalg::grouped_gemm(lad.problems);

  SymArgs s;
  s.block_offset = impl.block_offsets.device_data();
  s.num_kets = xv.tb.channel_num_kets;
  s.mpp = lad.pp.product.device_data();
  s.mhh = lad.hh.product.device_data();
  s.h_z = static_cast<double>(h_z);
  s.num_channels = nch;
  exec::parallel_for(
      "commutator::ladder_antisymmetrize", static_cast<exec::Index>(impl.two_body_size),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = find_block(s.block_offset, s.num_channels, e);
        const std::int64_t local = e - s.block_offset[ch];
        const std::int64_t n = s.num_kets[ch];
        const std::int64_t i = local % n;
        const std::int64_t j = local / n;
        if (i > j) return;  // handled together with (j, i)
        const std::int64_t base = s.block_offset[ch];
        const std::int64_t ij = base + i + j * n;
        const std::int64_t ji = base + j + i * n;
        const double a_pp = s.mpp[ij];
        const double b_pp = s.mpp[ji];
        const double a_hh = s.mhh[ij];
        const double b_hh = s.mhh[ji];
        s.mpp[ij] = a_pp + s.h_z * b_pp;
        s.mhh[ij] = a_hh + s.h_z * b_hh;
        if (i != j) {
          s.mpp[ji] = b_pp + s.h_z * a_pp;
          s.mhh[ji] = b_hh + s.h_z * a_hh;
        }
      });
  exec::fence("commutator::ladder_antisymmetrize");
}

/// Z2 += Mpp - Mhh.
void add_ladder_two_body(op::Operator& Z, Workspace::Impl& impl) {
  double* z2 = Z.two_body().matrix().buffer().device_data();
  const double* mpp = impl.ladder.pp.product.device_data();
  const double* mhh = impl.ladder.hh.product.device_data();
  exec::parallel_for(
      "commutator::comm222_pp_hhss", static_cast<exec::Index>(impl.two_body_size),
      IMSRG_LAMBDA(const exec::Index e) { z2[e] += mpp[e] - mhh[e]; });
  exec::fence("commutator::comm222_pp_hhss");
}

/// POD bundle of the 221 kernel.
struct Ladder221Args {
  TwoBodyView mpp;
  TwoBodyView mhh;
  const double* occ = nullptr;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* owner = nullptr;
  const std::int32_t* orbits = nullptr;
  MatrixView<double> z1;
  int num_orbits = 0;
};
static_assert(std::is_trivially_copyable_v<Ladder221Args>);

/// Z1 += sum_c sum_J (2J + 1) / (2 j_i + 1) (n_c Mpp~(ci,cj;J) + (1 - n_c) Mhh~(ci,cj;J)).
void add_221(op::Operator& Z, Workspace::Impl& impl, const op::Operator& X) {
  Ladder221Args a;
  a.mpp = impl.two_body_view(X);
  a.mpp.data = impl.ladder.pp.product.device_data();
  a.mhh = a.mpp;
  a.mhh.data = impl.ladder.hh.product.device_data();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.owner = impl.ob_owner.device_data();
  a.orbits = impl.ob_orbits.device_data();
  a.z1 = Z.one_body().device_view();
  a.num_orbits = impl.tables.orbits().num_orbits;
  exec::parallel_for(
      "commutator::comm221ss", static_cast<exec::Index>(impl.ob_orbits.size()),
      IMSRG_LAMBDA(const exec::Index e) {
        const int i = a.owner[e];
        const int j = a.orbits[e];
        const int ji = a.j2[i];
        const int jj = a.j2[j];
        double z = 0.0;
        for (int c = 0; c < a.num_orbits; ++c) {
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
            double s = 0.0;
            if (wp != 0.0) s += wp * a.mpp.get_unnormalized(c, i, c, j, two_J);
            if (wh != 0.0) s += wh * a.mhh.get_unnormalized(c, i, c, j, two_J);
            z += (two_J + 1) * s;
          }
        }
        a.z1(i, j) += z / (ji + 1);
      });
  exec::fence("commutator::comm221ss");
}

bool has_two_body(const op::Operator& X, const op::Operator& Y) {
  return X.two_body().allocated() && Y.two_body().allocated();
}

/// hZ = -hX hY from X and Y (Z's flag is validated against it by `check_term_inputs`).
int z_sign(const op::Operator& X, const op::Operator& Y) {
  return detail::commutator_hermiticity_sign(X, Y);
}

}  // namespace

void comm222_pp_hhss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  if (!has_two_body(X, Y) || !Z.two_body().allocated()) return;
  auto& impl = ws.impl();
  compute_intermediates(X, Y, z_sign(X, Y), impl);
  add_ladder_two_body(Z, impl);
}

void comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  if (!has_two_body(X, Y)) return;
  auto& impl = ws.impl();
  compute_intermediates(X, Y, z_sign(X, Y), impl);
  add_221(Z, impl, X);
}

void comm222_pp_hh_221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                         Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  if (!has_two_body(X, Y)) return;
  auto& impl = ws.impl();
  compute_intermediates(X, Y, z_sign(X, Y), impl);
  if (Z.two_body().allocated()) add_ladder_two_body(Z, impl);
  add_221(Z, impl, X);
}

}  // namespace imsrg::commutator::terms
