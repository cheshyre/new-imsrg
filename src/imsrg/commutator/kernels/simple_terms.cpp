// The one-, two-body-reduced and mixed IMSRG(2) terms: comm110ss, comm220ss, comm111ss, comm121ss,
// comm122ss (design/12 section 4.1). Kernel TU: the only place of this module (with ladder.cpp and
// particle_hole.cpp) that includes the Kokkos vocabulary.

#include <cmath>
#include <cstdint>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/constants.hpp"
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/commutator/detail/device_views.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"
#include "imsrg/exec/kokkos_backend.hpp"
#include "imsrg/linalg/blas.hpp"

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

}  // namespace

void comm110ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  // [X,Y] is anti-Hermitian (hZ = -hX hY = -1) -> the scalar part vanishes.
  if (detail::commutator_hermiticity_sign(X, Y) < 0) return;
  const auto& impl = ws.impl();
  OneBodyArgs a;
  a.x = std::as_const(X).one_body().device_view();
  a.y = std::as_const(Y).one_body().device_view();
  a.occ = impl.tables.orbits().occupation;
  a.j2 = impl.tables.orbits().j2;
  a.ch1 = impl.one_body_channels();
  const exec::Index norb = impl.tables.orbits().num_orbits;
  double sum = 0.0;
  exec::parallel_reduce(
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

void comm220ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  // [X,Y] is anti-Hermitian (hZ = -hX hY = -1) -> the scalar part vanishes.
  if (detail::commutator_hermiticity_sign(X, Y) < 0) return;
  if (!X.two_body().allocated() || !Y.two_body().allocated()) return;
  const auto& impl = ws.impl();
  const TwoBodyView xv = impl.two_body_view(X);
  const TwoBodyView yv = impl.two_body_view(Y);
  const double* occ = impl.tables.orbits().occupation;
  const std::int32_t* entry_channel = impl.entry_channel.device_data();
  const exec::Index nentries = static_cast<exec::Index>(impl.entry_channel.size());
  double sum = 0.0;
  exec::parallel_reduce(
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
        double s = 0.0;
        for (int k = 0; k < n; ++k) {
          const int pk = xv.tb.ket_p[kets[k]];
          const int qk = xv.tb.ket_q[kets[k]];
          const double nb_k = (1.0 - occ[pk]) * (1.0 - occ[qk]);
          const double w = nn_i * nb_k;
          if (w == 0.0) continue;
          s +=
              w *
              (xb[i + static_cast<std::int64_t>(k) * n] * yb[k + static_cast<std::int64_t>(i) * n] -
               yb[i + static_cast<std::int64_t>(k) * n] * xb[k + static_cast<std::int64_t>(i) * n]);
        }
        partial += (xv.tb.channel_two_j[ch] + 1) * s;
      },
      sum);
  Z.zero_body() += sum;
}

void comm111ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  // Host BLAS on the one-body matrices (tiny: norb^3). GPU backends will route gemm to device
  // BLAS; until then wait for pending device work before touching the data.
  exec::fence("commutator::comm111ss");
  const auto x1 = std::as_const(X).one_body().device_view();
  const auto y1 = std::as_const(Y).one_body().device_view();
  const auto z1 = Z.one_body().device_view();
  // Host BLAS only: on a device backend the buffers are not host-addressable (design/12 section 3).
  IMSRG_CHECK(Z.one_body().buffer().host_is_device(),
              "comm111ss: host BLAS on device memory; needs a device gemm (GPU backends)");
  linalg::gemm(1.0, x1, linalg::Op::NoTrans, y1, linalg::Op::NoTrans, 1.0, z1);
  linalg::gemm(-1.0, y1, linalg::Op::NoTrans, x1, linalg::Op::NoTrans, 1.0, z1);
}

void comm121ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
  // [X1,Y2] and [X2,Y1] each need only one two-body operand (a missing part reads as zero).
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
  exec::parallel_for(
      "commutator::comm121ss", nentries, IMSRG_LAMBDA(const exec::Index e) {
        const int i = owner[e];
        const int j = orbits[e];
        const int ji = a.j2[i];
        const int jj = a.j2[j];
        double z = 0.0;
        for (int p = 0; p < norb; ++p) {
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
        }
        z1(i, j) += z / (ji + 1);
      });
  exec::fence("commutator::comm121ss");
}

void comm122ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws) {
  detail::check_term_inputs(X, Y, Z, ws);
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
  const double hz = static_cast<double>(detail::commutator_hermiticity_sign(X, Y));
  const int nch = xv.tb.num_channels;
  const exec::Index total = static_cast<exec::Index>(impl.two_body_size);
  // One thread per element (r, c) with r <= c. B(r, c) is the bra part (lifted one-body list of
  // row r acting on column c of X, Y); the ket part of Z(r, c) is hZ B(c, r) by hermiticity of X,
  // Y (checked by check_term_inputs), and Z(c, r) = hZ Z(r, c). No division or sqrt.
  exec::parallel_for(
      "commutator::comm122ss", total, IMSRG_LAMBDA(const exec::Index e) {
        const int ch = detail::find_block(block_offset, nch, e);
        const std::int64_t off = block_offset[ch];
        const std::int64_t local = e - off;
        const int n = num_kets[ch];
        const int r = static_cast<int>(local % n);
        const int c = static_cast<int>(local / n);
        if (r > c) return;
        const int rb = row_base[ch];
        double b_rc = 0.0;
        double b_cr = 0.0;
        for (int side = 0; side < 2; ++side) {
          const int row = side == 0 ? r : c;
          const std::int64_t col = side == 0 ? c : r;
          double acc = 0.0;
          for (int t = entry_offset[rb + row]; t < entry_offset[rb + row + 1]; ++t) {
            const std::int64_t idx = off + lm[t] + col * n;
            double v = 0.0;
            if (yd != nullptr) v += x1(lfrom[t], lto[t]) * yd[idx];
            if (xd != nullptr) v -= y1(lfrom[t], lto[t]) * xd[idx];
            acc += lcoef[t] * v;
          }
          if (side == 0)
            b_rc = acc;
          else
            b_cr = acc;
        }
        const double z = b_rc + hz * b_cr;
        z2[off + r + static_cast<std::int64_t>(c) * n] += z;
        if (r < c) z2[off + c + static_cast<std::int64_t>(r) * n] += hz * z;
      });
  exec::fence("commutator::comm122ss");
}

}  // namespace imsrg::commutator::terms
