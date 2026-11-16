// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// K0: the reference [2,2]->3 kernel, a port of imsrg++ `comm223ss`
// (IMSRG3Commutators.cc:5012-5376), and the reference perturbative triples.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/exec/kokkos_backend.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {

namespace {

/// Everything the element function reads (raw-pointer views, captured by
/// value in kernels).
struct RefViews {
  CanonicalView cv;
  SixJ3View sj;
  basis::OrbitTableView ob;
  OrbitGroupView g;
  commutator::detail::TwoBodyView xv;
  commutator::detail::TwoBodyView yv;
};
static_assert(std::is_trivially_copyable_v<RefViews>);

/// Integer range [lo, hi] of the intermediate J' for the permutation `p` of
/// a triple with doubled j (ja, jb, jc) coupled to doubled total `two_j`; the
/// fixed `jfix` for ABC. Empty if lo > hi.
IMSRG_INLINE_FUNCTION void JRange(Perm p, int ja, int jb, int jc, int two_j,
                                  int jfix, int& lo, int& hi) noexcept {
  if (p == Perm::kABC) {
    lo = jfix;
    hi = jfix;
    return;
  }
  const int d1 = ja > jb ? ja - jb : jb - ja;
  const int d2 = two_j > jc ? two_j - jc : jc - two_j;
  lo = (d1 > d2 ? d1 : d2) / 2;
  const int u1 = ja + jb;
  const int u2 = two_j + jc;
  hi = (u1 < u2 ? u1 : u2) / 2;
}

/// The i-th of the permutations {ABC, CBA, ACB} (i = 0, 1, 2).
IMSRG_INLINE_FUNCTION Perm PermOf(int i) noexcept {
  return i == 0 ? Perm::kABC : (i == 1 ? Perm::kCBA : Perm::kACB);
}

/// Element z[bra_row, ket_row] of the K0 commutator for flat canonical rows
/// of the same channel: the sum over the permutations of bra and ket, the
/// intermediate J1', J2' and the internal orbit a (see Comm223Reference).
/// Cost O(9 (2 J' + 1)^2 n_a).
IMSRG_INLINE_FUNCTION double Element(const RefViews& v, std::int64_t bra_row,
                                     std::int64_t ket_row) noexcept {
  const CanonicalView& cv = v.cv;
  const std::int32_t tb = cv.row_triple[bra_row];
  const std::int32_t tk = cv.row_triple[ket_row];
  const int c = cv.tr_channel[tb];
  const int two_j = cv.ch_two_j[c];
  const int bi = cv.tr_a[tb];
  const int bj = cv.tr_b[tb];
  const int bk = cv.tr_c[tb];
  const int J1 = cv.row_two_jab[bra_row] / 2;
  const int ki = cv.tr_a[tk];
  const int kj = cv.tr_b[tk];
  const int kk = cv.tr_c[tk];
  const int J2 = cv.row_two_jab[ket_row] / 2;
  const std::int32_t* j2 = v.ob.j2;
  const std::int32_t* lo = v.ob.l;
  const std::int32_t* tzo = v.ob.tz2;

  double z = 0.0;
  for (int ip1 = 0; ip1 < 3; ++ip1) {
    const Perm p1 = PermOf(ip1);
    int i1 = 0;
    int i2 = 0;
    int i3 = 0;
    Permute(p1, bi, bj, bk, i1, i2, i3);
    const int j3 = j2[i3];
    int a1lo = 0;
    int a1hi = -1;
    JRange(p1, j2[i1], j2[i2], j3, two_j, J1, a1lo, a1hi);
    for (int Jp1 = a1lo; Jp1 <= a1hi; ++Jp1) {
      const double rec1 =
          PermutationPhase(p1) *
          RecouplingCoefficient(p1, TwoJ{j2[bi]}, TwoJ{j2[bj]}, TwoJ{j2[bk]},
                                Jp1, J1, TwoJ{two_j}, v.sj);
      if (rec1 == 0.0) continue;
      for (int ip2 = 0; ip2 < 3; ++ip2) {
        const Perm p2 = PermOf(ip2);
        int i4 = 0;
        int i5 = 0;
        int i6 = 0;
        Permute(p2, ki, kj, kk, i4, i5, i6);
        const int j6 = j2[i6];
        int a2lo = 0;
        int a2hi = -1;
        JRange(p2, j2[i4], j2[i5], j6, two_j, J2, a2lo, a2hi);
        const int tz2_a = tzo[i1] + tzo[i2] - tzo[i6];
        if (tz2_a != 1 && tz2_a != -1) continue;
        const int grp =
            OrbitGroupView::Group((lo[i1] + lo[i2] + lo[i6]) % 2, tz2_a);
        const int gbeg = v.g.offset[grp];
        const int gend = v.g.offset[grp + 1];
        for (int Jp2 = a2lo; Jp2 <= a2hi; ++Jp2) {
          const double rec2 =
              PermutationPhase(p2) *
              RecouplingCoefficient(p2, TwoJ{j2[ki]}, TwoJ{j2[kj]},
                                    TwoJ{j2[kk]}, Jp2, J2, TwoJ{two_j}, v.sj);
          if (rec2 == 0.0) continue;
          const double pref = rec1 * rec2 * v.sj.Hat(Jp1) * v.sj.Hat(Jp2);
          double acc = 0.0;
          for (int ia = gbeg; ia < gend; ++ia) {
            const int a = v.g.orbits[ia];
            const double six = v.sj(TwoJ{j6}, TwoJ{j2[a]}, TwoJ{2 * Jp1},
                                    TwoJ{j3}, TwoJ{two_j}, TwoJ{2 * Jp2});
            if (six == 0.0) continue;
            const double x1 = v.xv.get_unnormalized(i1, i2, i6, a, 2 * Jp1);
            const double y2 = v.yv.get_unnormalized(i3, a, i4, i5, 2 * Jp2);
            const double y1 = v.yv.get_unnormalized(i1, i2, i6, a, 2 * Jp1);
            const double x2 = v.xv.get_unnormalized(i3, a, i4, i5, 2 * Jp2);
            acc += six * (x1 * y2 - y1 * x2);
          }
          z += pref * acc;
        }
      }
    }
  }
  return z;
}

using RowPolicy =
    Kokkos::RangePolicy<exec::DefaultExecSpace, Kokkos::IndexType<std::int64_t>,
                        Kokkos::Schedule<Kokkos::Dynamic>>;

RefViews MakeViews(const KernelContext& ctx, const op::Operator& X,
                   const op::Operator& Y) {
  return RefViews{ctx.Space().device_view(),
                  ctx.SixJ().device_view(),
                  ctx.Orbits(),
                  ctx.Groups(),
                  ctx.TwoBody(X),
                  ctx.TwoBody(Y)};
}

/// One work item per flat canonical row: its lower part (iket <= ibra, or all
/// of the row with `both`) and the mirrored column part.
void Comm223Kernel(const RefViews v, double* z, int hz, bool both,
                   std::int64_t num_rows) {
  Kokkos::parallel_for(
      "imsrg3_layout::comm223_ref", RowPolicy(0, num_rows),
      IMSRG_LAMBDA(const std::int64_t row) {
        const int c = v.cv.tr_channel[v.cv.row_triple[row]];
        const std::int64_t n = v.cv.ch_dim[c];
        const std::int64_t r0 = v.cv.ch_row_offset[c];
        double* blk = z + v.cv.ch_block_offset[c];
        const std::int64_t ibra = row - r0;
        const std::int64_t last = both ? n - 1 : ibra;
        for (std::int64_t iket = 0; iket <= last; ++iket) {
          const double val = Element(v, row, r0 + iket);
          blk[ibra + iket * n] = val;
          if (!both && iket != ibra) blk[iket + ibra * n] = hz * val;
        }
      });
}

/// Sum over rows of the triples energy contributions (see TriplesReference).
double TriplesKernel(const RefViews v, const double* eps,
                     std::int64_t num_rows) {
  double total = 0.0;
  Kokkos::parallel_reduce(
      "imsrg3_layout::triples_ref", RowPolicy(0, num_rows),
      IMSRG_LAMBDA(const std::int64_t row, double& sum) {
        const std::int32_t tb = v.cv.row_triple[row];
        const int c = v.cv.tr_channel[tb];
        const int bi = v.cv.tr_a[tb];
        const int bj = v.cv.tr_b[tb];
        const int bk = v.cv.tr_c[tb];
        const double ni = v.ob.occupation[bi];
        const double nj = v.ob.occupation[bj];
        const double nk = v.ob.occupation[bk];
        const double occ_b = ni * nj * nk;
        const double unocc_b = (1.0 - ni) * (1.0 - nj) * (1.0 - nk);
        if (occ_b < 1.0e-8 && unocc_b < 1.0e-8) return;
        const double e_b = eps[bi] + eps[bj] + eps[bk];
        const double w_b = v.cv.tr_sym[tb] / 6.0;
        const double dim_j = v.cv.ch_two_j[c] + 1.0;
        const std::int64_t r0 = v.cv.ch_row_offset[c];
        const std::int64_t ibra = row - r0;
        for (std::int64_t iket = 0; iket < ibra; ++iket) {
          const std::int64_t kr = r0 + iket;
          const std::int32_t tk = v.cv.row_triple[kr];
          const int ki = v.cv.tr_a[tk];
          const int kj = v.cv.tr_b[tk];
          const int kk = v.cv.tr_c[tk];
          const double mi = v.ob.occupation[ki];
          const double mj = v.ob.occupation[kj];
          const double mk = v.ob.occupation[kk];
          const double occ_k = mi * mj * mk;
          const double unocc_k = (1.0 - mi) * (1.0 - mj) * (1.0 - mk);
          const double num = occ_b * unocc_k - occ_k * unocc_b;
          if (num < 1.0e-8 && num > -1.0e-8) continue;
          const double e_k = eps[ki] + eps[kj] + eps[kk];
          const double w_k = v.cv.tr_sym[tk] / 6.0;
          const double z = Element(v, row, kr);
          sum += dim_j * w_b * w_k * z * z * num / (e_b - e_k);
        }
      },
      total);
  return total;
}

}  // namespace

void Comm223Reference(const KernelContext& ctx, const op::Operator& X,
                      const op::Operator& Y, std::span<double> z,
                      bool both_triangles) {
  IMSRG_CHECK(static_cast<std::int64_t>(z.size()) == ctx.Space().total_size(),
              "Comm223Reference: z has the wrong size");
  const int hz = CommutatorHermiticity(X, Y);
  const RefViews v = MakeViews(ctx, X, Y);
  exec::DeviceBuffer<double> zbuf(z.size(), "imsrg3_layout::k0_z");
  Comm223Kernel(v, zbuf.device_data(), hz, both_triangles,
                ctx.Space().NumRows());
  exec::Fence("imsrg3_layout::comm223_ref");
  zbuf.CopyToHost(z);
}

double TriplesReference(const KernelContext& ctx, const op::Operator& X,
                        const op::Operator& Y,
                        std::span<const double> epsilon) {
  IMSRG_CHECK(
      static_cast<std::int64_t>(epsilon.size()) == ctx.Orbits().num_orbits,
      "TriplesReference: epsilon needs one value per orbit");
  const RefViews v = MakeViews(ctx, X, Y);
  exec::DeviceBuffer<double> eps(epsilon.size(), "imsrg3_layout::epsilon");
  eps.CopyFromHost(epsilon);
  return TriplesKernel(v, eps.device_data(), ctx.Space().NumRows());
}

}  // namespace imsrg::layout3
