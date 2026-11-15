// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_PANDYA_H_
#define IMSRG_COMMUTATOR_DETAIL_PANDYA_H_

/// @file
/// Precomputed data of the particle-hole term comm222_phss (design/12 section
/// 4.2; imsrg++ `comm222_phss`, reference/02 P-12). Everything is allocated
/// once per model space by `BuildPandyaData` (the scratch of the general path
/// at its first use, `EnsureGeneralScratch`); a call of the term only fills
/// the scratch buffers. No Kokkos.
///
/// Conventions. For every cross-coupled (Pandya) channel c of the model space
/// with Ncc kets (orbit pairs p <= q), the ordered-pair basis has dimension 2
/// Ncc: index of (p, q) is local(|pq>) for p <= q and local(|qp>) + Ncc for p >
/// q. The contraction runs over the list K of ordered pairs kappa = (a, b) with
/// n_a != n_b (`nk` entries, built from the kets of the channel with both
/// orders, weight D = n_a - n_b). K is stored in flipped pairs: the flip (b, a)
/// of the K entry kappa is kappa ^ 1 (the fused fill of the kernel relies on
/// it). Because of hermiticity and the flip symmetry only rows alpha of the
/// kets (i <= l) are computed; the remaining rows follow from the symmetries,
/// see `kernels/particle_hole.cc`.
///
/// Scratch layout (all column-major; element (r, s) of an R-row block at offset
/// + r + s R):
///   XK : Ncc x nk,   YK : nk x Ncc,   W : Ncc x 2 Ncc.
/// W's left half is XK YK. Its right half needs YK's flipped-pair partner,
/// which is a signed row permutation of YK (F Ybar F = hY Phi Ybar Phi): after
/// the first GEMM the row pairs (kappa, kappa ^ 1) of YK are swapped in place
/// (sign phi_kappa) and a second GEMM writes XK YK' into the right half, which
/// is the right half of XK [YK | YK_R] up to the column signs hY phi_beta. A
/// pairwise in-place pass then turns W into the stored half of the
/// Pandya-transformed result (ZB in imsrg++ terms), which the inverse
/// transform reads (no separate ZB buffer and no second YK half, M7-Q4).

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/linalg/blas.h"

namespace imsrg::commutator::detail {

/// Trivially copyable, Kokkos-free view of the Pandya data for kernels. Arrays
/// indexed by the cross-coupled channel index c in [0, num_cc) unless noted
/// (empty channels have Ncc = 0).
struct PandyaView {
  std::int32_t num_cc = 0;            ///< Number of cc channels.
  const std::int32_t* ncc = nullptr;  ///< Number of kets Ncc of channel c.
  const std::int32_t* nk =
      nullptr;  ///< Number of ordered pairs |K| of channel c.
  const std::int32_t* ket_off =
      nullptr;  ///< Offset of channel c in `ket_p`/`ket_q`.
  const std::int32_t* k_off =
      nullptr;  ///< Offset of channel c in `k_a`/`k_b`/`k_d`.
  const std::int32_t* ket_p = nullptr;  ///< First orbit of each cc ket (flat).
  const std::int32_t* ket_q = nullptr;  ///< Second orbit of each cc ket (flat).
  const std::int32_t* k_a = nullptr;    ///< First orbit of each K pair (flat).
  const std::int32_t* k_b = nullptr;    ///< Second orbit of each K pair (flat).
  const double* k_d = nullptr;  ///< Weight n_a - n_b of each K pair (flat).
  const std::int64_t* xk_off =
      nullptr;  ///< num_cc + 1 offsets of the XK blocks.
  const std::int64_t* yk_off =
      nullptr;  ///< num_cc + 1 offsets of the YK blocks.
  const std::int64_t* w_off = nullptr;  ///< num_cc + 1 offsets of the W blocks.
  const std::int32_t* ket_J_offsets =
      nullptr;  ///< cc `ket_J_offsets` (global kets).
  const std::int32_t* local_index_table = nullptr;  ///< cc `local_index_table`.
  /// Global two-body kets grouped by class (parity, Tz) of the pair, each
  /// class in ascending ket order (the order of the kets inside every
  /// two-body channel); the inverse transform runs over pairs of kets of a
  /// class (`kernels/particle_hole.cc`).
  const std::int32_t* class_kets = nullptr;
  const std::int32_t* class_offset = nullptr;  ///< num_classes + 1 offsets.
  std::int32_t num_classes = 0;                ///< Number of classes (6).
  double* xk = nullptr;                        ///< Scratch XK.
  double* yk = nullptr;                        ///< Scratch YK.
  /// Scratch W = XK YK, combined in place into the stored half of the
  /// Pandya-transformed result.
  double* w = nullptr;
};
static_assert(std::is_trivially_copyable_v<PandyaView>);

/// Scratch of the general (non-Hermitian) particle-hole path, allocated on the
/// first general call by `EnsureGeneralScratch` (see there for the layout).
struct PandyaGeneralData {
  std::vector<std::int64_t>
      g_off_host;  ///< Offsets (num_cc + 1) of the A and B blocks.
  std::vector<std::int64_t>
      gz_off_host;  ///< Offsets (num_cc + 1) of the Z blocks.
  exec::DeviceBuffer<std::int64_t> g_off, gz_off;  ///< Device copies.
  exec::DeviceBuffer<double> a, b, z;              ///< A, B and Z = A B.
  std::vector<linalg::GemmProblem> gemms;  ///< One GEMM per channel with
                                           ///< Ncc > 0 and |K| > 0.

  /// True once `EnsureGeneralScratch` has run. O(1).
  bool allocated() const noexcept { return !gz_off_host.empty(); }

  /// Size of A, B and Z in bytes (0 before the first general call). O(1).
  std::size_t scratch_bytes() const noexcept {
    return (a.size() + b.size() + z.size()) * sizeof(double);
  }
};

/// Device-resident data and scratch of the particle-hole term. Movable, not
/// copyable (the prebuilt GEMM list and the cached view hold pointers into the
/// scratch buffers, which survive moves because the buffers are heap
/// allocations).
struct PandyaData {
  PandyaData() = default;
  PandyaData(const PandyaData&) = delete;
  PandyaData& operator=(const PandyaData&) = delete;
  PandyaData(PandyaData&&) noexcept = default;
  PandyaData& operator=(PandyaData&&) noexcept = default;
  ~PandyaData() = default;

  std::int32_t num_cc = 0;  ///< Number of cc channels (including empty ones).
  std::vector<std::int32_t>
      ncc_host;  ///< Host copy of the per-channel ket count.
  std::vector<std::int32_t> nk_host;  ///< Host copy of the per-channel |K|.
  std::vector<std::int64_t>
      xk_off_host;  ///< Host XK block offsets (num_cc + 1).
  std::vector<std::int64_t>
      yk_off_host;  ///< Host YK block offsets (num_cc + 1).
  std::vector<std::int64_t> w_off_host;  ///< Host W block offsets (num_cc + 1).

  exec::DeviceBuffer<std::int32_t> ncc, nk, ket_off,
      k_off;  ///< Per-channel sizes and offsets.
  exec::DeviceBuffer<std::int32_t> ket_p, ket_q, k_a,
      k_b;                         ///< Flat orbit lists.
  exec::DeviceBuffer<double> k_d;  ///< Flat weights n_a - n_b.
  exec::DeviceBuffer<std::int64_t> xk_off, yk_off,
      w_off;  ///< Block offsets (num_cc + 1).
  exec::DeviceBuffer<std::int32_t> ket_J_offsets,
      local_index_table;                 ///< Device cc tables.
  exec::DeviceBuffer<double> xk, yk, w;  ///< Scratch buffers.
  exec::DeviceBuffer<std::int32_t> class_kets,
      class_offset;  ///< Kets by class (see `PandyaView::class_kets`).
  std::int64_t num_class_kets = 0;  ///< Size of `class_kets`.

  /// One GEMM W_left = XK YK per channel with Ncc > 0 and |K| > 0 (pointers
  /// into the scratch); `gemms_right` writes the right halves after the flip.
  std::vector<linalg::GemmProblem> gemms_right;
  /// One GEMM W = XK YK per channel with Ncc > 0 and |K| > 0 (pointers into the
  /// scratch).
  std::vector<linalg::GemmProblem> gemms;

  PandyaGeneralData general;  ///< Scratch of the general path (lazy).

  /// Trivially copyable view for kernels, cached at build time; valid while
  /// this object lives.
  PandyaView kernel_view;

  /// The kernel view. O(1).
  const PandyaView& view() const noexcept { return kernel_view; }

  /// Total size of the scratch buffers in bytes (XK + YK + W, plus the
  /// general-path scratch once allocated). O(1).
  std::size_t scratch_bytes() const noexcept {
    return (xk.size() + yk.size() + w.size()) * sizeof(double) +
           general.scratch_bytes();
  }
};

/// Builds the Pandya data of `ms`: the cc ket lists, the pair lists K (from the
/// reference occupations of `ms`), device copies of the cc index tables and the
/// scratch buffers with their block offsets. Needs an initialized
/// `exec::Runtime`. All allocation of the (anti-)Hermitian path happens here.
/// Memory: sum over cc channels of Ncc (2 nk + 2 Ncc) doubles, XK (Ncc nk) +
/// YK (Ncc nk) + W (2 Ncc^2); `scratch_bytes()` reports it.
PandyaData BuildPandyaData(const basis::ModelSpace& ms);

/// Allocates (once) the scratch of the general particle-hole path for the
/// channels of `pd`. General X or Y have no (anti-)Hermiticity to halve the
/// work, so the path computes the Pandya-transformed result for all ordered
/// pairs on both sides: Zbar(alpha, beta) = sum_kappa D_kappa (Xbar(alpha,
/// kappa) Ybar(kappa, beta) - Ybar(alpha, kappa) Xbar(kappa, beta)) for alpha,
/// beta in the 2 Ncc ordered pairs of the channel, as TWO GEMMs per channel
/// that share the operand buffers: Z (2Ncc x 2Ncc) = A0 B0 + A1 B1, A (2Ncc x
/// nk) and B (nk x 2Ncc), all column-major (A, B at the offset `g_off`, each 2
/// Ncc nk doubles; Z at `gz_off`, 4 Ncc^2). Memory sum over cc channels of Ncc
/// (4 nk + 4 Ncc) doubles, twice the Hermitian scratch of Ncc (2 nk + 2
/// Ncc); 4x the flops. Paid only by
/// programs that use general operators; included in `scratch_bytes()` once
/// allocated. Idempotent; strongly exception safe (a throwing allocation
/// leaves the scratch unallocated). Needs an initialized `exec::Runtime`.
void EnsureGeneralScratch(PandyaData& pd);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_PANDYA_H_
