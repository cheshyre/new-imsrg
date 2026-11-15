// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_PANDYA_H_
#define IMSRG_COMMUTATOR_DETAIL_PANDYA_H_

/// @file
/// Precomputed data of the particle-hole term comm222_phss (design/12 section
/// 4.2; imsrg++ `comm222_phss`, reference/02 P-12). Everything is allocated
/// once per model space by `BuildPandyaData`; a call of the term only fills
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
///   XK : Ncc x nk,   YK : nk x 2 Ncc,   W : Ncc x 2 Ncc,   ZB : Ncc x 2 Ncc.

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
  const std::int64_t* w_off =
      nullptr;  ///< num_cc + 1 offsets of the W and ZB blocks.
  const std::int32_t* ket_J_offsets =
      nullptr;  ///< cc `ket_J_offsets` (global kets).
  const std::int32_t* local_index_table = nullptr;  ///< cc `local_index_table`.
  double* xk = nullptr;                             ///< Scratch XK.
  double* yk = nullptr;                             ///< Scratch YK.
  double* w = nullptr;                              ///< Scratch W = XK YK.
  double* zb = nullptr;  ///< Scratch ZB, the Pandya-transformed result.
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
  std::vector<std::int64_t>
      w_off_host;  ///< Host W/ZB block offsets (num_cc + 1).

  exec::DeviceBuffer<std::int32_t> ncc, nk, ket_off,
      k_off;  ///< Per-channel sizes and offsets.
  exec::DeviceBuffer<std::int32_t> ket_p, ket_q, k_a,
      k_b;                         ///< Flat orbit lists.
  exec::DeviceBuffer<double> k_d;  ///< Flat weights n_a - n_b.
  exec::DeviceBuffer<std::int64_t> xk_off, yk_off,
      w_off;  ///< Block offsets (num_cc + 1).
  exec::DeviceBuffer<std::int32_t> ket_J_offsets,
      local_index_table;                     ///< Device cc tables.
  exec::DeviceBuffer<double> xk, yk, w, zb;  ///< Scratch buffers.

  /// One GEMM W = XK YK per channel with Ncc > 0 and |K| > 0 (pointers into the
  /// scratch).
  std::vector<linalg::GemmProblem> gemms;

  PandyaGeneralData general;  ///< Scratch of the general path (lazy).

  /// Trivially copyable view for kernels, cached at build time; valid while
  /// this object lives.
  PandyaView kernel_view;

  /// The kernel view. O(1).
  const PandyaView& view() const noexcept { return kernel_view; }

  /// Total size of the four scratch buffers in bytes (XK + YK + W + ZB). O(1).
  std::size_t scratch_bytes() const noexcept {
    return (xk.size() + yk.size() + w.size() + zb.size()) * sizeof(double);
  }
};

/// Builds the Pandya data of `ms`: the cc ket lists, the pair lists K (from the
/// reference occupations of `ms`), device copies of the cc index tables and the
/// scratch buffers with their block offsets. Needs an initialized
/// `exec::Runtime`. All allocation of the term happens here. Memory O(sum over
/// cc channels of Ncc (2 nk + 4 Ncc)) doubles: XK (Ncc nk) + YK (2 Ncc nk) + W
/// and ZB (2 Ncc^2 each), i.e. Ncc (3 nk + 4 Ncc); `scratch_bytes()` reports
/// it.
PandyaData BuildPandyaData(const basis::ModelSpace& ms);

/// Allocates (once) the scratch of the general particle-hole path for the
/// channels of `pd`. General X or Y have no (anti-)Hermiticity to halve the
/// work, so the path computes the Pandya-transformed result for all ordered
/// pairs on both sides: Zbar(alpha, beta) = sum_kappa D_kappa (Xbar(alpha,
/// kappa) Ybar(kappa, beta) - Ybar(alpha, kappa) Xbar(kappa, beta)) for alpha,
/// beta in the 2 Ncc ordered pairs of the channel, as ONE GEMM per channel
///   Z (2Ncc x 2Ncc) = A (2Ncc x 2nk) B (2nk x 2Ncc),
///   A = [ D Xbar(alpha, kappa) | -D Ybar(alpha, kappa) ],
///   B = [ Ybar(kappa, beta) ; Xbar(kappa, beta) ]   (stacked rows),
/// all column-major (A, B at the same offset `g_off`, each 4 Ncc nk doubles; Z
/// at `gz_off`, 4 Ncc^2). Memory O(sum over cc channels of Ncc (8 nk + 4
/// Ncc)) doubles, about 4x the Hermitian scratch, paid only by programs that
/// use general operators. Idempotent. Needs an initialized `exec::Runtime`.
void EnsureGeneralScratch(PandyaData& pd);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_PANDYA_H_
