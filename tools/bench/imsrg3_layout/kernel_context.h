// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_KERNEL_CONTEXT_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_KERNEL_CONTEXT_H_
/// @file
/// Device data shared by the three-body benchmark kernels of one canonical
/// space (benchmark prototype, design/14 section 2), and the kernel entry
/// points. Kernels live in kernels/*.cc (design/12 section 5 rules: value
/// captures, no allocation or exceptions inside kernels, raw-pointer views).

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

#include "imsrg/basis/device_tables.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/layouts.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {

/// Orbits grouped by (parity, tz): group g = 2 * parity + (tz2 + 1) / 2 holds
/// orbits[offset[g] .. offset[g + 1]) in ascending index order (all orbits of
/// the model space, the range of the internal index a of [2,2]->3).
struct OrbitGroupView {
  const std::int32_t* offset = nullptr;  ///< 5 entries.
  const std::int32_t* orbits = nullptr;  ///< All orbits, grouped.

  /// Group of orbits with parity `parity` (0, 1) and 2tz = `tz2` (-1, +1).
  static IMSRG_INLINE_FUNCTION constexpr std::int32_t Group(
      std::int32_t parity, std::int32_t tz2) noexcept {
    return 2 * parity + (tz2 + 1) / 2;
  }
};
static_assert(std::is_trivially_copyable_v<OrbitGroupView>);

/// Tables for the kernels: two-body views (through a `commutator::Workspace`),
/// orbit tables, orbit groups, the 3B 6j table, and the K1 work order. The
/// referenced objects must outlive the context. Needs an initialized
/// `exec::Runtime`. Not copyable.
class KernelContext {
 public:
  KernelContext(const CanonicalSpace& space, const PairThirdSpace& pt,
                const SixJ3Table& sixj);
  KernelContext(const KernelContext&) = delete;
  KernelContext& operator=(const KernelContext&) = delete;
  KernelContext(KernelContext&&) = delete;
  KernelContext& operator=(KernelContext&&) = delete;
  ~KernelContext();

  const CanonicalSpace& Space() const noexcept { return *space_; }
  const PairThirdSpace& PairThird() const noexcept { return *pt_; }
  const SixJ3Table& SixJ() const noexcept { return *sixj_; }

  /// View of the two-body part of `op` (normalized storage; reads any orbit
  /// order, `get_unnormalized` adds sqrt(2) per identical pair). `op` must
  /// live on the context's model space. O(1).
  commutator::detail::TwoBodyView TwoBody(const op::Operator& op) const;
  /// Device orbit table (j2, l, tz2, occupation, ...).
  basis::OrbitTableView Orbits() const noexcept;
  /// Device orbit groups.
  OrbitGroupView Groups() const noexcept;
  /// Flat triple indices ordered by estimated [2,2]->3 tile work, largest
  /// first (the K1 league order), device data; `NumTriples()` entries.
  const std::int32_t* WorkOrder() const noexcept;

 private:
  const CanonicalSpace* space_;
  const PairThirdSpace* pt_;
  const SixJ3Table* sixj_;
  std::unique_ptr<commutator::Workspace> ws_;
  exec::DeviceBuffer<std::int32_t> group_offset_, group_orbits_;
  exec::DeviceBuffer<std::int32_t> work_order_;
};

/// Hermiticity sign h_Z = -h_X h_Y of Z = [X, Y] for scalar X, Y of definite
/// hermiticity (IMSRG_CHECK).
int CommutatorHermiticity(const op::Operator& X, const op::Operator& Y);

/// K0, the reference [2,2]->3: the loop structure and arithmetic of imsrg++
/// `comm223ss` (IMSRG3Commutators.cc:5012-5376): for each canonical bra (ijk
/// J1) and ket (lmn J2) of a channel, the sum over the perms {ABC, CBA, ACB}
/// of both, J1', J2' and the internal orbit a of
///   rec_ijk rec_lmn sqrt((2J1'+1)(2J2'+1)) {j6 ja J1'; j3 J J2'}
///   (x_{12,6a} y_{3a,45} - y_{12,6a} x_{3a,45})
/// with unnormalized two-body elements. Overwrites `z` (canonical blocks,
/// `ctx.Space().total_size()` doubles). Computes iket <= ibra and mirrors with
/// h_Z, or both triangles explicitly if `both_triangles`. Parallel over (ch,
/// ibra). Cost ~ sum_ch N_A^2 9 J'^2 n_a / 2.
void Comm223Reference(const KernelContext& ctx, const op::Operator& X,
                      const op::Operator& Y, std::span<double> z,
                      bool both_triangles = false);

/// What `Comm223Tiled` computes.
enum class TiledMode : std::int8_t {
  kFull,   ///< W tiles and the layout epilogue (the [2,2]->3 term).
  kWOnly,  ///< Only the W tiles (z untouched; for timing the W part).
};

/// K1, the tiled [2,2]->3 for any layout: a team per bra triple T (largest
/// first), for each ket triple T' <= T of the channel the pair-third tile
///   W[(P z J1'), (Q w J2')] = sqrt((2J1'+1)(2J2'+1)) sum_a
///     {j_w j_a J1'; j_z J J2'} (X_{P,(w a)} Y~_{(z a),Q} - Y_{P,(w a)}
///     X~_{(z a),Q})
/// (P, Q normalized pairs; ~ adds sqrt(2) for z == a or w == a), then the
/// epilogue z_L[T, T'] = L_T W L_T'^T with the layout's outer map and the
/// mirror h_Z z_L[T, T']^T into [T', T]. Overwrites `z` (the layout's blocks).
/// Cost ~ sum_ch N_C^2 n_a / 2 for W plus the epilogue.
void Comm223Tiled(const KernelContext& ctx, const LayoutView& layout,
                  const op::Operator& X, const op::Operator& Y,
                  std::span<double> z, TiledMode mode = TiledMode::kFull);

/// Perturbative triples with K0 elements (imsrg++ comm223ss with
/// perturbative_triples, IMSRG3Commutators.cc:5343-5373):
///   E = sum_ch (2J+1) sum_{iket < ibra} w_bra w_ket z^2 f,
///   f = (n_ijk nbar_lmn - n_lmn nbar_ijk) / (e_ijk - e_lmn),
/// w = sym / 6, n_ijk = n_i n_j n_k, nbar = (1 - n_i)(1 - n_j)(1 - n_k)
/// (reference occupations), e_ijk = epsilon[i] + epsilon[j] + epsilon[k];
/// pairs with |n_ijk nbar_lmn - n_lmn nbar_ijk| < 1e-8 are skipped, as in
/// imsrg++. No three-body storage.
double TriplesReference(const KernelContext& ctx, const op::Operator& X,
                        const op::Operator& Y, std::span<const double> epsilon);

/// Perturbative triples through the K1 tiles of `layout`: only tile pairs (T,
/// T' < T) with nonzero f are computed; E = sum_ch (2J+1) sum_tiles f_TT'
/// sum_ij (weight_i weight_j z_ij)^2 with the layout's energy weights. No
/// three-body storage (one tile of scratch per team).
double TriplesTiled(const KernelContext& ctx, const LayoutView& layout,
                    const op::Operator& X, const op::Operator& Y,
                    std::span<const double> epsilon);

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_KERNEL_CONTEXT_H_
