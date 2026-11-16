// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_LAYOUTS_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_LAYOUTS_H_
/// @file
/// The three candidate storage layouts of a scalar three-body operator and the
/// `ThreeBodyLayout` concept they satisfy (benchmark prototype, design/14
/// section 2). Every layout stores one dense dim x dim block per canonical
/// channel (column-major, offsets `BlockOffsets()`, one buffer), and groups
/// its rows by the canonical triples of the channel.
///
/// * A, canonical: rows = the kept canonical kets |(ab) Jab, c; J>, a <= b <=
///   c (imsrg++ order and values, overcomplete when orbits coincide).
///   Conversions are the identity. Internal three-body sums weight triple T by
///   w_T = sym_T / 6.
/// * B, orthonormal: per triple, an orthonormal basis of the antisymmetric
///   subspace. For a < b = c, V_T = eigenvectors of eigenvalue 1 of Q = (1 +
///   M_ACB) / 2; for a = b = c, of Q = (1 + M_CBA + M_ACB) / 3, where M_p[J',
///   J] = PermutationPhase(p) RecouplingCoefficient(p, ja, jb, jc, J', J, 2J)
///   over the kept rows J', J of T; V_T = identity otherwise. With s_T =
///   sqrt(w_T): z_B[T, T'] = s_T s_T' V_T^T z_A[T, T'] V_T', and z_A[T, T'] =
///   V_T z_B[T, T'] V_T'^T / (s_T s_T'), so products are plain matrix products
///   and sum z_B^2 = sum w_T w_T' z_A^2.
/// * C, pair-third: rows = all pair-third states (pair_third.h), fully
///   antisymmetrized; z_C = E z_A E^T, and z_A is the sub-block of the rows
///   that are canonical kets (the ABC splitting).
///
/// Kernels see a layout only through `LayoutView`: per triple its layout rows
/// and the "outer map" L_T from the pair-third states of T to its layout rows
/// (A: R_T; B: s_T V_T^T R_T; C: E_T R_T), so that a [2,2]->3 tile is z_L[T,
/// T'] = L_T W[T, T'] L_T'^T for the pair-third intermediate W.

#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/exec/device_buffer.h"
#include "imsrg/linalg/block_sparse_matrix.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"
#include "tools/bench/imsrg3_layout/pair_third.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {

/// The candidate layouts.
enum class LayoutKind : std::int8_t { kA, kB, kC };

/// Device view of a layout for kernels (struct of arrays).
struct LayoutView {
  const std::int32_t* ch_dim = nullptr;           ///< Per canonical channel.
  const std::int32_t* ch_row_offset = nullptr;    ///< First flat layout row.
  const std::int64_t* ch_block_offset = nullptr;  ///< num_channels + 1.
  const std::int32_t* tr_row_begin = nullptr;  ///< Per triple, channel-local.
  const std::int32_t* tr_row_end = nullptr;
  CsrView outer;  ///< L: flat layout row -> channel-local pair-third states.
  const double* energy_weight =
      nullptr;  ///< Per flat layout row: A s_T, B 1, C s_T on canonical rows
                ///< (else 0); E_triples uses (weight_i weight_j z_ij)^2.
  std::int32_t num_channels = 0;
};
static_assert(std::is_trivially_copyable_v<LayoutView>);

/// Data common to the three layout classes (row tables, outer map, weights).
/// The layout classes fill it in their constructors.
struct LayoutTables {
  std::vector<std::int32_t> dim;           ///< Per channel (host).
  std::vector<std::int64_t> block_offset;  ///< num_channels + 1 (host).
  exec::DeviceBuffer<std::int32_t> ch_dim, ch_row_offset;
  exec::DeviceBuffer<std::int64_t> ch_block_offset;
  exec::DeviceBuffer<std::int32_t> tr_row_begin, tr_row_end;
  Csr outer;
  exec::DeviceBuffer<double> energy_weight;

  /// The device view.
  LayoutView device_view() const noexcept;
  /// Bytes of the row tables and the weights (not the outer map).
  std::int64_t Bytes() const noexcept;
};

/// What a layout provides to the benchmark harness and the kernels.
template <class L>
concept ThreeBodyLayout =
    requires(const L& l, std::int32_t ch, std::span<const double> in,
             std::span<double> out) {
      { L::kKind } -> std::convertible_to<LayoutKind>;
      { l.Space() } -> std::same_as<const CanonicalSpace&>;
      { l.Dims() } -> std::same_as<std::span<const std::int32_t>>;
      { l.BlockOffsets() } -> std::same_as<std::span<const std::int64_t>>;
      { l.total_size() } -> std::same_as<std::int64_t>;
      { l.Counts() } -> std::same_as<SpaceCounts>;
      { l.MetadataBytes() } -> std::same_as<std::int64_t>;
      { l.device_view() } -> std::same_as<LayoutView>;
      { l.Shapes() } -> std::same_as<std::vector<linalg::BlockShape>>;
      // Channel block conversions (host, dense column-major blocks of channel
      // `ch`): `in` and `out` hold dim_layout^2 resp. dim_A^2 values.
      l.ToCanonical(ch, in, out);
      l.FromCanonical(ch, in, out);
    };

/// Layout A (canonical). Metadata: the canonical space tables plus its outer
/// map R.
class LayoutA {
 public:
  static constexpr LayoutKind kKind = LayoutKind::kA;
  /// Builds the tables. `space` and `pt` must outlive the layout.
  LayoutA(const CanonicalSpace& space, const PairThirdSpace& pt);
  LayoutA(const LayoutA&) = delete;
  LayoutA& operator=(const LayoutA&) = delete;
  LayoutA(LayoutA&&) = default;
  LayoutA& operator=(LayoutA&&) = delete;
  ~LayoutA() = default;

  const CanonicalSpace& Space() const noexcept { return *space_; }
  std::span<const std::int32_t> Dims() const noexcept { return t_.dim; }
  std::span<const std::int64_t> BlockOffsets() const noexcept {
    return t_.block_offset;
  }
  std::int64_t total_size() const noexcept { return t_.block_offset.back(); }
  SpaceCounts Counts() const;
  std::int64_t MetadataBytes() const noexcept;
  LayoutView device_view() const noexcept { return t_.device_view(); }
  std::vector<linalg::BlockShape> Shapes() const;
  void ToCanonical(std::int32_t ch, std::span<const double> in,
                   std::span<double> out) const;
  void FromCanonical(std::int32_t ch, std::span<const double> in,
                     std::span<double> out) const;

 private:
  const CanonicalSpace* space_;
  LayoutTables t_;
};

/// Layout B (orthonormal antisymmetric basis per triple). Metadata: the
/// canonical tables plus V_T for the triples with b == c, and its outer map.
class LayoutB {
 public:
  static constexpr LayoutKind kKind = LayoutKind::kB;
  /// Builds V_T with `linalg::Eigh`; IMSRG_CHECK that every eigenvalue of Q
  /// is 0 or 1 within 1e-10 and Q is symmetric within 1e-12.
  LayoutB(const CanonicalSpace& space, const PairThirdSpace& pt,
          const SixJ3Table& sixj);
  LayoutB(const LayoutB&) = delete;
  LayoutB& operator=(const LayoutB&) = delete;
  LayoutB(LayoutB&&) = default;
  LayoutB& operator=(LayoutB&&) = delete;
  ~LayoutB() = default;

  const CanonicalSpace& Space() const noexcept { return *space_; }
  std::span<const std::int32_t> Dims() const noexcept { return t_.dim; }
  std::span<const std::int64_t> BlockOffsets() const noexcept {
    return t_.block_offset;
  }
  std::int64_t total_size() const noexcept { return t_.block_offset.back(); }
  SpaceCounts Counts() const;
  std::int64_t MetadataBytes() const noexcept;
  LayoutView device_view() const noexcept { return t_.device_view(); }
  std::vector<linalg::BlockShape> Shapes() const;
  void ToCanonical(std::int32_t ch, std::span<const double> in,
                   std::span<double> out) const;
  void FromCanonical(std::int32_t ch, std::span<const double> in,
                     std::span<double> out) const;

  /// Layout-B dimension of `space` without building V (rank of Q per triple
  /// from its trace, Q computed with angmom::Wigner6j; for the extrapolation).
  static SpaceCounts CountOnly(const CanonicalSpace& space);

 private:
  const CanonicalSpace* space_;
  LayoutTables t_;
  // Per flat triple: offset of V_T (n_T x d_T, column-major) in `v_`, or -1
  // for the identity, and d_T; the layout rows of T are its d_T basis vectors.
  std::vector<std::int64_t> v_offset_;
  std::vector<std::int32_t> v_dim_;
  std::vector<double> v_;
};

/// Layout C (fully antisymmetrized pair-third states). Metadata: the
/// pair-third tables, E, and the outer map E R.
class LayoutC {
 public:
  static constexpr LayoutKind kKind = LayoutKind::kC;
  LayoutC(const CanonicalSpace& space, const PairThirdSpace& pt);
  LayoutC(const LayoutC&) = delete;
  LayoutC& operator=(const LayoutC&) = delete;
  LayoutC(LayoutC&&) = default;
  LayoutC& operator=(LayoutC&&) = delete;
  ~LayoutC() = default;

  const CanonicalSpace& Space() const noexcept { return *space_; }
  std::span<const std::int32_t> Dims() const noexcept { return t_.dim; }
  std::span<const std::int64_t> BlockOffsets() const noexcept {
    return t_.block_offset;
  }
  std::int64_t total_size() const noexcept { return t_.block_offset.back(); }
  SpaceCounts Counts() const;
  std::int64_t MetadataBytes() const noexcept;
  LayoutView device_view() const noexcept { return t_.device_view(); }
  std::vector<linalg::BlockShape> Shapes() const;
  void ToCanonical(std::int32_t ch, std::span<const double> in,
                   std::span<double> out) const;
  void FromCanonical(std::int32_t ch, std::span<const double> in,
                     std::span<double> out) const;

 private:
  const CanonicalSpace* space_;
  const PairThirdSpace* pt_;
  LayoutTables t_;
  // For each flat canonical row, the channel-local layout row of its ABC
  // pair-third state (the gather of ToCanonical).
  std::vector<std::int32_t> canonical_row_;
};

static_assert(ThreeBodyLayout<LayoutA>);
static_assert(ThreeBodyLayout<LayoutB>);
static_assert(ThreeBodyLayout<LayoutC>);

/// imsrg++ norm of a three-body operator stored in layout `l` (D12): converts
/// every channel block to canonical form and applies `ImsrgppNorm`. O(total
/// conversion cost).
template <ThreeBodyLayout L>
double LayoutImsrgppNorm(const L& l, std::span<const double> blocks) {
  const CanonicalSpace& s = l.Space();
  const CanonicalView v = s.host_view();
  std::vector<double> canonical(static_cast<std::size_t>(s.total_size()));
  for (std::int32_t ch = 0; ch < s.num_channels(); ++ch) {
    const auto c = static_cast<std::size_t>(ch);
    const auto lo = static_cast<std::size_t>(l.BlockOffsets()[c]);
    const auto ln = static_cast<std::size_t>(l.BlockOffsets()[c + 1]) - lo;
    const auto co = static_cast<std::size_t>(v.ch_block_offset[ch]);
    const auto cn = static_cast<std::size_t>(v.ch_block_offset[ch + 1]) - co;
    l.ToCanonical(ch, blocks.subspan(lo, ln),
                  std::span<double>(canonical).subspan(co, cn));
  }
  return ImsrgppNorm(s, canonical);
}

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_LAYOUTS_H_
