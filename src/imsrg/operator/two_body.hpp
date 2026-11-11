#pragma once
/// @file
/// `TwoBody`: the J-coupled two-body part of an operator on a `linalg::BlockSparseMatrix`
/// (design/10 section 3.3, design/13 section 2, design/reference/01 sections 2.5 and 4).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/basis/two_body_channels.hpp"
#include "imsrg/linalg/block_sparse_matrix.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::op {

/// Two-body part V^J_{pq,rs} = <pq; J|V|rs; J> of a scalar operator.
///
/// Conventions (identical to imsrg++, design/00 D11):
/// * kets |pq; J> with p <= q are the kets of `basis::TwoBodyChannels`; the stored values are
///   NORMALIZED and antisymmetrized: |pq; J> carries 1/sqrt(1 + delta_pq), so the matrix product
///   over kets p <= q is the half-sum of the unnormalized formulas (`TwoBodyME.hh:36-50`);
/// * the "unnormalized" value of imsrg++'s `GetTBME*` is sqrt((1 + delta_pq)(1 + delta_rs)) times
///   the normalized one (`get_unnormalized`);
/// * kets with p > q follow from |qp; J> = exchange_phase(j_p, j_q, J) |pq; J>
///   (`basis::exchange_phase`); |pp; J> with odd J does not exist (Pauli).
///
/// Layout: for a scalar operator one dense column-major block per channel, ALL channels in
/// channel order (empty channels as 0 x 0 blocks), so block k is channel k (`block_index`). Both
/// triangles are stored. The blocks lie back to back in one device buffer (`matrix()`), so
/// whole-part operations are flat kernels and channel products are (grouped) gemms. Tensor
/// operators (later) will add off-diagonal channel blocks; `block_index` hides the difference.
///
/// Memory model: as `OneBody` (element accessors use the host mirror; sync explicitly on
/// backends with separate device memory).
///
/// Value semantics: copies are deep, moves are cheap. The part shares the model space with its
/// operator (`std::shared_ptr<const basis::ModelSpace>`); only its two-body channels are used.
class TwoBody {
 public:
  /// Empty part without blocks (particle_rank 1, or a default-constructed operator).
  TwoBody() = default;
  /// Zero part for `spec` on `ms`: one block per channel if spec.particle_rank >= 2, no blocks
  /// otherwise. Preconditions: `ms` non-null and `validate(spec)` passed (IMSRG_CHECK).
  TwoBody(std::shared_ptr<const basis::ModelSpace> ms, const OperatorSpec& spec);

  /// True if the part has blocks (particle_rank >= 2).
  bool allocated() const noexcept { return matrix_.num_blocks() > 0; }
  /// The two-body channels (kets, channel labels, local indices). Precondition: built from a
  /// model space (IMSRG_CHECK).
  const basis::TwoBodyChannels& channels() const;
  /// The hermiticity used by `set_normalized` and `add_normalized`.
  Hermiticity hermiticity() const noexcept { return hermiticity_; }
  /// Changes the hermiticity flag; the data are not touched.
  void set_hermiticity(Hermiticity h) noexcept { hermiticity_ = h; }

  /// Storage index of the block (ch_bra, ch_ket), or nullopt if that block is not stored
  /// (for scalars: ch_bra != ch_ket, or no blocks). Scalars: the index is ch_bra. O(1).
  std::optional<std::size_t> block_index(ChannelIndex ch_bra, ChannelIndex ch_ket) const noexcept;

  /// Stored (normalized) element (i, j) of block (ch_bra, ch_ket) with channel-local ket indices;
  /// 0 if the block is not stored. Host mirror. Precondition: i, j inside the channels
  /// (IMSRG_ASSERT). O(1).
  double get(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i, LocalKetIndex j) const;
  /// Sets the stored element (i, j) of block (ch_bra, ch_ket) only (no partner, no phases).
  /// Host mirror. Precondition: the block is stored and i, j are inside it (IMSRG_CHECK). O(1).
  void set(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i, LocalKetIndex j, double v);

  /// Normalized <ab; J|V|cd; J> for orbits in any order (exchange phases applied, imsrg++
  /// `GetTBME_J_norm`). Returns 0 if a ket does not exist (Pauli, |j_a - j_b| <= J <= j_a + j_b
  /// violated) or bra and ket are not connected by the operator (different parity or Tz for a
  /// scalar), or the part is not allocated. Host mirror. O(1).
  double get_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J) const;
  /// sqrt((1 + delta_ab)(1 + delta_cd)) * get_normalized(a, b, c, d, J): imsrg++ `GetTBME_J`.
  double get_unnormalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J) const;
  /// Sets the normalized <ab; J|V|cd; J> = v (orbits in any order, exchange phases applied) and,
  /// for bra != ket, its partner <cd; J|V|ab; J> = h v (h = hermiticity_sign; no partner for
  /// non-Hermitian parts). imsrg++ `SetTBME` semantics. Host mirror. O(1).
  /// @throws InvariantError if a ket does not exist or bra and ket are not connected.
  void set_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J, double v);
  /// Adds v to the normalized <ab; J|V|cd; J> and h v to its partner (as `set_normalized`). Unlike
  /// imsrg++ `AddToTBME`, the anti-Hermitian sign is applied to the partner (design/13 section 2,
  /// design/30 section 6). Host mirror. O(1).
  /// @throws InvariantError if a ket does not exist or bra and ket are not connected.
  void add_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d, TwoJ J, double v);
  /// Sets every element to zero (device kernel).
  void set_zero();

  /// Host-mirror view of the block of channel `ch` (scalar operators). Precondition: allocated.
  linalg::MatrixView<double> host_block(ChannelIndex ch);
  /// Read-only host-mirror view of the block of channel `ch`.
  linalg::MatrixView<const double> host_block(ChannelIndex ch) const;
  /// Device view of the block of channel `ch`, for kernels and BLAS (host-accessible backends).
  linalg::MatrixView<double> device_block(ChannelIndex ch);
  /// Read-only device view of the block of channel `ch`.
  linalg::MatrixView<const double> device_block(ChannelIndex ch) const;
  /// All blocks; their row and column block labels are channel indices.
  linalg::BlockSparseMatrix<double>& matrix() noexcept { return matrix_; }
  /// All blocks, read-only.
  const linalg::BlockSparseMatrix<double>& matrix() const noexcept { return matrix_; }
  /// Copies the device data to the host mirror (see the memory model).
  void sync_to_host() { matrix_.buffer().sync_to_host(); }
  /// Copies the host mirror to the device data (see the memory model).
  void sync_to_device() { matrix_.buffer().sync_to_device(); }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  Hermiticity hermiticity_ = Hermiticity::Hermitian;
  linalg::BlockSparseMatrix<double> matrix_;
};

}  // namespace imsrg::op
