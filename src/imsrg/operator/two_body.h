// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_TWO_BODY_H_
#define IMSRG_OPERATOR_TWO_BODY_H_
/// @file
/// `TwoBody`: the J-coupled two-body part of an operator on a
/// `linalg::BlockSparseMatrix` (design/10 section 3.3, design/13 section 2,
/// design/reference/01 sections 2.5 and 4).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/linalg/block_sparse_matrix.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::op {

/// Two-body part V^J_{pq,rs} = <pq; J|V|rs; J> of a scalar operator, or the
/// reduced elements <pq; J_bra||V||rs; J_ket> of a tensor operator.
///
/// Conventions (identical to imsrg++, design/00 D11):
/// * kets |pq; J> with p <= q are the kets of `basis::TwoBodyChannels`; the
///   stored values are NORMALIZED and antisymmetrized: |pq; J> carries 1/sqrt(1
///   + delta_pq), so the matrix product over kets p <= q is the half-sum of the
///   unnormalized formulas (`TwoBodyME.hh:36-50`);
/// * the "unnormalized" value of imsrg++'s `GetTBME*` is sqrt((1 + delta_pq)(1
///   + delta_rs)) times the normalized one (`get_unnormalized`);
/// * kets with p > q follow from |qp; J> = ExchangePhase(j_p, j_q, J) |pq; J>
///   (`basis::ExchangePhase`); |pp; J> with odd J does not exist (Pauli).
///
/// Layout: for a scalar operator one dense column-major block per channel, ALL
/// channels in channel order (empty channels as 0 x 0 blocks), so block k is
/// channel k (`block_index`). Both triangles are stored. The blocks lie back to
/// back in one device buffer (`matrix()`), so whole-part operations are flat
/// kernels and channel products are (grouped) gemms.
///
/// Tensor layout (non-scalar spec, design/00 D25, imsrg++
/// `TwoBodyME::Allocate`): one dense block per channel pair (ch_bra, ch_ket)
/// with ch_bra <= ch_ket, |J_bra - J_ket| <= rank_J <= J_bra + J_ket,
/// |Tz_bra - Tz_ket| == delta_Tz and parity_bra + parity_ket + parity even,
/// in lexicographic (ch_bra, ch_ket) order (imsrg++'s map order; empty
/// channels give 0-size blocks). Diagonal blocks store both triangles. The
/// elements are reduced and normalized; blocks with ch_bra > ch_ket follow
/// from the hermiticity, <cd J_ket||V||ab J_bra> = h (-1)^(J_bra - J_ket)
/// <ab J_bra||V||cd J_ket> (h = +1 Hermitian, -1 anti-Hermitian), which the
/// tensor element getters apply.
///
/// Memory model: as `OneBody` (element accessors use the host mirror; sync
/// explicitly on backends with separate device memory).
///
/// Value semantics: copies are deep, moves are cheap. The part shares the model
/// space with its operator (`std::shared_ptr<const basis::ModelSpace>`); only
/// its two-body channels are used.
class TwoBody {
 public:
  /// Empty part without blocks (particle_rank 1, or a default-constructed
  /// operator).
  TwoBody() = default;
  /// Zero part for `spec` on `ms`: if spec.particle_rank >= 2 one block per
  /// channel (scalar) or per allocated channel pair (tensor, see above), no
  /// blocks otherwise. Preconditions: `ms` non-null and
  /// `Validate(spec)` passed (IMSRG_CHECK).
  TwoBody(std::shared_ptr<const basis::ModelSpace> ms,
          const OperatorSpec& spec);

  TwoBody(const TwoBody&) = default;
  TwoBody& operator=(const TwoBody&) = default;
  TwoBody(TwoBody&&) = default;
  TwoBody& operator=(TwoBody&&) = default;

  /// True if the part has blocks (particle_rank >= 2).
  bool allocated() const noexcept { return matrix_.num_blocks() > 0; }
  /// The two-body channels (kets, channel labels, local indices). Precondition:
  /// built from a model space (IMSRG_CHECK).
  const basis::TwoBodyChannels& channels() const;
  /// The hermiticity used by `set_normalized` and `AddNormalized`.
  Hermiticity hermiticity() const noexcept { return hermiticity_; }
  /// Changes the hermiticity flag; the data are not touched.
  void set_hermiticity(Hermiticity h) noexcept { hermiticity_ = h; }

  /// True if the part uses the tensor layout (built for a non-scalar spec).
  bool is_tensor() const noexcept { return tensor_; }
  /// Tensor rank J of the operator the part was built for (0 for scalars).
  std::int32_t rank_J() const noexcept { return rank_J_; }

  /// Storage index of the block (ch_bra, ch_ket), or nullopt if that block is
  /// not stored (scalars: ch_bra != ch_ket, or no blocks; tensors: not an
  /// allocated pair with ch_bra <= ch_ket). Scalars: the index is ch_bra, O(1);
  /// tensors: O(log(number of blocks)).
  std::optional<std::size_t> block_index(ChannelIndex ch_bra,
                                         ChannelIndex ch_ket) const noexcept;

  /// Stored (normalized) element (i, j) of block (ch_bra, ch_ket) with
  /// channel-local ket indices; 0 if the block is not stored. Host mirror.
  /// Precondition: i, j inside the channels (IMSRG_ASSERT). O(1).
  double get(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i,
             LocalKetIndex j) const;
  /// Sets the stored element (i, j) of block (ch_bra, ch_ket) only (no partner,
  /// no phases). Host mirror. Precondition: the block is stored and i, j are
  /// inside it (IMSRG_CHECK). O(1).
  void set(ChannelIndex ch_bra, ChannelIndex ch_ket, LocalKetIndex i,
           LocalKetIndex j, double v);

  /// Normalized <ab; J|V|cd; J> for orbits in any order (exchange phases
  /// applied, imsrg++ `GetTBME_J_norm`). Returns 0 if a ket does not exist
  /// (Pauli, |j_a - j_b| <= J <= j_a + j_b violated) or bra and ket are not
  /// connected by the operator (different parity or Tz for a scalar), or the
  /// part is not allocated. Host mirror. O(1).
  double get_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                        TwoJ J) const;
  /// sqrt((1 + delta_ab)(1 + delta_cd)) * get_normalized(a, b, c, d, J):
  /// imsrg++ `GetTBME_J`.
  double get_unnormalized(OrbitIndex a, OrbitIndex b, OrbitIndex c,
                          OrbitIndex d, TwoJ J) const;
  /// Sets the normalized <ab; J|V|cd; J> = v (orbits in any order, exchange
  /// phases applied) and, for bra != ket, its partner <cd; J|V|ab; J> = h v (h
  /// = hermiticity_sign; no partner for non-Hermitian parts). imsrg++ `SetTBME`
  /// semantics. Host mirror. O(1).
  /// @throws InvariantError if a ket does not exist or bra and ket are not
  ///         connected.
  void set_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                      TwoJ J, double v);
  /// Adds v to the normalized <ab; J|V|cd; J> and h v to its partner (as
  /// `set_normalized`). Unlike imsrg++ `AddToTBME`, the anti-Hermitian sign is
  /// applied to the partner (design/13 section 2, design/30 section 6). Host
  /// mirror. O(1).
  /// @throws InvariantError if a ket does not exist or bra and ket are not
  ///         connected.
  void AddNormalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                     TwoJ J, double v);
  /// Tensor parts: normalized reduced <ab; J_bra||V||cd; J_ket> for orbits in
  /// any order (exchange phases applied); a pair (ch_bra, ch_ket) with ch_bra >
  /// ch_ket is read from the stored transposed block with the hermiticity
  /// phase h (-1)^(J_bra - J_ket) (imsrg++ `GetTBME_J_norm(J_bra, J_ket,
  /// ...)`). Scalar parts: `get_normalized(a, b, c, d, J_bra)` if J_bra ==
  /// J_ket, else 0. Returns 0 for non-existing kets or pairs that are not
  /// connected. Host mirror. O(log blocks).
  double get_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                        TwoJ J_bra, TwoJ J_ket) const;
  /// sqrt((1 + delta_ab)(1 + delta_cd)) times the tensor `get_normalized`
  /// (imsrg++ `GetTBME_J(J_bra, J_ket, ...)`).
  double get_unnormalized(OrbitIndex a, OrbitIndex b, OrbitIndex c,
                          OrbitIndex d, TwoJ J_bra, TwoJ J_ket) const;
  /// Tensor parts: sets the normalized reduced <ab; J_bra||V||cd; J_ket> = v
  /// (orbits in any order) in its stored block (through the hermiticity
  /// phase if ch_bra > ch_ket) and, inside a diagonal block, the partner
  /// element with h. Scalar parts: `set_normalized(a, b, c, d, J)` with J_bra
  /// == J_ket required. Host mirror.
  /// @throws InvariantError if a ket does not exist or the pair is not
  ///         connected.
  void set_normalized(OrbitIndex a, OrbitIndex b, OrbitIndex c, OrbitIndex d,
                      TwoJ J_bra, TwoJ J_ket, double v);

  /// Sets every element to zero (device kernel).
  void set_zero();

  /// Host-mirror view of the stored block (ch_bra, ch_ket). Precondition: the
  /// block is stored (IMSRG_CHECK).
  linalg::MatrixView<double> host_block(ChannelIndex ch_bra,
                                        ChannelIndex ch_ket);
  /// Read-only host-mirror view of the stored block (ch_bra, ch_ket).
  linalg::MatrixView<const double> host_block(ChannelIndex ch_bra,
                                              ChannelIndex ch_ket) const;
  /// Device view of the stored block (ch_bra, ch_ket).
  linalg::MatrixView<double> device_block(ChannelIndex ch_bra,
                                          ChannelIndex ch_ket);
  /// Read-only device view of the stored block (ch_bra, ch_ket).
  linalg::MatrixView<const double> device_block(ChannelIndex ch_bra,
                                                ChannelIndex ch_ket) const;

  /// Host-mirror view of the block of channel `ch` (scalar operators).
  /// Precondition: allocated.
  linalg::MatrixView<double> host_block(ChannelIndex ch);
  /// Read-only host-mirror view of the block of channel `ch`.
  linalg::MatrixView<const double> host_block(ChannelIndex ch) const;
  /// Device view of the block of channel `ch`, for kernels and BLAS
  /// (host-accessible backends).
  linalg::MatrixView<double> device_block(ChannelIndex ch);
  /// Read-only device view of the block of channel `ch`.
  linalg::MatrixView<const double> device_block(ChannelIndex ch) const;
  /// All blocks; their row and column block labels are channel indices.
  linalg::BlockSparseMatrix<double>& matrix() noexcept { return matrix_; }
  /// All blocks, read-only.
  const linalg::BlockSparseMatrix<double>& matrix() const noexcept {
    return matrix_;
  }
  /// Copies the device data to the host mirror (see the memory model).
  void SyncToHost() const { matrix_.buffer().SyncToHost(); }
  /// Copies the host mirror to the device data (see the memory model).
  void SyncToDevice() { matrix_.buffer().SyncToDevice(); }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  Hermiticity hermiticity_ = Hermiticity::kHermitian;
  bool tensor_ = false;
  std::int32_t rank_J_ = 0;
  linalg::BlockSparseMatrix<double> matrix_;
};

/// True if an operator of kind `spec` connects the orbits a and b (in either
/// order): triangle(j_a, j_b, rank_J), l_a + l_b + parity even and |tz2_a -
/// tz2_b| == 2 delta_Tz (imsrg++ `Operator::SetUpOneBodyChannels`). For a
/// scalar spec: equal (2j, parity, tz2). O(1).
bool ConnectsOrbits(const OperatorSpec& spec, const basis::Orbit& a,
                    const basis::Orbit& b) noexcept;

/// True if a tensor of kind `spec` connects the channels `bra` and `ket` (in
/// either order): |J_bra - J_ket| <= rank_J <= J_bra + J_ket, |Tz_bra -
/// Tz_ket| == delta_Tz and parity_bra + parity_ket + parity even (imsrg++
/// `TwoBodyME::Allocate`). For a scalar spec: bra == ket. O(1).
bool ConnectsChannels(const OperatorSpec& spec,
                      const basis::TwoBodyChannel& bra,
                      const basis::TwoBodyChannel& ket) noexcept;

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_TWO_BODY_H_
