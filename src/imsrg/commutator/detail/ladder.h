// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_LADDER_H_
#define IMSRG_COMMUTATOR_DETAIL_LADDER_H_

/// @file
/// Precomputed data of the particle-particle / hole-hole ladder terms
/// (comm222_pp_hhss, comm221ss and their fusion). For every two-body channel
/// with N kets the terms need
///   Mpp = X P_pp Y - Y P_pp X   and   Mhh = X P_hh Y - Y P_hh X,
/// with P_pp = diag((1 - n_p)(1 - n_q)) and P_hh = diag(n_p n_q) over the
/// normalized kets. P_pp and P_hh are non-zero only on the kets of the lists Kp
/// and Kh, so the products are small dense gemms on gathered operands. All
/// buffers are allocated once, when the workspace is built.

#include <cstdint>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

/// The gathered operands of one projector (pp or hh) over all channels. Channel
/// c has `count[c]` listed kets (local indices `list[list_offset[c] ..)` with
/// weights `weight[..]`), and the operands Xg (N x count, column major, `Xg(i,
/// m) = w_m X(i, K_m)`) and Yg (count x N, `Yg(m, j) = Y(K_m, j)`) at
/// `gather_offset[c]` (both N * count elements).
struct LadderProjector {
  std::vector<std::int32_t> count_host;          ///< Listed kets per channel.
  std::vector<std::int64_t> gather_offset_host;  ///< num_channels + 1 offsets
                                                 ///< of the gathered operands.
  exec::DeviceBuffer<std::int32_t> list;  ///< Concatenated local ket indices.
  exec::DeviceBuffer<double> weight;  ///< Projector weight of each listed ket.
  exec::DeviceBuffer<std::int64_t>
      list_offset;  ///< num_channels + 1 offsets into `list`.
  exec::DeviceBuffer<std::int64_t>
      gather_offset;              ///< Device copy of `gather_offset_host`.
  exec::DeviceBuffer<double> xg;  ///< Gathered X operands, all channels.
  exec::DeviceBuffer<double> yg;  ///< Gathered Y operands, all channels.
  exec::DeviceBuffer<double>
      product;  ///< Mpp or Mhh, laid out like a two-body operator.
};

/// Device-resident data of the ladder terms.
struct LadderData {
  LadderProjector pp;  ///< Weights (1 - n_p)(1 - n_q).
  LadderProjector hh;  ///< Weights n_p n_q.
  /// Scratch for the per-call gemm descriptions (capacity reserved at
  /// construction).
  std::vector<linalg::GemmProblem> problems;

  /// Total size in bytes of the scratch buffers (Xg, Yg and product of both
  /// projectors). O(1).
  std::size_t scratch_bytes() const noexcept {
    std::size_t n = 0;
    for (const LadderProjector* p : {&pp, &hh})
      n += p->xg.size() + p->yg.size() + p->product.size();
    return n * sizeof(double);
  }
};

/// Builds the ladder data of `ms` (lists, offsets and all scratch buffers).
/// Needs an initialized `exec::Runtime`. The scratch products use the layout of
/// `op::TwoBody` (dense column-major N x N blocks back to back in channel
/// order, the same offsets as `Workspace::Impl:: block_offsets`). Complexity
/// O(number of kets + buffer sizes).
LadderData BuildLadderData(const basis::ModelSpace& ms);

/// The raw ladder products W_pp = X P_pp Y and W_hh = X P_hh Y (no
/// antisymmetrization) of every channel, written into
/// `impl.ladder.{pp,hh}.product` (layout of `op::TwoBody`). X and Y must have
/// an allocated two-body part and live on the workspace's model space. Shared
/// by the ladder terms (which then antisymmetrize, kernels/ladder.cc) and the
/// factorized double commutator. The products are complete when it returns
/// (host BLAS). Complexity: grouped gemms of sum_c N_c^2 (|Kp| + |Kh|) flops.
void LadderProducts(const op::Operator& X, const op::Operator& Y,
                    Workspace::Impl& impl);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_LADDER_H_
