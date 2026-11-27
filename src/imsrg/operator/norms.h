// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_NORMS_H_
#define IMSRG_OPERATOR_NORMS_H_
/// @file
/// Operator norms with the imsrg++ conventions (design/00 D12, design/13
/// section 2, design/reference/01 section 3.4). They drive the Magnus step
/// size, the Omega splitting and the BCH termination, so they must equal
/// imsrg++'s exactly (tested against the golden norms at 1e-10 relative).
#include <cstdint>

#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// Weight of the reduced one-body matrix elements of a tensor operator in the
/// one-body norm (scalar operators ignore it). Design/30 section 6, AL-A-Q6.
enum class TensorOneBodyNorm : std::int8_t {
  /// sum over connected (p, q) of O_pq^2 (2 j_p + 1)(2 j_q + 1) / (2 L + 1),
  /// L = `spec().rank_J` (undoubled): the number of magnetic substates, always
  /// positive. The default.
  kDegeneracyWeighted,
  /// imsrg++ `Operator::OneBodyNorm` for tensors: the integer degeneracy
  /// (2 j_p + 1) ((min(2 j_q, 2 j_p + L) - max(-2 j_q, 2 j_p - L)) / 2 + 1),
  /// which mixes 2j with the undoubled L and can be negative (then the norm
  /// is NaN). Only for comparisons with imsrg++ / the fork 79e9330.
  kImsrgppInteger,
};

/// imsrg++ `Operator::OneBodyNorm` (Operator.cc:1666-1682) for scalar
/// operators: sqrt( sum_p sum_{q with (l, 2j, tz2) of p} O_pq^2 (2 j_p + 1) ).
/// Entries between orbits of different (l, 2j, tz2) are ignored, as in imsrg++.
/// Tensor operators: sum over the connected (p, q) weighted as `mode` says.
/// Host computation, O(norb^2); syncs the one-body part to the host.
double OneBodyNorm(
    const Operator& op,
    TensorOneBodyNorm mode = TensorOneBodyNorm::kDegeneracyWeighted);

/// imsrg++ `TwoBodyME::Norm` (TwoBodyME.cc:781-803) for scalar operators: sqrt(
/// sum_channels ((2J + 1) ||V^J||_F)^2 ) over the stored normalized blocks
/// (note the weight (2J + 1)^2). 0 if the two-body part is not allocated. One
/// device reduction per channel block.
double TwoBodyNorm(const Operator& op);

/// Three-body norm: `ImsrgppNorm` of the three-body part (imsrg++
/// `ThreeBodyStorage_pn::Norm`, the sum of squares over the stored packed
/// lower triangle of the canonical pn blocks); 0 without a three-body part.
/// HIDDEN COLLECTIVE if the three-body part is distributed (see
/// `ImsrgppNorm`): every rank must call it.
double ThreeBodyNorm(const Operator& op);

/// imsrg++ `Operator::Norm` for number-conserving operators: sqrt(n1^2 + n2^2 +
/// n3^2) of the norms above. The zero-body part is EXCLUDED.
/// HIDDEN COLLECTIVE for an operator with a distributed three-body part (M12b
/// review note M4): every rank of the distribution's communicator must call
/// it, and a norm must never be computed inside a rank-0-only branch (e.g. a
/// writer that runs on the root only). The result has identical bits on all
/// ranks. `mode` selects the tensor one-body weight (scalars ignore it).
double Norm(const Operator& op,
            TensorOneBodyNorm mode = TensorOneBodyNorm::kDegeneracyWeighted);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_NORMS_H_
