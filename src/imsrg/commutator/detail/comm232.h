// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_COMM232_H_
#define IMSRG_COMMUTATOR_DETAIL_COMM232_H_

/// @file
/// Internal entry points of the N7 term comm232ss ([X2, Y3] + [X3, Y2] -> Z2,
/// imsrg3.h; implementation and conventions in kernels/comm232.cc), for
/// `terms::Comm232ss` and the tests (design/12 section 4.2: every kernel
/// phase passes the same tests). Internal to the commutator module and its
/// tests; no Kokkos.

#include <cstdint>

#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

/// Scratch budget of `terms::Comm232ss` for the pair-third blocks (bytes, all
/// block sets of one channel batch together): 512 MiB, so emax 3 at E3max 9
/// runs in one batch (two block sets for an anti-Hermitian X and a Hermitian
/// Y: 365 MB for O16, 505 MB for He4; 8.8 MB for O16 at emax 2, E3max 6).
inline constexpr std::int64_t kComm232MaxBlockBytes = std::int64_t{1} << 29;

/// comm232ss evaluated element by element: one work item per stored Z2
/// element, three-body elements read with `ThreeBodyElement` (imsrg++
/// `GetME_pn`, recoupled from the canonical kets) inside the sums over c, J',
/// (a b) and J3. Same contract as `terms::Comm232ss` (adds into Z, any
/// hermiticity). Complexity per element O(norb * J'-range * kets *
/// J3-range) element reads of cost O(J range^2 log triples).
void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);

/// comm232ss through the pair-third blocks of the three-body operands (the
/// path of `terms::Comm232ss`, which passes `kComm232MaxBlockBytes`): the
/// active rows of W = E B3 E^T per three-body channel, built per call for
/// batches of consecutive channels whose blocks (one set per operand, two
/// for a non-Hermitian three-body part) fit `max_block_bytes` together; a
/// single channel larger than the budget forms its own batch. The element
/// kernel runs once per batch (same result for any budget up to rounding).
/// Same contract as `terms::Comm232ss`. Complexity: the block build O(sum
/// over channels A S nnz(E row)^2) plus, per batch, O(elements * norb *
/// J'-range * J3-range * (rows + log S)) (S pair-third states of a channel,
/// A of them with an active pair).
void Comm232ssPairThird(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx,
                        std::int64_t max_block_bytes);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_COMM232_H_
