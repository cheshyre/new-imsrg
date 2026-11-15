// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_N7_COMMON_H_
#define IMSRG_COMMUTATOR_DETAIL_N7_COMMON_H_

/// @file
/// Host-side helpers shared by the N7 term functions (imsrg3.h): operand
/// checks and the construction of the kernel views. Internal to the
/// commutator module; no Kokkos.

#include <cstdint>
#include <type_traits>

#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

/// The output two-body blocks of Z: the stored normalized elements, block of
/// channel ch (n x n, column-major) at `data + block_offset[ch]`; kets of a
/// channel in the order of `basis::TwoBodyTableView::channel_kets`.
struct TwoBodyOutView {
  double* data = nullptr;                      ///< Z's two-body data.
  const std::int64_t* block_offset = nullptr;  ///< num_channels + 1 entries.
};
static_assert(std::is_trivially_copyable_v<TwoBodyOutView>);

/// Checks the operands of an N7 term `term` and returns the kernel views:
/// X, Y and Z non-empty, scalar and number-conserving, on model spaces with
/// the basis, reference and partition of the context's; Z's hermiticity that
/// of [X, Y]; every three-body part present laid out as the context's kept
/// space (same block offsets).
/// @throws NotImplementedError for non-scalar or non-number-conserving
///         operands, InputError for empty ones.
/// @throws InvariantError for a different model space, layout or Z
///         hermiticity.
N7Views PrepareN7(const char* term, const op::Operator& X,
                  const op::Operator& Y, const op::Operator& Z,
                  const ThreeBodyContext& ctx);

/// The output view of Z's three-body part. Precondition (InvariantError): Z
/// has a three-body part laid out as the context's kept space.
ThreeBodyOutView ThreeBodyOut(const char* term, op::Operator& Z,
                              const ThreeBodyContext& ctx);

/// The output view of Z's two-body part. Precondition (InvariantError): Z
/// has a two-body part laid out as the context's model space.
TwoBodyOutView TwoBodyOut(const char* term, op::Operator& Z,
                          const ThreeBodyContext& ctx);

/// Z's one-body data (norb x norb, column-major), device pointer.
double* OneBodyOut(op::Operator& Z);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_N7_COMMON_H_
