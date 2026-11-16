// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_SUPPORT_ROW_MASK_H_
#define TESTS_SUPPORT_ROW_MASK_H_

/// @file
/// Test helpers for the row-range (distribution) tests of the three-body
/// kernels: poison the elements an operand must not read, and check that an
/// output was written only inside the rows it owns.

#include "imsrg/basis/three_body_rows.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::testing {

/// Sets every stored element of `b` that is on no line of `lines` to a quiet
/// NaN. An element (i, j) of a channel (packed: i >= j; dense: any) is on a
/// line of a set S iff row i or row j of that channel is in S. Host
/// operation on the host mirror (synchronised from the device first with
/// `SyncToHost()`), followed by `SyncToDevice()`. Precondition: `b` is laid out
/// on the same `ThreeBodySpace` as `lines` was built for (IMSRG_CHECK on the
/// channel dimensions). O(size).
void MaskOutsideLines(op::ThreeBody& b, const basis::RowSet& lines);

/// True iff every stored element in a row NOT in `rows` is exactly 0 (a NaN
/// counts as nonzero). Row k holds the stored elements of column k: z(m, k),
/// m >= k packed, all m dense. Syncs to the host first. Same precondition as
/// `MaskOutsideLines`. O(size).
bool OnlyRowsNonzero(const op::ThreeBody& z, const basis::RowSet& rows);

}  // namespace imsrg::testing

#endif  // TESTS_SUPPORT_ROW_MASK_H_
