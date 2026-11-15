// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_COMM232_H_
#define IMSRG_COMMUTATOR_DETAIL_COMM232_H_

/// @file
/// Internal entry points of the N7 term comm232ss ([X2, Y3] + [X3, Y2] -> Z2,
/// imsrg3.h; implementation and conventions in kernels/comm232.cc), for the
/// tests (design/12 section 4.2: every kernel phase passes the same tests).
/// Internal to the commutator module and its tests; no Kokkos.

#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::detail {

/// comm232ss evaluated element by element: one work item per stored Z2
/// element, three-body elements read with `ThreeBodyElement` (imsrg++
/// `GetME_pn`, recoupled from the canonical kets) inside the sums over c, J',
/// (a b) and J3. Same contract as `terms::Comm232ss` (adds into Z, any
/// hermiticity). Complexity per element O(norb * J'-range * kets *
/// J3-range) element reads of cost O(J range^2 log triples).
void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_COMM232_H_
