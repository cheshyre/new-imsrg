// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_DETAIL_GENERATOR_THREE_BODY_H_
#define IMSRG_FLOW_DETAIL_GENERATOR_THREE_BODY_H_
/// @file
/// Device kernel of the three-body part of the single-reference generator
/// (imsrg++ ConstructGenerator_SingleRef_3body). Host-callable, no Kokkos in
/// this header.

#include <cstdint>
#include <span>

#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/flow/generator.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::flow::detail {

/// Writes the three-body part eta3 of the generator for the rows of `rows`.
///
/// For every STORED element z(m, k) of eta3 (packed lower triangle, m >= k)
/// of a row k in `rows`: with T(m), T(k) the orbit triples of the two rows,
/// if T(m) is all-particle (no core orbit) and T(k) all-core, the element is
/// f(H3(m, k), d(T(m), T(k))); if T(m) is all-core and T(k) all-particle, it
/// is -f(H3(k, m), d(T(k), T(m))) (anti-Hermitian triangle); otherwise, and on
/// the diagonal, it is 0. Here f is `GeneratorFunction(kind, ., ., nq)` and d
/// the floored `Denominators::three_body(bra triple, ket triple)` with the
/// bra the all-particle triple; the operation order is that of the host
/// formulas, so the values are bitwise those of the former host loop on the
/// OpenMP backend. Elements of rows outside `rows` are not touched.
///
/// Conventions: `f_diag` is the one-body diagonal per orbit (MeV),
/// `orbit_is_core[o]` is 1 for a core orbit and 0 otherwise, `cutoff` the
/// positive denominator floor. The kernel reads and writes the device buffers
/// (`H3`'s device data must be current; `eta3`'s device data are written and
/// its host mirror is stale on return).
/// Preconditions: H3 Hermitian, eta3 anti-Hermitian (both packed), both on
/// the three-body space `ts`; `rows` built on `ts`; `kind` one of the kinds
/// with a generator function (not Hartree-Fock).
/// @throws InvariantError on a violated precondition.
/// Complexity: O(stored elements of the row set) work, O(rows + triples)
/// host table construction.
void FillThreeBodyGenerator(const op::ThreeBody& H3, op::ThreeBody& eta3,
                            const basis::ThreeBodySpace& ts,
                            const basis::RowSet& rows,
                            std::span<const double> f_diag,
                            std::span<const std::int32_t> orbit_is_core,
                            double cutoff, GeneratorKind kind, int nq);

}  // namespace imsrg::flow::detail

#endif  // IMSRG_FLOW_DETAIL_GENERATOR_THREE_BODY_H_
