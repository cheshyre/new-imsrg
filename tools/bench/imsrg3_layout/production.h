// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_PRODUCTION_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_PRODUCTION_H_
/// @file
/// The production [2,2]->3 term (`commutator::Comm223`, three_body.h) in the
/// terms of the layout benchmark: the production model space of a benchmark
/// canonical space and the conversion of a production three-body part to the
/// benchmark's dense canonical blocks, for the checks against K0 and the
/// timings against K1.

#include <memory>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/three_body.h"
#include "tools/bench/imsrg3_layout/canonical_space.h"

namespace imsrg::layout3 {

/// The model space whose kept three-body space has the rows of a
/// `CanonicalSpace(ms, de3max, occnat3)`: `ms` (built `WithThreeBody`) with
/// its three-body cuts (E3max, emax_3body) plus the configuration cuts
/// `de3max` and `occnat3` (reference occupations). Basis, reference and
/// partition are those of `ms`.
std::shared_ptr<const basis::ModelSpace> ProductionModelSpace(
    const std::shared_ptr<const basis::ModelSpace>& ms, double de3max,
    double occnat3);

/// The dense canonical blocks of `space` (layout of `CanonicalSpace`, both
/// triangles) holding the full matrices of the production part `z`. The
/// nonempty channels of z's kept space must be the channels of `space` in
/// order, with the same (2J, parity, 2Tz), dimensions, row orbits and Jab
/// (IMSRG_CHECK). O(total size).
std::vector<double> ToCanonicalDense(const op::ThreeBody& z,
                                     const CanonicalSpace& space);

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_PRODUCTION_H_
