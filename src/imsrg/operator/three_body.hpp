#pragma once
/// @file
/// `ThreeBody`: placeholder for the three-body part of an operator (design/00 D15).

namespace imsrg::op {

/// Placeholder for the three-body part of an `Operator`. Its storage layout is chosen by the
/// IMSRG(3) benchmark of milestone M10 (design/14); until then it holds nothing, and operators
/// with particle_rank 3 are rejected by `validate` (NotImplementedError). Whatever the layout,
/// its norm must reproduce imsrg++'s Frobenius norm over the canonical pn storage (design/00
/// D12). Three-nucleon INPUT interactions do not use this type: they live in separate input
/// containers (NO2B, me3j) that are converted by explicit functions (D15).
class ThreeBody {};

}  // namespace imsrg::op
