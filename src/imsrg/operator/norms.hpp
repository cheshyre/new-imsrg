#pragma once
/// @file
/// Operator norms with the imsrg++ conventions (design/00 D12, design/13 section 2,
/// design/reference/01 section 3.4). They drive the Magnus step size, the Omega splitting and the
/// BCH termination, so they must equal imsrg++'s exactly (tested against the golden norms at
/// 1e-10 relative).
#include "imsrg/operator/operator.hpp"

namespace imsrg::op {

/// imsrg++ `Operator::OneBodyNorm` (Operator.cc:1666-1682) for scalar operators:
/// sqrt( sum_p sum_{q with (l, 2j, tz2) of p} O_pq^2 (2 j_p + 1) ). Entries between orbits of
/// different (l, 2j, tz2) are ignored, as in imsrg++. Host computation, O(norb^2); syncs the
/// one-body part to the host.
double one_body_norm(const Operator& op);

/// imsrg++ `TwoBodyME::Norm` (TwoBodyME.cc:781-803) for scalar operators:
/// sqrt( sum_channels ((2J + 1) ||V^J||_F)^2 ) over the stored normalized blocks (note the
/// weight (2J + 1)^2). 0 if the two-body part is not allocated. One device reduction per
/// channel block.
double two_body_norm(const Operator& op);

/// Three-body norm (imsrg++ Frobenius norm over its canonical pn storage); 0 while operators
/// have no three-body part (until M10).
double three_body_norm(const Operator& op);

/// imsrg++ `Operator::Norm` for number-conserving operators: sqrt(n1^2 + n2^2 + n3^2) of the
/// norms above. The zero-body part is EXCLUDED.
double norm(const Operator& op);

}  // namespace imsrg::op
