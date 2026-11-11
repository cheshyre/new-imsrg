#pragma once

/// @file
/// Precomputed data of the particle-particle / hole-hole ladder term (comm222_pp_hh and the
/// shared 221 intermediates). Filled in by task T3b; the struct is empty until then.

#include "imsrg/basis/model_space.hpp"

namespace imsrg::commutator::detail {

/// Device-resident data for the ladder terms (placeholder until T3b).
struct LadderData {};

/// Builds the ladder data of `ms`. Placeholder until T3b: returns an empty struct. O(1).
LadderData build_ladder_data(const basis::ModelSpace& ms);

}  // namespace imsrg::commutator::detail
