#pragma once

/// @file
/// Precomputed data of the particle-hole (Pandya) term comm222_ph. Filled in by task T4; the
/// struct is empty until then.

#include "imsrg/basis/model_space.hpp"

namespace imsrg::commutator::detail {

/// Device-resident data for the particle-hole term (placeholder until T4).
struct PandyaData {};

/// Builds the Pandya data of `ms`. Placeholder until T4: returns an empty struct. O(1).
PandyaData build_pandya_data(const basis::ModelSpace& ms);

}  // namespace imsrg::commutator::detail
