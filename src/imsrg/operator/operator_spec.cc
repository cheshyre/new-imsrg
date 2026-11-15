// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/operator_spec.h"

#include <string>

#include "imsrg/base/error.h"

namespace imsrg::op {

char HermiticityCode(Hermiticity h) noexcept {
  switch (h) {
    case Hermiticity::kHermitian:
      return 'H';
    case Hermiticity::kAntiHermitian:
      return 'A';
    case Hermiticity::kNonHermitian:
      return 'N';
  }
  return 'N';
}

Hermiticity HermiticityFromCode(char c) {
  switch (c) {
    case 'H':
      return Hermiticity::kHermitian;
    case 'A':
      return Hermiticity::kAntiHermitian;
    case 'N':
      return Hermiticity::kNonHermitian;
    default:
      throw InputError(std::string("invalid hermiticity code '") + c +
                       "' (expected H, A or N)");
  }
}

void Validate(const OperatorSpec& s) {
  if (s.particle_rank < 1 || s.particle_rank > 3) {
    throw InputError("operator particle_rank must be 1, 2 or 3, got " +
                     std::to_string(s.particle_rank));
  }
  if (s.rank_J < 0) {
    throw InputError("operator rank_J must be >= 0, got " +
                     std::to_string(s.rank_J));
  }
  if (s.delta_Tz < 0) {
    throw InputError("operator delta_Tz must be >= 0, got " +
                     std::to_string(s.delta_Tz));
  }
  if (s.delta_particle_number < -1 || s.delta_particle_number > 1) {
    throw InputError("operator delta_particle_number must be -1, 0 or 1, got " +
                     std::to_string(s.delta_particle_number));
  }
  if (s.rank_J != 0)
    throw NotImplementedError(
        "tensor operators (rank_J > 0) are not implemented");
  if (s.parity != Parity::kEven) {
    throw NotImplementedError("parity-changing operators are not implemented");
  }
  if (s.delta_Tz != 0)
    throw NotImplementedError("Tz-changing operators are not implemented");
  if (s.delta_particle_number != 0) {
    throw NotImplementedError(
        "non-number-conserving operators are not implemented");
  }
}

}  // namespace imsrg::op
