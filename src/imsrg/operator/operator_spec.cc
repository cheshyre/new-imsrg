// SPDX-License-Identifier: GPL-3.0-only
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
  if (s.delta_Tz > 2) {
    throw InputError("operator delta_Tz must be 0, 1 or 2, got " +
                     std::to_string(s.delta_Tz));
  }
  if (!s.is_scalar()) {
    if (s.particle_rank > 2) {
      throw NotImplementedError(
          "tensor operators (rank_J > 0, parity- or Tz-changing) with a "
          "three-body part are not implemented");
    }
    if (s.hermiticity == Hermiticity::kNonHermitian) {
      throw NotImplementedError(
          "non-Hermitian tensor operators are not implemented (imsrg++ "
          "stores only ch_bra <= ch_ket blocks of (anti-)Hermitian tensors)");
    }
  }
  if (s.delta_particle_number != 0) {
    throw NotImplementedError(
        "non-number-conserving operators are not implemented");
  }
}

}  // namespace imsrg::op
