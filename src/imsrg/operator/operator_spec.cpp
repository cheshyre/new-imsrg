#include "imsrg/operator/operator_spec.hpp"

#include <string>

#include "imsrg/base/error.hpp"

namespace imsrg::op {

char hermiticity_code(Hermiticity h) noexcept {
  switch (h) {
    case Hermiticity::Hermitian:
      return 'H';
    case Hermiticity::AntiHermitian:
      return 'A';
    case Hermiticity::NonHermitian:
      return 'N';
  }
  return 'N';
}

Hermiticity hermiticity_from_code(char c) {
  switch (c) {
    case 'H':
      return Hermiticity::Hermitian;
    case 'A':
      return Hermiticity::AntiHermitian;
    case 'N':
      return Hermiticity::NonHermitian;
    default:
      throw InputError(std::string("invalid hermiticity code '") + c + "' (expected H, A or N)");
  }
}

void validate(const OperatorSpec& s) {
  if (s.particle_rank < 1 || s.particle_rank > 3) {
    throw InputError("operator particle_rank must be 1, 2 or 3, got " +
                     std::to_string(s.particle_rank));
  }
  if (s.rank_J < 0) {
    throw InputError("operator rank_J must be >= 0, got " + std::to_string(s.rank_J));
  }
  if (s.delta_Tz < 0) {
    throw InputError("operator delta_Tz must be >= 0, got " + std::to_string(s.delta_Tz));
  }
  if (s.delta_particle_number < -1 || s.delta_particle_number > 1) {
    throw InputError("operator delta_particle_number must be -1, 0 or 1, got " +
                     std::to_string(s.delta_particle_number));
  }
  if (s.rank_J != 0) throw NotImplementedError("tensor operators (rank_J > 0) are not implemented");
  if (s.parity != Parity::Even) {
    throw NotImplementedError("parity-changing operators are not implemented");
  }
  if (s.delta_Tz != 0) throw NotImplementedError("Tz-changing operators are not implemented");
  if (s.delta_particle_number != 0) {
    throw NotImplementedError("non-number-conserving operators are not implemented");
  }
  if (s.particle_rank == 3) {
    throw NotImplementedError("three-body operators (particle_rank 3) are not implemented yet");
  }
}

}  // namespace imsrg::op
