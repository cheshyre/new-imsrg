#pragma once
/// @file
/// `OperatorSpec`: rank and symmetry description of an operator (design/10 section 3.3).
///
/// The C++ namespace of the `operator` module is `imsrg::op` (`operator` is a C++ keyword).
#include <cstdint>
#include <type_traits>

#include "imsrg/base/types.hpp"

namespace imsrg::op {

/// What an `Operator` represents. Only scalar, number-conserving operators with particle_rank 1
/// or 2 are implemented now; the other fields exist so that tensor (rank_J > 0), parity-changing,
/// Tz-changing and odd-particle-number ("dagger") operators can be added later without changing
/// the type. `validate` rejects what is not implemented. Trivially copyable value type.
struct OperatorSpec {
  /// Highest stored n-body part: 1 (0B + 1B), 2 (0B + 1B + 2B) or 3 (adds a 3B part, milestone
  /// M10). imsrg++ `particle_rank`.
  std::int32_t particle_rank = 2;
  /// Spherical tensor rank J of the operator; 0 for scalars. imsrg++ `rank_J`.
  std::int32_t rank_J = 0;
  /// Parity of the operator; Even conserves parity. imsrg++ `parity` (0 even, 1 odd).
  Parity parity = Parity::Even;
  /// Magnitude of the isospin-projection change |Tz_bra - Tz_ket| (0 for isospin-conserving
  /// operators). imsrg++ `rank_T`, which is |delta Tz|, not an isotensor rank.
  std::int32_t delta_Tz = 0;
  /// O^dagger = +O (Hermitian), -O (anti-Hermitian), or neither (non-Hermitian).
  Hermiticity hermiticity = Hermiticity::Hermitian;
  /// Change of the particle number: 0 for number-conserving operators; +1 or -1 for the
  /// odd-particle-number ("dagger", imsrg++ odd `legs`) operators of later milestones.
  std::int32_t delta_particle_number = 0;

  /// A scalar (rank_J 0, even parity, delta_Tz 0), number-conserving spec.
  static constexpr OperatorSpec scalar(std::int32_t particle_rank = 2,
                                       Hermiticity hermiticity = Hermiticity::Hermitian) noexcept {
    OperatorSpec s;
    s.particle_rank = particle_rank;
    s.hermiticity = hermiticity;
    return s;
  }
  /// True for rank_J 0, even parity and delta_Tz 0 (imsrg++ "scalar": non-reduced storage).
  constexpr bool is_scalar() const noexcept {
    return rank_J == 0 && parity == Parity::Even && delta_Tz == 0;
  }
  /// True if the operator conserves the particle number.
  constexpr bool is_number_conserving() const noexcept { return delta_particle_number == 0; }
  /// True if the symmetry labels (rank_J, parity, delta_Tz, delta_particle_number) agree, i.e.
  /// the two specs give the same storage layout up to particle_rank and hermiticity.
  constexpr bool same_symmetry(const OperatorSpec& o) const noexcept {
    return rank_J == o.rank_J && parity == o.parity && delta_Tz == o.delta_Tz &&
           delta_particle_number == o.delta_particle_number;
  }
  /// Field-wise equality.
  friend constexpr bool operator==(const OperatorSpec&, const OperatorSpec&) = default;
};
static_assert(std::is_trivially_copyable_v<OperatorSpec>);

/// The factor h of O_ba = h O_ab used by imsrg++: +1 Hermitian, -1 anti-Hermitian, 0
/// non-Hermitian (no relation between O_ab and O_ba). O(1).
constexpr std::int32_t hermiticity_sign(Hermiticity h) noexcept {
  return h == Hermiticity::Hermitian ? 1 : (h == Hermiticity::AntiHermitian ? -1 : 0);
}

/// Hermiticity of a linear combination a X + b Y (a, b real and nonzero) of operators with
/// hermiticities `x` and `y`: the common value if they agree, NonHermitian otherwise. O(1).
constexpr Hermiticity combined_hermiticity(Hermiticity x, Hermiticity y) noexcept {
  return x == y ? x : Hermiticity::NonHermitian;
}

/// One-letter code of the OperatorDump format (design/15): 'H', 'A' or 'N'.
char hermiticity_code(Hermiticity h) noexcept;

/// Inverse of `hermiticity_code`.
/// @throws InputError for any other character.
Hermiticity hermiticity_from_code(char c);

/// Checks a spec before an operator is built from it.
/// @throws InputError if a value is invalid: particle_rank outside 1..3, rank_J < 0,
///         delta_Tz < 0, or delta_particle_number outside -1..1.
/// @throws NotImplementedError for valid specs that are not implemented yet: non-scalar
///         (tensor, parity- or Tz-changing), non-number-conserving, or particle_rank 3.
void validate(const OperatorSpec& spec);

}  // namespace imsrg::op
