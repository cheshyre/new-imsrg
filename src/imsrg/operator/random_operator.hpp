#pragma once
/// @file
/// Random operators with an explicit seed, for property tests and for golden data exchanged
/// with imsrg-devel through OperatorDump (design/30 section 1). imsrg++ uses a static incrementing
/// seed, so its random operators depend on call order; ours depend only on the arguments.
#include <cstdint>
#include <memory>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::op {

/// Options of `random_operator`. Values are uniform in [-scale, scale).
struct RandomOperatorOptions {
  /// Seed of the generator. The same seed gives the same operator on every platform.
  std::uint64_t seed = 1;
  /// Scale of the zero-body part (0: zero-body part 0). Ignored (0) unless Hermitian.
  double zero_body_scale = 0.0;
  /// Scale of the one-body elements.
  double one_body_scale = 1.0;
  /// Scale of the two-body elements (normalized storage).
  double two_body_scale = 1.0;
};

/// A random operator of kind `spec` on `ms` with the symmetries of its storage:
/// * one-body elements only between orbits with equal (l, 2j, tz2) (scalar, isospin-conserving
///   selection rule); two-body elements in every channel block (only existing kets are stored,
///   so Pauli and the channel quantum numbers hold by construction);
/// * Hermitian: O_ba = O_ab; anti-Hermitian: O_ba = -O_ab with a zero diagonal; non-Hermitian:
///   independent elements (both for 1B and for each 2B block).
/// Generator: splitmix64 seeded with `options.seed`; a uniform double is (x >> 11) * 2^-53,
/// mapped to [-scale, scale). Values are drawn in a fixed order: the zero-body part (if its scale
/// is nonzero), then the one-body elements (a ascending, b ascending, b >= a unless
/// non-Hermitian, diagonal skipped if anti-Hermitian), then the two-body blocks in channel order
/// (row i ascending, column j ascending, the same triangle rules). The result is therefore
/// reproducible across platforms and compilers (no std:: distributions).
/// @throws InputError / NotImplementedError from `validate(spec)`.
Operator random_operator(std::shared_ptr<const basis::ModelSpace> ms, const OperatorSpec& spec,
                         const RandomOperatorOptions& options = {});

}  // namespace imsrg::op
