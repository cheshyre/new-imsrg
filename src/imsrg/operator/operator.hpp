#pragma once
/// @file
/// `Operator`: a value-semantic many-body operator built by composition (design/00 D8,
/// design/10 section 3.3). Arithmetic, norms, normal ordering etc. are free functions in the
/// other headers of this module.
#include <memory>
#include <optional>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/one_body.hpp"
#include "imsrg/operator/operator_spec.hpp"
#include "imsrg/operator/three_body.hpp"
#include "imsrg/operator/two_body.hpp"

namespace imsrg::op {

/// A many-body operator O = O0 + sum O_ab {a+ b} + (1/4) sum V_abcd {a+ b+ d c} (+ 3B later) on a
/// model space: a zero-body constant, a `OneBody`, a `TwoBody` (allocated iff particle_rank >= 2)
/// and an optional `ThreeBody` (M10), described by an `OperatorSpec`.
///
/// The operator does not record with respect to which reference it is normal ordered: the
/// normal-ordering functions say it in their names, and the model space's reference is the one
/// they use by default. Combining two operators checks that their model spaces are the same
/// object or structurally equal (`same_model_space`).
///
/// Regular value type: copy = deep copy of all parts, move = cheap (the model space is shared).
/// The parts keep the shapes given at construction: assigning a part of another shape through
/// the mutable accessors is a precondition violation. Needs an initialized `exec::Runtime` (except
/// for default-constructed operators); operators must be destroyed before the Runtime.
class Operator {
 public:
  /// Empty operator without a model space; only assignment, destruction and `empty()` are valid.
  Operator() = default;
  /// The zero operator of kind `spec` on `ms`.
  /// @throws InputError / NotImplementedError from `validate(spec)`.
  /// @throws InvariantError if `ms` is null.
  Operator(std::shared_ptr<const basis::ModelSpace> ms, const OperatorSpec& spec);

  /// True for a default-constructed (or moved-from) operator.
  bool empty() const noexcept { return ms_ == nullptr; }
  /// Rank and symmetry description.
  const OperatorSpec& spec() const noexcept { return spec_; }
  /// Shorthand for spec().hermiticity.
  Hermiticity hermiticity() const noexcept { return spec_.hermiticity; }
  /// Changes the hermiticity in the spec and in the parts; the data are not touched (imsrg++
  /// `SetHermitian` / `SetAntiHermitian` / `SetNonHermitian`).
  void set_hermiticity(Hermiticity h) noexcept;
  /// The model space. Precondition: !empty() (IMSRG_CHECK).
  const basis::ModelSpace& model_space() const;
  /// The shared model space (null for an empty operator).
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr() const noexcept { return ms_; }

  /// Zero-body part (a constant, MeV for Hamiltonians).
  double& zero_body() noexcept { return zero_body_; }
  /// Zero-body part, read-only.
  double zero_body() const noexcept { return zero_body_; }
  /// One-body part.
  OneBody& one_body() noexcept { return one_body_; }
  /// One-body part, read-only.
  const OneBody& one_body() const noexcept { return one_body_; }
  /// Two-body part (not allocated for particle_rank 1).
  TwoBody& two_body() noexcept { return two_body_; }
  /// Two-body part, read-only.
  const TwoBody& two_body() const noexcept { return two_body_; }
  /// Three-body part: always empty until M10.
  std::optional<ThreeBody>& three_body() noexcept { return three_body_; }
  /// Three-body part, read-only.
  const std::optional<ThreeBody>& three_body() const noexcept { return three_body_; }

  /// Sets all parts to zero (imsrg++ `Erase`).
  void set_zero();
  /// Syncs the device data of all parts to their host mirrors (see `OneBody`).
  void sync_to_host();
  /// Syncs the host mirrors of all parts to the device data (see `OneBody`).
  void sync_to_device();

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  OperatorSpec spec_;
  double zero_body_ = 0.0;
  OneBody one_body_;
  TwoBody two_body_;
  std::optional<ThreeBody> three_body_;
};

/// True if `a` and `b` are defined on the same model space: the same object, or structurally
/// equal (`basis::ModelSpace::operator==`: same basis, occupations and partition). Empty
/// operators are never on the same model space. O(1) for the same object, O(norb) otherwise.
bool same_model_space(const Operator& a, const Operator& b) noexcept;

}  // namespace imsrg::op
