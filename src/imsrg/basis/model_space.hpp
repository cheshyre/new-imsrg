#pragma once
/// @file
/// The ModelSpace bundle: basis, reference, partition and the channel structures built from them
/// (design/10 section 3.2).

#include <cstdint>
#include <memory>
#include <string_view>

#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/basis/two_body_channels.hpp"

namespace imsrg::basis {

/// Immutable bundle of everything that defines the many-body basis of an operator: the
/// single-particle basis, the reference occupations, the core/valence/q-space partition, the
/// two-body channels and their ket subsets. Shared as `std::shared_ptr<const ModelSpace>`.
///
/// The parts are held by `shared_ptr<const ...>`, so copies are cheap and model spaces derived
/// with `with_reference` share the occupation-independent structures (the basis and the two-body
/// channels). There are no back-pointers, so copying and moving are always correct (imsrg++'s
/// ModelSpace copy is unsound, design/reference/01 section 9 B.20).
///
/// Later milestones add the particle-hole (Pandya) channels, the optional three-body channels
/// (`with_three_body`), device mirrors of the index tables, and the angular-momentum tables; the
/// accessors below stay as they are.
class ModelSpace {
 public:
  /// Assembles a model space from its parts and builds the two-body channels and ket subsets.
  /// @throws InputError if `reference` or `partition` does not have one entry per orbit.
  static std::shared_ptr<const ModelSpace> create(SingleParticleBasis basis, Reference reference,
                                                  Partition partition);

  /// The construction of the imsrg++ driver (imsrg++.cc:331-366, `ModelSpace(emax, reference,
  /// valence_space)` followed by `SetLmax` and `SetHbarOmega`): `reference` is a nucleus name or
  /// "default" (then `default_reference_nucleus(valence_space)`); the reference occupations are
  /// its naive filling (`Reference::from_nucleus`) and the partition is
  /// `Partition::from_definition(define_valence_space(valence_space, basis, nucleus))`.
  /// Occupation files and custom valence spaces use `create` with the corresponding parts.
  /// @throws InputError for unknown names, a nucleus that does not fit the basis, etc.
  static std::shared_ptr<const ModelSpace> from_names(const BasisTruncation& truncation,
                                                      double hw_MeV, std::string_view reference,
                                                      std::string_view valence_space);

  /// The single-particle basis.
  const SingleParticleBasis& basis() const noexcept { return *basis_; }
  /// The reference occupations.
  const Reference& reference() const noexcept { return *reference_; }
  /// The core/valence/q-space partition.
  const Partition& partition() const noexcept { return *partition_; }
  /// The two-body kets and channels (independent of reference and partition).
  const TwoBodyChannels& two_body() const noexcept { return *two_body_; }
  /// The per-channel ket subsets (pp, hh, ph, cc, vc, qc, vv, qv, qq) for this reference and
  /// partition.
  const TwoBodyKetSubsets& two_body_subsets() const noexcept { return *two_body_subsets_; }

  /// Number of orbits.
  std::int32_t num_orbits() const noexcept { return basis_->num_orbits(); }

  /// A model space with new reference occupations and the same partition (imsrg++
  /// `SetReference(string)` keeps core and valence). Shares the basis and the two-body channels.
  /// @throws InputError if `reference` has the wrong number of orbits.
  std::shared_ptr<const ModelSpace> with_reference(Reference reference) const;

  /// A model space with new reference occupations and a new partition (e.g.
  /// `Partition::with_holes_as_core`, the imsrg++ `SetReference(map)` rule used after the HF
  /// re-filling). Shares the basis and the two-body channels.
  /// @throws InputError if a size differs from the number of orbits.
  std::shared_ptr<const ModelSpace> with_reference_and_partition(Reference reference,
                                                                 Partition partition) const;

  /// Structural equality: same basis, occupations and partition (operators defined on equal
  /// model spaces may be combined).
  friend bool operator==(const ModelSpace& a, const ModelSpace& b) noexcept;

 private:
  ModelSpace() = default;

  std::shared_ptr<const SingleParticleBasis> basis_;
  std::shared_ptr<const Reference> reference_;
  std::shared_ptr<const Partition> partition_;
  std::shared_ptr<const TwoBodyChannels> two_body_;
  std::shared_ptr<const TwoBodyKetSubsets> two_body_subsets_;
};

}  // namespace imsrg::basis
