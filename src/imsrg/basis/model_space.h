// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASIS_MODEL_SPACE_H_
#define IMSRG_BASIS_MODEL_SPACE_H_
/// @file
/// The ModelSpace bundle: basis, reference, partition and the channel
/// structures built from them (design/10 section 3.2).

#include <cstdint>
#include <memory>
#include <string_view>

#include "imsrg/basis/particle_hole_channels.h"
#include "imsrg/basis/partition.h"
#include "imsrg/basis/reference.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/basis/three_body_distribution.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/basis/two_body_channels.h"

namespace imsrg::basis {

/// Immutable bundle of everything that defines the many-body basis of an
/// operator: the single-particle basis, the reference occupations, the
/// core/valence/q-space partition, the two-body channels and their ket subsets.
/// Shared as `std::shared_ptr<const ModelSpace>`.
///
/// The parts are held by `shared_ptr<const ...>`, so copies are cheap and model
/// spaces derived with `WithReference` share the occupation-independent
/// structures (the basis and the two-body channels). There are no
/// back-pointers, so copying and moving are always correct (imsrg++'s
/// ModelSpace copy is unsound, design/reference/01 section 9 B.20).
///
/// The particle-hole (Pandya) channel structure is always built (it is cheap);
/// the three-body channels only on request (`WithThreeBody`). Device mirrors
/// of the index tables are a separate object, `DeviceTables(const ModelSpace&)`
/// (device_tables.h), so that building a ModelSpace never needs the Kokkos
/// runtime (design/QUESTIONS.md M3-Q7). The angular-momentum tables are added
/// by a later milestone.
class ModelSpace {
 public:
  ModelSpace(const ModelSpace&) = default;
  ModelSpace& operator=(const ModelSpace&) = default;
  ModelSpace(ModelSpace&&) = default;
  ModelSpace& operator=(ModelSpace&&) = default;

  /// Assembles a model space from its parts and builds the two-body channels
  /// and ket subsets.
  /// @throws InputError if `reference` or `partition` does not have one entry
  ///         per orbit.
  static std::shared_ptr<const ModelSpace> Create(SingleParticleBasis basis,
                                                  Reference reference,
                                                  Partition partition);

  /// The construction of the imsrg++ driver (imsrg++.cc:331-366,
  /// `ModelSpace(emax, reference, valence_space)` followed by `SetLmax` and
  /// `SetHbarOmega`): `reference` is a nucleus name or "default" (then
  /// `DefaultReferenceNucleus(valence_space)`); the reference occupations are
  /// its naive filling (`Reference::FromNucleus`) and the partition is
  /// `Partition::FromDefinition(DefineValenceSpace(valence_space, basis,
  /// nucleus))`. Occupation files and custom valence spaces use `create` with
  /// the corresponding parts.
  /// @throws InputError for unknown names, a nucleus that does not fit the
  ///         basis, etc.
  static std::shared_ptr<const ModelSpace> FromNames(
      const BasisTruncation& truncation, double hw_MeV,
      std::string_view reference, std::string_view valence_space);

  /// The single-particle basis.
  const SingleParticleBasis& basis() const noexcept { return *basis_; }
  /// The reference occupations.
  const Reference& reference() const noexcept { return *reference_; }
  /// The core/valence/q-space partition.
  const Partition& partition() const noexcept { return *partition_; }
  /// The two-body kets and channels (independent of reference and partition).
  const TwoBodyChannels& two_body() const noexcept { return *two_body_; }
  /// The per-channel ket subsets (pp, hh, ph, cc, vc, qc, vv, qv, qq) for this
  /// reference and partition.
  const TwoBodyKetSubsets& two_body_subsets() const noexcept {
    return *two_body_subsets_;
  }
  /// The particle-hole (cross-coupled, Pandya) channels (independent of
  /// reference and partition).
  const ParticleHoleChannels& particle_hole() const noexcept {
    return *particle_hole_;
  }
  /// The three-body channels, or nullptr unless built with `WithThreeBody`.
  const ThreeBodyChannels* three_body() const noexcept {
    return three_body_.get();
  }

  /// The kept canonical three-body space (the 3B kets that pass the
  /// configuration cuts for this model space's reference), or nullptr unless
  /// built with `WithThreeBody`.
  const ThreeBodySpace* three_body_space() const noexcept {
    return three_body_space_.get();
  }

  /// The rows of the three-body space stored on this rank:
  /// `RowSet::Whole` of `three_body_space()` unless the model space is
  /// distributed (`WithThreeBodyDistribution`), then this rank's part of the
  /// partition. An empty set for a model space without three-body space.
  /// O(1) (cached at construction).
  const RowSet& three_body_rows() const noexcept { return *three_body_rows_; }

  /// The three-body row distribution, or nullptr if the three-body space is
  /// not distributed (every rank holds all rows).
  const ThreeBodyDistribution* three_body_distribution() const noexcept {
    return three_body_distribution_.get();
  }

  /// Number of orbits.
  std::int32_t num_orbits() const noexcept { return basis_->num_orbits(); }

  /// A model space with new reference occupations and the same partition
  /// (imsrg++ `SetReference(string)` keeps core and valence). Shares the basis
  /// and the two-body channels.
  /// @throws InputError if `reference` has the wrong number of orbits.
  std::shared_ptr<const ModelSpace> WithReference(Reference reference) const;

  /// A model space with new reference occupations and a new partition (e.g.
  /// `Partition::WithHolesAsCore`, the imsrg++ `SetReference(map)` rule used
  /// after the HF re-filling). Shares the basis and the two-body channels.
  /// @throws InputError if a size differs from the number of orbits.
  std::shared_ptr<const ModelSpace> WithReferenceAndPartition(
      Reference reference, Partition partition) const;

  /// The same model space with three-body channels and the kept three-body
  /// space for `cuts`. The channels depend only on the basis and the cuts, so
  /// derived model spaces (`WithReference`) share them; the kept space also
  /// depends on the reference (dE3max, occupations) and is rebuilt for every
  /// derived reference. Cost: the 3B ket enumeration.
  std::shared_ptr<const ModelSpace> WithThreeBody(
      const ThreeBodyCuts& cuts) const;

  /// The same model space with its three-body rows distributed as `d`: this
  /// rank stores `d.partition->part(d.comm.rank())`. Distributing a model
  /// space that is already distributed replaces the distribution.
  /// `WithReference`/`WithReferenceAndPartition` of a distributed space drop
  /// the distribution (the kept space depends on the reference, so the
  /// partition would no longer cover it): distribute the derived space again.
  /// @throws InputError if there is no three-body space, the partition is
  ///         null, or `partition->num_parts() != comm.size()`.
  std::shared_ptr<const ModelSpace> WithThreeBodyDistribution(
      ThreeBodyDistribution d) const;

  /// The same model space with the three-body rows not distributed (all rows
  /// local). Used by `op::GatherToRoot`.
  std::shared_ptr<const ModelSpace> WithoutThreeBodyDistribution() const;

  /// Structural equality: same basis, occupations and partition, the same
  /// three-body cuts (or both without three-body channels) and the same
  /// three-body distribution (both undistributed, or the same rank and the
  /// same partition content). Operators on equal model spaces may be
  /// combined.
  friend bool operator==(const ModelSpace& a, const ModelSpace& b) noexcept;

 private:
  ModelSpace();

  std::shared_ptr<const SingleParticleBasis> basis_;
  std::shared_ptr<const Reference> reference_;
  std::shared_ptr<const Partition> partition_;
  std::shared_ptr<const TwoBodyChannels> two_body_;
  std::shared_ptr<const TwoBodyKetSubsets> two_body_subsets_;
  std::shared_ptr<const ParticleHoleChannels> particle_hole_;
  std::shared_ptr<const ThreeBodyChannels> three_body_;     // may be null
  std::shared_ptr<const ThreeBodySpace> three_body_space_;  // may be null
  std::shared_ptr<const ThreeBodyDistribution> three_body_distribution_;
  std::shared_ptr<const RowSet> three_body_rows_;  // never null
};

}  // namespace imsrg::basis

#endif  // IMSRG_BASIS_MODEL_SPACE_H_
