// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/model_space.h"

#include <memory>
#include <string>
#include <utility>

#include "imsrg/base/error.h"
#include "imsrg/basis/nucleus.h"
#include "imsrg/basis/valence_space.h"

namespace imsrg::basis {

namespace {

void CheckSizes(const SingleParticleBasis& basis, const Reference& reference,
                const Partition& partition) {
  if (reference.num_orbits() != basis.num_orbits() ||
      partition.num_orbits() != basis.num_orbits()) {
    throw InputError("ModelSpace: the reference (" +
                     std::to_string(reference.num_orbits()) +
                     " orbits) and the partition (" +
                     std::to_string(partition.num_orbits()) +
                     " orbits) must match the basis (" +
                     std::to_string(basis.num_orbits()) + " orbits)");
  }
}

}  // namespace

std::shared_ptr<const ModelSpace> ModelSpace::Create(SingleParticleBasis basis,
                                                     Reference reference,
                                                     Partition partition) {
  CheckSizes(basis, reference, partition);
  auto basis_ptr =
      std::make_shared<const SingleParticleBasis>(std::move(basis));
  auto two_body = std::make_shared<const TwoBodyChannels>(*basis_ptr);
  std::shared_ptr<ModelSpace> ms(new ModelSpace());
  ms->basis_ = std::move(basis_ptr);
  ms->reference_ = std::make_shared<const Reference>(std::move(reference));
  ms->partition_ = std::make_shared<const Partition>(std::move(partition));
  ms->two_body_ = std::move(two_body);
  ms->two_body_subsets_ = std::make_shared<const TwoBodyKetSubsets>(
      *ms->two_body_, *ms->reference_, *ms->partition_);
  ms->particle_hole_ =
      std::make_shared<const ParticleHoleChannels>(*ms->two_body_);
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::FromNames(
    const BasisTruncation& truncation, double hw_MeV,
    std::string_view reference, std::string_view valence_space) {
  SingleParticleBasis basis =
      SingleParticleBasis::HarmonicOscillator(truncation, hw_MeV);
  const Nucleus nucleus = reference == "default"
                              ? DefaultReferenceNucleus(valence_space)
                              : Nucleus::Parse(reference);
  Reference ref = Reference::FromNucleus(basis, nucleus);
  Partition partition = Partition::FromDefinition(
      basis, DefineValenceSpace(valence_space, basis, nucleus));
  return Create(std::move(basis), std::move(ref), std::move(partition));
}

std::shared_ptr<const ModelSpace> ModelSpace::WithReference(
    Reference reference) const {
  return WithReferenceAndPartition(std::move(reference), *partition_);
}

std::shared_ptr<const ModelSpace> ModelSpace::WithReferenceAndPartition(
    Reference reference, Partition partition) const {
  CheckSizes(*basis_, reference, partition);
  std::shared_ptr<ModelSpace> ms(new ModelSpace());
  ms->basis_ = basis_;
  ms->two_body_ = two_body_;
  ms->particle_hole_ = particle_hole_;
  ms->three_body_ = three_body_;
  ms->reference_ = std::make_shared<const Reference>(std::move(reference));
  ms->partition_ = std::make_shared<const Partition>(std::move(partition));
  ms->two_body_subsets_ = std::make_shared<const TwoBodyKetSubsets>(
      *ms->two_body_, *ms->reference_, *ms->partition_);
  if (three_body_ != nullptr) {
    ms->three_body_space_ = std::make_shared<const ThreeBodySpace>(
        *three_body_, *basis_, *ms->reference_, three_body_->cuts());
  }
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::WithThreeBody(
    const ThreeBodyCuts& cuts) const {
  std::shared_ptr<ModelSpace> ms(new ModelSpace(*this));
  ms->three_body_ = std::make_shared<const ThreeBodyChannels>(*basis_, cuts);
  ms->three_body_space_ = std::make_shared<const ThreeBodySpace>(
      *ms->three_body_, *basis_, *reference_, ms->three_body_->cuts());
  return ms;
}

bool operator==(const ModelSpace& a, const ModelSpace& b) noexcept {
  if (&a == &b) return true;
  const bool same_3b = (a.three_body_ == nullptr && b.three_body_ == nullptr) ||
                       (a.three_body_ != nullptr && b.three_body_ != nullptr &&
                        a.three_body_->cuts() == b.three_body_->cuts());
  return same_3b && *a.basis_ == *b.basis_ && *a.reference_ == *b.reference_ &&
         *a.partition_ == *b.partition_;
}

}  // namespace imsrg::basis
