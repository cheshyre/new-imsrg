#include "imsrg/basis/model_space.hpp"

#include <memory>
#include <string>
#include <utility>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/valence_space.hpp"

namespace imsrg::basis {

namespace {

void check_sizes(const SingleParticleBasis& basis, const Reference& reference,
                 const Partition& partition) {
  if (reference.num_orbits() != basis.num_orbits() ||
      partition.num_orbits() != basis.num_orbits()) {
    throw InputError("ModelSpace: the reference (" + std::to_string(reference.num_orbits()) +
                     " orbits) and the partition (" + std::to_string(partition.num_orbits()) +
                     " orbits) must match the basis (" + std::to_string(basis.num_orbits()) +
                     " orbits)");
  }
}

}  // namespace

std::shared_ptr<const ModelSpace> ModelSpace::create(SingleParticleBasis basis, Reference reference,
                                                     Partition partition) {
  check_sizes(basis, reference, partition);
  auto basis_ptr = std::make_shared<const SingleParticleBasis>(std::move(basis));
  auto two_body = std::make_shared<const TwoBodyChannels>(*basis_ptr);
  std::shared_ptr<ModelSpace> ms(new ModelSpace());
  ms->basis_ = std::move(basis_ptr);
  ms->reference_ = std::make_shared<const Reference>(std::move(reference));
  ms->partition_ = std::make_shared<const Partition>(std::move(partition));
  ms->two_body_ = std::move(two_body);
  ms->two_body_subsets_ =
      std::make_shared<const TwoBodyKetSubsets>(*ms->two_body_, *ms->reference_, *ms->partition_);
  ms->particle_hole_ = std::make_shared<const ParticleHoleChannels>(*ms->two_body_);
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::from_names(const BasisTruncation& truncation,
                                                         double hw_MeV, std::string_view reference,
                                                         std::string_view valence_space) {
  SingleParticleBasis basis = SingleParticleBasis::harmonic_oscillator(truncation, hw_MeV);
  const Nucleus nucleus =
      reference == "default" ? default_reference_nucleus(valence_space) : Nucleus::parse(reference);
  Reference ref = Reference::from_nucleus(basis, nucleus);
  Partition partition =
      Partition::from_definition(basis, define_valence_space(valence_space, basis, nucleus));
  return create(std::move(basis), std::move(ref), std::move(partition));
}

std::shared_ptr<const ModelSpace> ModelSpace::with_reference(Reference reference) const {
  return with_reference_and_partition(std::move(reference), *partition_);
}

std::shared_ptr<const ModelSpace> ModelSpace::with_reference_and_partition(
    Reference reference, Partition partition) const {
  check_sizes(*basis_, reference, partition);
  std::shared_ptr<ModelSpace> ms(new ModelSpace());
  ms->basis_ = basis_;
  ms->two_body_ = two_body_;
  ms->particle_hole_ = particle_hole_;
  ms->three_body_ = three_body_;
  ms->reference_ = std::make_shared<const Reference>(std::move(reference));
  ms->partition_ = std::make_shared<const Partition>(std::move(partition));
  ms->two_body_subsets_ =
      std::make_shared<const TwoBodyKetSubsets>(*ms->two_body_, *ms->reference_, *ms->partition_);
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::with_three_body(const ThreeBodyCuts& cuts) const {
  std::shared_ptr<ModelSpace> ms(new ModelSpace(*this));
  ms->three_body_ = std::make_shared<const ThreeBodyChannels>(*basis_, cuts);
  return ms;
}

bool operator==(const ModelSpace& a, const ModelSpace& b) noexcept {
  if (&a == &b) return true;
  const bool same_3b = (a.three_body_ == nullptr && b.three_body_ == nullptr) ||
                       (a.three_body_ != nullptr && b.three_body_ != nullptr &&
                        a.three_body_->cuts().e3max == b.three_body_->cuts().e3max &&
                        a.three_body_->cuts().emax_3body == b.three_body_->cuts().emax_3body);
  return same_3b && *a.basis_ == *b.basis_ && *a.reference_ == *b.reference_ &&
         *a.partition_ == *b.partition_;
}

}  // namespace imsrg::basis
