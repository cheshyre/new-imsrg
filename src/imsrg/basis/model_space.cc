// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/model_space.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/error.h"
#include "imsrg/basis/nucleus.h"
#include "imsrg/basis/valence_space.h"

namespace imsrg::basis {

namespace {

bool SameRows(const RowSet& a, const RowSet& b) {
  const auto ra = a.ranges();
  const auto rb = b.ranges();
  if (ra.size() != rb.size()) return false;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    if (ra[i].channel != rb[i].channel ||
        ra[i].triple_begin != rb[i].triple_begin ||
        ra[i].triple_end != rb[i].triple_end ||
        ra[i].row_begin != rb[i].row_begin || ra[i].row_end != rb[i].row_end) {
      return false;
    }
  }
  return true;
}

bool SameDistribution(const ThreeBodyDistribution* a,
                      const ThreeBodyDistribution* b) {
  if (a == nullptr || b == nullptr) return a == b;
  if (a->comm.rank() != b->comm.rank() || a->comm.size() != b->comm.size()) {
    return false;
  }
  if (a->partition == b->partition) return true;
  if (a->partition->num_parts() != b->partition->num_parts()) return false;
  for (std::int32_t p = 0; p < a->partition->num_parts(); ++p) {
    if (!SameRows(a->partition->part(p), b->partition->part(p))) return false;
  }
  return true;
}

/// The shared empty row set of a model space without three-body space.
std::shared_ptr<const RowSet> EmptyRows() {
  static const auto* const kEmpty =
      new std::shared_ptr<const RowSet>(std::make_shared<const RowSet>());
  return *kEmpty;
}

/// Whole rows of `space` (the empty set if null).
std::shared_ptr<const RowSet> WholeRows(const ThreeBodySpace* space) {
  return space == nullptr
             ? EmptyRows()
             : std::make_shared<const RowSet>(RowSet::Whole(*space));
}

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

ModelSpace::ModelSpace() : three_body_rows_(EmptyRows()) {}

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
    ms->three_body_rows_ = WholeRows(ms->three_body_space_.get());
  }
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::WithThreeBody(
    const ThreeBodyCuts& cuts) const {
  std::shared_ptr<ModelSpace> ms(new ModelSpace(*this));
  ms->three_body_ = std::make_shared<const ThreeBodyChannels>(*basis_, cuts);
  ms->three_body_space_ = std::make_shared<const ThreeBodySpace>(
      *ms->three_body_, *basis_, *reference_, ms->three_body_->cuts());
  ms->three_body_distribution_.reset();
  ms->three_body_rows_ = WholeRows(ms->three_body_space_.get());
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::WithThreeBodyDistribution(
    ThreeBodyDistribution d) const {
  if (three_body_space_ == nullptr) {
    throw InputError(
        "ModelSpace::WithThreeBodyDistribution: the model space has no "
        "three-body space");
  }
  if (d.partition == nullptr) {
    throw InputError(
        "ModelSpace::WithThreeBodyDistribution: null row partition");
  }
  if (d.partition->num_parts() != d.comm.size()) {
    throw InputError(
        "ModelSpace::WithThreeBodyDistribution: the partition has " +
        std::to_string(d.partition->num_parts()) +
        " parts but the communicator has " + std::to_string(d.comm.size()) +
        " ranks");
  }
  // The partition must cover THIS space's kept rows exactly once: all parts'
  // ranges together are valid (FromRanges: channels, triple alignment, no
  // overlap) and their row counts sum to the space's row count. The check is
  // local and identical on every rank (same partition), so all ranks throw
  // together.
  {
    std::vector<RowRange> all;
    for (std::int32_t p = 0; p < d.partition->num_parts(); ++p) {
      const auto r = d.partition->part(p).ranges();
      all.insert(all.end(), r.begin(), r.end());
    }
    const RowSet covered =
        RowSet::FromRanges(*three_body_space_, std::move(all));
    if (covered.NumRows() != RowSet::Whole(*three_body_space_).NumRows()) {
      throw InputError(
          "ModelSpace::WithThreeBodyDistribution: the partition does not "
          "cover the model space's kept three-body rows (" +
          std::to_string(covered.NumRows()) + " rows in the parts, " +
          std::to_string(RowSet::Whole(*three_body_space_).NumRows()) +
          " in the space)");
    }
  }
  std::shared_ptr<ModelSpace> ms(new ModelSpace(*this));
  ms->three_body_rows_ =
      std::make_shared<const RowSet>(d.partition->part(d.comm.rank()));
  ms->three_body_distribution_ =
      std::make_shared<const ThreeBodyDistribution>(std::move(d));
  return ms;
}

std::shared_ptr<const ModelSpace> ModelSpace::WithoutThreeBodyDistribution()
    const {
  std::shared_ptr<ModelSpace> ms(new ModelSpace(*this));
  ms->three_body_distribution_.reset();
  ms->three_body_rows_ = WholeRows(three_body_space_.get());
  return ms;
}

bool operator==(const ModelSpace& a, const ModelSpace& b) noexcept {
  if (&a == &b) return true;
  const bool same_3b = (a.three_body_ == nullptr && b.three_body_ == nullptr) ||
                       (a.three_body_ != nullptr && b.three_body_ != nullptr &&
                        a.three_body_->cuts() == b.three_body_->cuts());
  return same_3b &&
         SameDistribution(a.three_body_distribution_.get(),
                          b.three_body_distribution_.get()) &&
         *a.basis_ == *b.basis_ && *a.reference_ == *b.reference_ &&
         *a.partition_ == *b.partition_;
}

}  // namespace imsrg::basis
