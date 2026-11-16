// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/support/commutator_test_support.h"

#include <algorithm>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "imsrg/basis/reference.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/operator/operator_spec.h"

#ifndef IMSRG_GOLDEN_DIR
#define IMSRG_GOLDEN_DIR ""
#endif

namespace imsrg::testing {

std::shared_ptr<const basis::ModelSpace> MakeModelSpace(
    int emax, std::string_view reference) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{emax}, 16.0,
                                      reference, reference);
}

std::shared_ptr<const basis::ModelSpace> FractionalModelSpace(
    int emax, std::string_view reference) {
  const auto ms = MakeModelSpace(emax, reference);
  const basis::Reference& ref = ms->reference();
  std::vector<double> occ(ref.occupations().begin(), ref.occupations().end());
  for (const int tz2 : {-1, +1}) {
    const auto species = [&](OrbitIndex a) {
      return ms->basis().orbit(a).tz2.twice == tz2;
    };
    OrbitIndex last_hole{-1};
    for (const OrbitIndex h : ref.holes())
      if (species(h)) last_hole = h;
    OrbitIndex first_particle{-1};
    for (const OrbitIndex p : ref.particles())
      if (species(p)) {
        first_particle = p;
        break;
      }
    if (last_hole.value() >= 0)
      occ[static_cast<std::size_t>(last_hole.value())] = (tz2 < 0) ? 0.6 : 0.7;
    if (first_particle.value() >= 0)
      occ[static_cast<std::size_t>(first_particle.value())] =
          (tz2 < 0) ? 0.25 : 0.15;
  }
  return ms->WithReference(basis::Reference::FromOccupations(
      ms->basis(), std::move(occ), "fractional"));
}

op::Operator RandomOperator(std::shared_ptr<const basis::ModelSpace> ms,
                            Hermiticity h, std::uint64_t seed,
                            int particle_rank) {
  op::Operator op(ms, op::OperatorSpec::Scalar(particle_rank, h));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  const int sgn = op::HermiticitySign(h);
  const auto& basis = ms->basis();
  const std::int32_t norb = ms->num_orbits();

  op.SyncToHost();
  op::OneBody& ob = op.one_body();
  for (std::int32_t a = 0; a < norb; ++a) {
    for (std::int32_t b = (h == Hermiticity::kNonHermitian ? 0 : a); b < norb;
         ++b) {
      const basis::Orbit& oa = basis.orbit(OrbitIndex{a});
      const basis::Orbit& obb = basis.orbit(OrbitIndex{b});
      if (oa.l != obb.l || oa.j2 != obb.j2 || oa.tz2 != obb.tz2) continue;
      if (a == b && h == Hermiticity::kAntiHermitian) continue;
      const double v = uni(rng);
      ob.set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      if (a != b && h != Hermiticity::kNonHermitian)
        ob.set_element(OrbitIndex{b}, OrbitIndex{a}, sgn * v);
    }
  }
  if (particle_rank >= 2) {
    op::TwoBody& tb = op.two_body();
    const auto& channels = tb.channels();
    for (std::int32_t c = 0; c < channels.num_channels(); ++c) {
      const ChannelIndex ch{c};
      const std::int32_t n = channels.channel(ch).num_kets;
      for (std::int32_t i = 0; i < n; ++i) {
        for (std::int32_t j = (h == Hermiticity::kNonHermitian ? 0 : i); j < n;
             ++j) {
          if (i == j && h == Hermiticity::kAntiHermitian) continue;
          const double v = uni(rng);
          tb.set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
          if (i != j && h != Hermiticity::kNonHermitian)
            tb.set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i}, sgn * v);
        }
      }
    }
  }
  op.SyncToDevice();
  return op;
}

double RelativeDeviation(const op::Operator& z, const op::Operator& z_ref) {
  const double diff = op::MaxAbsDifference(z, z_ref).max();
  return diff / std::max(1.0, op::MaxAbs(z_ref).max());
}

std::filesystem::path GoldenPath(std::string_view relative) {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) /
         std::filesystem::path(relative);
}

op::Operator ReadGolden(std::string_view relative,
                        std::shared_ptr<const basis::ModelSpace> ms) {
  return op::ReadOperatorDump(GoldenPath(relative), std::move(ms));
}

std::shared_ptr<const basis::ModelSpace> GoldenModelSpace(
    std::string_view relative) {
  return op::ModelSpaceForDump(
      op::ReadOperatorDumpHeader(GoldenPath(relative)));
}

}  // namespace imsrg::testing
