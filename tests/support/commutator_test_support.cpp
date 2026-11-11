#include "commutator_test_support.hpp"

#include <algorithm>
#include <random>
#include <vector>

#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/operator/operator_spec.hpp"

#ifndef IMSRG_GOLDEN_DIR
#define IMSRG_GOLDEN_DIR ""
#endif

namespace imsrg::testing {

std::shared_ptr<const basis::ModelSpace> model_space(int emax, std::string_view reference) {
  return basis::ModelSpace::from_names(basis::BasisTruncation{emax}, 16.0, reference, reference);
}

std::shared_ptr<const basis::ModelSpace> fractional_model_space(int emax,
                                                                std::string_view reference) {
  const auto ms = model_space(emax, reference);
  const basis::Reference& ref = ms->reference();
  std::vector<double> occ(ref.occupations().begin(), ref.occupations().end());
  for (const int tz2 : {-1, +1}) {
    const auto species = [&](OrbitIndex a) { return ms->basis().orbit(a).tz2.twice == tz2; };
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
      occ[static_cast<std::size_t>(first_particle.value())] = (tz2 < 0) ? 0.25 : 0.15;
  }
  return ms->with_reference(
      basis::Reference::from_occupations(ms->basis(), std::move(occ), "fractional"));
}

op::Operator random_operator(std::shared_ptr<const basis::ModelSpace> ms, Hermiticity h,
                             std::uint64_t seed, int particle_rank) {
  op::Operator op(ms, op::OperatorSpec::scalar(particle_rank, h));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  const int sgn = op::hermiticity_sign(h);
  const auto& basis = ms->basis();
  const std::int32_t norb = ms->num_orbits();

  op.sync_to_host();
  op::OneBody& ob = op.one_body();
  for (std::int32_t a = 0; a < norb; ++a) {
    for (std::int32_t b = (h == Hermiticity::NonHermitian ? 0 : a); b < norb; ++b) {
      const basis::Orbit& oa = basis.orbit(OrbitIndex{a});
      const basis::Orbit& obb = basis.orbit(OrbitIndex{b});
      if (oa.l != obb.l || oa.j2 != obb.j2 || oa.tz2 != obb.tz2) continue;
      if (a == b && h == Hermiticity::AntiHermitian) continue;
      const double v = uni(rng);
      ob.set_element(OrbitIndex{a}, OrbitIndex{b}, v);
      if (a != b && h != Hermiticity::NonHermitian)
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
        for (std::int32_t j = (h == Hermiticity::NonHermitian ? 0 : i); j < n; ++j) {
          if (i == j && h == Hermiticity::AntiHermitian) continue;
          const double v = uni(rng);
          tb.set(ch, ch, LocalKetIndex{i}, LocalKetIndex{j}, v);
          if (i != j && h != Hermiticity::NonHermitian)
            tb.set(ch, ch, LocalKetIndex{j}, LocalKetIndex{i}, sgn * v);
        }
      }
    }
  }
  op.sync_to_device();
  return op;
}

double relative_deviation(const op::Operator& z, const op::Operator& z_ref) {
  const double diff = op::max_abs_difference(z, z_ref).max();
  return diff / std::max(1.0, op::max_abs(z_ref).max());
}

std::filesystem::path golden_path(std::string_view relative) {
  return std::filesystem::path(IMSRG_GOLDEN_DIR) / std::filesystem::path(relative);
}

op::Operator read_golden(std::string_view relative, std::shared_ptr<const basis::ModelSpace> ms) {
  return op::read_operator_dump(golden_path(relative), std::move(ms));
}

std::shared_ptr<const basis::ModelSpace> golden_model_space(std::string_view relative) {
  return op::model_space_for_dump(op::read_operator_dump_header(golden_path(relative)));
}

}  // namespace imsrg::testing
