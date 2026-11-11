#pragma once

/// @file
/// Test helpers shared by the commutator and commutator_ref tests: model spaces with closed-shell
/// and fractional (open-shell) references, random operators with an explicit seed (design/30
/// section 1), golden-file access and the term tolerance of design/30 section 2.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::testing {

/// Single-reference model space: HO basis e <= emax, hw = 16 MeV, the naive filling of nucleus
/// `reference` (fractional occupations for a partially filled orbit, e.g. "O18" or "He6"), no
/// valence space.
std::shared_ptr<const basis::ModelSpace> model_space(int emax, std::string_view reference);

/// A model space at `emax` whose reference has explicitly fractional occupations in several
/// orbits on both sides of the Fermi surface (an ensemble reference): starting from the naive
/// filling of `reference`, the last hole orbit of each species gets 0.6 (protons) / 0.7
/// (neutrons) and the first particle orbit of each species gets 0.25 / 0.15.
std::shared_ptr<const basis::ModelSpace> fractional_model_space(int emax,
                                                                std::string_view reference);

/// A random scalar operator on `ms` with the given hermiticity and particle_rank (1 or 2):
/// zero-body part 0; one-body entries uniform in [-1, 1] between orbits of equal (l, 2j, 2tz),
/// two-body blocks uniform in [-1, 1] over all kets of each channel, with O_ba = h O_ab imposed
/// (zero diagonal for anti-Hermitian; independent entries for NonHermitian). Deterministic for
/// a given seed (std::mt19937_64). Synced to the device.
op::Operator random_operator(std::shared_ptr<const basis::ModelSpace> ms, Hermiticity h,
                             std::uint64_t seed, int particle_rank = 2);

/// max part difference / max(1, max |reference part|): the design/30 section 2 term criterion is
/// `relative_deviation(z, z_ref) <= 1e-10`. Both operators on the same model space with the same
/// symmetry (InvariantError otherwise).
double relative_deviation(const op::Operator& z, const op::Operator& z_ref);

/// Path of a golden file: <IMSRG_GOLDEN_DIR>/<relative> (IMSRG_GOLDEN_DIR is a compile
/// definition of the test target).
std::filesystem::path golden_path(std::string_view relative);

/// Reads a golden OperatorDump into an operator on `ms` (occupations checked).
op::Operator read_golden(std::string_view relative, std::shared_ptr<const basis::ModelSpace> ms);

/// The model space described by the header of a golden dump (`op::model_space_for_dump`).
std::shared_ptr<const basis::ModelSpace> golden_model_space(std::string_view relative);

}  // namespace imsrg::testing
