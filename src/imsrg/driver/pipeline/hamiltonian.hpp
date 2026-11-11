#pragma once
/// @file
/// Pipeline stage 1 of `driver::run` (design/13 section 9, design/reference/03 section 1.2 steps
/// 5-11): model space, bare Hamiltonian, Hartree-Fock and normal ordering; and the construction of
/// the requested operators in the same basis.
#include <memory>
#include <optional>
#include <string>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/driver/config/schema.hpp"
#include "imsrg/hf/hartree_fock.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::driver::pipeline {

/// The normal-ordered Hamiltonian that enters the IMSRG flow, plus what later stages need.
struct PreparedHamiltonian {
  /// Model space of the bare Hamiltonian (oscillator basis; reference occupations of the config).
  std::shared_ptr<const basis::ModelSpace> ms_bare;
  /// Normal-ordered Hamiltonian in the HF basis (or the oscillator basis for basis.kind =
  /// oscillator), on the model space whose occupations are those of the HF solution (equal to
  /// `ms_bare` unless HF re-filled orbits). The Lawson constant -1.5 beta_cm beta_cm_hw is already
  /// subtracted from its zero-body part (imsrg++.cc:733).
  op::Operator hno;
  /// The HF solution (absent for basis.kind = oscillator).
  std::optional<hf::HFResult> hf;
  /// E_HF (or the zero-body part of the normal-ordered oscillator-basis H), MeV.
  double e_reference = 0.0;
  /// Mass number used for the kinetic energy (config basis.mass_number).
  int mass_number = 0;
  /// Proton number of the reference (for Rp2 and friends).
  int proton_number = 0;
};

/// Builds the model space, reads the interaction(s), adds the kinetic energy (intrinsic Trel with
/// the config's A, or lab-frame T) and the Lawson term (|beta_cm| > 1e-6: beta_cm * HCM(hw =
/// beta_cm_hw)), solves Hartree-Fock (linear mixing, occupations unfrozen unless
/// hartree_fock.freeze_occupations) and returns the normal-ordered Hamiltonian (NO2B for a 3N
/// input). Logs E_HF and its parts.
/// @throws InputError for unreadable input files; ConfigError ("E-PLANNED ...") for options that
///         are not implemented yet (NAT basis, formats other than me2j / no2b, ...).
PreparedHamiltonian prepare_hamiltonian(const RunConfig& config);

/// The operator `name` (an `OperatorRequest::canonical_name`, e.g. "Rp2") built on the bare model
/// space, transformed to the HF basis if HF was solved, and normal ordered with respect to the
/// reference of `ph.hno`'s model space, so it can be transformed with the flow's Omegas
/// (imsrg++.cc:1386-1440).
/// @throws ConfigError ("E-PLANNED ...") for operators that are not implemented yet.
op::Operator prepare_operator(const OperatorRequest& request, const PreparedHamiltonian& ph,
                              const RunConfig& config);

}  // namespace imsrg::driver::pipeline
