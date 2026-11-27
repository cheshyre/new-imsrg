// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Pipeline stage 1: model space, bare Hamiltonian, Hartree-Fock, normal
// ordering, operators.
#include "imsrg/driver/pipeline/hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/driver/config/compat.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/operator_frame.h"
#include "imsrg/io/me2j.h"
#include "imsrg/io/no2b.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/mbpt/natural_orbitals.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/no2b.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/operator/truncate.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/kinetic.h"
#include "imsrg/opslib/nreft.h"
#include "imsrg/opslib/radius.h"

namespace imsrg::driver::pipeline {
namespace {

[[noreturn]] void Planned(const std::string& what) {
  throw ConfigError("E-PLANNED: " + what + " is not implemented yet");
}

/// Throws ConfigError("E-PLANNED: ...") for every config option stage 1 does
/// not support yet.
void RejectPlanned(const RunConfig& c) {
  if (c.reference.occupations_file) Planned("reference.occupations_file");
  if (c.valence.custom) Planned("valence.custom");
  if (c.basis.emax_unocc) Planned("basis.emax_unocc");
  if (c.natural_orbitals.use_occupations) {
    Planned("natural_orbitals.use_occupations");
  }
  if (c.natural_orbitals.order != NatOrder::kOccupation) {
    Planned("natural_orbitals.order other than \"occupation\"");
  }
  if (c.hamiltonian.trap_hw) Planned("hamiltonian.trap_hw");
  if (c.interaction.two_body.format != TwoBodyFormat::kMe2j) {
    Planned("interaction.two_body.format other than me2j");
  }
  if (c.interaction.three_body &&
      c.interaction.three_body->type != ThreeBodyType::kNo2b) {
    Planned("interaction.three_body.type other than no2b");
  }
}

/// basis.kind = "nat" (imsrg++ `HFMBPT::GetNaturalOrbitals` +
/// `GetNormalOrderedH`): natural orbitals of the MBPT(2) density matrix of the
/// HF-basis Hamiltonian in `out`, the mean-field state of the reference
/// occupations in that basis (`hf::SolutionInBasis`) and the Hamiltonian
/// normal ordered in it. Replaces `out.hno` and `out.e_reference` and fills
/// `out.nat` (its C maps the oscillator basis to the NAT basis).
/// With `natural_orbitals.use_hf_reference` the occupied orbitals are reset
/// to the HF orbitals (`mbpt::UseHfForHoleStates`), and the compat option
/// `natural_orbitals.sort` orders the NAT orbitals by |occupation| (default)
/// or by signed occupation.
void ToNaturalOrbitalBasis(const op::Operator& H, const op::No2bThreeBody* v3,
                           const hf::HFOptions& ho, const RunConfig& config,
                           PreparedHamiltonian& out) {
  IMSRG_CHECK(out.hf.has_value(), "the NAT basis is built on an HF solution");
  op::Operator hno = out.hno;  // Mbpt2DensityMatrix syncs its argument
  const linalg::Matrix<double> rho_hf = mbpt::Mbpt2DensityMatrix(hno);
  mbpt::NaturalOrbitals nat = mbpt::ComputeNaturalOrbitals(
      rho_hf, out.hno.model_space(),
      config.natural_orbitals.sort == NatSort::kSignedOccupation
          ? mbpt::NatSorting::kSignedDescending
          : mbpt::NatSorting::kAbsDescending);
  {
    const auto& basis = out.hno.model_space().basis();
    for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
      const basis::Orbit& o = basis.orbit(OrbitIndex(a));
      log::Debug(
          "NAT orbit {:3d}: n {} l {} 2j {} 2tz {:2d} occupation {:.16e}", a,
          o.n, o.l, o.j2.twice, o.tz2.twice,
          nat.occupations[static_cast<std::size_t>(a)]);
    }
  }
  if (config.natural_orbitals.use_hf_reference) {
    log::Info("NAT basis: setting occupied states back to HF");
    mbpt::UseHfForHoleStates(nat, out.hno.model_space());
  }
  const linalg::Matrix<double> c_ho2nat = mbpt::Matmul(out.hf->C, nat.c_hf2nat);
  hf::HFResult r =
      hf::SolutionInBasis(H, v3, c_ho2nat, out.hf->occupations, ho);
  out.hno = hf::NormalOrderedHamiltonian(H, v3, r, ho);
  out.e_reference = r.e_hf;
  log::Info("NAT basis: E_HF = {:.8f} MeV (HF basis {:.8f} MeV)", r.e_hf,
            out.hf->e_hf);
  out.nat = std::move(r);
  out.nat_occupations = nat.occupations;
}

}  // namespace

std::shared_ptr<const basis::ModelSpace> MakeImsrgModelSpace(
    const RunConfig& config, const basis::ModelSpace& from) {
  const ImsrgSpaceConfig& sp = *config.imsrg.space;
  if (sp.e2max != 2 * sp.emax) {
    Planned("imsrg.space.e2max < 2*imsrg.space.emax");
  }
  RunConfig c = config;
  c.basis.emax = sp.emax;
  const auto naive = MakeModelSpace(c);
  const auto& basis = naive->basis();
  std::vector<double> occ(static_cast<std::size_t>(basis.num_orbits()));
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
    occ[static_cast<std::size_t>(a)] = from.reference().occupation(
        from.basis().index_of(basis.orbit(OrbitIndex(a))));
  }
  // Holes outside the IMSRG space would be lost silently.
  double n_from = 0.0, n_to = 0.0;
  for (std::int32_t a = 0; a < from.num_orbits(); ++a) {
    n_from += (from.basis().orbit(OrbitIndex(a)).j2.twice + 1) *
              from.reference().occupation(OrbitIndex(a));
  }
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
    n_to += (basis.orbit(OrbitIndex(a)).j2.twice + 1) *
            occ[static_cast<std::size_t>(a)];
  }
  if (std::abs(n_from - n_to) > 1.0e-8) {
    throw ConfigError("imsrg.space.emax = " + std::to_string(sp.emax) +
                      " drops occupied orbits of the reference");
  }
  return naive->WithReference(basis::Reference::FromOccupations(
      basis, std::move(occ), from.reference().name()));
}

std::shared_ptr<const basis::ModelSpace> MakeModelSpace(
    const RunConfig& config) {
  const BasisConfig& b = config.basis;
  const std::int32_t lmax = b.lmax.value_or(-1);
  const std::string reference = config.reference.nucleus.value_or("default");
  std::string valence;
  if (config.valence.space) {
    valence = *config.valence.space;
  } else if (config.reference.nucleus) {
    valence = *config.reference.nucleus;
  } else {
    throw ConfigError(
        "either reference.nucleus or valence.space must be given");
  }
  return basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = b.emax, .lmax = lmax}, b.hw, reference,
      valence);
}

namespace {

/// Adds the kinetic energy of the configuration to `H`.
void AddKineticEnergy(op::Operator& H, const RunConfig& config,
                      const std::shared_ptr<const basis::ModelSpace>& ms) {
  const BasisConfig& b = config.basis;
  switch (config.hamiltonian.kinetic) {
    case KineticEnergy::kIntrinsic:
      H += opslib::TRel(ms, b.mass_number);
      break;
    case KineticEnergy::kLab:
      H += opslib::KineticEnergy(ms);
      break;
    case KineticEnergy::kNone:
      break;
  }
}

/// Reads the normal-ordered two-body part of the three-body interaction
/// (`v3` is engaged iff `config.interaction.three_body` is set).
void ReadThreeBody(const RunConfig& config,
                   std::optional<op::No2bThreeBody>& v3) {
  const BasisConfig& b = config.basis;
  if (const auto& t = config.interaction.three_body) {
    const std::int32_t l3 =
        std::min(b.lmax3.value_or(b.lmax.value_or(b.emax)), b.emax);
    v3.emplace(op::No2bTruncation{
        .emax = b.emax, .e2max = 2 * b.emax, .e3max = b.e3max, .lmax = l3});
    io::No2bReadOptions ro;
    ro.file = {.e1max = t->e1max,
               .e2max = t->e2max.value_or(2 * t->e1max),
               .e3max = t->e3max,
               .lmax = -1};
    ro.precision = t->no2b_precision == No2bPrecision::kHalf
                       ? io::No2bPrecision::kHalf
                       : io::No2bPrecision::kSingle;
    io::ReadNo2b(t->file, ro, *v3);
  }
}

/// Fills `out.hno`, `out.e_reference` (and `out.hf`, `out.nat`) from the bare
/// Hamiltonian `H`: HF or NAT solution, or the plain oscillator basis.
void SolveReferenceState(const op::Operator& H, const op::No2bThreeBody* v3p,
                         const RunConfig& config, PreparedHamiltonian& out) {
  const BasisConfig& b = config.basis;
  if (b.kind == BasisKind::kHf || b.kind == BasisKind::kNat) {
    hf::HFOptions ho;
    ho.mixing = config.hartree_fock.mixing == HartreeFockMixing::kDiis
                    ? hf::Mixing::kDiis
                    : hf::Mixing::kLinear;
    ho.freeze_occupations = config.hartree_fock.freeze_occupations;
    ho.e3max = b.e3max;
    hf::HFResult r = hf::Solve(H, v3p, ho);
    log::Info(
        "HF: E = {:.8f} MeV (e1 = {:.7f}, e2 = {:.7f}, e3 = {:.7f}), {} "
        "iterations{}",
        r.e_hf, r.e1, r.e2, r.e3, r.iterations,
        r.converged ? "" : " (NOT converged)");
    out.hno = hf::NormalOrderedHamiltonian(H, v3p, r, ho);
    out.e_reference = r.e_hf;
    out.hf = std::move(r);
    if (b.kind == BasisKind::kNat) {
      ToNaturalOrbitalBasis(H, v3p, ho, config, out);
    }
  } else {
    out.hno = v3p ? op::NormalOrder(H, *v3p) : op::NormalOrder(H);
    out.e_reference = out.hno.zero_body();
    log::Info("Oscillator basis: E_ref = {:.8f} MeV", out.e_reference);
  }
}

}  // namespace

PreparedHamiltonian PrepareHamiltonian(const RunConfig& config) {
  RejectPlanned(config);
  const BasisConfig& b = config.basis;
  const auto& tb = config.interaction.two_body;
  if (tb.file.empty()) throw ConfigError("interaction.two_body.file is empty");

  auto ms = MakeModelSpace(config);

  op::Operator H = io::ReadMe2j(
      tb.file.front(),
      io::Me2jReadOptions{
          .file = {.e1max = tb.e1max, .e2max = tb.e2max, .lmax = tb.lmax}},
      ms);

  AddKineticEnergy(H, config, ms);
  // Lawson term: beta_cm * HCM(hw = beta_cm_hw); its constant is removed after
  // normal ordering.
  const double beta = config.hamiltonian.beta_cm;
  const bool lawson = std::abs(beta) > 1.0e-6;
  if (lawson) {
    H += beta * opslib::HCm(ms, b.mass_number, config.hamiltonian.beta_cm_hw);
  }

  std::optional<op::No2bThreeBody> v3;
  ReadThreeBody(config, v3);
  const op::No2bThreeBody* v3p = v3 ? &*v3 : nullptr;

  PreparedHamiltonian out;
  out.ms_bare = ms;
  out.mass_number = b.mass_number;
  out.proton_number = ms->reference().target_Z();

  SolveReferenceState(H, v3p, config, out);
  {
    // MBPT(2) correlation energy of the Hamiltonian entering the flow
    // (imsrg++.cc:739); Mp2Energy syncs its argument, so use a copy.
    op::Operator h = out.hno;
    out.e_mp2 = mbpt::Mp2Energy(h);
    log::Info("EMP2 = {:.8f} MeV", out.e_mp2);
  }
  if (lawson) out.hno -= 1.5 * beta * config.hamiltonian.beta_cm_hw;
  if (config.imsrg.space) {
    // Cut the normal-ordered pieces (no re-normal-ordering, imsrg++.cc:927).
    out.ms_normal_ordering = out.hno.model_space_ptr();
    out.hno = op::Truncate(out.hno,
                           MakeImsrgModelSpace(config, out.hno.model_space()));
    op::Operator h = out.hno;
    log::Info("IMSRG space: emax = {} ({} orbits), EMP2 = {:.8f} MeV",
              config.imsrg.space->emax, out.hno.model_space().num_orbits(),
              mbpt::Mp2Energy(h));
    // imsrg++.cc:946-957 prints MBPT(3) in the IMSRG space as well (a cheap
    // check against archived logs, e.g. the 27Al runs of design/50).
    const mbpt::Mp3Energy e3 = mbpt::ComputeMp3Energy(h);
    log::Info(
        "IMSRG space: E3_pp = {:.6f} E3_hh = {:.6f} E3_ph = {:.6f} "
        "EMP3 = {:.6f} MeV",
        e3.pp, e3.hh, e3.ph, e3.total);
  }
  return out;
}

op::Operator PrepareOperator(const OperatorRequest& request,
                             const PreparedHamiltonian& ph,
                             const RunConfig& config) {
  return PrepareOperator(request, MakeOperatorFrame(ph, config));
}

}  // namespace imsrg::driver::pipeline
