// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Pipeline stage 1: model space, bare Hamiltonian, Hartree-Fock, normal
// ordering, operators.
#include "imsrg/driver/pipeline/hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/io/me2j.h"
#include "imsrg/io/no2b.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/basis_transform.h"
#include "imsrg/operator/no2b.h"
#include "imsrg/operator/normal_ordering.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/kinetic.h"
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
  if (c.basis.kind == BasisKind::kNat) Planned("basis.kind = \"nat\"");
  if (c.hamiltonian.trap_hw) Planned("hamiltonian.trap_hw");
  if (c.interaction.two_body.format != TwoBodyFormat::kMe2j) {
    Planned("interaction.two_body.format other than me2j");
  }
  if (c.interaction.three_body &&
      c.interaction.three_body->type != ThreeBodyType::kNo2b) {
    Planned("interaction.three_body.type other than no2b");
  }
}

}  // namespace

PreparedHamiltonian PrepareHamiltonian(const RunConfig& config) {
  RejectPlanned(config);
  const BasisConfig& b = config.basis;
  const auto& tb = config.interaction.two_body;
  if (tb.file.empty()) throw ConfigError("interaction.two_body.file is empty");

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
  auto ms = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = b.emax, .lmax = lmax}, b.hw, reference,
      valence);

  op::Operator H = io::ReadMe2j(
      tb.file.front(),
      io::Me2jReadOptions{
          .file = {.e1max = tb.e1max, .e2max = tb.e2max, .lmax = tb.lmax}},
      ms);

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
  // Lawson term: beta_cm * HCM(hw = beta_cm_hw); its constant is removed after
  // normal ordering.
  const double beta = config.hamiltonian.beta_cm;
  const bool lawson = std::abs(beta) > 1.0e-6;
  if (lawson) {
    H += beta * opslib::HCm(ms, b.mass_number, config.hamiltonian.beta_cm_hw);
  }

  std::optional<op::No2bThreeBody> v3;
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
  const op::No2bThreeBody* v3p = v3 ? &*v3 : nullptr;

  PreparedHamiltonian out;
  out.ms_bare = ms;
  out.mass_number = b.mass_number;
  out.proton_number = ms->reference().target_Z();

  if (b.kind == BasisKind::kHf) {
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
  } else {
    out.hno = v3p ? op::NormalOrder(H, *v3p) : op::NormalOrder(H);
    out.e_reference = out.hno.zero_body();
    log::Info("Oscillator basis: E_ref = {:.8f} MeV", out.e_reference);
  }
  if (lawson) out.hno -= 1.5 * beta * config.hamiltonian.beta_cm_hw;
  return out;
}

op::Operator PrepareOperator(const OperatorRequest& request,
                             const PreparedHamiltonian& ph,
                             const RunConfig& /*config*/) {
  const auto& ms = ph.ms_bare;
  const int A = ph.mass_number;
  const int Z = ph.proton_number;
  op::Operator o = [&]() -> op::Operator {
    switch (request.kind) {
      case OperatorKind::kRp2:
        return opslib::Rp2(ms, A, Z);
      case OperatorKind::kRn2:
        return opslib::Rn2(ms, A, Z);
      case OperatorKind::kRm2:
        return opslib::Rm2(ms, A);
      case OperatorKind::kTcm:
        return opslib::TCm(ms, A);
      case OperatorKind::kTrel:
        return opslib::TRel(ms, A);
      case OperatorKind::kTlab:
        return opslib::KineticEnergy(ms);
      case OperatorKind::kHcm: {
        std::optional<double> hw;
        if (!request.params.empty()) {
          if (const auto* d = std::get_if<double>(&request.params.front())) {
            hw = *d;
          } else if (const auto* i =
                         std::get_if<std::int64_t>(&request.params.front())) {
            hw = static_cast<double>(*i);
          }
        }
        return opslib::HCm(ms, A, hw);
      }
      default:
        Planned("operator '" + request.canonical_name + "'");
    }
  }();
  if (ph.hf) o = hf::ToHfBasis(o, *ph.hf);
  const auto hno_ms = ph.hno.model_space_ptr();
  if (hno_ms != o.model_space_ptr()) o = op::WithModelSpace(o, hno_ms);
  return op::NormalOrder(o);
}

}  // namespace imsrg::driver::pipeline
