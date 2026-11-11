// Pipeline stage 1: model space, bare Hamiltonian, Hartree-Fock, normal ordering, operators.
#include "imsrg/driver/pipeline/hamiltonian.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "imsrg/base/error.hpp"
#include "imsrg/base/log.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/driver/operators/operator_request.hpp"
#include "imsrg/io/me2j.hpp"
#include "imsrg/io/no2b.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/basis_transform.hpp"
#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/normal_ordering.hpp"
#include "imsrg/opslib/center_of_mass.hpp"
#include "imsrg/opslib/kinetic.hpp"
#include "imsrg/opslib/radius.hpp"

namespace imsrg::driver::pipeline {
namespace {

[[noreturn]] void planned(const std::string& what) {
  throw ConfigError("E-PLANNED: " + what + " is not implemented yet");
}

/// Throws ConfigError("E-PLANNED: ...") for every config option stage 1 does not support yet.
void reject_planned(const RunConfig& c) {
  if (c.reference.occupations_file) planned("reference.occupations_file");
  if (c.valence.custom) planned("valence.custom");
  if (c.basis.emax_unocc) planned("basis.emax_unocc");
  if (c.basis.kind == BasisKind::Nat) planned("basis.kind = \"nat\"");
  if (c.hamiltonian.trap_hw) planned("hamiltonian.trap_hw");
  if (c.interaction.two_body.format != TwoBodyFormat::Me2j) {
    planned("interaction.two_body.format other than me2j");
  }
  if (c.interaction.three_body && c.interaction.three_body->type != ThreeBodyType::No2b) {
    planned("interaction.three_body.type other than no2b");
  }
}

}  // namespace

PreparedHamiltonian prepare_hamiltonian(const RunConfig& config) {
  reject_planned(config);
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
    throw ConfigError("either reference.nucleus or valence.space must be given");
  }
  auto ms = basis::ModelSpace::from_names(basis::BasisTruncation{.emax = b.emax, .lmax = lmax},
                                          b.hw, reference, valence);

  op::Operator H = io::read_me2j(
      tb.file.front(),
      io::Me2jReadOptions{.file = {.e1max = tb.e1max, .e2max = tb.e2max, .lmax = tb.lmax}}, ms);

  switch (config.hamiltonian.kinetic) {
    case KineticEnergy::Intrinsic:
      H += opslib::t_rel(ms, b.mass_number);
      break;
    case KineticEnergy::Lab:
      H += opslib::kinetic_energy(ms);
      break;
    case KineticEnergy::None:
      break;
  }
  // Lawson term: beta_cm * HCM(hw = beta_cm_hw); its constant is removed after normal ordering.
  const double beta = config.hamiltonian.beta_cm;
  const bool lawson = std::abs(beta) > 1e-6;
  if (lawson) {
    H += beta * opslib::h_cm(ms, b.mass_number, config.hamiltonian.beta_cm_hw);
  }

  std::optional<op::No2bThreeBody> v3;
  if (const auto& t = config.interaction.three_body) {
    const std::int32_t l3 = std::min(b.lmax3.value_or(b.lmax.value_or(b.emax)), b.emax);
    v3.emplace(
        op::No2bTruncation{.emax = b.emax, .e2max = 2 * b.emax, .e3max = b.e3max, .lmax = l3});
    io::No2bReadOptions ro;
    ro.file = {
        .e1max = t->e1max, .e2max = t->e2max.value_or(2 * t->e1max), .e3max = t->e3max, .lmax = -1};
    ro.precision = t->no2b_precision == No2bPrecision::Half ? io::No2bPrecision::Half
                                                            : io::No2bPrecision::Single;
    io::read_no2b(t->file, ro, *v3);
  }
  const op::No2bThreeBody* v3p = v3 ? &*v3 : nullptr;

  PreparedHamiltonian out;
  out.ms_bare = ms;
  out.mass_number = b.mass_number;
  out.proton_number = ms->reference().target_Z();

  if (b.kind == BasisKind::Hf) {
    hf::HFOptions ho;
    ho.freeze_occupations = config.hartree_fock.freeze_occupations;
    ho.e3max = b.e3max;
    hf::HFResult r = hf::solve(H, v3p, ho);
    log::info("HF: E = {:.8f} MeV (e1 = {:.7f}, e2 = {:.7f}, e3 = {:.7f}), {} iterations{}", r.e_hf,
              r.e1, r.e2, r.e3, r.iterations, r.converged ? "" : " (NOT converged)");
    out.hno = hf::normal_ordered_hamiltonian(H, v3p, r, ho);
    out.e_reference = r.e_hf;
    out.hf = std::move(r);
  } else {
    out.hno = v3p ? op::normal_order(H, *v3p) : op::normal_order(H);
    out.e_reference = out.hno.zero_body();
    log::info("Oscillator basis: E_ref = {:.8f} MeV", out.e_reference);
  }
  if (lawson) out.hno -= 1.5 * beta * config.hamiltonian.beta_cm_hw;
  return out;
}

op::Operator prepare_operator(const OperatorRequest& request, const PreparedHamiltonian& ph,
                              const RunConfig& /*config*/) {
  const auto& ms = ph.ms_bare;
  const int A = ph.mass_number;
  const int Z = ph.proton_number;
  op::Operator o = [&]() -> op::Operator {
    switch (request.kind) {
      case OperatorKind::Rp2:
        return opslib::rp2(ms, A, Z);
      case OperatorKind::Rn2:
        return opslib::rn2(ms, A, Z);
      case OperatorKind::Rm2:
        return opslib::rm2(ms, A);
      case OperatorKind::Tcm:
        return opslib::t_cm(ms, A);
      case OperatorKind::Trel:
        return opslib::t_rel(ms, A);
      case OperatorKind::Tlab:
        return opslib::kinetic_energy(ms);
      case OperatorKind::Hcm: {
        std::optional<double> hw;
        if (!request.params.empty()) {
          if (const auto* d = std::get_if<double>(&request.params.front())) {
            hw = *d;
          } else if (const auto* i = std::get_if<std::int64_t>(&request.params.front())) {
            hw = static_cast<double>(*i);
          }
        }
        return opslib::h_cm(ms, A, hw);
      }
      default:
        planned("operator '" + request.canonical_name + "'");
    }
  }();
  if (ph.hf) o = hf::to_hf_basis(o, *ph.hf);
  const auto hno_ms = ph.hno.model_space_ptr();
  if (hno_ms != o.model_space_ptr()) o = op::with_model_space(o, hno_ms);
  return op::normal_order(o);
}

}  // namespace imsrg::driver::pipeline
