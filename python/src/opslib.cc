// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the operator library (opslib): kinetic energy, center of mass
/// and radius operators, the building blocks of a scripted pipeline.

#include <optional>

#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/kinetic.h"
#include "imsrg/opslib/radius.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;

namespace {

/// Binds the kinetic-energy and center-of-mass kinetic operators.
void BindKineticOperators(nb::module_& m) {
  m.def(
      "kinetic_energy",
      [](const PyModelSpace& ms) {
        EnsureRuntime();
        return Result(opslib::KineticEnergy(ms.model_space()));
      },
      nb::arg("model_space"),
      "Laboratory-frame kinetic energy T (one-body, MeV).");
  m.def(
      "t_cm",
      [](const PyModelSpace& ms, int mass_number) {
        EnsureRuntime();
        GilFree r;
        return Result(opslib::TCm(ms.model_space(), mass_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"),
      "Center-of-mass kinetic energy T_cm of a system of A nucleons (MeV).");
  m.def(
      "t_rel",
      [](const PyModelSpace& ms, int mass_number) {
        EnsureRuntime();
        GilFree r;
        return Result(opslib::TRel(ms.model_space(), mass_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"),
      "Intrinsic kinetic energy T_rel = T - T_cm (MeV), the kinetic term of "
      "the driver's Hamiltonian.");
  m.def(
      "r2_cm",
      [](const PyModelSpace& ms, int mass_number) {
        EnsureRuntime();
        GilFree r;
        return Result(opslib::R2Cm(ms.model_space(), mass_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"),
      "Center-of-mass radius squared (fm^2).");
}

/// Binds the center-of-mass Hamiltonian and the radius operators.
void BindRadiusOperators(nb::module_& m) {
  m.def(
      "h_cm",
      [](const PyModelSpace& ms, int mass_number,
         std::optional<double> hbar_omega_trap) {
        EnsureRuntime();
        GilFree r;
        return Result(
            opslib::HCm(ms.model_space(), mass_number, hbar_omega_trap));
      },
      nb::arg("model_space"), nb::arg("mass_number"),
      nb::arg("hbar_omega_trap") = nb::none(),
      "Center-of-mass Hamiltonian (Lawson term, MeV); trap frequency "
      "defaults to the basis frequency.");
  m.def(
      "rp2",
      [](const PyModelSpace& ms, int mass_number, int proton_number) {
        EnsureRuntime();
        GilFree r;
        return Result(
            opslib::Rp2(ms.model_space(), mass_number, proton_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"), nb::arg("proton_number"),
      "Intrinsic point-proton radius squared (fm^2), without the spin-orbit "
      "correction (D35).");
  m.def(
      "rn2",
      [](const PyModelSpace& ms, int mass_number, int proton_number) {
        EnsureRuntime();
        GilFree r;
        return Result(
            opslib::Rn2(ms.model_space(), mass_number, proton_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"), nb::arg("proton_number"),
      "Intrinsic point-neutron radius squared (fm^2).");
  m.def(
      "rm2",
      [](const PyModelSpace& ms, int mass_number) {
        EnsureRuntime();
        GilFree r;
        return Result(opslib::Rm2(ms.model_space(), mass_number));
      },
      nb::arg("model_space"), nb::arg("mass_number"),
      "Intrinsic matter radius squared (fm^2).");
}

}  // namespace

void BindOpslib(nb::module_& m) {
  BindKineticOperators(m);
  BindRadiusOperators(m);
}

}  // namespace imsrg::py
