// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of hf and mbpt: Hartree-Fock with options and results, the MBPT
/// energies and the natural-orbital basis (design/18 sections 2.4 and 2.5).
/// All heavy calls release the GIL.

#include <cstring>
#include <string>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/base/error.h"
#include "imsrg/hf/hartree_fock.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/mbpt/energies.h"
#include "imsrg/mbpt/natural_orbitals.h"
#include "imsrg/operator/no2b.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

using Array2 = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

/// Copies a column-major matrix into a new F-ordered numpy array.
Array2 ToNumpy(const linalg::Matrix<double>& a) {
  const auto rows = static_cast<std::size_t>(a.rows());
  const auto cols = static_cast<std::size_t>(a.cols());
  auto* data = new double[rows * cols + 1];
  for (std::size_t j = 0; j < cols; ++j)
    for (std::size_t i = 0; i < rows; ++i)
      data[i + j * rows] =
          a(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j));
  nb::capsule owner(data,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return Array2(data, {rows, cols}, owner,
                {1, static_cast<std::int64_t>(rows)});
}

/// Copies any 2-D float64 array (any layout) into a column-major matrix.
linalg::Matrix<double> FromNumpy(
    const nb::ndarray<const double, nb::ndim<2>>& a) {
  linalg::Matrix<double> m(static_cast<std::int64_t>(a.shape(0)),
                           static_cast<std::int64_t>(a.shape(1)));
  const double* p = a.data();
  for (std::size_t i = 0; i < a.shape(0); ++i)
    for (std::size_t j = 0; j < a.shape(1); ++j)
      m(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j)) =
          p[static_cast<std::int64_t>(i) * a.stride(0) +
            static_cast<std::int64_t>(j) * a.stride(1)];
  return m;
}

/// The C++ operator for a call that takes it by non-const reference (MBPT
/// syncs the host mirror as a side effect): pushes host edits first.
op::Operator& Mutable(PyOperator& o) {
  Input(o);
  return o;
}

}  // namespace

void BindHf(nb::module_& m) {
  nb::enum_<hf::Mixing>(m, "Mixing",
                        "How successive HF densities are combined.")
      .value("Linear", hf::Mixing::kLinear)
      .value("Diis", hf::Mixing::kDiis);

  nb::class_<hf::HFOptions>(m, "HFOptions", "Options of hf.solve.")
      .def(
          "__init__",
          [](hf::HFOptions* self, hf::Mixing mixing, double tolerance,
             int max_iterations, double density_mixing, bool freeze_occupations,
             int diis_start, int diis_subspace, int e3max) {
            hf::HFOptions o;
            o.mixing = mixing;
            o.tolerance = tolerance;
            o.max_iterations = max_iterations;
            o.density_mixing = density_mixing;
            o.freeze_occupations = freeze_occupations;
            o.diis_start = diis_start;
            o.diis_subspace = diis_subspace;
            o.e3max = e3max;
            new (self) hf::HFOptions(o);
          },
          nb::kw_only(), nb::arg("mixing") = hf::Mixing::kLinear,
          nb::arg("tolerance") = 1.0e-8, nb::arg("max_iterations") = 1000,
          nb::arg("density_mixing") = 0.2,
          nb::arg("freeze_occupations") = false, nb::arg("diis_start") = 2,
          nb::arg("diis_subspace") = 8, nb::arg("e3max") = 12)
      .def_rw("mixing", &hf::HFOptions::mixing,
              "Linear density mixing (imsrg++) or Pulay DIIS on the Fock "
              "matrix.")
      .def_rw("tolerance", &hf::HFOptions::tolerance)
      .def_rw("max_iterations", &hf::HFOptions::max_iterations)
      .def_rw("density_mixing", &hf::HFOptions::density_mixing,
              "Linear mixing: weight of the OLD density (imsrg++ 0.2).")
      .def_rw("freeze_occupations", &hf::HFOptions::freeze_occupations)
      .def_rw("diis_start", &hf::HFOptions::diis_start)
      .def_rw("diis_subspace", &hf::HFOptions::diis_subspace)
      .def_rw("e3max", &hf::HFOptions::e3max,
              "Only three-body monopole terms with e_a + e_b + e_c <= e3max "
              "enter.");

  nb::class_<hf::HFResult>(m, "HFResult",
                           "Result of a Hartree-Fock calculation (host data; "
                           "matrices are numpy copies).")
      .def_ro("e_hf", &hf::HFResult::e_hf, "E_HF = e1 + e2 + e3 (MeV).")
      .def_ro("e1", &hf::HFResult::e1)
      .def_ro("e2", &hf::HFResult::e2)
      .def_ro("e3", &hf::HFResult::e3)
      .def_ro("iterations", &hf::HFResult::iterations)
      .def_ro("converged", &hf::HFResult::converged)
      .def_prop_ro(
          "spe", [](const hf::HFResult& r) { return r.spe; },
          "HF single-particle energies by HF orbit (MeV).")
      .def_prop_ro("occupations",
                   [](const hf::HFResult& r) { return r.occupations; })
      .def_prop_ro("ediff_history",
                   [](const hf::HFResult& r) { return r.ediff_history; })
      .def_prop_ro(
          "c", [](const hf::HFResult& r) { return ToNumpy(r.C); },
          nb::rv_policy::move, "C(ho orbit, hf orbit), column-major copy.")
      .def_prop_ro(
          "rho", [](const hf::HFResult& r) { return ToNumpy(r.rho); },
          nb::rv_policy::move, "Density matrix in the oscillator basis.")
      .def_prop_ro(
          "fock", [](const hf::HFResult& r) { return ToNumpy(r.F); },
          nb::rv_policy::move, "Fock matrix in the oscillator basis.");

  m.def(
      "hf_solve",
      [](const PyOperator& h, const op::No2bThreeBody* v3,
         const hf::HFOptions& options) {
        const op::Operator& x = Input(h);
        nb::gil_scoped_release r;
        return hf::Solve(x, v3, options);
      },
      nb::arg("hamiltonian"), nb::arg("v3").none() = nb::none(),
      nb::arg("options") = hf::HFOptions{},
      "Spherical Hartree-Fock for the Hamiltonian (one-body kinetic part and "
      "two-body interaction; optional NO2B three-body input). The reference "
      "is that of the model space. Returns an HFResult.");
  m.def(
      "hf_solution_in_basis",
      [](const PyOperator& h, const op::No2bThreeBody* v3,
         const nb::ndarray<const double, nb::ndim<2>>& c_ho2basis,
         std::vector<double> occupations, const hf::HFOptions& options) {
        const op::Operator& x = Input(h);
        const linalg::Matrix<double> c = FromNumpy(c_ho2basis);
        nb::gil_scoped_release r;
        return hf::SolutionInBasis(x, v3, c, occupations, options);
      },
      nb::arg("hamiltonian"), nb::arg("v3").none(), nb::arg("c_ho2basis"),
      nb::arg("occupations"), nb::arg("options") = hf::HFOptions{},
      "Mean-field state of a given orthogonal basis (no iteration), e.g. the "
      "natural-orbital basis.");
  m.def(
      "hf_normal_ordered_hamiltonian",
      [](const PyOperator& h, const op::No2bThreeBody* v3,
         const hf::HFResult& res, const hf::HFOptions& options) {
        const op::Operator& x = Input(h);
        nb::gil_scoped_release r;
        return Result(hf::NormalOrderedHamiltonian(x, v3, res, options));
      },
      nb::arg("hamiltonian"), nb::arg("v3").none(), nb::arg("result"),
      nb::arg("options") = hf::HFOptions{},
      "The Hamiltonian in the HF basis, normal ordered with respect to the "
      "HF reference, three-body part truncated at the normal-ordered "
      "two-body level.");
  m.def(
      "hf_to_hf_basis",
      [](const PyOperator& o, const hf::HFResult& res) {
        const op::Operator& x = Input(o);
        nb::gil_scoped_release r;
        return Result(hf::ToHfBasis(x, res));
      },
      nb::arg("operator"), nb::arg("result"),
      "The operator transformed to the HF basis (no normal ordering).");

  // ---- mbpt
  nb::class_<mbpt::Mp3Energy>(m, "Mp3Energy", "MBPT(3) energy terms (MeV).")
      .def_ro("pp", &mbpt::Mp3Energy::pp)
      .def_ro("hh", &mbpt::Mp3Energy::hh)
      .def_ro("ph", &mbpt::Mp3Energy::ph)
      .def_ro("total", &mbpt::Mp3Energy::total);
  m.def(
      "mp2_energy",
      [](PyOperator& h) {
        op::Operator& x = Mutable(h);
        nb::gil_scoped_release r;
        return mbpt::Mp2Energy(x);
      },
      nb::arg("hamiltonian"),
      "MBPT(2) correlation energy (MeV) of a normal-ordered Hamiltonian in "
      "the basis where the one-body part is diagonal (HF basis).");
  m.def(
      "mp3_energy",
      [](PyOperator& h) {
        op::Operator& x = Mutable(h);
        nb::gil_scoped_release r;
        return mbpt::ComputeMp3Energy(x);
      },
      nb::arg("hamiltonian"),
      "MBPT(3) energy terms (pp, hh, ph, total; MeV), same preconditions as "
      "mp2_energy.");

  nb::class_<mbpt::NaturalOrbitals>(m, "NaturalOrbitals",
                                    "Natural orbitals of a density matrix.")
      .def_prop_ro(
          "c_hf2nat",
          [](const mbpt::NaturalOrbitals& n) { return ToNumpy(n.c_hf2nat); },
          nb::rv_policy::move,
          "Row = HF orbit, column = NAT orbit; block diagonal in "
          "(l, j, tz).")
      .def_prop_ro("occupations", [](const mbpt::NaturalOrbitals& n) {
        return n.occupations;
      });
  m.def(
      "mbpt2_density_matrix",
      [](PyOperator& h) {
        op::Operator& x = Mutable(h);
        linalg::Matrix<double> rho;
        {
          nb::gil_scoped_release r;
          rho = mbpt::Mbpt2DensityMatrix(x);
        }
        return ToNumpy(rho);
      },
      nb::arg("hamiltonian"),
      "MBPT(2) one-body density matrix in the HF basis (norb x norb).");
  m.def(
      "natural_orbitals",
      [](const nb::ndarray<const double, nb::ndim<2>>& rho_hf,
         const PyModelSpace& ms) {
        const linalg::Matrix<double> rho = FromNumpy(rho_hf);
        nb::gil_scoped_release r;
        return mbpt::ComputeNaturalOrbitals(rho, *ms.ms);
      },
      nb::arg("rho_hf"), nb::arg("model_space"),
      "Diagonalizes each (l, j, tz) block of the density matrix; checks the "
      "particle number.");
  m.def(
      "reference_density_ho",
      [](const nb::ndarray<const double, nb::ndim<2>>& c_ho2nat,
         const PyModelSpace& ms) {
        const linalg::Matrix<double> c = FromNumpy(c_ho2nat);
        return ToNumpy(mbpt::ReferenceDensityHo(c, *ms.ms));
      },
      nb::arg("c_ho2nat"), nb::arg("model_space"),
      "Oscillator-basis reference density of the NAT basis.");
  m.def(
      "to_nat_basis",
      [](PyOperator& h, const nb::ndarray<const double, nb::ndim<2>>& c) {
        op::Operator& x = Mutable(h);
        const linalg::Matrix<double> cm = FromNumpy(c);
        nb::gil_scoped_release r;
        return Result(mbpt::ToNatBasis(x, cm));
      },
      nb::arg("hamiltonian"), nb::arg("c_hf2nat"),
      "Transforms the one- and two-body parts of H (HF basis) to the NAT "
      "basis.");
}

}  // namespace imsrg::py
