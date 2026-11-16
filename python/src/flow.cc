// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Bindings of the flow module: generators, the Magnus solver (with
/// hunter-gatherer), the ODE solvers (`flow_rk4`, `flow`) and the BCH
/// transform/product with the factorized 3f2 options (design/18 section 2.7).
/// Long calls release the GIL; status callbacks re-acquire it.

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "imsrg/base/error.h"
#include "imsrg/flow/bch.h"
#include "imsrg/flow/generator.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/flow/ode.h"
#include "python/src/bindings.h"

namespace imsrg::py {

namespace nb = nanobind;
namespace {

using StatusFn = std::function<void(const flow::FlowStatus&)>;

/// Wraps a Python callable as the status callback of a solver; the callable
/// runs with the GIL held. An exception of the callable propagates out of the
/// solver's `Solve` (the solver is left at a step boundary) and is re-raised
/// by `solve()`.
StatusFn WrapCallback(const std::optional<nb::callable>& cb) {
  std::shared_ptr<nb::object> holder;
  if (cb) holder = std::make_shared<nb::object>(*cb);
  return [holder](const flow::FlowStatus& st) {
    nb::gil_scoped_acquire gil;
    // Ctrl-C: the Python handlers run here (a step boundary); a pending
    // KeyboardInterrupt aborts the solve and is re-raised by `solve()`.
    ThrowIfSignalPending();
    if (holder) (*holder)(st);
  };
}

/// A Magnus solver owned by Python (releasable with the Runtime).
struct PyMagnus : Releasable {
  BusyFlag busy;  ///< one thread at a time
  std::unique_ptr<flow::MagnusSolver> s;
  bool CanRelease() const noexcept override { return true; }
  void Release() noexcept override { s.reset(); }
  flow::MagnusSolver& get() {
    if (!s) throw StaleHostError("the solver's device data were released");
    return *s;
  }
};

/// An ODE solver owned by Python.
struct PyOde : Releasable {
  BusyFlag busy;  ///< one thread at a time
  std::unique_ptr<flow::OdeSolver> s;
  bool CanRelease() const noexcept override { return true; }
  void Release() noexcept override { s.reset(); }
  flow::OdeSolver& get() {
    if (!s) throw StaleHostError("the solver's device data were released");
    return *s;
  }
};

std::vector<op::Operator> Inputs(const std::vector<PyOperator>& ops) {
  std::vector<op::Operator> out;
  for (const PyOperator& o : ops) out.push_back(Input(o));
  return out;
}

}  // namespace

void BindFlow(nb::module_& m) {
  using flow::GeneratorKind;
  nb::enum_<GeneratorKind>(m, "GeneratorKind", "IMSRG generator types.")
      .value("White", GeneratorKind::kWhite)
      .value("Atan", GeneratorKind::kAtan)
      .value("ImaginaryTime", GeneratorKind::kImaginaryTime)
      .value("Wegner", GeneratorKind::kWegner)
      .value("QtransferAtan", GeneratorKind::kQtransferAtan)
      .value("ShellModel", GeneratorKind::kShellModel)
      .value("ShellModelAtan", GeneratorKind::kShellModelAtan)
      .value("ShellModelWegner", GeneratorKind::kShellModelWegner)
      .value("ShellModelImaginaryTime", GeneratorKind::kShellModelImaginaryTime)
      .value("ShellModelAtanNpnh", GeneratorKind::kShellModelAtanNpnh)
      .value("HartreeFock", GeneratorKind::kHartreeFock)
      .value("OnePa", GeneratorKind::kOnePa);
  nb::enum_<flow::Partitioning>(m, "Partitioning",
                                "Energy-denominator partitioning.")
      .value("EpsteinNesbet", flow::Partitioning::kEpsteinNesbet)
      .value("MollerPlesset", flow::Partitioning::kMollerPlesset)
      .value("MpIsospin", flow::Partitioning::kMpIsospin);
  nb::enum_<flow::StopReason>(m, "StopReason", "Why a solve stopped.")
      .value("Converged", flow::StopReason::kConverged)
      .value("SmaxReached", flow::StopReason::kSmaxReached)
      .value("Diverged", flow::StopReason::kDiverged);
  nb::enum_<flow::OdeMethod>(
      m, "OdeMethod",
      "Rk4 = imsrg++ `flow_RK4` (fixed step), Dopri5 = imsrg++ `flow` "
      "(adaptive Dormand-Prince 5(4)).")
      .value("Rk4", flow::OdeMethod::kRk4)
      .value("Dopri5", flow::OdeMethod::kDopri5);

  nb::class_<flow::GeneratorOptions>(m, "GeneratorOptions",
                                     "Options of compute_generator.")
      .def(
          "__init__",
          [](flow::GeneratorOptions* self, GeneratorKind kind,
             flow::Partitioning partitioning, double cutoff, double delta,
             bool delta_all, std::optional<int> delta_orbit, int qtransfer_n,
             bool only_1b_eta) {
            flow::GeneratorOptions o;
            o.kind = kind;
            o.partitioning = partitioning;
            o.denominator_cutoff = cutoff;
            o.denominator_delta = delta;
            o.denominator_delta_all = delta_all;
            if (delta_orbit)
              o.denominator_delta_orbit = OrbitIndex{*delta_orbit};
            o.qtransfer_n = qtransfer_n;
            o.only_1b_eta = only_1b_eta;
            new (self) flow::GeneratorOptions(o);
          },
          nb::kw_only(), nb::arg("kind") = GeneratorKind::kAtan,
          nb::arg("partitioning") = flow::Partitioning::kEpsteinNesbet,
          nb::arg("denominator_cutoff") = 1.0e-6,
          nb::arg("denominator_delta") = 0.0,
          nb::arg("denominator_delta_all") = false,
          nb::arg("denominator_delta_orbit").none() = nb::none(),
          nb::arg("qtransfer_n") = 1, nb::arg("only_1b_eta") = false)
      .def_rw("kind", &flow::GeneratorOptions::kind)
      .def_rw("partitioning", &flow::GeneratorOptions::partitioning)
      .def_rw("denominator_cutoff", &flow::GeneratorOptions::denominator_cutoff)
      .def_rw("denominator_delta", &flow::GeneratorOptions::denominator_delta)
      .def_rw("denominator_delta_all",
              &flow::GeneratorOptions::denominator_delta_all)
      .def_prop_rw(
          "denominator_delta_orbit",
          [](const flow::GeneratorOptions& o) -> std::optional<int> {
            if (!o.denominator_delta_orbit) return std::nullopt;
            return o.denominator_delta_orbit->value();
          },
          [](flow::GeneratorOptions& o, std::optional<int> v) {
            if (v) {
              o.denominator_delta_orbit = OrbitIndex{*v};
            } else {
              o.denominator_delta_orbit.reset();
            }
          })
      .def_rw("qtransfer_n", &flow::GeneratorOptions::qtransfer_n)
      .def_rw("only_1b_eta", &flow::GeneratorOptions::only_1b_eta);

  m.def("generator_function", &flow::GeneratorFunction, nb::arg("kind"),
        nb::arg("h_od"), nb::arg("denominator"), nb::arg("qtransfer_n") = 1,
        "The generator function f(H_od, Delta) of `kind` (white: H/Delta, "
        "atan: atan(2H/Delta)/2, ...).");
  m.def("floor_denominator", &flow::FloorDenominator, nb::arg("delta"),
        nb::arg("cutoff"),
        "|delta| < cutoff -> sign(delta) * cutoff (sign of 0 is +).");
  m.def(
      "compute_generator",
      [](const PyOperator& h, const flow::GeneratorOptions& options) {
        const op::Operator& x = Input(h);
        GilFree r;
        return Result(flow::ComputeGenerator(x, options));
      },
      nb::arg("hamiltonian"), nb::arg("options") = flow::GeneratorOptions{},
      "The generator eta(H) of the Hamiltonian (anti-Hermitian, same ranks "
      "as H; one-body only for the HartreeFock kind).");

  nb::class_<flow::Denominators>(
      m, "Denominators",
      "Floored energy denominators of a Hamiltonian (read-only tables).")
      .def(
          "__init__",
          [](flow::Denominators* self, const PyOperator& h,
             const flow::GeneratorOptions& options) {
            new (self) flow::Denominators(Input(h), options);
          },
          nb::arg("hamiltonian"), nb::arg("options") = flow::GeneratorOptions{})
      .def(
          "one_body",
          [](const flow::Denominators& d, int i, int j) {
            return d.one_body(OrbitIndex{i}, OrbitIndex{j});
          },
          nb::arg("i"), nb::arg("j"), "One-body denominator Delta(i, j).")
      .def(
          "two_body",
          [](const flow::Denominators& d, int channel, int bra, int ket) {
            return d.two_body(ChannelIndex{channel}, LocalKetIndex{bra},
                              LocalKetIndex{ket});
          },
          nb::arg("channel"), nb::arg("bra"), nb::arg("ket"),
          "Two-body denominator of local kets (bra, ket) of a channel.")
      .def(
          "monopole",
          [](const flow::Denominators& d, int a, int b) {
            return d.monopole(OrbitIndex{a}, OrbitIndex{b});
          },
          nb::arg("a"), nb::arg("b"),
          "Unnormalized J-averaged monopole V-bar_abab.");

  nb::class_<flow::SolverOptions>(m, "SolverOptions",
                                  "Step control of the flow solvers "
                                  "(defaults of the imsrg++ executable).")
      .def(
          "__init__",
          [](flow::SolverOptions* self, double smax, double dsmax, double ds_0,
             double domega, double omega_norm_max, double eta_criterion,
             bool magnus_adaptive, bool hunter_gatherer) {
            flow::SolverOptions o;
            o.smax = smax;
            o.dsmax = dsmax;
            o.ds_0 = ds_0;
            o.domega = domega;
            o.omega_norm_max = omega_norm_max;
            o.eta_criterion = eta_criterion;
            o.magnus_adaptive = magnus_adaptive;
            o.hunter_gatherer = hunter_gatherer;
            new (self) flow::SolverOptions(o);
          },
          nb::kw_only(), nb::arg("smax") = 200.0, nb::arg("dsmax") = 0.5,
          nb::arg("ds_0") = 0.5, nb::arg("domega") = 0.1,
          nb::arg("omega_norm_max") = 0.25, nb::arg("eta_criterion") = 1.0e-6,
          nb::arg("magnus_adaptive") = true, nb::arg("hunter_gatherer") = false)
      .def_rw("smax", &flow::SolverOptions::smax)
      .def_rw("dsmax", &flow::SolverOptions::dsmax)
      .def_rw("ds_0", &flow::SolverOptions::ds_0)
      .def_rw("domega", &flow::SolverOptions::domega)
      .def_rw("omega_norm_max", &flow::SolverOptions::omega_norm_max)
      .def_rw("eta_criterion", &flow::SolverOptions::eta_criterion)
      .def_rw("magnus_adaptive", &flow::SolverOptions::magnus_adaptive)
      .def_rw("hunter_gatherer", &flow::SolverOptions::hunter_gatherer);

  nb::class_<flow::BchOptions>(
      m, "BchOptions",
      "Options of bch_transform and bch_product (imsrg++ defaults); "
      "`factorized` switches on the IMSRG(3f2) correction.")
      .def(
          "__init__",
          [](flow::BchOptions* self, double transform_threshold,
             double product_threshold, int max_iterations, int warn_iterations,
             bool factorized, commutator::FactorizedOptions terms) {
            flow::BchOptions o;
            o.transform_threshold = transform_threshold;
            o.product_threshold = product_threshold;
            o.max_iterations = max_iterations;
            o.warn_iterations = warn_iterations;
            o.factorized = factorized;
            o.factorized_terms = terms;
            new (self) flow::BchOptions(o);
          },
          nb::kw_only(), nb::arg("transform_threshold") = 1.0e-9,
          nb::arg("product_threshold") = 1.0e-4, nb::arg("max_iterations") = 40,
          nb::arg("warn_iterations") = 12, nb::arg("factorized") = false,
          nb::arg("factorized_terms") = commutator::FactorizedOptions{})
      .def_rw("transform_threshold", &flow::BchOptions::transform_threshold)
      .def_rw("product_threshold", &flow::BchOptions::product_threshold)
      .def_rw("max_iterations", &flow::BchOptions::max_iterations)
      .def_rw("warn_iterations", &flow::BchOptions::warn_iterations)
      .def_rw("factorized", &flow::BchOptions::factorized)
      .def_rw("factorized_terms", &flow::BchOptions::factorized_terms);

  nb::class_<flow::BchReport>(m, "BchReport", "Diagnostics of bch_transform.")
      .def_ro("nested_commutators", &flow::BchReport::nested_commutators)
      .def_ro("converged", &flow::BchReport::converged)
      .def_ro("factorized_corrections",
              &flow::BchReport::factorized_corrections);

  nb::class_<flow::FlowStatus>(m, "FlowStatus",
                               "One line of the flow log (state after a "
                               "step; istep 0 = initial state of a stage).")
      .def_ro("stage", &flow::FlowStatus::stage)
      .def_ro("istep", &flow::FlowStatus::istep)
      .def_ro("s", &flow::FlowStatus::s)
      .def_ro("ds", &flow::FlowStatus::ds)
      .def_ro("e0", &flow::FlowStatus::e0)
      .def_ro("norm_h", &flow::FlowStatus::norm_h)
      .def_ro("norm_eta_1b", &flow::FlowStatus::norm_eta_1b)
      .def_ro("norm_eta_2b", &flow::FlowStatus::norm_eta_2b)
      .def_ro("norm_eta", &flow::FlowStatus::norm_eta)
      .def_ro("norm_omega_1b", &flow::FlowStatus::norm_omega_1b)
      .def_ro("norm_omega_2b", &flow::FlowStatus::norm_omega_2b)
      .def_ro("norm_omega", &flow::FlowStatus::norm_omega)
      .def_ro("num_omegas", &flow::FlowStatus::num_omegas)
      .def_ro("wall_seconds", &flow::FlowStatus::wall_seconds)
      .def("__repr__", [](const flow::FlowStatus& st) {
        return "FlowStatus(stage=" + std::to_string(st.stage) +
               ", istep=" + std::to_string(st.istep) +
               ", s=" + std::to_string(st.s) + ", e0=" + std::to_string(st.e0) +
               ", norm_eta=" + std::to_string(st.norm_eta) + ")";
      });

  nb::class_<flow::SolveResult>(m, "SolveResult", "Summary of one solve.")
      .def_ro("steps", &flow::SolveResult::steps)
      .def_ro("s", &flow::SolveResult::s)
      .def_ro("e0", &flow::SolveResult::e0)
      .def_ro("norm_eta", &flow::SolveResult::norm_eta)
      .def_ro("num_gathers", &flow::SolveResult::num_gathers)
      .def_ro("reason", &flow::SolveResult::reason);

  nb::class_<PyMagnus>(
      m, "MagnusSolver",
      "The Magnus IMSRG(2) solver (Euler steps on Omega through the BCH "
      "product), optionally with hunter-gatherer Omega. Operators returned "
      "by its accessors are copies (device current: call sync_to_host() "
      "before taking numpy views).")
      .def(
          "__init__",
          [](PyMagnus* self, const PyOperator& h0,
             const flow::SolverOptions& solver,
             const flow::GeneratorOptions& generator,
             const flow::BchOptions& bch,
             const commutator::CommutatorOptions& comm) {
            EnsureRuntime();
            op::Operator h = Input(h0);
            // Build first: if this throws, nanobind never runs the destructor
            // of `self`, so it must not be registered yet.
            auto solver_ptr = std::make_unique<flow::MagnusSolver>(
                std::move(h), solver, generator, bch, comm);
            new (self) PyMagnus();
            self->s = std::move(solver_ptr);
          },
          nb::arg("h0"), nb::kw_only(),
          nb::arg("solver") = flow::SolverOptions{},
          nb::arg("generator") = flow::GeneratorOptions{},
          nb::arg("bch") = flow::BchOptions{},
          nb::arg("commutator") = commutator::CommutatorOptions{},
          "H0 must be scalar, Hermitian, particle rank 2 and normal ordered "
          "with respect to the reference of its model space.")
      .def(
          "solve",
          [](PyMagnus& self, std::optional<nb::callable> on_status) {
            BusyScope busy(self.busy);
            flow::MagnusSolver& s = self.get();
            const StatusFn cb = WrapCallback(on_status);
            GilFree r;
            return s.Solve(cb);
          },
          nb::arg("on_status").none() = nb::none(),
          "Runs the Magnus loop until convergence, s >= smax or divergence. "
          "`on_status(FlowStatus)` is called for the initial state and after "
          "every step; an exception it raises aborts the solve and is "
          "re-raised (the solver stays consistent).")
      .def(
          "transform",
          [](PyMagnus& self, const PyOperator& o) {
            BusyScope busy(self.busy);
            flow::MagnusSolver& s = self.get();
            const op::Operator& x = Input(o);
            GilFree r;
            return Result(s.Transform(x));
          },
          nb::arg("operator"),
          "Applies the accumulated Omega stack e^Omega O e^-Omega to an "
          "operator on the solver's model space.")
      .def(
          "set_generator",
          [](PyMagnus& self, const flow::GeneratorOptions& g) {
            BusyScope busy(self.busy);
            self.get().set_generator(g);
          },
          "Changes the generator for the next solve (a second stage).")
      .def(
          "set_smax",
          [](PyMagnus& self, double v) {
            BusyScope busy(self.busy);
            self.get().set_smax(v);
          },
          "Changes smax for the next solve.")
      .def(
          "set_solver_options",
          [](PyMagnus& self, const flow::SolverOptions& o) {
            BusyScope busy(self.busy);
            self.get().set_solver_options(o);
          },
          "Replaces the step options for the next solve.")
      .def_prop_ro(
          "hamiltonian",
          [](PyMagnus& self) {
            BusyScope busy(self.busy);
            return PyOperator(self.get().hamiltonian());
          },
          "A copy of H(s).")
      .def_prop_ro(
          "eta",
          [](PyMagnus& self) {
            BusyScope busy(self.busy);
            return PyOperator(self.get().eta());
          },
          "A copy of eta(s).")
      .def_prop_ro(
          "omegas",
          [](PyMagnus& self) {
            BusyScope busy(self.busy);
            std::vector<PyOperator> out;
            for (const auto& o : self.get().omegas()) out.emplace_back(o);
            return out;
          },
          "Copies of the Omega stack, oldest first.")
      .def_prop_ro("s",
                   [](PyMagnus& self) {
                     BusyScope busy(self.busy);
                     return self.get().s();
                   })
      .def_prop_ro("num_gathers",
                   [](PyMagnus& self) {
                     BusyScope busy(self.busy);
                     return self.get().num_gathers();
                   })
      .def_prop_ro("solver_options",
                   [](PyMagnus& self) {
                     BusyScope busy(self.busy);
                     return self.get().solver_options();
                   })
      .def_prop_ro("generator_options", [](PyMagnus& self) {
        BusyScope busy(self.busy);
        return self.get().generator_options();
      });

  nb::class_<PyOde>(
      m, "OdeSolver",
      "Solvers of dH/ds = [eta(H), H]: fixed-step RK4 (imsrg++ `flow_RK4`) "
      "and adaptive Dormand-Prince (imsrg++ `flow`). Operators evolve "
      "alongside H with the same eta.")
      .def(
          "__init__",
          [](PyOde* self, flow::OdeMethod method, const PyOperator& h0,
             const std::vector<PyOperator>& operators,
             const flow::SolverOptions& solver, double ode_tolerance,
             const flow::GeneratorOptions& generator,
             const commutator::CommutatorOptions& comm) {
            EnsureRuntime();
            op::Operator h = Input(h0);
            std::vector<op::Operator> ops = Inputs(operators);
            auto solver_ptr = std::make_unique<flow::OdeSolver>(
                method, std::move(h), std::move(ops), solver, ode_tolerance,
                generator, comm);
            new (self) PyOde();
            self->s = std::move(solver_ptr);
          },
          nb::arg("method"), nb::arg("h0"), nb::kw_only(),
          nb::arg("operators") = std::vector<PyOperator>{},
          nb::arg("solver") = flow::SolverOptions{},
          nb::arg("ode_tolerance") = 1.0e-6,
          nb::arg("generator") = flow::GeneratorOptions{},
          nb::arg("commutator") = commutator::CommutatorOptions{})
      .def(
          "solve",
          [](PyOde& self, std::optional<nb::callable> on_status) {
            BusyScope busy(self.busy);
            flow::OdeSolver& s = self.get();
            const StatusFn cb = WrapCallback(on_status);
            GilFree r;
            return s.Solve(cb);
          },
          nb::arg("on_status").none() = nb::none(),
          "Integrates from the current s to smax (Rk4: or until converged).")
      .def(
          "set_generator",
          [](PyOde& self, const flow::GeneratorOptions& g) {
            BusyScope busy(self.busy);
            self.get().set_generator(g);
          },
          "Changes the generator for the next solve (a second stage).")
      .def(
          "set_solver_options",
          [](PyOde& self, const flow::SolverOptions& o) {
            BusyScope busy(self.busy);
            self.get().set_solver_options(o);
          },
          "Replaces the step options for the next solve.")
      .def_prop_ro(
          "hamiltonian",
          [](PyOde& self) {
            BusyScope busy(self.busy);
            return PyOperator(self.get().hamiltonian());
          },
          "A copy of H(s).")
      .def_prop_ro("num_operators",
                   [](PyOde& self) {
                     BusyScope busy(self.busy);
                     return self.get().num_operators();
                   })
      .def(
          "flowing_operator",
          [](PyOde& self, std::size_t i) {
            BusyScope busy(self.busy);
            flow::OdeSolver& s = self.get();
            if (i >= s.num_operators())
              throw imsrg::InputError("operator index out of range");
            return PyOperator(s.flowing_operator(i));
          },
          nb::arg("i"), "A copy of the i-th flowing operator O_i(s).")
      .def_prop_ro("eta",
                   [](PyOde& self) {
                     BusyScope busy(self.busy);
                     return PyOperator(self.get().eta());
                   })
      .def_prop_ro("s", [](PyOde& self) {
        BusyScope busy(self.busy);
        return self.get().s();
      });

  m.def(
      "bch_transform",
      [](const PyOperator& x, const PyOperator& omega,
         const flow::BchOptions& bch, const commutator::CommutatorOptions& comm,
         std::shared_ptr<PyWorkspace> workspace) {
        const op::Operator& xi = Input(x);
        const op::Operator& oi = Input(omega);
        RequireSameModelSpace(xi, oi);
        if (!workspace) workspace = WorkspaceFor(xi.model_space_ptr());
        GilFree r;
        std::lock_guard lock(workspace->mutex());
        flow::BchReport report;
        op::Operator z =
            flow::BchTransform(xi, oi, bch, comm, workspace->ws(), &report);
        return std::make_pair(Result(std::move(z)), report);
      },
      nb::arg("operator"), nb::arg("omega"), nb::kw_only(),
      nb::arg("bch") = flow::BchOptions{},
      nb::arg("commutator") = commutator::CommutatorOptions{},
      nb::arg("workspace").none() = nb::none(),
      "Returns (Z, BchReport) with Z = e^Omega X e^-Omega = X + [Omega, X] + "
      "1/2! [Omega, [Omega, X]] + ... (imsrg++ Standard_BCH_Transform, "
      "truncated at the IMSRG(2) level; with bch.factorized the 3f2 "
      "correction enters, Omega must be anti-Hermitian then).");
  m.def(
      "bch_product",
      [](const PyOperator& x, const PyOperator& y, const flow::BchOptions& bch,
         const commutator::CommutatorOptions& comm,
         std::shared_ptr<PyWorkspace> workspace) {
        const op::Operator& xi = Input(x);
        const op::Operator& yi = Input(y);
        RequireSameModelSpace(xi, yi);
        if (!workspace) workspace = WorkspaceFor(xi.model_space_ptr());
        GilFree r;
        std::lock_guard lock(workspace->mutex());
        return Result(flow::BchProduct(xi, yi, bch, comm, workspace->ws()));
      },
      nb::arg("x"), nb::arg("y"), nb::kw_only(),
      nb::arg("bch") = flow::BchOptions{},
      nb::arg("commutator") = commutator::CommutatorOptions{},
      nb::arg("workspace").none() = nb::none(),
      "Z with e^Z = e^X e^Y for small X (imsrg++ BCH_Product: series in "
      "ad_Y to 8th order plus the 1/12 [[Y,X],X] term).");
}

}  // namespace imsrg::py
