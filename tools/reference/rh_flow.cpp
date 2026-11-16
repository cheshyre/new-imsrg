// rh_flow.cpp -- `refharness flow`: single-reference Magnus IMSRG run with a full-precision trajectory.
//
//   refharness flow emax=4 reference=O16 [generator=atan] [smax=200] [ds_0=0.5] [dsmax=0.5]
//       [domega=0.1] [omega_norm_max=0.25] [eta_criterion=1e-6] [--imsrg3 [--n7]] [--factorized]
//       [--hunter-gatherer] [--omega-norm-max X] [--factorized-1b-intermediates 0|1]
//       [--factorized-2b-intermediates 0|1]
//       [2bme=... file2e1max=... hw=16] out=DIR
//
// Pipeline = imsrg++.cc (method=magnus, valence_space == reference, basis=HF, 2N only):
// ModelSpace -> bare H (me2j) -> HFMBPT (unfrozen occupations) -> GetNormalOrderedH ->
// IMSRGSolver set-up (imsrg++.cc:1029-1090) -> a verbatim copy of IMSRGSolver::Solve_magnus_euler
// (IMSRGSolver.cc:280-366, all solver members are public) that prints every step.
//
//   --imsrg3        Commutator::SetUseIMSRG3(true) with a pn-mode rank-3 H (IMSRG(3))
//   --n7            with --imsrg3: Commutator::SetUseIMSRG3N7(true)
//   --factorized    BCH::SetUseFactorizedCorrection(true): IMSRG(2) + factorized
//                   [Eta,[Eta,Gamma]_3b] correction = "IMSRG(3f2)". imsrg++ has no switch for this.
//   --hunter-gatherer  IMSRGSolver::SetHunterGatherer(true): when |Omega| > omega_norm_max the hunter
//                   is gathered (GatherOmega) instead of pushing a new Omega (NewOmega), and the
//                   solve ends with one more gather (IMSRGSolver::Solve, IMSRGSolver.cc:236-239).
//                   Default off (imsrg++ default hunter_gatherer=false).
//   --perturbative-triples  after the solve (and after the final gather with --hunter-gatherer) call
//                   IMSRGSolver::CalculatePerturbativeTriples() (IMSRGSolver.cc:1172-1214; uses Omega,
//                   FlowingOps[0] and H_0; imsrg++ raises E3max to 3 emax inside). Prints
//                   `perturbative_triples <dE>` and `e0_plus_triples <E0 + dE>` (%.17g) to stdout and adds
//                   perturbative_triples / e0_plus_triples to result.json (only when the flag is given).
//   --omega-norm-max X  same as omega_norm_max=X (the dashed form wins if both are given).
//   --factorized-1b-intermediates B / --factorized-2b-intermediates B  (B = 0 or 1)
//                   Commutator::FactorizedDoubleCommutator::SetUse_1b_Intermediates / _2b_ (only
//                   used with --factorized). Not given = imsrg++ defaults (both on).
//                   The owner's IMSRG(3f2) recipe (D32): --factorized --hunter-gatherer
//                   --omega-norm-max 0.1 --factorized-1b-intermediates 1 --factorized-2b-intermediates 0.
//
// Output: DIR/trajectory.csv  (header `step,s,E0,norm_eta,norm_omega`, %.17g, step 0 = initial H;
//         norm_eta is |Eta| after the generator update of that step, norm_omega = |Omega.back()|),
//         DIR/result.json (E_HF, E0, steps, final s, final norms, wall time, parameters; with
//         --hunter-gatherer also n_gathers and norm_omega_gathered).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "IMSRG.hh"
#include "rh_common.hh"

namespace rh {

int cmd_flow(const Args& a) {
  SpaceParams sp = space_params(a);
  InteractionParams ip = interaction_params(a);
  const std::string out = a.str("out", ".");
  const bool imsrg3 = a.flag("imsrg3"), n7 = a.flag("n7"), fact = a.flag("factorized");
  const bool hunter = a.flag("hunter-gatherer");
  const std::string gen = a.str("generator", "atan");
  const double smax = a.real("smax", 200.0), ds0 = a.real("ds_0", 0.5), dsmax = a.real("dsmax", 0.5),
               domega = a.real("domega", 0.1),
               omax = a.has("omega-norm-max") ? a.real("omega-norm-max", 0.25) : a.real("omega_norm_max", 0.25),
               eta_crit = a.real("eta_criterion", 1e-6);
  const std::string denom = a.str("denominator_partitioning", "Epstein_Nesbet");

  double t0 = omp_get_wtime();
  auto msp = make_modelspace(sp);
  ModelSpace& ms = *msp;
  BareH b = make_bare_hamiltonian(ms, ip);
  HFMBPT hf(b.H);
  hf.UnFreezeOccupations();
  hf.Solve();
  Operator HNO = hf.GetNormalOrderedH(2);
  HNO.SetModelSpace(ms);  // imsrg++.cc:994-1000 (emax_imsrg == emax)
  if (imsrg3) {
    HNO.ThreeBody.SetMode("pn");
    HNO.SetParticleRank(3);
  }
  const double ehf = hf.EHF;

  IMSRGSolver solver(HNO);
  solver.SetMethod("magnus");
  solver.SetDenominatorPartitioning(denom);
  solver.SetEtaCriterion(eta_crit);
  solver.max_omega_written = 500;
  solver.SetHunterGatherer(hunter);
  solver.SetSmax(smax);
  solver.SetFlowFile("");
  solver.SetDs(ds0);
  solver.SetDsmax(dsmax);
  solver.SetdOmega(domega);
  solver.SetOmegaNormMax(omax);
  BCH::SetUseBruecknerBCH(false);
  Commutator::SetUseIMSRG3(imsrg3);
  Commutator::SetUseIMSRG3N7(imsrg3 && n7);
  if (fact) BCH::SetUseFactorizedCorrection(true);
  // Process-wide switches in imsrg++ (both default true); touched only when given.
  if (a.has("factorized-1b-intermediates"))
    Commutator::FactorizedDoubleCommutator::SetUse_1b_Intermediates(a.flag("factorized-1b-intermediates"));
  if (a.has("factorized-2b-intermediates"))
    Commutator::FactorizedDoubleCommutator::SetUse_2b_Intermediates(a.flag("factorized-2b-intermediates"));
  solver.SetGenerator(gen);

  std::ofstream csv(out + "/trajectory.csv");
  csv << "step,s,E0,norm_eta,norm_omega\n";
  auto row = [&](int istep) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%d,%.17g,%.17g,%.17g,%.17g\n", istep, solver.s,
                  solver.FlowingOps[0].ZeroBody, solver.Eta.Norm(), solver.Omega.back().Norm());
    csv << buf << std::flush;
    std::cerr << buf;
  };

  // ---- copy of IMSRGSolver::Solve_magnus_euler (IMSRGSolver.cc:280-366) ----
  solver.istep = 0;
  solver.generator.Update(solver.FlowingOps[0], solver.Eta);
  if (std::abs(solver.omega_norm_max - solver.norm_domega) < 1e-6) solver.norm_domega += 1e-4;
  solver.Elast = solver.H_0->ZeroBody;
  solver.cumulative_error = 0;
  row(0);
  int last_step = 0;
  int n_gathers = 0;
  bool diverged = false;
  for (solver.istep = 1; solver.s < solver.smax; ++solver.istep) {
    double norm_eta = solver.Eta.Norm();
    if (norm_eta < solver.eta_criterion) break;
    if (norm_eta > 1e12 or std::abs(solver.Elast) > 1e9) {
      diverged = true;
      break;
    }
    double norm_omega = solver.Omega.back().Norm();
    if (norm_omega > solver.omega_norm_max) {
      if (hunter) {
        solver.GatherOmega();
        ++n_gathers;
      } else {
        solver.NewOmega();
      }
      norm_omega = 0;
    }
    if (solver.magnus_adaptive)
      solver.ds = std::min({solver.norm_domega / norm_eta,
                            solver.norm_domega / norm_eta / (norm_omega + 1.0e-9),
                            (solver.omega_norm_max + 1e-5) / norm_eta, solver.ds_max});
    solver.ds = std::min(solver.ds, solver.smax - solver.s);
    solver.s += solver.ds;
    solver.Eta *= solver.ds;
    solver.Omega.back() = BCH::BCH_Product(solver.Eta, solver.Omega.back());
    if ((solver.Omega.size() + solver.n_omega_written) < 2)
      solver.FlowingOps[0] = BCH::BCH_Transform(*solver.H_0, solver.Omega.back());
    else
      solver.FlowingOps[0] = BCH::BCH_Transform(solver.H_saved, solver.Omega.back());
    if (norm_eta < 1.0 and solver.generator.GetType() == "shell-model-atan")
      solver.generator.SetDenominatorCutoff(1e-6);
    solver.generator.Update(solver.FlowingOps[0], solver.Eta);
    row(solver.istep);
    last_step = solver.istep;
    solver.Elast = solver.FlowingOps[0].ZeroBody;
  }
  // ---- end of copy ----
  // Values of the last step, before the final gather erases the hunter.
  const double norm_omega_final = solver.Omega.back().Norm();
  double norm_omega_gathered = 0;
  if (hunter) {  // IMSRGSolver::Solve (IMSRGSolver.cc:236-239): one more gather after the method.
    solver.GatherOmega();
    ++n_gathers;
    norm_omega_gathered = solver.Omega[solver.Omega.size() - 2].Norm();
    // D48 (same line as the D35 patch of IMSRGSolver::Solve): H(s) := H_saved = e^Omega H_0 e^-Omega with
    // the gathered Omega, so E0 and every transformed operator come from one Omega.
    if (!diverged) solver.FlowingOps[0] = solver.H_saved;  // not after a divergence, as the patch
  }

  const bool triples = a.flag("perturbative-triples");
  double d_triples = 0, e0_triples = 0;
  if (triples) {
    d_triples = solver.CalculatePerturbativeTriples();
    e0_triples = solver.FlowingOps[0].ZeroBody + d_triples;
    std::printf("perturbative_triples %s\n", fmt17(d_triples).c_str());
    std::printf("e0_plus_triples %s\n", fmt17(e0_triples).c_str());
  }

  Json j;
  j.str("source", reference_version() + " refharness flow");
  j.integer("emax", ms.GetEmax());
  j.str("reference", sp.reference);
  j.str("generator", gen);
  j.str("variant", imsrg3 ? (n7 ? "IMSRG3-N7" : "IMSRG3") : (fact ? "IMSRG3f2" : "IMSRG2"));
  j.num("smax", smax);
  j.num("EHF", ehf);
  j.num("E0", solver.FlowingOps[0].ZeroBody);
  j.integer("steps", last_step);
  j.num("s_final", solver.s);
  j.num("norm_eta_final", solver.Eta.Norm());
  j.num("norm_omega_final", norm_omega_final);
  j.boolean("diverged", diverged);
  if (hunter || a.has("omega-norm-max") || a.has("factorized-1b-intermediates") ||
      a.has("factorized-2b-intermediates")) {  // only new-style runs, so default result.json is unchanged
    j.boolean("hunter_gatherer", hunter);
    j.num("omega_norm_max", omax);
    j.integer("n_gathers", n_gathers);  // includes the final gather of Solve()
    j.num("norm_omega_gathered", norm_omega_gathered);
    j.boolean("factorized_1b_intermediates", Commutator::FactorizedDoubleCommutator::use_1b_intermediates);
    j.boolean("factorized_2b_intermediates", Commutator::FactorizedDoubleCommutator::use_2b_intermediates);
  }
  if (triples) {
    j.num("perturbative_triples", d_triples);
    j.num("e0_plus_triples", e0_triples);
  }
  j.integer("omp_threads", omp_get_max_threads());
  j.num("wall_time_s", omp_get_wtime() - t0);
  j.write(out + "/result.json");
  return 0;
}

}  // namespace rh
