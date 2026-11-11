// rh_flow.cpp -- `refharness flow`: single-reference Magnus IMSRG run with a full-precision trajectory.
//
//   refharness flow emax=4 reference=O16 [generator=atan] [smax=200] [ds_0=0.5] [dsmax=0.5]
//       [domega=0.1] [omega_norm_max=0.25] [eta_criterion=1e-6] [--imsrg3 [--n7]] [--factorized]
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
//
// Output: DIR/trajectory.csv  (header `step,s,E0,norm_eta,norm_omega`, %.17g, step 0 = initial H;
//         norm_eta is |Eta| after the generator update of that step, norm_omega = |Omega.back()|),
//         DIR/result.json (E_HF, E0, steps, final s, final norms, wall time, parameters).
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
  const std::string gen = a.str("generator", "atan");
  const double smax = a.real("smax", 200.0), ds0 = a.real("ds_0", 0.5), dsmax = a.real("dsmax", 0.5),
               domega = a.real("domega", 0.1), omax = a.real("omega_norm_max", 0.25),
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
  solver.SetHunterGatherer(false);
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
      solver.NewOmega();
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
  j.num("norm_omega_final", solver.Omega.back().Norm());
  j.boolean("diverged", diverged);
  j.integer("omp_threads", omp_get_max_threads());
  j.num("wall_time_s", omp_get_wtime() - t0);
  j.write(out + "/result.json");
  return 0;
}

}  // namespace rh
