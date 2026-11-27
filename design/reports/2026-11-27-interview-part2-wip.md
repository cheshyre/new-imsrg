# Part-2 interview (2026-11-27 JST) — working notes, folded into the final docs at the end

Answers so far (owner chose the recommendation unless noted):
* AL-A-Q6: emulate imsrg++'s tensor 1B norm (NaN -> 40-term BCH) in v1 only; v2/v3 use
  (2j_p+1)(2j_q+1)/(2L+1); Phase 0 checks which archived J=5 logs hit it.
* GPU BLAS: rocBLAS stays the hip default; Kokkos Kernels a tested option.
* AL-E-Q2: few-ulp (4 ulp) tolerance in the five 3B-generator Bitwise tests on device builds only.
* Fork tensor goldens (1.7 MB): keep.
* AL-D1/C2: store H_0 (emax 10) + NAT C in checkpoints, checkpoint on SIGTERM; before v2/v3 flows.
* AL-E-Q5: kokkos-hip with OpenMP host space.
* AL-E-Q3: frontier-cpu* presets switch to serial libsci + IMSRG_BLAS_CONCURRENT (after the
  Python two-thread test passes).
* Owner: "Interview me on all questions." for the rest; answers:
  C1 e2max cut later; C3 keep stable_sort; C4 v2/v3 use domega 0.1; C5 fix doc, accept rest;
  A-Q1 store random inputs in every golden script (filler); A-Q2 enough for the farm;
  A-Q3 keep explicit sum, devel bug in design/30 §6; A-Q4 keep rewrite rule;
  A-Q5 storage = plain Edmonds, document in design/15; B2 keep rule + mpmath, into design/30 §2;
  B3 analytic q->0 limit on the default path, NaN under compat (filler); B4 confirmed;
  D2/D3 confirmed, verify D3 in Phase 0; E-Q1 KK compile check in ci.sh frontier-hip;
  E-Q4 one flow per GCD; F2 fork names under compat (driver and farm); F3 refit after
  Phase 0 timing; F4 add a per-item log (name, times, BCH terms, Omega norm).
* Order: overlap phases; v1 must be complete, v3 may be partial.
* Scale: owner (verbatim): "Not a must this weekend. Run each variant as its own ~600-1,200-node
  job; document how 19,320 ranks would be reached. Use the GPU resources efficiently with what
  you have available."
* Paused (owner) before: v2/v3 acceptance, miss-1e-4 policy, results location, never-do list,
  logistics (C), write-up (D). Owner asked for: v1/v2/v3 differences, node-hour estimates.
* NAT sort (owner, verbatim): "To be clear NAT orbitals should be sorted by |occupation| not
  signed occupation. Verify that this is what is done in v2 and v3." Checked: it is NOT. The
  default is `mbpt::NatSorting::kSignedDescending` (imsrg-devel `DiagonalizeRho`); |occupation|
  (`kAbsDescending`) is only switched on by the compat option `nat_sort_by_abs_occupation`
  (src/imsrg/driver/config/compat.h:40, driver/pipeline/hamiltonian.cc:86). To do (Phase 0,
  before any v2/v3 flow): |occupation| becomes the default NAT order for every run, taken out of
  the compat profile; signed order stays only as an explicit option for imsrg-devel comparisons;
  record the devel behaviour as an old-code bug in design/30 §6.
