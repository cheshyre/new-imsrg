# Hand-off: Lead B, 27Al weekend Phase 0 (defaults, restarts, farm log)

(Kept in the branch: the agent hook does not allow writes to
`$IMSRG_REFERENCE_ROOT/plans/` from an agent clone.)

Branch: feature/al27-defaults-restart (hub). Base develop e5ecee9. Node chain: imsrg-b.
Brief: design/NEXT_SESSION_PROMPT.md "Brief B".

## Decisions (key names)
* `natural_orbitals.sort` = "abs_occupation" (default) | "signed_occupation" (imsrg-devel
  comparisons). `mbpt::ComputeNaturalOrbitals` default -> kAbsDescending.
  CompatOptions::nat_sort_by_abs_occupation removed (now default).
* Tensor 1B norm: `op::TensorOneBodyNorm {kDegeneracyWeighted (default, (2jp+1)(2jq+1)/(2L+1)),
  kImsrgppInteger}`; `flow::BchOptions::tensor_one_body_norm`; compat option
  `tensor_one_body_norm_imsrgpp` (fork profile on).
* Restart: checkpoint dir gets `prepared/` (H_0 IMSRG basis, C, occupations, HF scalars,
  preparation hash); restart loads it and skips PrepareHamiltonian.
  `checkpoint.prepared_from = "<other ckpt dir>"` starts a fresh flow from another run's
  prepared/ (v3 from v2), refused unless PreparationHash matches.
  SIGTERM: handler sets flag; per-step allreduce; checkpoint + typed error; run_flow.sh
  traps TERM and forwards it.
* Farm log per item; file names `<stem><name>.snt` under compat, `<stem>_<name>.snt` otherwise.

## Tasks
- [x] T1 nat-sort + tensor norm (t1 head 90f14ac, merged c328be2; own CI PASS)
- [x] T2 farm log + file names (t2 head d545053, merged 5c553de; own CI PASS)
- [x] T3 prepared-H0 restart + SIGTERM (t3 head c952615, merged 5170aa0; report pending)
- [x] T4 v2/v3 configs (all four `--validate` OK after merge); run_flow.sh done by T3
- [ ] CI frontier-cpu on the merged head, ctest -L slow, report

## Notes
* First merged CI run (2026-11-26 23:xx) FAILED only by timeouts of driver_mpi_2ranks /
  driver_mpi_it11_2ranks while 3 worker CIs shared the node and the allocation was ending;
  `ctest -L mpi` alone on the fresh node: 22/22 pass incl. the two slow MPI tests.
* The tensor writer (devel format) still thresholds with imsrg++'s integer norm (falls back
  to the weighted one when NaN) so the devel-format golden stays byte-equal (T1).
* The integer degeneracy is already negative at emax 4 (g9/2 -> s1/2, L = 4), not first
  at emax 10 as AL-A-Q6 says (T1 test).
