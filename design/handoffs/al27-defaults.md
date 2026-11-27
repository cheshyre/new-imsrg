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
- [ ] T1 nat-sort + tensor norm (implementer, branch feature/al27-defaults-restart-t1)
- [ ] T2 farm log + file names (implementer, branch feature/al27-defaults-restart-t2)
- [ ] T3 prepared-H0 restart + SIGTERM (implementer opus, branch feature/al27-defaults-restart-t3)
- [ ] T4 v2/v3 configs + run_flow.sh (lead)
- [ ] merge, CI frontier-cpu, ctest -L slow, report
