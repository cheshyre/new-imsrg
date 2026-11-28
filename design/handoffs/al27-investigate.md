# Hand-off: Lead E, 27Al weekend investigation of the variants

Branch: feature/al27-investigate-variants (hub), base develop 59eb116. Node chain: imsrg-e
(`IMSRG_SLURM_JOBNAME=imsrg-e`). Deadline: report and push by Mon 2026-11-30 07:00 EST.
Runs (never in git): `$IMSRG_WORK/runs/al27/investigate/` (configs in `configs/`, decomposition
farms in `compat/`, generated tables in `md/`, build and CI logs in `logs/`).
Jobs: design/benchmarks/al27-jobs.md, section "Lead E (investigation)": 6018030 (0.56 node-hours),
6018194 (4.95 node-hours); total 5.5 of 40 node-hours.

## Tasks
- [x] 1. v2 - v1 per compat sub-option. Debug keys `run.compat_enable` / `run.compat_disable`
  (merged from feature/al27-investigate-compat-overrides 2dde92b; design/16 V-34, design/50 sec. 5),
  `tools/al27/make_compat_decomp.sh` (24 directories), farm 6018194 (v2 Omegas, 136 items x 12
  variants x 2 Hamiltonians), `tools/al27/compat_decomp.py` (tested). Result in al27-variants.md
  4.3 and the new design/benchmarks/al27-variants-decomposition.md: M15-1 is the largest
  operator-side part at every rank L >= 1 (1b only), AL-F1 adds 1b + 2b for odd L, cross term
  1-1.5 %, the D72 norm / 9j / writer do not matter; controls bit-identical.
- [x] 2. ODE IMSRG(2) with FM0p_150 + FPhipp0p_0 for DN2LOGO (job 6018030): v3 0.08-0.4 %, v2 4-8 %
  (median, dominant elements); al27-variants.md 5.2; `tools/al27/ode_check.py` (tested; reproduces
  the N2LOsat table).
- [x] 3. Tensor operator in the ODE check: not supported (`flow::OdeSolver` rejects non-scalar
  operators, src/imsrg/flow/ode.cc:214-220); WE-E-1; 5.2 text corrected (it gave cost as reason).
- [x] 4. WE-R-1 fixed (merged from feature/al27-investigate-flowfile-restart 8c713a7 + baca471
  field-count check): FlowLog keeps rows (stage, istep) <= checkpoint on restart; tests
  `Checkpoint.RestartedFlowFileIsContinuous`, `Checkpoint.FlowFileTruncatedUnlessRestart`.
- [x] Weekend report (design/reports/2026-11-27-weekend.md) and design/30 row AL-A-Q6 updated.
- CI: `CI PASS frontier-cpu` at baca471 (all src changes) and at d22241a (branch head, 2026-11-27
  ~23:00 EST). `pytest tools/al27`: 46 passed.

## Decisions
* The new compat keys are not hashed (FlowConfigHash / PreparationHash): no compat option changes
  H_0 or the flow of the scalar H; domega (fork_step_defaults) is hashed through the resolved
  imsrg.step.domega. Existing checkpoints stay valid.
* Farmed options: writer, D72 norm, M15-1, 9j threshold, AL-F1 (fork_step_defaults acts on the flow
  only, fork_operator_file_names on names only).
* The per-column tables of Lead D's decomposition farm 6016893 moved from al27-variants.md 4.2 to
  al27-variants-decomposition.md section 4 (size limit; al27-variants.md is 91 kB). Lead D's
  template placeholder DECOMPTABLES (handoffs/al27-variants.md) no longer has a target in
  al27-variants.md: regenerate those tables into the companion file.

## Notes for the next session
* The M15-1 two-body differences are exactly 0 in every item: consistent with the bug acting in
  the final re-normal-ordering of the evolved operator to the core (the bare NREFT operators are
  one-body, so the bare normal ordering has nothing to contract); not verified in the code.
* The ODE flow could carry tensors only after WE-E-1 is implemented (owner decision).
