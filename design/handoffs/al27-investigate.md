# Hand-off: Lead E, 27Al weekend investigation of the variants

Branch: feature/al27-investigate-variants (hub), base develop 59eb116. Node chain: imsrg-e
(`IMSRG_SLURM_JOBNAME=imsrg-e`). Deadline: report and push by Mon 2026-11-30 07:00 EST.
Runs (never in git): `$IMSRG_WORK/runs/al27/investigate/` (configs in `configs/`).
Jobs: design/benchmarks/al27-jobs.md, section "Lead E (investigation)".

## Tasks
- [ ] 1. v2 - v1 per compat sub-option: debug keys `run.compat_enable` / `run.compat_disable`
  (implementer, branch feature/al27-investigate-compat-overrides), frontier-hip build, farms with
  v2 Omegas and one sub-option at a time (136 items = 34 columns x q 0/150/450/900, as the
  compatop farm 6016893), both Hamiltonians; table in al27-variants.md (or
  al27-variants-decomposition.md).
- [ ] 2. ODE IMSRG(2) with FM0p_150 + FPhipp0p_0 for DN2LOGO (job 6018030, config
  `investigate/configs/DN2LOGO-ode-ops.toml` = scan DN2LOGO-ode.toml + the two operators, binary of
  the N2LOsat ODE-ops run `imsrg.gpu-20261127-202543`), compare with v2/v3.
- [ ] 3. Tensor operator in the ODE check: NOT supported (`flow::OdeSolver` throws InputError for a
  non-scalar operator, src/imsrg/flow/ode.cc:214-220); question WE-E-1; al27-variants.md 5.2 says
  "a tensor would make the ODE flow too slow", which is wrong (it is unsupported) -> correct it.
- [ ] 4. WE-R-1 flow file continuous across restarts (implementer, branch
  feature/al27-investigate-flowfile-restart).

## Decisions
* The new compat keys are not hashed (FlowConfigHash / PreparationHash): no compat option changes
  H_0 or the flow of the scalar H; domega (fork_step_defaults) is hashed through the resolved
  imsrg.step.domega. Existing checkpoints stay valid.
