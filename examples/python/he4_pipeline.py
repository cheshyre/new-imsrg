"""A scripted IMSRG(2) pipeline in about 20 lines of imsrg calls (design/18 section 2.9).

Read the 2N interaction, build the Hamiltonian H = T_rel + V, solve Hartree-Fock,
normal order, run the Magnus flow with the atan generator, and transform the
point-proton radius operator with the accumulated Omega. It is the scripted
equivalent of ``imsrg --config he4.toml`` (the same steps the C++ driver runs, which
``imsrg.pipeline.run`` calls directly); ``tests/python/test_example_pipeline.py``
checks that both give the same E0.

Usage::

    python examples/python/he4_pipeline.py VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz

Operators are device-resident after every library call. Numpy views of their data
need ``op.sync_to_host()`` first (``imsrg.StaleHostError`` otherwise on GPU builds);
this script only reads scalars (``zero_body``, energies), which never need it.
"""

import argparse

import imsrg
from imsrg import flow, hf, io
from imsrg import operators as ops


def run(me2j_file, *, emax=2, hw=16.0, nucleus="He4", mass_number=4, proton_number=2,
        file_e1max=14, file_e2max=28, file_lmax=14, smax=200.0, verbose=False):
    """Runs the pipeline and returns a dict with E_HF, E0, MP2 and the Rp2 values (fm^2)."""
    imsrg.Runtime.ensure()  # optional: imsrg.Runtime(num_threads=8) to choose the threads
    ms = imsrg.ModelSpace(emax, nucleus, nucleus, hw=hw)

    # Hamiltonian: the file holds the 2N interaction only, the kinetic term is added here.
    h = io.read_me2j(me2j_file, ms, file_e1max=file_e1max, file_e2max=file_e2max, file_lmax=file_lmax)
    h += ops.t_rel(ms, mass_number)
    rp2 = ops.rp2(ms, mass_number, proton_number)  # observable, built in the oscillator basis

    # Hartree-Fock, then the Hamiltonian and the observable in the HF basis, normal ordered.
    res = hf.solve(h, mixing="linear")
    if verbose:
        print(f"E_HF = {res.e_hf:.8f} MeV ({res.iterations} iterations, converged={res.converged})")
    hno = res.normal_ordered_hamiltonian(h)
    rp2_hf = res.transform(rp2).normal_ordered()

    # IMSRG(2) flow: Magnus expansion with the atan generator.
    solver = flow.MagnusSolver(
        hno,
        solver=flow.SolverOptions(smax=smax),
        generator=flow.generator("atan"),
    )

    def log(st):
        if verbose and st.istep % 10 == 0:
            print(f"  s = {st.s:9.4f}   E0 = {st.e0:.8f}   |eta| = {st.norm_eta:.3e}")

    result = solver.solve(on_status=log)
    rp2_evolved = solver.transform(rp2_hf)  # e^Omega Rp2 e^-Omega
    if verbose:
        print(f"E0 = {result.e0:.8f} MeV after {result.steps} steps ({result.reason.name})")
        print(f"<Rp2> = {rp2_hf.zero_body:.6f} (HF) -> {rp2_evolved.zero_body:.6f} fm^2 (IMSRG)")
    return {
        "e_hf": res.e_hf,
        "e0": result.e0,
        "steps": result.steps,
        "converged": result.reason == flow.StopReason.Converged,
        "rp2_reference": rp2_hf.zero_body,
        "rp2_imsrg": rp2_evolved.zero_body,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("me2j_file")
    p.add_argument("--emax", type=int, default=2)
    p.add_argument("--hw", type=float, default=16.0)
    args = p.parse_args(argv)
    run(args.me2j_file, emax=args.emax, hw=args.hw, verbose=True)


if __name__ == "__main__":
    main()
