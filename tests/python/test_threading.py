"""Misuse across threads, released objects, exit hook and Ctrl-C (design/18 sections 4.1, 4.3).

Tests that finalize the Runtime or send signals run in subprocesses; the in-process
tests only provoke refusals, which leave the Runtime intact.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import imsrg
from imsrg import flow


def _env():
    env = dict(os.environ)
    root = str(Path(imsrg.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(script, timeout=180):
    return subprocess.run(
        [sys.executable, "-c", script], env=_env(), capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def h_o16(golden, o16_emax2):
    return imsrg.read_operator_dump(str(golden / "hf" / "emax2_O16_2N" / "HNO.dump"), o16_emax2)


# ------------------------------------------------------------------ threads


def test_finalize_and_reuse_are_refused_while_a_solve_runs(h_o16):
    s = flow.MagnusSolver(h_o16, solver=flow.SolverOptions(smax=6.0))
    started, release = threading.Event(), threading.Event()
    outcome = {}

    def work():
        def on_status(st):
            started.set()
            if st.istep == 1:
                release.wait(30)  # keep the solve in flight (inside the callback)

        try:
            outcome["result"] = s.solve(on_status)
        except BaseException as exc:  # pragma: no cover
            outcome["error"] = exc

    t = threading.Thread(target=work)
    t.start()
    try:
        assert started.wait(30)
        time.sleep(0.2)
        # the Runtime cannot be finalized under a running call, forced or not
        for force in (False, True):
            with pytest.raises(imsrg.RuntimeError, match="call is running|alive"):
                imsrg.Runtime.finalize(force=force)
        # the same solver cannot be used from a second thread (busy flag), ...
        with pytest.raises(imsrg.RuntimeError, match="another thread"):
            s.solve()
        with pytest.raises(imsrg.RuntimeError, match="another thread"):
            s.hamiltonian
        with pytest.raises(imsrg.RuntimeError, match="another thread"):
            s.transform(h_o16)
        # ... while a different solver works in parallel
        other = flow.MagnusSolver(h_o16, solver=flow.SolverOptions(smax=0.5))
        assert other.solve().s == pytest.approx(0.5)
    finally:
        release.set()
        t.join(60)
    assert "error" not in outcome, outcome
    assert outcome["result"].s == pytest.approx(6.0)
    assert imsrg.Runtime.is_initialized()
    assert s.s == pytest.approx(6.0)  # usable again once the call is over


def test_finalize_from_inside_a_callback_is_refused(h_o16):
    s = flow.MagnusSolver(h_o16, solver=flow.SolverOptions(smax=1.0))
    seen = []

    def evil(st):
        try:
            imsrg.Runtime.finalize(force=True)
        except imsrg.RuntimeError as exc:
            seen.append(str(exc))
            raise

    with pytest.raises(imsrg.RuntimeError, match="call is running"):
        s.solve(evil)
    assert seen and imsrg.Runtime.is_initialized()
    # re-entrant use of the same solver from its own callback (same thread) is allowed
    s2 = flow.MagnusSolver(h_o16, solver=flow.SolverOptions(smax=0.5))
    energies = []
    s2.solve(lambda st: energies.append(s2.hamiltonian.zero_body))
    assert len(energies) >= 2


def test_two_threads_on_distinct_solvers_and_operators(h_o16):
    results = []

    def work(smax):
        s = flow.MagnusSolver(h_o16, solver=flow.SolverOptions(smax=smax))
        results.append(s.solve().s)

    ts = [threading.Thread(target=work, args=(x,)) for x in (1.0, 2.0)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    assert sorted(results) == pytest.approx([1.0, 2.0])


# -------------------------------------------------------- released objects

_RELEASED_SCRIPT = """
import imsrg
from imsrg import flow, commutators as comm
ms = imsrg.ModelSpace(1, "He4", "He4", hw=16.0)
op = imsrg.random_operator(ms, seed=1)
other = imsrg.random_operator(ms, seed=2)
solver = flow.MagnusSolver(op, solver=flow.SolverOptions(smax=0.5))
ode = flow.OdeSolver(flow.OdeMethod.Rk4, op)
ws = comm.CommutatorWorkspace(ms)
comm.commutator(op, other)
imsrg.Runtime.finalize(force=True)

calls = {
    "Operator": [
        ("model_space", lambda: op.model_space), ("spec", lambda: op.spec),
        ("residency", lambda: op.residency), ("zero_body", lambda: op.zero_body),
        ("zero_body=", lambda: setattr(op, "zero_body", 1.0)),
        ("one_body", lambda: op.one_body), ("one_body_view", lambda: op.one_body_view(writable=True)),
        ("two_body_block", lambda: op.two_body_block(0)), ("sync_to_host", lambda: op.sync_to_host()),
        ("sync_to_device", lambda: op.sync_to_device()), ("set_zero", lambda: op.set_zero()),
        ("copy", lambda: op.copy()), ("norm", lambda: op.norm()),
        ("one_body_norm", lambda: op.one_body_norm()), ("two_body_norm", lambda: op.two_body_norm()),
        ("three_body_norm", lambda: op.three_body_norm()),
        ("normal_ordered", lambda: op.normal_ordered()), ("undo_normal_order", lambda: op.undo_normal_order()),
        ("+", lambda: op + other), ("-", lambda: op - other), ("neg", lambda: -op),
        ("*", lambda: op * 2.0), ("rmul", lambda: 2.0 * op), ("/", lambda: op / 2.0),
        ("+=", lambda: op.__iadd__(other)), ("-=", lambda: op.__isub__(other)),
        ("*=", lambda: op.__imul__(2.0)), ("/=", lambda: op.__itruediv__(2.0)),
        ("other + op", lambda: other + op), ("copy.copy", lambda: __import__("copy").copy(op)),
        ("commutator", lambda: comm.commutator(op, other)),
        ("edit", lambda: op.edit().__enter__()),
    ],
    "MagnusSolver": [
        ("solve", lambda: solver.solve()), ("transform", lambda: solver.transform(op)),
        ("hamiltonian", lambda: solver.hamiltonian), ("eta", lambda: solver.eta),
        ("omegas", lambda: solver.omegas), ("s", lambda: solver.s),
        ("num_gathers", lambda: solver.num_gathers), ("solver_options", lambda: solver.solver_options),
        ("generator_options", lambda: solver.generator_options),
        ("set_smax", lambda: solver.set_smax(1.0)),
        ("set_generator", lambda: solver.set_generator(flow.GeneratorOptions())),
        ("set_solver_options", lambda: solver.set_solver_options(flow.SolverOptions())),
    ],
    "OdeSolver": [
        ("solve", lambda: ode.solve()), ("hamiltonian", lambda: ode.hamiltonian), ("eta", lambda: ode.eta),
        ("s", lambda: ode.s), ("num_operators", lambda: ode.num_operators),
        ("flowing_operator", lambda: ode.flowing_operator(0)),
        ("set_generator", lambda: ode.set_generator(flow.GeneratorOptions())),
        ("set_solver_options", lambda: ode.set_solver_options(flow.SolverOptions())),
    ],
    "CommutatorWorkspace": [("model_space", lambda: ws.model_space),
                            ("commutator", lambda: comm.commutator(other, other, workspace=ws))],
}
import sys
bad = 0
for cls, items in calls.items():
    for name, f in items:
        try:
            f()
        except imsrg.StaleHostError:
            pass
        except BaseException as e:
            bad += 1
            print("WRONG", cls, name, type(e).__name__, e)
        else:
            bad += 1
            print("NO ERROR", cls, name)
# every public member of the bound classes is covered above
import re
covered = {n.rstrip("=") for n, _ in calls["Operator"]}
for cls, obj in (("Operator", imsrg.Operator), ("MagnusSolver", flow.MagnusSolver),
                 ("OdeSolver", flow.OdeSolver), ("CommutatorWorkspace", comm.CommutatorWorkspace)):
    names = {n for n in dir(obj) if not n.startswith("_")}
    got = {n.rstrip("=") for n, _ in calls[cls]}
    missing = names - got
    if missing:
        bad += 1
        print("UNCOVERED", cls, sorted(missing))
print("bad", bad)
"""


def test_every_member_of_a_released_object_raises_stale_host_error():
    out = _run(_RELEASED_SCRIPT)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "bad 0", out.stdout


# ------------------------------------------------------------------- exit hook

_EXIT_SCRIPT = """
import imsrg
from imsrg import commutators as comm, flow
ms = imsrg.ModelSpace(1, "He4", "He4", hw=16.0)   # module level, outlives everything
a = imsrg.random_operator(ms, seed=1)
h = imsrg.random_operator(ms, seed=2)
z = comm.commutator(a, h)                         # fills the workspace cache (Kokkos buffers)
ws = comm.CommutatorWorkspace(ms)                 # and an explicit workspace
solver = flow.MagnusSolver(h, solver=flow.SolverOptions(smax=0.2))
solver.solve()
print("done")
"""


def test_exit_with_live_model_space_workspaces_and_solvers():
    out = _run(_EXIT_SCRIPT)
    assert out.returncode == 0, (out.returncode, out.stderr[-3000:])
    assert out.stdout.strip().splitlines()[-1] == "done"
    assert "terminate" not in out.stderr and "abort" not in out.stderr.lower()


def test_deleting_the_model_space_frees_the_cached_workspace():
    imsrg.commutators.clear_workspace_cache()
    base = imsrg._core.live_objects()
    ms = imsrg.ModelSpace(1, "He4", "He4", hw=16.0)
    a, h = imsrg.random_operator(ms, seed=1), imsrg.random_operator(ms, seed=2)
    imsrg.commutators.commutator(a, h)
    assert imsrg._core.live_objects() == base + 3  # a, h and the cached workspace
    del a, h
    assert imsrg._core.live_objects() == base + 1  # the workspace stays cached while ms lives
    del ms
    assert imsrg._core.live_objects() == base  # `del ms` dropped the workspace and the model space


# ---------------------------------------------------------------------- Ctrl-C

_SIGINT_SOLVE = """
import imsrg, sys
from imsrg import flow
ms = imsrg.ModelSpace(2, "O16", "O16", hw=16.0)
h = imsrg.read_operator_dump(sys.argv[1], ms)
s = flow.MagnusSolver(h, solver=flow.SolverOptions(smax=500.0))
def on_status(st):
    if st.istep == 1:
        print("ready", flush=True)
try:
    s.solve(on_status)
    print("finished")
except KeyboardInterrupt:
    print("interrupted", s.s > 0, flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ctrl_c_interrupts_a_solve(golden):
    dump = str(golden / "hf" / "emax2_O16_2N" / "HNO.dump")
    p = subprocess.Popen(
        [sys.executable, "-c", _SIGINT_SOLVE, dump], env=_env(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        assert p.stdout.readline().strip() == "ready"
        p.send_signal(signal.SIGINT)
        out, err = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert out.strip().splitlines()[-1] == "interrupted True", (out, err[-2000:])
    assert p.returncode == 0


_SIGINT_RUN = """
import imsrg, sys
cfg = imsrg.load_config(file=sys.argv[1])
print("ready", flush=True)
imsrg.pipeline.run(cfg)
print("finished")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ctrl_c_stops_a_pipeline_run(me2j_file, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        f'''[basis]
emax = 2
hw = 16.0
[reference]
nucleus = "He4"
[interaction.two_body]
file = "{me2j_file}"
e1max = 14
e2max = 28
lmax = 14
[hartree_fock]
mixing = "linear"
[imsrg.step]
smax = 5000.0
eta_criterion = 1e-30
[output]
directory = "{tmp_path / 'out'}"
'''
    )
    p = subprocess.Popen(
        [sys.executable, "-c", _SIGINT_RUN, str(cfg)], env=_env(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        assert p.stdout.readline().strip() == "ready"
        time.sleep(2.5)
        assert p.poll() is None, "the run ended before the signal (too fast for this test)"
        t0 = time.time()
        p.send_signal(signal.SIGINT)
        p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode in (-signal.SIGINT, 130, 1) and p.returncode != 0
    assert time.time() - t0 < 30  # stopped promptly, not after the run
