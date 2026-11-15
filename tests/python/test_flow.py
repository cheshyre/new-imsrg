"""flow: generators, Magnus (incl. hunter-gatherer), ODE solvers, BCH transform and
product (incl. the factorized 3f2 options) against the imsrg++ goldens."""

import csv
import json
import math
import threading
from pathlib import Path

import pytest

import imsrg
from imsrg import flow

O16 = "commutator/emax2_O16"
BCH = "factorized/bch_emax2_O16"

# (golden file, transform?, a, scale_a, b, scale_b, factorized): tests/golden/factorized README
BCH_CASES = [
    ("transform_H__eta_white_x0.0929__d32", True, "H", 1.0, "eta_white", 0.0929, True),
    ("transform_H__eta_white_x1.86__d32", True, "H", 1.0, "eta_white", 1.86, True),
    ("transform_H__eta_white_x1.86__none", True, "H", 1.0, "eta_white", 1.86, False),
    ("product_eta_white_x0.0464__random_A_x0.000415__d32", False, "eta_white", 0.0464, "random_A", 0.000415, True),
    ("product_eta_white_x0.0929__random_A_x0.00829__d32", False, "eta_white", 0.0929, "random_A", 0.00829, True),
    ("product_eta_white_x0.0929__random_A_x0.00829__none", False, "eta_white", 0.0929, "random_A", 0.00829, False),
]


@pytest.fixture(scope="module")
def o16(golden):
    out = {"H": imsrg.read_operator_dump(str(golden / O16 / "H.dump"))}
    ms = out["H"].model_space
    for n in ("eta_white", "eta_atan", "random_A"):
        out[n] = imsrg.read_operator_dump(str(golden / O16 / f"{n}.dump"), ms)
    return out


def _rel_dev(a, b):
    return (a - b).norm() / max(1.0, b.norm())


# ---------------------------------------------------------------- generators


@pytest.mark.parametrize("kind", ["white", "atan"])
def test_generator_matches_imsrgpp(o16, kind):
    eta = flow.compute_generator(o16["H"], flow.generator(kind))
    assert eta.spec.hermiticity == imsrg.Hermiticity.AntiHermitian
    assert _rel_dev(eta, o16[f"eta_{kind}"]) < 1e-10


def test_generator_functions_and_options():
    G = flow.GeneratorKind
    assert flow.generator_function(G.White, 2.0, 4.0) == pytest.approx(0.5)
    assert flow.generator_function(G.Atan, 2.0, 4.0) == pytest.approx(0.5 * math.atan(1.0))
    assert flow.generator_function(G.Wegner, 2.0, 4.0) == pytest.approx(8.0)
    assert flow.generator_function(G.ImaginaryTime, 2.0, -4.0) == pytest.approx(-2.0)
    with pytest.raises(imsrg.NotImplementedError):
        flow.generator_function(G.HartreeFock, 1.0, 1.0)
    assert flow.floor_denominator(0.0, 1e-6) == 1e-6
    assert flow.floor_denominator(-1e-9, 1e-6) == -1e-6  # D21: the sign is kept
    assert flow.floor_denominator(-3.0, 1e-6) == -3.0
    g = flow.generator("shell-model-atan", partitioning="moller_plesset", denominator_delta_orbit=3)
    assert g.kind == G.ShellModelAtan and g.partitioning == flow.Partitioning.MollerPlesset
    assert g.denominator_delta_orbit == 3
    g.denominator_delta_orbit = None
    assert g.denominator_delta_orbit is None and g.denominator_cutoff == 1e-6
    with pytest.raises(ValueError, match="GeneratorKind"):
        flow.generator("nonsense")
    bad = flow.generator("white", denominator_cutoff=-1.0)
    with pytest.raises(imsrg.InputError):
        flow.compute_generator(imsrg.random_operator(imsrg.ModelSpace(1, "He4", "He4", hw=16.0)), bad)


def test_denominators(o16):
    h = o16["H"]
    d = flow.Denominators(h, flow.generator("white", partitioning="MollerPlesset"))
    for i, j in ((0, 3), (1, 5), (2, 7)):
        assert d.one_body(i, j) == pytest.approx(-d.one_body(j, i))
    f = h.sync_to_host().one_body
    assert d.one_body(0, 3) == pytest.approx(f[0, 0] - f[3, 3]) or abs(d.one_body(0, 3)) == 1e-6
    assert math.isfinite(d.monopole(0, 1)) and d.two_body(0, 0, 0) != 0.0


# ----------------------------------------------------------------------- BCH


@pytest.mark.parametrize("case", BCH_CASES, ids=[c[0] for c in BCH_CASES])
def test_bch_matches_imsrgpp(golden, o16, case):
    name, transform, a, sa, b, sb, fact = case
    ref = imsrg.read_operator_dump(str(golden / BCH / f"{name}.dump"), o16["H"].model_space)
    x, y = sa * o16[a], sb * o16[b]
    opts = flow.BchOptions(factorized=fact)
    if transform:
        z, report = flow.bch_transform_with_report(x, y, bch=opts)
        assert report.nested_commutators >= 1
        assert (report.factorized_corrections > 0) == fact
        assert (flow.bch_transform(x, y, bch=opts) - z).norm() == 0.0
    else:
        z = flow.bch_product(x, y, bch=opts)
    assert _rel_dev(z, ref) < 1e-10


def test_bch_options_and_trivial_cases(o16):
    o = flow.BchOptions()
    assert (o.transform_threshold, o.product_threshold, o.max_iterations, o.factorized) == (1e-9, 1e-4, 40, False)
    assert o.factorized_terms.one_body_intermediates and not o.factorized_terms.two_body_intermediates
    h, eta = o16["H"], o16["eta_white"]
    zero = imsrg.Operator(h.model_space, imsrg.OperatorSpec(hermiticity=imsrg.Hermiticity.AntiHermitian))
    assert (flow.bch_transform(h, zero) - h).norm() == 0.0  # e^0 H e^0 = H
    # with all factorized topologies off the 3f2 series is the plain one
    off = flow.BchOptions(factorized=True, factorized_terms=imsrg.commutators.FactorizedOptions(one_body_intermediates=False))
    x = 0.05 * eta
    assert (flow.bch_transform(h, x, bch=off) - flow.bch_transform(h, x)).norm() < 1e-12 * h.norm()


# -------------------------------------------------------------------- Magnus


def test_magnus_flow_on_o16(o16):
    h0 = o16["H"]
    seen = []
    s = flow.MagnusSolver(h0, solver=flow.SolverOptions(smax=3.0), generator=flow.generator("atan"))
    assert s.s == 0.0 and s.eta.norm() == 0.0 and len(s.omegas) == 1
    r = s.solve(on_status=seen.append)
    assert isinstance(seen[0], flow.FlowStatus) and seen[0].istep == 0 and seen[0].s == 0.0
    assert [st.istep for st in seen] == list(range(r.steps + 1))
    e0 = [st.e0 for st in seen]
    assert all(b <= a + 1e-12 for a, b in zip(e0, e0[1:])) and e0[-1] < e0[0] - 1.0
    assert r.reason == flow.StopReason.SmaxReached and r.s == pytest.approx(3.0)
    assert r.e0 == pytest.approx(e0[-1]) and s.hamiltonian.zero_body == pytest.approx(r.e0)
    assert s.s == pytest.approx(3.0) and s.num_gathers == 0
    # the accumulated Omega reproduces H(s) up to the BCH truncation, and eta is anti-Hermitian
    assert s.transform(h0).zero_body == pytest.approx(r.e0, abs=1e-3)
    assert s.eta.spec.hermiticity == imsrg.Hermiticity.AntiHermitian
    # copies: accessors return device-current operators
    assert s.hamiltonian.residency == imsrg.Residency.DeviceCurrent
    assert s.generator_options.kind == flow.GeneratorKind.Atan
    assert s.solver_options.smax == 3.0
    # a second stage continues the flow
    s.set_smax(4.0)
    s.set_generator(flow.generator("white"))
    r2 = s.solve()
    assert r2.s == pytest.approx(4.0) and r2.e0 <= r.e0 + 1e-9


def test_magnus_callback_exception_propagates(o16):
    s = flow.MagnusSolver(o16["H"], solver=flow.SolverOptions(smax=2.0))

    def stop(st):
        if st.istep == 2:
            raise KeyError("stop here")

    with pytest.raises(KeyError, match="stop here"):
        s.solve(stop)
    assert 0.0 < s.s < 2.0  # stopped at a step boundary, the solver is consistent
    r = s.solve()  # and usable: continues to smax
    assert r.s == pytest.approx(2.0)


def test_solver_option_validation(o16):
    with pytest.raises(imsrg.InputError):
        flow.MagnusSolver(o16["H"], solver=flow.SolverOptions(dsmax=0.0))
    s = flow.MagnusSolver(o16["H"])
    with pytest.raises(imsrg.InputError):
        s.set_solver_options(flow.SolverOptions(omega_norm_max=-1.0))
    opts = flow.SolverOptions()
    assert (opts.smax, opts.dsmax, opts.domega, opts.omega_norm_max, opts.eta_criterion) == (200.0, 0.5, 0.1, 0.25, 1e-6)
    assert opts.magnus_adaptive and not opts.hunter_gatherer


def test_solve_releases_the_gil(o16):
    s = flow.MagnusSolver(o16["H"], solver=flow.SolverOptions(smax=4.0))
    done = threading.Event()
    t = threading.Thread(target=lambda: (s.solve(), done.set()))
    ticks = 0
    t.start()
    while not done.is_set():
        ticks += 1
        done.wait(0.001)
    t.join()
    assert ticks > 20  # the main thread ran while the solver worked


def test_hunter_gatherer_he4_matches_imsrgpp(golden, he4_emax2):
    gdir = golden / "e2e" / "IT-2" / "hunter_gatherer" / "He4_emax2_omax_0.10"
    gold = json.loads((gdir / "result.json").read_text())
    rows = list(csv.DictReader((gdir / "trajectory.csv").open()))
    hno = he4_emax2[3]
    s = flow.MagnusSolver(
        hno,
        solver=flow.SolverOptions(omega_norm_max=0.1, hunter_gatherer=True),
        generator=flow.generator("atan"),
    )
    got = []
    r = s.solve(lambda st: got.append((st.s, st.e0, st.norm_eta, st.norm_omega)))
    assert r.reason == flow.StopReason.Converged
    assert r.steps == gold["steps"] and r.num_gathers == gold["n_gathers"] == s.num_gathers
    assert r.e0 == pytest.approx(gold["E0"], abs=1e-9)
    assert len(s.omegas) == 2 and s.omegas[-1].norm() == 0.0  # {gatherer, zero hunter}
    assert s.omegas[0].norm() == pytest.approx(gold["norm_omega_gathered"], abs=1e-6)
    assert len(got) == len(rows)
    for g, row in zip(got, rows):
        for val, key in zip(g, ("s", "E0", "norm_eta", "norm_omega")):
            assert val == pytest.approx(float(row[key]), abs=1e-9)


# ----------------------------------------------------------------------- ODE


def test_ode_solvers_agree_with_magnus(o16):
    h0 = o16["H"]
    opts = flow.SolverOptions(smax=5.0)
    gen = flow.generator("atan")
    e0 = {}
    for name in ("flow_rk4", "flow"):
        o = flow.ode_solver(name, h0, solver=opts, generator=gen)
        calls = []
        r = o.solve(calls.append)
        e0[name] = r.e0
        assert o.s == pytest.approx(5.0) and o.hamiltonian.zero_body == pytest.approx(r.e0)
        assert calls and calls[0].istep == 0
        if name == "flow_rk4":
            assert r.steps == 10  # ds = dsmax = 0.5
            assert len(calls) == r.steps + 1
    mag = flow.MagnusSolver(h0, solver=opts, generator=gen).solve()
    assert e0["flow_rk4"] < h0.zero_body - 1.0
    assert e0["flow_rk4"] == pytest.approx(mag.e0, abs=5e-2)
    assert e0["flow"] == pytest.approx(mag.e0, abs=5e-2)


def test_ode_flowing_operators(o16):
    h0 = o16["H"]
    r2 = imsrg.random_operator(h0.model_space, seed=3)
    o = flow.OdeSolver(flow.OdeMethod.Rk4, h0, operators=[h0.copy(), r2], solver=flow.SolverOptions(smax=2.0))
    assert o.num_operators == 2
    o.solve()
    # an operator that is H itself flows exactly like H
    assert (o.flowing_operator(0) - o.hamiltonian).norm() < 1e-9 * o.hamiltonian.norm()
    assert (o.flowing_operator(1) - r2).norm() > 0
    assert o.eta.spec.hermiticity == imsrg.Hermiticity.AntiHermitian
    with pytest.raises(imsrg.InputError):
        o.flowing_operator(2)
    with pytest.raises(ValueError):
        flow.ode_solver("euler", h0)
    assert flow.ode_solver("rk4", h0).num_operators == 0


def test_stale_solver_objects_after_the_state_of_the_operator(o16):
    s = flow.MagnusSolver(o16["H"], solver=flow.SolverOptions(smax=1.0))
    s.solve()
    h = s.hamiltonian
    with imsrg.strict_residency():
        with pytest.raises(imsrg.StaleHostError):
            h.one_body
        assert h.sync_to_host().one_body.shape == (12, 12)
