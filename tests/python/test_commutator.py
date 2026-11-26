"""commutator: options and term mask, golden terms of imsrg++, workspaces, and the
factorized double commutator (design/18 section 2.6)."""

import re
from pathlib import Path

import pytest

import imsrg
from imsrg import commutators as comm
from imsrg.commutators import Term, TermMask

HE4 = "commutator/emax1_He4"
O16 = "commutator/emax2_O16"


def _dump(golden, rel, ms=None):
    return imsrg.read_operator_dump(str(golden / (rel + ".dump")), ms)


def _rel_dev(a, b):
    return (a - b).norm() / max(1.0, b.norm())


def _he4_cases(golden_dir):
    cases = []
    for p in sorted((golden_dir / HE4).glob("Z_comm*.dump")):
        m = re.match(r"Z_(comm\w+?)__(random_\w+)__(random_\w+)\.dump", p.name)
        cases.append((m.group(1), m.group(2), m.group(3)))
    return cases


def pytest_generate_tests(metafunc):
    if "he4_case" in metafunc.fixturenames:
        golden = Path(__file__).resolve().parents[2] / "tests" / "golden"
        metafunc.parametrize("he4_case", _he4_cases(golden), ids=lambda c: "-".join(c))


@pytest.fixture(scope="module")
def he4(golden):
    x = {"random_A1": _dump(golden, f"{HE4}/random_A1")}
    ms = x["random_A1"].model_space
    for n in ("random_A2", "random_H1", "random_H2"):
        x[n] = _dump(golden, f"{HE4}/{n}", ms)
    return x


def test_every_term_matches_imsrgpp(golden, he4, he4_case):
    term_name, xn, yn = he4_case
    term = comm.term_from_name(term_name)
    assert term is not None, term_name
    ref = _dump(golden, f"{HE4}/Z_{term_name}__{xn}__{yn}", he4[xn].model_space)
    z = comm.commutator(he4[xn], he4[yn], terms=[term])
    assert z.spec.hermiticity == ref.spec.hermiticity
    assert _rel_dev(z, ref) < 1e-10, term_name


def test_all_terms_equal_the_sum_of_the_terms(he4):
    x, y = he4["random_A1"], he4["random_H1"]
    full = comm.commutator(x, y)
    assert full.spec.hermiticity == imsrg.Hermiticity.Hermitian  # (A, H) -> H
    total = None
    # The IMSRG(2) terms (the IMSRG(3) ones need a three-body space).
    for t in TermMask.imsrg2().terms():
        zt = comm.commutator(x, y, terms=[t])
        total = zt if total is None else total + zt
    assert _rel_dev(total, full) < 1e-12
    # the same through the options object, and via a TermMask of everything
    opts = comm.CommutatorOptions(terms=TermMask.imsrg2())
    assert (comm.commutator(x, y, options=opts) - full).norm() == 0.0


def test_full_commutator_matches_the_golden_all(golden):
    o16 = {"H": _dump(golden, f"{O16}/H")}
    for n in ("eta_white", "random_A", "random_H"):
        o16[n] = _dump(golden, f"{O16}/{n}", o16["H"].model_space)
    for xn, yn in (("eta_white", "H"), ("random_A", "random_H")):
        ref = _dump(golden, f"{O16}/Z_all__{xn}__{yn}", o16[xn].model_space)
        z = comm.commutator(o16[xn], o16[yn])
        assert _rel_dev(z, ref) < 1e-10
    # the combined 222_pp_hh + 221 ladder of imsrg++ is the mask {222_pp_hh, 221}
    ref = _dump(golden, f"{O16}/Z_comm222_pp_hh_221ss__random_A__random_H", o16["random_A"].model_space)
    z = comm.commutator(o16["random_A"], o16["random_H"], terms=["Comm222PpHhss", "Comm221ss"])
    assert _rel_dev(z, ref) < 1e-10


def test_term_mask_api():
    m = TermMask.none()
    assert m.empty() and m.bits == 0 and m.terms() == []
    m.set(Term.Comm110ss)
    m.set(Term.Comm222Phss)
    assert m.test(Term.Comm110ss) and not m.test(Term.Comm111ss)
    assert m.terms() == [Term.Comm110ss, Term.Comm222Phss]
    m.reset(Term.Comm110ss)
    assert m == TermMask.only(Term.Comm222Phss) == TermMask.of([Term.Comm222Phss])
    assert (m | TermMask.only(Term.Comm111ss)).bits == m.bits | TermMask.only(Term.Comm111ss).bits
    assert (TermMask.imsrg2() & m) == m
    assert len(TermMask.imsrg2().terms()) == 8 and TermMask.imsrg2() != m
    assert hash(m) == hash(TermMask.only(Term.Comm222Phss))
    assert "comm222_phss" in repr(m)
    assert comm.term_name(Term.Comm110ss) == "comm110ss"
    assert comm.term_from_name("comm222_pp_hhss") == Term.Comm222PpHhss
    assert comm.term_from_name("nonsense") is None
    assert comm.CommutatorOptions().terms == TermMask.imsrg2()
    o = comm.CommutatorOptions(terms=m)
    o.terms = TermMask.none()
    assert o.terms.empty()


def test_empty_mask_gives_zero(he4):
    z = comm.commutator(he4["random_A1"], he4["random_H1"], terms=[])
    assert z.norm() == 0.0 and z.zero_body == 0.0


def test_hermiticity_of_the_result(he4):
    a, h = he4["random_A1"], he4["random_H1"]
    assert comm.commutator(a, a).spec.hermiticity == imsrg.Hermiticity.AntiHermitian
    assert comm.commutator(h, h).spec.hermiticity == imsrg.Hermiticity.AntiHermitian
    assert comm.commutator(h, a).spec.hermiticity == imsrg.Hermiticity.Hermitian
    # [X, X] = 0
    assert comm.commutator(h, h).norm() < 1e-9 * h.norm()


def test_non_hermitian_operands(he4):
    # The general (non-Hermitian) commutator paths exist since D36/M7-Q2; the C++ tests check
    # the kernels against commutator_ref. Here the binding passes a general operand through and
    # returns a general Z that is antisymmetric under exchange of the operands.
    x = he4["random_A1"].copy()
    ms = x.model_space
    nh = imsrg.Operator(ms, imsrg.OperatorSpec(hermiticity=imsrg.Hermiticity.NonHermitian))
    h1 = he4["random_H1"].sync_to_host()  # library results are device-current (GPU-portable)
    x.sync_to_host()
    nh.one_body_view(writable=True)[:] = x.one_body + h1.one_body
    z = comm.commutator(nh, x)
    assert z.spec.hermiticity == imsrg.Hermiticity.NonHermitian
    assert z.norm() > 0.0
    assert (z + comm.commutator(x, nh)).norm() < 1e-12 * z.norm()


def test_unsupported_requests_raise(he4):
    ms = he4["random_A1"].model_space
    with pytest.raises(imsrg.NotImplementedError):  # tensors with a 3B part are not implemented
        imsrg.Operator(ms, imsrg.OperatorSpec(rank_J=2, particle_rank=3))
    with pytest.raises(TypeError):
        comm.commutator(he4["random_A1"], he4["random_H1"], terms=[Term.Comm110ss], options=comm.CommutatorOptions())
    with pytest.raises(ValueError):
        comm.commutator(he4["random_A1"], he4["random_H1"], terms=["comm999"])


def test_explicit_workspace_and_cache(he4):
    a, h = he4["random_A1"], he4["random_H1"]
    ws = comm.CommutatorWorkspace(a.model_space)
    assert ws.model_space == a.model_space
    z1 = comm.commutator(a, h, workspace=ws)
    z2 = comm.commutator(a, h)
    z3 = comm.commutator(a, h, workspace=ws)
    assert (z1 - z2).norm() == 0.0 and (z1 - z3).norm() == 0.0
    comm.clear_workspace_cache()
    assert (comm.commutator(a, h) - z1).norm() == 0.0


def test_host_edits_are_pushed_before_the_commutator(he4):
    a, h = he4["random_A1"].copy(), he4["random_H1"]
    ref = comm.commutator(a, h)
    a.sync_to_host()
    w = a.one_body_view(writable=True)
    w[0, 0] += 1.0
    assert a.residency == imsrg.Residency.HostCurrent
    edited = comm.commutator(a, h)
    assert a.residency == imsrg.Residency.HostCurrent  # the writable view is alive
    del w
    comm.commutator(a, h)
    assert a.residency == imsrg.Residency.Both
    assert (edited - ref).norm() > 1e-6


def test_factorized_double_commutator_matches_imsrgpp(golden):
    he6 = {"random_A1": _dump(golden, "factorized/emax2_He6/random_A1")}
    for n in ("random_A2", "random_H1"):
        he6[n] = _dump(golden, f"factorized/emax2_He6/{n}", he6["random_A1"].model_space)
    for yn in ("random_H1", "random_A2"):
        ref = _dump(golden, f"factorized/emax2_He6/Z_d32__random_A1__{yn}", he6["random_A1"].model_space)
        z = comm.factorized_double_commutator(he6["random_A1"], he6[yn])
        assert z.spec.particle_rank == 2
        assert z.spec.hermiticity == he6[yn].spec.hermiticity
        assert _rel_dev(z, ref) < 1e-10
    ms = _dump(golden, f"{O16}/H").model_space
    eta, h = _dump(golden, f"{O16}/eta_white", ms), _dump(golden, f"{O16}/H", ms)
    ref = _dump(golden, "factorized/emax2_O16/Z_d32__eta_white__H", ms)
    assert _rel_dev(comm.factorized_double_commutator(eta, h), ref) < 1e-10
    # no topology selected: zero; the two-body intermediates are not implemented
    off = comm.FactorizedOptions(one_body_intermediates=False)
    assert comm.factorized_double_commutator(eta, h, options=off).norm() == 0.0
    with pytest.raises(imsrg.NotImplementedError):
        comm.factorized_double_commutator(eta, h, options=comm.FactorizedOptions(two_body_intermediates=True))
    # eta must be anti-Hermitian
    with pytest.raises(imsrg.NotImplementedError):
        comm.factorized_double_commutator(h, h)
