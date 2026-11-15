"""hf (options incl. mixing, results, normal ordering) and mbpt (MP2/MP3, natural
orbitals) against the imsrg++ golden data (tests/golden/hf)."""

import json

import numpy as np
import pytest

import imsrg
from imsrg import hf, io, mbpt


@pytest.fixture(scope="module")
def h_o16(golden, o16_emax2):
    return imsrg.read_operator_dump(str(golden / "hamiltonian" / "emax2_O16" / "H.dump"), o16_emax2)


@pytest.fixture(scope="module")
def gold_2n(golden):
    return json.loads((golden / "hf" / "emax2_O16_2N" / "hf.json").read_text())


@pytest.fixture(scope="module")
def hf_2n(h_o16):
    return hf.solve(h_o16)


def test_hf_energy_and_result_fields(hf_2n, gold_2n):
    r = hf_2n
    assert r.converged and r.iterations == gold_2n["iterations"] == 20
    for key, val in (("EHF", r.e_hf), ("e1hf", r.e1), ("e2hf", r.e2), ("e3hf", r.e3)):
        assert val == pytest.approx(gold_2n[key], abs=1e-8), key
    assert r.e_hf == pytest.approx(r.e1 + r.e2 + r.e3, abs=1e-10)
    n = 12
    assert len(r.spe) == n and len(r.occupations) == n and len(r.ediff_history) > 0
    for a in (r.c, r.rho, r.fock):
        assert a.shape == (n, n) and a.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(r.c.T @ r.c, np.eye(n), atol=1e-12)  # orthogonal
    np.testing.assert_allclose(r.rho, r.rho.T, atol=1e-12)
    # F is diagonal in the HF basis with the single-particle energies on the diagonal
    f_hf = r.c.T @ r.fock @ r.c
    np.testing.assert_allclose(np.diag(f_hf), r.spe, atol=1e-6)
    np.testing.assert_allclose(f_hf - np.diag(np.diag(f_hf)), 0.0, atol=1e-6)


def test_options_and_mixing(h_o16, gold_2n):
    o = hf.HFOptions()
    assert (o.mixing, o.density_mixing, o.tolerance, o.max_iterations) == (hf.Mixing.Linear, 0.2, 1e-8, 1000)
    # keyword form with a string mixing, options form, and DIIS
    lin = hf.solve(h_o16, mixing="linear", density_mixing=0.2)
    assert lin.iterations == gold_2n["iterations"]
    diis = hf.solve(h_o16, options=hf.HFOptions(mixing=hf.Mixing.Diis))
    assert diis.converged and diis.e_hf == pytest.approx(gold_2n["EHF"], abs=1e-6)
    assert hf.solve(h_o16, mixing="DIIS").iterations == diis.iterations
    # a different density mixing converges to the same minimum
    slow = hf.solve(h_o16, density_mixing=0.5)
    assert slow.e_hf == pytest.approx(gold_2n["EHF"], abs=1e-6)
    # not converged within the iteration limit is reported, not raised
    short = hf.solve(h_o16, max_iterations=3)
    assert not short.converged and short.iterations == 3
    with pytest.raises(ValueError, match="Mixing"):
        hf.solve(h_o16, mixing="anderson")
    with pytest.raises(TypeError):
        hf.solve(h_o16, options=o, tolerance=1e-6)


def test_normal_ordered_hamiltonian_matches_the_golden_dump(golden, h_o16, hf_2n, gold_2n, o16_emax2):
    hno = hf_2n.normal_ordered_hamiltonian(h_o16)
    ref = imsrg.read_operator_dump(str(golden / "hf" / "emax2_O16_2N" / "HNO.dump"), o16_emax2)
    assert hno.zero_body == pytest.approx(gold_2n["HNO_zero_body"], abs=1e-8)
    assert (hno - ref).norm() < 1e-7 * ref.norm()
    same = hf.normal_ordered_hamiltonian(h_o16, hf_2n)
    assert (same - hno).norm() == 0.0
    # transform + normal order of a bare operator is the same chain
    t_hf = hf_2n.transform(h_o16)
    assert t_hf.zero_body == h_o16.zero_body  # no normal ordering in transform


def test_mbpt_energies_match_the_golden_values(h_o16, hf_2n, gold_2n):
    hno = hf_2n.normal_ordered_hamiltonian(h_o16)
    assert mbpt.mp2(hno) == pytest.approx(gold_2n["EMP2"], abs=1e-8)
    e3 = mbpt.mp3(hno)
    assert e3.pp == pytest.approx(gold_2n["EMP3_pp"], abs=1e-8)
    assert e3.hh == pytest.approx(gold_2n["EMP3_hh"], abs=1e-8)
    assert e3.ph == pytest.approx(gold_2n["EMP3_ph"], abs=1e-8)
    assert e3.total == pytest.approx(gold_2n["EMP3"], abs=1e-8)
    assert e3.total == pytest.approx(e3.pp + e3.hh + e3.ph, abs=1e-12)


def test_natural_orbitals(h_o16, hf_2n, o16_emax2):
    hno = hf_2n.normal_ordered_hamiltonian(h_o16)
    nat = mbpt.natural_orbitals(hno)
    c = nat.c_hf2nat
    n = o16_emax2.num_orbits
    assert c.shape == (n, n)
    np.testing.assert_allclose(c.T @ c, np.eye(n), atol=1e-12)
    occ = np.asarray(nat.occupations)
    orbits = o16_emax2.orbits()
    assert sum((o.two_j + 1) * x for o, x in zip(orbits, occ)) == pytest.approx(16.0, abs=1e-6)
    assert np.all(occ > -1e-6) and np.all(occ < 1 + 1e-6)
    # the density matrix is symmetric and block diagonal in (l, j, tz)
    rho = mbpt.mbpt2_density_matrix(hno)
    np.testing.assert_allclose(rho, rho.T, atol=1e-10)
    # the NAT basis Hamiltonian is a unitary transformation: its E0 is unchanged
    h_nat = mbpt.to_nat_basis(hno, c)
    assert h_nat.zero_body == pytest.approx(hno.zero_body, abs=1e-10)
    # the HO reference density of the NAT basis has trace A
    c_ho2nat = hf_2n.c @ c
    rho_ho = mbpt.reference_density_ho(c_ho2nat, o16_emax2)
    assert sum((o.two_j + 1) * rho_ho[i, i] for i, o in enumerate(orbits)) == pytest.approx(16.0, abs=1e-6)
    # the mean-field state of that basis
    sol = hf.solution_in_basis(h_o16, None, c_ho2nat, occ)
    assert sol.converged and sol.iterations == 0
    assert sol.e_hf == pytest.approx(sol.e1 + sol.e2 + sol.e3)


def test_hf_with_three_body_input(golden, h_o16, o16_emax2, no2b_file):
    gold = json.loads((golden / "hf" / "emax2_O16_2N3N" / "hf.json").read_text())
    v3 = io.read_no2b(no2b_file, o16_emax2, file_e1max=6, file_e2max=12, file_e3max=18, e3max=12)
    r = hf.solve(h_o16, v3)
    assert r.iterations == gold["iterations"] == 22
    assert r.e_hf == pytest.approx(gold["EHF"], abs=1e-8)
    assert r.e3 == pytest.approx(gold["e3hf"], abs=1e-8)
    hno = r.normal_ordered_hamiltonian(h_o16, v3)
    assert hno.zero_body == pytest.approx(gold["HNO_zero_body"], abs=1e-8)
    assert mbpt.mp2(hno) == pytest.approx(gold["EMP2"], abs=1e-8)
    # e3max cuts the three-body monopole: e3max below the smallest e_a+e_b+e_c changes E_HF
    r0 = hf.solve(h_o16, v3, e3max=0)
    assert r0.e3 != r.e3

