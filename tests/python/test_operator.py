"""Operators: dump norms against golden, numpy views, arithmetic, I/O."""

import json

import numpy as np
import pytest

import imsrg


@pytest.mark.parametrize("name", ["H", "T", "V2", "V3_NO2B"])
def test_dump_norms_match_golden(golden, o16_emax2, name):
    ref = json.loads((golden / "hamiltonian" / "emax2_O16" / "norms.json").read_text())[name]
    o = imsrg.read_operator_dump(str(golden / "hamiltonian" / "emax2_O16" / f"{name}.dump"), o16_emax2)
    assert o.zero_body == pytest.approx(ref["zero_body"], rel=1e-10, abs=1e-12)
    for key, val in (
        ("one_body_norm", o.one_body_norm()),
        ("two_body_norm", o.two_body_norm()),
        ("three_body_norm", o.three_body_norm()),
        ("norm", o.norm()),
    ):
        assert val == pytest.approx(ref[key], rel=1e-10, abs=1e-12), key


def test_dump_without_model_space(golden):
    o = imsrg.read_operator_dump(str(golden / "hamiltonian" / "emax2_O16" / "H.dump"))
    assert o.model_space.num_orbits == 12
    header = imsrg.read_operator_dump_header(str(golden / "hamiltonian" / "emax2_O16" / "H.dump"))
    assert header["emax"] == 2 and len(header["orbits"]) == 12


def test_numpy_view_round_trip(o16_emax2):
    o = imsrg.random_operator(o16_emax2, seed=7)
    v = o.one_body
    assert v.shape == (12, 12) and v.flags["F_CONTIGUOUS"]
    assert not v.flags.writeable
    np.testing.assert_allclose(v, v.T, atol=1e-14)  # Hermitian
    # Zero copy: a writable view edits the operator; the read-only view sees it.
    w = o.one_body_view(writable=True)
    old = float(w[0, 0])
    w[0, 0] = old + 1.0
    o.sync_to_device()
    assert o.one_body[0, 0] == pytest.approx(old + 1.0)
    blk = o.two_body_block(0)
    assert blk.shape[0] == blk.shape[1] == o16_emax2.channel(0).num_kets
    # the view keeps the operator alive
    view = imsrg.random_operator(o16_emax2, seed=2).one_body
    assert np.isfinite(view).all()


def test_arithmetic_and_norms(o16_emax2):
    a = imsrg.random_operator(o16_emax2, seed=1, zero_body_scale=1.0)
    b = imsrg.random_operator(o16_emax2, seed=2)
    assert (a + a).norm() == pytest.approx(2 * a.norm(), rel=1e-12)
    assert (a - a).norm() == pytest.approx(0.0, abs=1e-12)
    assert (2.0 * a).norm() == pytest.approx((a * 2.0).norm())
    assert (-a).norm() == pytest.approx(a.norm())
    assert (a / 4.0).norm() == pytest.approx(a.norm() / 4)
    c = a.copy()
    c += b
    assert c.norm() == pytest.approx((a + b).norm(), rel=1e-12)
    c -= b
    assert (c - a).norm() == pytest.approx(0.0, abs=1e-10)
    c.set_zero()
    assert c.norm() == 0.0


def test_random_operator_is_seeded(o16_emax2):
    a = imsrg.random_operator(o16_emax2, seed=5)
    b = imsrg.random_operator(o16_emax2, seed=5)
    assert (a - b).norm() == 0.0
    assert (a - imsrg.random_operator(o16_emax2, seed=6)).norm() > 0


def test_normal_ordering_round_trip(o16_emax2):
    a = imsrg.random_operator(o16_emax2, seed=3)
    back = a.normal_ordered().undo_normal_order()
    assert (back - a).norm() < 1e-9 * a.norm()


def test_dump_write_read_round_trip(tmp_path, o16_emax2):
    a = imsrg.random_operator(o16_emax2, seed=4, zero_body_scale=1.0)
    p = tmp_path / "x.dump"
    imsrg.write_operator_dump(str(p), a, source="pytest")
    b = imsrg.read_operator_dump(str(p), o16_emax2)
    assert (a - b).norm() < 1e-12 * a.norm()


def test_bad_inputs_raise(tmp_path, o16_emax2):
    with pytest.raises(imsrg.InputError):
        imsrg.read_operator_dump(str(tmp_path / "missing.dump"), o16_emax2)
    with pytest.raises(imsrg.ImsrgError):
        o16_emax2.channel(-1)


def test_runtime_state():
    imsrg.Runtime.ensure()
    assert imsrg.Runtime.is_initialized()
    assert imsrg.Runtime.concurrency() >= 1
    with pytest.raises(imsrg.RuntimeError):
        imsrg.Runtime()  # already initialized
