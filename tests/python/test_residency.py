"""StaleHostError state machine of design/18 section 2.2 (host/device residency).

The default policy ("Auto") only enforces on GPU builds, where the host mirror is
not the device data; the tests switch the policy to Strict so that the state
machine is exercised on the OpenMP build as well.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import imsrg
from imsrg import Residency


@pytest.fixture()
def strict():
    with imsrg.strict_residency():
        yield


@pytest.fixture()
def op(o16_emax2):
    return imsrg.random_operator(o16_emax2, seed=11)


def test_default_policy_matches_the_build():
    assert imsrg.residency_policy() == imsrg.ResidencyPolicy.Auto
    # OpenMP/Serial: the host mirror aliases the device data, nothing to enforce.
    assert imsrg.residency_enforced() == (imsrg.Runtime.execution_space() in ("Cuda", "HIP", "SYCL"))


def test_initial_states(o16_emax2, op):
    assert imsrg.Operator(o16_emax2).residency == Residency.Both  # zero operator
    assert op.residency == Residency.DeviceCurrent  # library result
    assert (op + op).residency == Residency.DeviceCurrent
    assert op.normal_ordered().residency == Residency.DeviceCurrent


def test_stale_view_raises_and_refreshed_view_works(strict, op):
    assert imsrg.residency_enforced()
    with pytest.raises(imsrg.StaleHostError, match="sync_to_host"):
        op.one_body
    with pytest.raises(imsrg.StaleHostError):
        op.two_body_block(0)
    with pytest.raises(imsrg.StaleHostError):
        op.one_body_view(writable=True)
    assert issubclass(imsrg.StaleHostError, imsrg.ImsrgError)
    # device-side operations never need the host mirror
    assert op.norm() > 0 and (op + op).norm() > 0
    assert op.residency == Residency.DeviceCurrent

    assert op.sync_to_host() is op  # returns the operator
    assert op.residency == Residency.Both
    v = op.one_body
    assert v.shape == (12, 12) and not v.flags.writeable
    np.testing.assert_allclose(v, v.T, atol=1e-14)
    assert op.two_body_block(0).ndim == 2


def test_writable_view_marks_the_device_stale_and_compute_pushes(strict, op):
    ref = op.copy()
    op.sync_to_host()
    w = op.one_body_view(writable=True)
    assert op.residency == Residency.HostCurrent
    w[0, 0] += 3.0
    w[1, 1] += 3.0
    # a read-only view of a HostCurrent operator is fine (the host data are current)
    assert op.one_body[0, 0] == w[0, 0]
    # a compute call pushes the edits to the device first
    diff = op - ref
    assert op.residency == Residency.Both
    assert diff.one_body_norm() > 1.0  # the pushed edits are in the result
    assert diff.two_body_norm() == 0.0
    w[0, 0] -= 3.0
    w[1, 1] -= 3.0
    assert (op - ref).norm() == pytest.approx(0.0, abs=1e-12)


def test_sync_to_device_and_noops(strict, op):
    op.sync_to_host()
    w = op.one_body_view(writable=True)
    w[0, 0] += 1.0
    op.sync_to_host()  # host holds edits: must not clobber them
    assert op.residency == Residency.HostCurrent
    assert op.sync_to_device() is op
    assert op.residency == Residency.Both
    op.sync_to_device()  # no-op when both are current
    assert op.residency == Residency.Both


def test_in_place_ops_and_set_zero_make_the_device_current(strict, op):
    op.sync_to_host()
    op += op
    assert op.residency == Residency.DeviceCurrent
    with pytest.raises(imsrg.StaleHostError):
        op.one_body
    op.sync_to_host()
    op *= 0.5
    assert op.residency == Residency.DeviceCurrent
    op.set_zero()
    assert op.residency == Residency.DeviceCurrent
    op.sync_to_host()
    assert not np.any(op.one_body)


def test_copy_inherits_the_state_and_is_deep(strict, op):
    for c in (op.copy(), copy.copy(op), copy.deepcopy(op)):
        assert c.residency == Residency.DeviceCurrent
    op.sync_to_host()
    c = op.copy()
    assert c.residency == Residency.Both
    w = c.one_body_view(writable=True)
    w[0, 0] += 5.0
    assert op.one_body[0, 0] != c.one_body[0, 0]
    # copying a HostCurrent operator pushes its edits first
    c2 = c.copy()
    assert c.residency == Residency.Both and c2.residency == Residency.Both
    assert c2.one_body[0, 0] == c.one_body[0, 0]


def test_edit_context(strict, op):
    ref = op.copy()
    with op.edit() as e:
        assert e.residency == Residency.Both
        w = e.one_body_view(writable=True)
        w[2, 2] += 2.0
    assert op.residency == Residency.Both
    assert (op - ref).one_body_norm() > 1.0
    with op.edit() as e:
        e.one_body_view(writable=True)[2, 2] -= 2.0
    assert (op - ref).norm() == pytest.approx(0.0, abs=1e-12)


def test_implicit_policy_syncs_by_itself(op):
    with imsrg.residency_policy_scope(imsrg.ResidencyPolicy.Implicit):
        assert not imsrg.residency_enforced()
        assert op.one_body.shape == (12, 12)
        assert op.residency == Residency.Both
    assert imsrg.residency_policy() == imsrg.ResidencyPolicy.Auto


def test_writers_do_not_need_a_sync(strict, tmp_path, o16_emax2):
    a = imsrg.random_operator(o16_emax2, seed=4)
    p = tmp_path / "x.dump"
    imsrg.write_operator_dump(str(p), a)
    assert a.residency == Residency.DeviceCurrent
    b = imsrg.read_operator_dump(str(p), o16_emax2)
    assert (a - b).norm() < 1e-12 * a.norm()


_FORCE_SCRIPT = """
import sys, numpy as np
import imsrg
ms = imsrg.ModelSpace(2, "O16", "O16", hw=16.0)
a = imsrg.random_operator(ms, seed=1)
b = a + a
assert imsrg._core.live_objects() >= 2
try:
    imsrg.Runtime.finalize()
except imsrg.RuntimeError as e:
    print("plain finalize refused:", "alive" in str(e))
a.sync_to_host()
view = a.one_body                      # a numpy view blocks a forced release
try:
    imsrg.Runtime.finalize(force=True)
except imsrg.RuntimeError as e:
    print("force with view refused:", "view" in str(e))
del view
imsrg.Runtime.finalize(force=True)
print("finalized:", not imsrg.Runtime.is_initialized())
for f in (lambda: a.norm(), lambda: a + a, lambda: a.one_body, lambda: b.copy()):
    try:
        f()
        print("no error")
    except imsrg.StaleHostError:
        print("stale")
print("repr:", repr(a))
"""


def test_force_finalize_frees_live_operators():
    env = dict(os.environ)
    root = str(Path(imsrg.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _FORCE_SCRIPT], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert "plain finalize refused: True" in lines
    assert "force with view refused: True" in lines
    assert "finalized: True" in lines
    assert lines.count("stale") == 4
    assert "repr: Operator(<released>)" in lines
