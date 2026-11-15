"""Python bindings of the imsrg rewrite (nanobind). See design/18-python-api.md."""

import atexit as _atexit
import contextlib as _contextlib
import gc as _gc

from . import _core
from ._core import (  # noqa: F401
    Channel,
    ConfigError,
    ConvergenceError,
    FileFormatError,
    Hermiticity,
    ImsrgError,
    InputError,
    InputFileNotFoundError,
    InvariantError,
    ModelSpace,
    ModelSpaceError,
    NotImplementedError,
    NumericalError,
    Operator,
    OperatorSpec,
    Orbit,
    Parity,
    Residency,
    ResidencyPolicy,
    RunConfig,
    Runtime,
    RuntimeError,
    StaleHostError,
    SymmetryError,
    load_config,
    random_operator,
    read_operator_dump,
    read_operator_dump_header,
    residency_enforced,
    residency_policy,
    set_residency_policy,
    write_operator_dump,
)

__version__ = "0.1.0"


@_contextlib.contextmanager
def residency_policy_scope(policy):
    """Context manager that sets the residency policy and restores the old one.

    ``with imsrg.residency_policy_scope(imsrg.ResidencyPolicy.Strict):`` makes a
    host view of a device-current operator raise ``StaleHostError`` also on
    builds where the host mirror aliases the device data, so a script can be
    tested for portability to GPU nodes.
    """
    old = residency_policy()
    set_residency_policy(policy)
    try:
        yield
    finally:
        set_residency_policy(old)


def strict_residency():
    """``residency_policy_scope(ResidencyPolicy.Strict)``."""
    return residency_policy_scope(ResidencyPolicy.Strict)


class _EditContext:
    """``with op.edit() as e:`` -- syncs to the host on enter, to the device on exit."""

    def __init__(self, op):
        self._op = op

    def __enter__(self):
        return self._op.sync_to_host()

    def __exit__(self, *exc):
        self._op.sync_to_device()
        return False


def _edit(self):
    """Context manager for host edits: ``sync_to_host()`` on enter,
    ``sync_to_device()`` on exit. Take writable views inside the block."""
    return _EditContext(self)


Operator.edit = _edit

from . import (  # noqa: E402,F401
    commutators,
    flow,
    hf,
    io,
    mbpt,
    operators,
    pipeline,
)


def _finalize_at_exit():
    _gc.collect()
    _core._atexit()


_atexit.register(_finalize_at_exit)
