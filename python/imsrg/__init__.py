"""Python bindings of the imsrg rewrite (nanobind). See design/18-python-api.md."""

import atexit as _atexit
import gc as _gc

from . import _core
from ._core import (  # noqa: F401
    Channel,
    ConfigError,
    Hermiticity,
    ImsrgError,
    InputError,
    InvariantError,
    ModelSpace,
    NotImplementedError,
    NumericalError,
    Operator,
    OperatorSpec,
    Orbit,
    Parity,
    RunConfig,
    Runtime,
    RuntimeError,
    load_config,
    random_operator,
    read_operator_dump,
    read_operator_dump_header,
    write_operator_dump,
)

__version__ = "0.1.0"


def _finalize_at_exit():
    _gc.collect()
    _core._atexit()


_atexit.register(_finalize_at_exit)
