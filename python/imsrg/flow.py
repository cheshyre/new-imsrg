"""The IMSRG flow (design/18 section 2.7): generators, the Magnus solver
(optionally hunter-gatherer), the ODE solvers and the BCH functions.

Enum arguments also accept strings: ``generator("shell-model-atan")``,
``ode_solver("flow_rk4", h0)``.
"""

from . import _core
from ._core import (  # noqa: F401
    BchOptions,
    BchReport,
    Denominators,
    FlowStatus,
    GeneratorKind,
    GeneratorOptions,
    MagnusSolver,
    OdeMethod,
    OdeSolver,
    Partitioning,
    SolveResult,
    SolverOptions,
    StopReason,
    bch_product,
    compute_generator,
    floor_denominator,
    generator_function,
)
from ._util import to_enum

__all__ = [
    "BchOptions",
    "BchReport",
    "Denominators",
    "FlowStatus",
    "GeneratorKind",
    "GeneratorOptions",
    "MagnusSolver",
    "OdeMethod",
    "OdeSolver",
    "Partitioning",
    "SolveResult",
    "SolverOptions",
    "StopReason",
    "bch_product",
    "bch_transform",
    "bch_transform_with_report",
    "compute_generator",
    "floor_denominator",
    "generator",
    "generator_function",
    "ode_solver",
]

_ODE_ALIASES = {"flow_rk4": OdeMethod.Rk4, "rk4": OdeMethod.Rk4, "flow": OdeMethod.Dopri5}


def generator(kind="atan", *, partitioning="epstein-nesbet", **kwargs):
    """``GeneratorOptions`` with `kind` and `partitioning` given as strings or enums."""
    return GeneratorOptions(
        kind=to_enum(GeneratorKind, kind),
        partitioning=to_enum(Partitioning, partitioning),
        **kwargs,
    )


def ode_solver(method, h0, **kwargs):
    """An ``OdeSolver``; `method` is ``OdeMethod``, "flow_rk4" or "flow" (the
    imsrg++ names), "rk4" or "dopri5"."""
    if isinstance(method, str) and method.lower() in _ODE_ALIASES:
        method = _ODE_ALIASES[method.lower()]
    return OdeSolver(to_enum(OdeMethod, method), h0, **kwargs)


def bch_transform_with_report(operator, omega, **kwargs):
    """Returns ``(Z, BchReport)``; see ``bch_transform``."""
    return _core.bch_transform(operator, omega, **kwargs)


def bch_transform(operator, omega, **kwargs):
    """Z = e^Omega X e^-Omega through the IMSRG(2)-truncated BCH series.

    Keyword arguments: ``bch`` (``BchOptions``, ``factorized=True`` adds the
    IMSRG(3f2) correction), ``commutator`` (``CommutatorOptions``),
    ``workspace``. Use ``bch_transform_with_report`` for the diagnostics.
    """
    return _core.bch_transform(operator, omega, **kwargs)[0]
