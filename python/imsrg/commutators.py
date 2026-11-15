"""The IMSRG(2) commutator (design/18 section 2.6).

``commutator(x, y)`` evaluates Z = [X, Y]; the terms are chosen with a
``TermMask`` (or any iterable of ``Term``).
"""

from . import _core
from ._core import (  # noqa: F401
    CommutatorOptions,
    CommutatorWorkspace,
    FactorizedOptions,
    Term,
    TermMask,
    clear_workspace_cache,
    factorized_double_commutator,
    term_from_name,
    term_name,
)
from ._util import to_enum

__all__ = [
    "CommutatorOptions",
    "CommutatorWorkspace",
    "FactorizedOptions",
    "Term",
    "TermMask",
    "clear_workspace_cache",
    "commutator",
    "factorized_double_commutator",
    "term_from_name",
    "term_name",
]


def _mask(terms):
    if terms is None:
        return TermMask.imsrg2()
    if isinstance(terms, TermMask):
        return terms
    return TermMask.of([to_enum(Term, t) for t in terms])


def commutator(x, y, *, terms=None, options=None, workspace=None):
    """Z = [X, Y] restricted to a set of terms.

    Parameters
    ----------
    x, y : Operator
        Scalar, number-conserving, particle rank <= 2. The hermiticity of Z
        follows from those of X and Y. (NonHermitian operands raise
        ``NotImplementedError`` in this version of the core.)
    terms : TermMask or iterable of Term or str, optional
        Terms to evaluate (default: all IMSRG(2) terms). Convenience for
        ``options=CommutatorOptions(terms=...)``.
    options : CommutatorOptions, optional
    workspace : CommutatorWorkspace, optional
        Reuse a workspace; by default one per model space is cached.
    """
    if options is not None and terms is not None:
        raise TypeError("pass either options or terms, not both")
    if options is None:
        options = CommutatorOptions(terms=_mask(terms))
    return _core.commutator(x, y, options=options, workspace=workspace)
