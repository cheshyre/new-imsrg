"""Hartree-Fock (design/18 section 2.4).

``solve(h, v3, mixing="linear", ...)`` runs spherical HF; the result object
transforms operators to the HF basis and builds the normal-ordered Hamiltonian.
"""

from . import _core
from ._core import HFOptions, HFResult, Mixing  # noqa: F401
from ._util import to_enum

__all__ = [
    "HFOptions",
    "HFResult",
    "Mixing",
    "normal_ordered_hamiltonian",
    "solution_in_basis",
    "solve",
    "to_hf_basis",
]


def _options(options, kwargs):
    if options is not None:
        if kwargs:
            raise TypeError("pass either options or keyword arguments, not both")
        return options
    if "mixing" in kwargs:
        kwargs = dict(kwargs, mixing=to_enum(Mixing, kwargs["mixing"]))
    return HFOptions(**kwargs)


def solve(hamiltonian, v3=None, *, options=None, **kwargs):
    """Solve spherical Hartree-Fock for `hamiltonian`.

    Parameters
    ----------
    hamiltonian : Operator
        One-body kinetic part plus two-body interaction on the model space
        whose reference is the HF reference.
    v3 : No2bThreeBody, optional
        NO2B three-nucleon input (from ``io.read_no2b``).
    options : HFOptions, optional
        Or keyword arguments of ``HFOptions``: ``mixing`` ("linear", "diis" or
        ``Mixing``), ``density_mixing`` (weight of the old density, 0.2),
        ``tolerance``, ``max_iterations``, ``freeze_occupations``, ``e3max``.

    Returns
    -------
    HFResult
        With ``e_hf``, ``e1``..``e3``, ``spe``, ``c``, ``rho``, ``fock``,
        ``iterations``, ``converged``, and the methods ``transform(op)`` and
        ``normal_ordered_hamiltonian(h, v3)``.
    """
    return _core.hf_solve(hamiltonian, v3, _options(options, kwargs))


def solution_in_basis(hamiltonian, v3, c_ho2basis, occupations, *, options=None, **kwargs):
    """Mean-field state of a given orthogonal basis (no iteration)."""
    return _core.hf_solution_in_basis(
        hamiltonian, v3, c_ho2basis, list(occupations), _options(options, kwargs)
    )


def normal_ordered_hamiltonian(hamiltonian, result, v3=None, *, options=None, **kwargs):
    """The Hamiltonian in the HF basis of `result`, normal ordered with respect
    to the HF reference, three-body part truncated at the two-body level."""
    return _core.hf_normal_ordered_hamiltonian(hamiltonian, v3, result, _options(options, kwargs))


def to_hf_basis(operator, result):
    """`operator` transformed to the HF basis of `result` (no normal ordering)."""
    return _core.hf_to_hf_basis(operator, result)


def _transform(self, operator):
    """Transforms `operator` to this HF basis (see to_hf_basis)."""
    return to_hf_basis(operator, self)


def _noh(self, hamiltonian, v3=None, **kwargs):
    """See hf.normal_ordered_hamiltonian."""
    return normal_ordered_hamiltonian(hamiltonian, self, v3, **kwargs)


HFResult.transform = _transform
HFResult.normal_ordered_hamiltonian = _noh
