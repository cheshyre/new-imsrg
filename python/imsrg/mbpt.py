"""Many-body perturbation theory and natural orbitals (design/18 section 2.5).

The Hamiltonian must be normal ordered and in the basis where the one-body part
is diagonal (the HF basis).
"""

from . import _core
from ._core import (  # noqa: F401
    Mp3Energy,
    NaturalOrbitals,
    mbpt2_density_matrix,
    reference_density_ho,
    to_nat_basis,
)

__all__ = [
    "Mp3Energy",
    "NaturalOrbitals",
    "mbpt2_density_matrix",
    "mp2",
    "mp3",
    "natural_orbitals",
    "reference_density_ho",
    "to_nat_basis",
]


def mp2(hamiltonian):
    """MBPT(2) correlation energy in MeV."""
    return _core.mp2_energy(hamiltonian)


def mp3(hamiltonian):
    """MBPT(3) energy terms: an ``Mp3Energy`` with ``pp``, ``hh``, ``ph``, ``total``."""
    return _core.mp3_energy(hamiltonian)


def natural_orbitals(hamiltonian, model_space=None):
    """Natural orbitals of the MBPT(2) density matrix of `hamiltonian`.

    Returns ``NaturalOrbitals`` with ``c_hf2nat`` and ``occupations``.
    """
    ms = model_space if model_space is not None else hamiltonian.model_space
    return _core.natural_orbitals(_core.mbpt2_density_matrix(hamiltonian), ms)
