"""Physical operators of the operator library (kinetic energy, center of mass,
radii). Each takes a ModelSpace (and the mass/proton number where needed) and
returns an Operator on that model space."""

from ._core import (  # noqa: F401
    h_cm,
    kinetic_energy,
    r2_cm,
    rm2,
    rn2,
    rp2,
    t_cm,
    t_rel,
)

__all__ = ["h_cm", "kinetic_energy", "r2_cm", "rm2", "rn2", "rp2", "t_cm", "t_rel"]
