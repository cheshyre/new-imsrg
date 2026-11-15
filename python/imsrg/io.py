"""Readers and writers (design/18 section 2.3).

``read_me2j`` and ``read_no2b`` read the interaction files, ``write_tokyo``,
``write_nushellx_int`` and ``write_nushellx_sps`` write valence-space
interactions. All of them release the GIL. Operators the writers receive may be
device-resident: they refresh their host mirror themselves.
"""

from ._core import (  # noqa: F401
    No2bPrecision,
    No2bThreeBody,
    ValenceWriteMode,
    ValenceWriteOptions,
    me2j_value_count,
    no2b_value_count,
    read_me2j,
    read_no2b,
    read_operator_dump as read_operator,
    write_nushellx_int,
    write_nushellx_sps,
    write_operator_dump as write_operator,
    write_tokyo,
)

__all__ = [
    "No2bPrecision",
    "No2bThreeBody",
    "ValenceWriteMode",
    "ValenceWriteOptions",
    "me2j_value_count",
    "no2b_value_count",
    "read_me2j",
    "read_no2b",
    "read_operator",
    "write_nushellx_int",
    "write_nushellx_sps",
    "write_operator",
    "write_tokyo",
]
