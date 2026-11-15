"""Small helpers shared by the Python layer (design/18 section 1.3)."""

import re


def to_enum(enum_cls, value):
    """Return `value` as a member of the nanobind enum `enum_cls`.

    Accepts a member, or a string spelled in any of the usual ways: the member
    name ("ShellModelAtan") or lower case with separators ("shell-model-atan",
    "shell_model_atan"). Raises ValueError listing the valid names otherwise.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = re.sub(r"[^a-z0-9]", "", value.lower())
        for member in enum_cls:
            if re.sub(r"[^a-z0-9]", "", member.name.lower()) == key:
                return member
    names = ", ".join(m.name for m in enum_cls)
    raise ValueError(f"{value!r} is not a valid {enum_cls.__name__} (one of: {names})")
