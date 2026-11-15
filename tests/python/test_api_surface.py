"""API surface hygiene (design/18 section 5.1, tier 1): docstrings, enum round trips,
error mapping and the exception hierarchy."""

import builtins
import copy
import inspect
import re
from pathlib import Path

import pytest

import imsrg
from imsrg import _core, flow, hf, io


def _own_doc(obj):
    """The docstring without nanobind's generated signature lines."""
    doc = inspect.getdoc(obj) or ""
    lines = [ln for ln in doc.splitlines() if ln.strip()]
    kept = [
        ln
        for ln in lines
        if not re.match(r"^\s*(\w+\.)?\w+\(.*\)( -> .*)?$", ln) and not ln.startswith("Overloaded function")
    ]
    return "\n".join(kept).strip()


def _public_members(cls):
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if name in dir(Exception) or name in ("name", "value"):  # builtin / enum plumbing
            continue
        yield name, inspect.getattr_static(cls, name)


def test_every_public_symbol_of_core_has_a_docstring():
    missing = []
    for name in dir(_core):
        if name.startswith("_"):
            continue
        obj = getattr(_core, name)
        if not _own_doc(obj):
            missing.append(name)
        if inspect.isclass(obj) and not issubclass(obj, BaseException):
            for mname, member in _public_members(obj):
                if issubclass(obj, __import__("enum").Enum) and isinstance(getattr(obj, mname, None), obj):
                    continue
                if mname in ("edit", "transform", "normal_ordered_hamiltonian") and obj is not _core.Operator:
                    pass
                if not _own_doc(member):
                    missing.append(f"{name}.{mname}")
    assert not missing, "public symbols of imsrg._core without a docstring: " + ", ".join(sorted(missing))


def test_python_layer_functions_are_documented():
    for mod in (io, hf, flow, imsrg.commutators, imsrg.mbpt, imsrg.pipeline, imsrg.operators):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            assert _own_doc(obj), f"{mod.__name__}.{name}"


def test_enums_round_trip():
    for enum in (
        imsrg.Parity, imsrg.Hermiticity, imsrg.Residency, imsrg.ResidencyPolicy, hf.Mixing, flow.GeneratorKind,
        flow.Partitioning, flow.StopReason, flow.OdeMethod, imsrg.commutators.Term, io.No2bPrecision,
        io.ValenceWriteMode,
    ):
        for member in enum:
            assert enum(member.value) is member
            assert enum[member.name] is member
            assert copy.copy(member) is member and copy.deepcopy(member) is member


def test_repr_is_stable_and_informative(o16_emax2):
    assert repr(imsrg.Orbit(0, 1, 3, -1)).startswith("Orbit(")
    assert repr(imsrg.OperatorSpec()).startswith("OperatorSpec(particle_rank=2")
    assert repr(imsrg.commutators.TermMask.only(imsrg.commutators.Term.Comm110ss)) == "TermMask([comm110ss])"
    op = imsrg.random_operator(o16_emax2)
    assert repr(op) == repr(op) and "norb=12" in repr(op)


# ------------------------------------------------------------- exception hierarchy

HIERARCHY = {
    "ImsrgError": builtins.RuntimeError,
    "InputError": imsrg.ImsrgError,
    "ConfigError": imsrg.InputError,
    "InputFileNotFoundError": imsrg.InputError,
    "FileFormatError": imsrg.InputError,
    "ModelSpaceError": imsrg.ImsrgError,
    "SymmetryError": imsrg.ImsrgError,
    "InvariantError": imsrg.ImsrgError,
    "NumericalError": imsrg.ImsrgError,
    "ConvergenceError": imsrg.NumericalError,
    "RuntimeError": imsrg.ImsrgError,
    "NotImplementedError": imsrg.ImsrgError,
    "StaleHostError": imsrg.ImsrgError,
}


@pytest.mark.parametrize("name,base", HIERARCHY.items())
def test_exception_hierarchy(name, base):
    cls = getattr(imsrg, name)
    assert issubclass(cls, base) and issubclass(cls, imsrg.ImsrgError) and _own_doc(cls)


def test_builtin_bases():
    assert issubclass(imsrg.NotImplementedError, builtins.NotImplementedError)
    assert issubclass(imsrg.RuntimeError, builtins.RuntimeError)
    assert issubclass(imsrg.InputFileNotFoundError, FileNotFoundError)
    assert issubclass(imsrg.ImsrgError, builtins.RuntimeError)


def test_missing_file_is_a_file_not_found_error(tmp_path, o16_emax2):
    missing = tmp_path / "nope"
    for call in (
        lambda: io.read_me2j(missing, o16_emax2, file_e1max=2),
        lambda: io.read_no2b(missing, o16_emax2, file_e1max=2, file_e2max=4, file_e3max=6, e3max=6),
        lambda: imsrg.read_operator_dump(str(missing), o16_emax2),
        lambda: imsrg.read_operator_dump_header(str(missing)),
        lambda: imsrg.load_config(file=str(missing)),
    ):
        with pytest.raises(FileNotFoundError) as info:
            call()
        assert isinstance(info.value, imsrg.InputError) and str(missing) in str(info.value)


def test_malformed_file_is_a_file_format_error(tmp_path, o16_emax2):
    bad = tmp_path / "bad.me2j"
    bad.write_text("header\n1.0 2.0\n")
    with pytest.raises(imsrg.FileFormatError):
        io.read_me2j(bad, o16_emax2, file_e1max=2)
    junk = tmp_path / "junk.dump"
    junk.write_text("not a dump\n")
    with pytest.raises(imsrg.FileFormatError):
        imsrg.read_operator_dump(str(junk), o16_emax2)
    with pytest.raises(imsrg.InputError):  # still an InputError for old callers
        imsrg.read_operator_dump(str(junk), o16_emax2)


def test_error_mapping_of_minimal_failing_calls(o16_emax2):
    op = imsrg.random_operator(o16_emax2)
    he4 = imsrg.ModelSpace(1, "He4", "He4", hw=16.0)
    other = imsrg.random_operator(he4)
    with pytest.raises(imsrg.ModelSpaceError):
        op + other
    with pytest.raises(imsrg.ModelSpaceError):
        imsrg.commutators.commutator(op, other)
    with pytest.raises(imsrg.ModelSpaceError):
        flow.bch_product(op, other)
    ws = imsrg.commutators.CommutatorWorkspace(he4)
    with pytest.raises(imsrg.ModelSpaceError):
        imsrg.commutators.commutator(op, op, workspace=ws)
    with pytest.raises(imsrg.ImsrgError):
        o16_emax2.channel(10**6)
    with pytest.raises(imsrg.NotImplementedError):
        imsrg.Operator(o16_emax2, imsrg.OperatorSpec(rank_J=2))
    with pytest.raises(builtins.NotImplementedError):  # also catchable as the builtin
        imsrg.Operator(o16_emax2, imsrg.OperatorSpec(rank_J=2))
    with pytest.raises(imsrg.ConfigError):
        imsrg.load_config(text="[basis]\nemaxx = 2\n")
    with pytest.raises(imsrg.InputError):
        flow.MagnusSolver(op, solver=flow.SolverOptions(dsmax=0.0))
    with pytest.raises(imsrg.RuntimeError):
        imsrg.Runtime()  # already initialized
    with imsrg.strict_residency():
        with pytest.raises(imsrg.StaleHostError):
            op.one_body


def test_hf_raise_on_failure(golden, o16_emax2):
    h = imsrg.read_operator_dump(str(golden / "hamiltonian" / "emax2_O16" / "H.dump"), o16_emax2)
    with pytest.raises(imsrg.ConvergenceError) as info:
        hf.solve(h, max_iterations=2, raise_on_failure=True)
    assert not info.value.result.converged and isinstance(info.value, imsrg.NumericalError)
    assert not hf.solve(h, max_iterations=2).converged  # default: reported, not raised


def test_stubs_in_sync_if_present():
    stub = Path(imsrg.__file__).parent / "_core.pyi"
    if not stub.exists():
        pytest.skip("no type stubs are generated yet")
    text = stub.read_text()
    for name in dir(_core):
        if not name.startswith("_"):
            assert re.search(rf"\b{name}\b", text), f"{name} missing in _core.pyi"
