"""io: me2j and NO2B readers, tokyo and NuShellX writers (design/18 section 2.3)."""

import re

import pytest

import imsrg
from imsrg import io

HNO_DUMP = "hf/emax2_O16_2N/HNO.dump"


def test_value_counts_match_the_documented_reference_values():
    assert io.me2j_value_count(14, 28, 14) == 168066656
    assert io.no2b_value_count(6, 12, 18) == 32685840
    # defaults of the truncation (e2max = 2 e1max, lmax = e1max)
    assert io.me2j_value_count(4) == io.me2j_value_count(4, 8, 4)


def test_read_me2j_against_the_harness_dump(golden, me2j_file, o16_emax2):
    v = io.read_me2j(me2j_file, o16_emax2, file_e1max=14, file_e2max=28, file_lmax=14)
    ref = imsrg.read_operator_dump(str(golden / "hamiltonian" / "emax2_O16" / "V2.dump"), o16_emax2)
    assert v.spec.hermiticity == imsrg.Hermiticity.Hermitian and v.spec.particle_rank == 2
    assert v.zero_body == 0.0 and v.one_body_norm() == 0.0
    assert (v - ref).norm() < 1e-10 * ref.norm()
    # str paths work as well, and the model space truncates the file
    assert io.read_me2j(str(me2j_file), o16_emax2, file_e1max=14, file_e2max=28, file_lmax=14).norm() == v.norm()


def test_read_me2j_errors(tmp_path, o16_emax2):
    with pytest.raises(imsrg.InputError):
        io.read_me2j(tmp_path / "missing.me2j", o16_emax2, file_e1max=2)
    short = tmp_path / "short.me2j"
    short.write_text("header\n1.0 2.0\n")
    with pytest.raises(imsrg.InputError):
        io.read_me2j(short, o16_emax2, file_e1max=2)


def test_read_no2b_fills_the_container(no2b_file, o16_emax2):
    v3 = io.read_no2b(no2b_file, o16_emax2, file_e1max=6, file_e2max=12, file_e3max=18, e3max=12)
    assert (v3.emax, v3.e2max, v3.e3max, v3.lmax) == (2, 4, 12, 2)
    assert v3.num_channels > 0 and v3.size > 0
    # the file truncation matters: a wrong one is detected from the file size
    with pytest.raises(imsrg.InputError, match="file3e"):
        io.read_no2b(no2b_file, o16_emax2, file_e1max=4, file_e2max=8, file_e3max=12, e3max=12)
    with pytest.raises(imsrg.InputError):
        io.read_no2b(no2b_file.parent / "missing.stream.bin", o16_emax2, file_e1max=6, file_e2max=12, file_e3max=18, e3max=12)


def test_no2b_container_constructor():
    v3 = io.No2bThreeBody(2, 4, 6, 2)
    assert v3.size > 0 and v3.e3max == 6
    with pytest.raises(imsrg.InputError):
        io.No2bThreeBody(-1, 0, 0, 0)


def _numbers_match(a, b, tol=1e-10):
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > tol:
                return False
        except ValueError:
            return False
    return True


def _compare(path, golden_path):
    got = path.read_text().splitlines()
    ref = golden_path.read_text().splitlines()
    assert len(got) == len(ref), f"{path.name}: {len(got)} lines, golden {len(ref)}"
    for i, (g, r) in enumerate(zip(got[1:], ref[1:]), start=2):  # line 1: version string
        assert _numbers_match(g, r), f"{path.name}:{i}: {g!r} != {r!r}"


@pytest.fixture(scope="module")
def sd_shell_hamiltonian(golden):
    ms = imsrg.ModelSpace(2, "O16", "sd-shell", hw=16.0, lmax=2)
    return imsrg.read_operator_dump(str(golden / HNO_DUMP), ms)


@pytest.mark.parametrize(
    "stem,mode",
    [
        ("O16_sdshell", io.ValenceWriteMode.Hamiltonian),
        ("O16_sdshell_op", io.ValenceWriteMode.IsospinAveragedOperator),
    ],
)
def test_valence_writers_match_imsrgpp(golden, tmp_path, sd_shell_hamiltonian, stem, mode):
    opts = io.ValenceWriteOptions(
        mode=mode, file_2n="none", file_3n="none", e1max=2, e2max=4, e3max=6,
        hbar_omega=16.0, a_ref=16, z_ref=8, a_target=16.0,
        version=re.sub(r"\s+", " ", (golden / "io" / f"{stem}.snt").read_text().splitlines()[0]),
    )
    int_opts = io.ValenceWriteOptions(**{k: getattr(opts, k) for k in (
        "mode", "file_2n", "file_3n", "e1max", "e2max", "e3max", "hbar_omega", "a_target", "version")})
    h = sd_shell_hamiltonian
    io.write_tokyo(tmp_path / f"{stem}.snt", h, opts)
    io.write_nushellx_int(tmp_path / f"{stem}.int", h, int_opts)
    io.write_nushellx_sps(tmp_path / f"{stem}.sp", h)
    for ext in ("snt", "int", "sp"):
        _compare(tmp_path / f"{stem}.{ext}", golden / "io" / f"{stem}.{ext}")


def test_writers_accept_a_host_edited_operator(golden, tmp_path, sd_shell_hamiltonian):
    h = sd_shell_hamiltonian.copy()
    h.sync_to_host()
    h.one_body_view(writable=True)[:] *= 1.0  # HostCurrent: the writer must not clobber it
    io.write_tokyo(tmp_path / "x.snt", h, io.ValenceWriteOptions(hbar_omega=16.0))
    assert (tmp_path / "x.snt").stat().st_size > 0


def test_write_errors(tmp_path, sd_shell_hamiltonian):
    with pytest.raises(imsrg.RuntimeError):
        io.write_tokyo(tmp_path / "no" / "such" / "dir.snt", sd_shell_hamiltonian)


def test_options_defaults_and_attributes():
    o = io.ValenceWriteOptions()
    assert o.mode == io.ValenceWriteMode.Hamiltonian and o.hbar_omega == 0.0
    o.hbar_omega = 20.0
    o.file_2n = "x.me2j"
    assert (o.hbar_omega, o.file_2n) == (20.0, "x.me2j")
