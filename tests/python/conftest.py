"""pytest fixtures of the Python bindings tests.

The extension is found through PYTHONPATH, or <IMSRG_BUILD_DIR>/python, or build/*/python.
Tests skip (with a message) when imsrg._core was not built.
"""

import glob
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"


def _add_build_dir_to_path():
    cands = []
    if os.environ.get("IMSRG_BUILD_DIR"):
        cands.append(Path(os.environ["IMSRG_BUILD_DIR"]) / "python")
    cands += [Path(p) for p in sorted(glob.glob(str(ROOT / "build" / "*" / "python")))]
    for c in cands:
        if glob.glob(str(c / "imsrg" / "_core*")):
            sys.path.insert(0, str(c))
            return


try:
    import imsrg  # noqa: F401
except ImportError:
    _add_build_dir_to_path()

try:
    import imsrg
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"imsrg extension not built: {exc}", allow_module_level=True)


@pytest.fixture(scope="session")
def golden():
    return GOLDEN


@pytest.fixture(scope="session")
def o16_emax2():
    return imsrg.ModelSpace(2, "O16", "O16", hw=16.0)


def _data_dir():
    cands = []
    if os.environ.get("IMSRG_TEST_DATA_DIR"):
        cands.append(Path(os.environ["IMSRG_TEST_DATA_DIR"]))
    cands += [ROOT.parents[2], Path("/Users/cheshyre/Downloads/new-imsrg")]
    for c in cands:
        if (c / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz").exists():
            return c
    return None


ME2J_NAME = "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz"
NO2B_NAME = "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin"


@pytest.fixture(scope="session")
def me2j_file():
    d = _data_dir()
    if d is None:
        pytest.skip("big 2N input file not found (set IMSRG_TEST_DATA_DIR)")
    return d / ME2J_NAME


@pytest.fixture(scope="session")
def no2b_file():
    d = _data_dir()
    if d is None or not (d / NO2B_NAME).exists():
        pytest.skip("big 3N input file not found (set IMSRG_TEST_DATA_DIR)")
    return d / NO2B_NAME
