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
