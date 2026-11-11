"""Config loading from examples/config/a_*.toml."""

import glob
import tomllib
from pathlib import Path

import pytest

import imsrg

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted(glob.glob(str(ROOT / "examples" / "config" / "a_*.toml")))


@pytest.mark.parametrize("path", EXAMPLES)
def test_load_example(path):
    cfg = imsrg.load_config(file=path)
    d = cfg.to_dict()
    assert isinstance(d, dict) and "basis" in d
    # canonical TOML re-loads to the same resolved config
    again = imsrg.load_config(text=cfg.to_toml(include_provenance=False))
    assert again.to_dict() == d


def test_text_and_set_override():
    text = Path(EXAMPLES[0]).read_text()
    cfg = imsrg.load_config(text=text, set=["basis.emax=4"])
    assert cfg.to_dict()["basis"]["emax"] == 4
    tomllib.loads(cfg.to_toml())


def test_bad_config_raises_config_error():
    with pytest.raises(imsrg.ConfigError):
        imsrg.load_config(text="[basis]\nemaxx = 2\n")
    assert issubclass(imsrg.ConfigError, imsrg.InputError)
