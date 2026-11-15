"""Running pipelines the way the ``imsrg`` executable does (design/18 section 2.8).

``run(config)`` calls the C++ driver, so there is exactly one pipeline
implementation; the results the driver writes (``results.json`` and the flow
CSV) are returned as a ``RunResult``.

>>> cfg = imsrg.load_config(file="he4.toml")          # doctest: +SKIP
>>> res = imsrg.pipeline.run(cfg)                     # doctest: +SKIP
>>> res.e0, res.operators["Rp2"]["imsrg"]             # doctest: +SKIP
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core

__all__ = ["RunResult", "load_flow", "run"]


def load_flow(path):
    """Reads a flow CSV written by the driver into ``{column: ndarray}``."""
    path = Path(path)
    with path.open() as f:
        header = f.readline().strip().split(",")
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {name: data[:, i] for i, name in enumerate(header)}


@dataclass
class RunResult:
    """Outputs of one pipeline run.

    Attributes
    ----------
    config : RunConfig
        The resolved configuration that was run.
    directory : pathlib.Path
        The output directory.
    results : dict
        The parsed ``results.json`` (schema in design/16 section 2.10).
    """

    config: object
    directory: Path
    results: dict

    @property
    def e_reference(self):
        """Reference energy in MeV (E_HF in the HF basis)."""
        return self.results["e_reference"]

    @property
    def e_mp2(self):
        """MBPT(2) correlation energy of the normal-ordered Hamiltonian (MeV)."""
        return self.results["e_mp2"]

    @property
    def e_hf(self):
        """Hartree-Fock energy (MeV), or None if no HF was run."""
        hf = self.results.get("hf")
        return None if hf is None else hf["e_hf"]

    @property
    def e0(self):
        """Zero-body part of the evolved Hamiltonian (MeV); the reference energy
        for an HF-only task (no flow stages), None if the run wrote no flow result."""
        imsrg = self.results.get("imsrg")
        return None if imsrg is None else imsrg["e0"]

    @property
    def operators(self):
        """``{name: {"reference": float, "imsrg": float, "file": str}}`` of the
        requested operators (expectation values in the reference state before
        and after the flow)."""
        return self.results.get("operators", {})

    @property
    def valence_interaction(self):
        """Path of the written valence-space interaction, or None."""
        return self.results.get("valence_interaction")

    def flow(self):
        """The flow CSV as ``{column: ndarray}`` (one entry per step)."""
        return load_flow(self.config.flow_file)


def run(config):
    """Runs the pipeline for `config` and returns a ``RunResult``.

    Parameters
    ----------
    config : RunConfig or str or pathlib.Path
        A resolved config (``imsrg.load_config``), or the path of a TOML file.

    Raises
    ------
    ConfigError
        For options that are not implemented yet or inconsistent input.
    InputError
        For unreadable inputs or unwritable outputs.
    NumericalError
        If the flow diverged (``results.json`` is still written).
    """
    if isinstance(config, (str, Path)):
        config = _core.load_config(file=str(config))
    _core.run_config(config)
    directory = Path(config.output_directory)
    results = json.loads((directory / "results.json").read_text())
    return RunResult(config=config, directory=directory, results=results)
