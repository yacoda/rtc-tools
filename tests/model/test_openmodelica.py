"""
OpenModelicaFlattenerMixin with the real OpenModelica compiler (omc).

This test runs in the OpenModelica CI job (modelica-conformance in .github/workflows/ci.yml, tox
environment omc), next to tests/modelica/test_omc_conformance.py, and is gated the same way: it
only runs when OMC_VERSION is set and the omc the flattener runs (RTCTOOLS_OMC, default omc on
the PATH) has that version, and is skipped otherwise. It requires the Modelica Standard Library:

    omc tests/modelica/install_msl.mos

- The flattening: omc writes <model>.bmo for examples, also from a relative model folder, and
  with the cache a second flattening reuses it. With an rtc-tools-channel-flow whose library
  omc refuses, channel_pulse is an expected failure.
- End to end: the basic example solved through the flattener, so pymoca reading the .bmo, gives
  the results of the Modelica sources read by pymoca, with ModelicaMixin and SimulationProblem.
  pymoca reads Base Modelica with its unquote_identifiers option; these tests are skipped with a
  pymoca without it, and start running once pymoca has the option.
"""

import functools
import importlib.metadata
import importlib.resources
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rtctools.model._flatten import tool_command
from rtctools.model._pymoca_basemodelica import UNQUOTE_OPTION
from rtctools.model.openmodelica_flattener_mixin import OpenModelicaFlattenerMixin

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# The omc version of the CI job, declared in ci.yml
OMC_VERSION = os.environ.get("OMC_VERSION")


@functools.cache
def flattener_omc_version():
    """The version of the omc the flattener runs, or None if there is none."""
    try:
        result = subprocess.run(
            tool_command("RTCTOOLS_OMC", "omc") + ["--version"], capture_output=True, text=True
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip().removeprefix("OpenModelica ") or None


def skip_reason():
    if OMC_VERSION is None:
        return "OMC_VERSION is not set, see .github/workflows/ci.yml"
    if flattener_omc_version() != OMC_VERSION:
        return f"requires omc {OMC_VERSION}, found {flattener_omc_version()}"
    return None


pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=str(skip_reason()))


def library_folders():
    """The Modelica library folders of installed packages, as ModelicaMixin finds them"""
    return [
        str(importlib.resources.files(ep.module).joinpath(ep.attr))
        for ep in importlib.metadata.entry_points(group="rtctools.libraries.modelica")
        if ep.name == "library_folder"
    ]


@pytest.fixture
def folder():
    with tempfile.TemporaryDirectory() as temporary:
        yield Path(temporary)


def copy_example(example, target, parts=("model",)):
    for part in parts:
        shutil.copytree(EXAMPLES / example / part, target / part)
    return target


class Loaded:
    """What comes after the flattener: returns the folder and name it is given"""

    def pymoca_model(self, model_folder, model_name, compiler_options):
        return model_folder, model_name


def flatten(model_folder, libraries, cache=False):
    problem = type("Problem", (OpenModelicaFlattenerMixin, Loaded), {})()
    options = {"library_folders": libraries, "cache": cache}
    assert problem.pymoca_model(model_folder, "Example", options) == (
        model_folder,
        "Example",
    )
    return Path(model_folder) / "Example.bmo"


def check_bmo(path, variable):
    text = path.read_text()
    assert text.startswith("//! base "), text[:200]
    assert "package 'Example'" in text
    assert "model 'Example'" in text
    assert "end 'Example';" in text
    assert f"'{variable}'" in text


# rtc-tools-channel-flow up to 1.5.0 calls max() with two arrays, which omc refuses (a known
# failure in tests/modelica/test_omc_conformance.py, CHECK_DELTARES_MAX)
DELTARES_MAX = r"No matching function found for max\(\S+\.H_b\["


@pytest.mark.parametrize(
    "example, variable", [("basic", "storage.V"), ("channel_pulse", "Inflow_Q")]
)
def test_flatten(folder, example, variable):
    # The channel flow library comes from the entry point of rtc-tools-channel-flow
    assert any("rtctools_channel_flow" in f for f in library_folders())
    copy_example(example, folder)
    try:
        path = flatten(str(folder / "model"), library_folders())
    except RuntimeError as error:
        if re.search(DELTARES_MAX, str(error)):
            pytest.xfail("this rtc-tools-channel-flow calls max() with two arrays")
        raise
    check_bmo(path, variable)


def test_flatten_relative_folders(folder, monkeypatch):
    # omc runs in a temporary folder, so the flattener must resolve relative folders
    copy_example("basic", folder)
    monkeypatch.chdir(folder)
    libraries = [os.path.relpath(f, folder) for f in library_folders()]
    check_bmo(flatten("model", libraries), "storage.V")


def test_flatten_cached(folder, caplog):
    # The second flattening with the cache reuses the file of the first
    copy_example("basic", folder)
    path = flatten(str(folder / "model"), library_folders(), cache=True)
    check_bmo(path, "storage.V")
    assert (folder / "model" / "Example.bmo_cache").is_file()
    caplog.set_level("INFO", "rtctools")
    caplog.clear()
    assert flatten(str(folder / "model"), library_folders(), cache=True) == path
    assert "Reusing" in caplog.text


def pymoca_reads_base_modelica():
    from pymoca.backends.casadi._options import _get_default_options

    return UNQUOTE_OPTION in _get_default_options()


needs_unquote = pytest.mark.skipif(
    not pymoca_reads_base_modelica(),
    reason=f"pymoca has no {UNQUOTE_OPTION} option, so it does not read Base Modelica",
)


def basic_optimization(flattener, folder):
    from rtctools.optimization.collocated_integrated_optimization_problem import (
        CollocatedIntegratedOptimizationProblem,
    )
    from rtctools.optimization.csv_mixin import CSVMixin
    from rtctools.optimization.modelica_mixin import ModelicaMixin

    class Example(*flattener, CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
        # examples/basic/src/example.py
        def objective(self, ensemble_member):
            return self.integral("Q_release", ensemble_member=ensemble_member)

        def path_constraints(self, ensemble_member):
            constraints = super().path_constraints(ensemble_member)
            constraints.append((self.state("storage.V"), 380000, 420000))
            return constraints

    copy_example("basic", folder, ("model", "input"))
    (folder / "output").mkdir()
    problem = Example(
        model_folder=str(folder / "model"),
        input_folder=str(folder / "input"),
        output_folder=str(folder / "output"),
    )
    assert problem.optimize()
    return problem.extract_results()


def basic_simulation(flattener, folder):
    from rtctools.simulation.simulation_problem import SimulationProblem

    class Example(*flattener, SimulationProblem):
        pass

    copy_example("basic", folder)
    problem = Example(model_folder=str(folder / "model"))
    problem.setup_experiment(0.0, 12 * 3600.0, 3600.0)
    problem.set_var("storage.V", 400000.0)
    problem.set_var("Q_in", 1.5)
    problem.set_var("Q_release", 1.0)
    problem.initialize()
    volumes = [problem.get_var("storage.V")]
    for step in range(12):
        problem.set_var("Q_in", 1.5 + 0.1 * step)
        problem.set_var("Q_release", 1.0 + 0.05 * step)
        problem.update(-1)
        volumes.append(problem.get_var("storage.V"))
    return {"storage.V": np.array(volumes)}


@needs_unquote
@pytest.mark.parametrize("run", [basic_optimization, basic_simulation])
def test_basic_same_results(folder, run):
    plain = run((), folder / "plain")
    flattened = run((OpenModelicaFlattenerMixin,), folder / "flattened")
    assert (folder / "flattened" / "model" / "Example.bmo").is_file()
    assert not (folder / "plain" / "model" / "Example.bmo").exists()
    for name in ("storage.V", "Q_release", "Q_in", "V_storage"):
        if name in plain:
            np.testing.assert_allclose(flattened[name], plain[name], rtol=1e-6, atol=1e-6)
