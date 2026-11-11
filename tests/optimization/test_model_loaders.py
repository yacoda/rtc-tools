"""
The lacemodelica and Rumoca model loaders, compared with pymoca on the examples.

The integration tests are skipped without the tools: the ``lacemodelica`` executable on the PATH
and a CasADi with ONNX import, or the Rumoca executable (``RTCTOOLS_RUMOCA``, default ``rumoca``).
The Base Modelica files in ``data/base_modelica`` are OpenModelica's exports of the examples
(``OMCallBack.exportBaseModelica`` with ``--baseModelica``).
"""

import importlib.metadata
import importlib.resources
import os
import shutil
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from rtctools._internal import lacemodelica_loader, rumoca_loader

EXAMPLES = Path(__file__).parents[2] / "examples"
BASE_MODELICA = Path(__file__).parent / "data" / "base_modelica"
TOLERANCE = 1e-9


def compiler_options():
    """The defaults of ModelicaMixin.compiler_options, without the cache"""
    library_folders = [
        str(importlib.resources.files(ep.module).joinpath(ep.attr))
        for ep in importlib.metadata.entry_points(group="rtctools.libraries.modelica")
        if ep.name == "library_folder"
    ]
    return {
        "expand_vectors": True,
        "library_folders": library_folders,
        "eliminate_constant_assignments": True,
        "replace_constant_values": True,
        "replace_constant_expressions": True,
        "replace_parameter_expressions": True,
        "eliminable_variable_expression": r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z",
        "expand_mx": True,
        "detect_aliases": True,
        "allow_derivative_aliases": False,
        "cache": False,
    }


def pymoca_model(folder, **options):
    import pymoca.backends.casadi.api

    options = {**compiler_options(), **options}
    return pymoca.backends.casadi.api.transfer_model(str(folder), "Example", options)


def pymoca_reads_base_modelica():
    from pymoca.backends.casadi._options import _get_default_options

    return "basemodelica" in _get_default_options()


def stable_random(name, salt):
    """A number in (0.5, 1.5) that only depends on the name, so both models get the same"""
    h = 2166136261 + salt
    for c in name.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return 0.5 + h / 2**32


def alias_key(model, name):
    """Alias detection may keep another member of a group: the smallest name, and the sign"""
    members = {
        m.lstrip("-"): -1 if m.startswith("-") else 1 for m in model.alias_relation.aliases(name)
    }
    members.setdefault(name, 1)
    key = min(members)
    return key, members[key] * members[name]


def keys(model, group):
    return sorted(alias_key(model, v.symbol.name())[0] for v in getattr(model, group))


def variable_keys(model):
    """
    The keys of the differential and of the algebraic variables.

    A state of which no equation has the derivative counts as algebraic: pymoca makes every
    element of an array a state, where OpenModelica and Rumoca only make those that are.
    """
    used = ca.DM(model.dae_residual_function.sparsity_jac(2, 0), 1).full().any(axis=0)
    differential = [v for v, u in zip(model.states, used, strict=True) if u]
    algebraic = [v for v, u in zip(model.states, used, strict=True) if not u] + model.alg_states
    return [
        sorted(alias_key(model, v.symbol.name())[0] for v in g) for g in (differential, algebraic)
    ]


def residuals(model, function, salt):
    """The sorted absolute values of a residual at a random point"""

    def point(group, prefix=""):
        values = []
        for v in getattr(model, group):
            key, sign = alias_key(model, v.symbol.name())
            values.append(sign * stable_random(prefix + key, salt))
        return ca.DM(values)

    args = [
        ca.DM(stable_random("time", salt)),
        point("states"),
        # A derivative goes by the key of its state
        point("states", "der:"),
        point("alg_states"),
        point("inputs"),
        ca.DM([float(ca.evalf(ca.MX(c.value))) for c in model.constants]),
        ca.DM([stable_random(p.symbol.name(), salt) for p in model.parameters]),
    ]
    return np.sort(np.abs(np.array(function(*args)).ravel()))


def assert_same_model(reference, model):
    differential, algebraic = variable_keys(model)
    assert differential == variable_keys(reference)[0]
    assert algebraic == variable_keys(reference)[1]
    for group in ("inputs", "parameters"):
        assert keys(model, group) == keys(reference, group), group
    for function in ("dae_residual_function", "initial_residual_function"):
        f, g = getattr(reference, function), getattr(model, function)
        assert f.n_out() == g.n_out(), function
        if f.n_out() == 0:
            continue
        for salt in range(3):
            x, y = residuals(reference, f, salt), residuals(model, g, salt)
            assert x.shape == y.shape, function
            np.testing.assert_allclose(y, x, rtol=TOLERANCE, atol=TOLERANCE, err_msg=function)


# Without the tools


def test_lacemodelica_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Example.bmo"):
        lacemodelica_loader.transfer_model(str(tmp_path), "Example", {})


def test_lacemodelica_missing_executable(tmp_path, monkeypatch):
    (tmp_path / "Example.bmo").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="lacemodelica executable on the PATH"):
        lacemodelica_loader.transfer_model(str(tmp_path), "Example", {})


def test_lacemodelica_casadi_without_onnx(tmp_path, monkeypatch):
    (tmp_path / "Example.bmo").write_text("")
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/lacemodelica")
    monkeypatch.delattr(ca, "GraphBuilder", raising=False)
    with pytest.raises(ImportError, match="CasADi with ONNX import"):
        lacemodelica_loader.transfer_model(str(tmp_path), "Example", {})


def test_rumoca_missing_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("RTCTOOLS_RUMOCA", str(tmp_path / "rumoca"))
    with pytest.raises(FileNotFoundError, match="set RTCTOOLS_RUMOCA"):
        rumoca_loader.transfer_model(str(EXAMPLES / "basic" / "model"), "Example", {})


def test_rumoca_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model Example"):
        rumoca_loader.transfer_model(str(tmp_path), "Example", {})


# With the tools


@pytest.mark.skipif(shutil.which("lacemodelica") is None, reason="lacemodelica not on the PATH")
@pytest.mark.parametrize("example", ["basic", "channel_pulse"])
def test_lacemodelica_as_pymoca(example):
    folder = BASE_MODELICA / example
    try:
        model = lacemodelica_loader.transfer_model(str(folder), "Example", compiler_options())
    except ImportError as error:
        pytest.skip(str(error))
    if pymoca_reads_base_modelica():
        # The same Base Modelica file, so that only the frontend differs
        reference = pymoca_model(folder, basemodelica=True)
    elif example == "basic":
        reference = pymoca_model(EXAMPLES / example / "model")
    else:
        pytest.skip("OpenModelica flattens this example otherwise than pymoca does")
    assert_same_model(reference, model)


@pytest.mark.skipif(
    shutil.which(os.environ.get("RTCTOOLS_RUMOCA", "rumoca")) is None, reason="Rumoca not found"
)
@pytest.mark.parametrize(
    "example",
    [
        "basic",
        pytest.param(
            "channel_pulse",
            marks=pytest.mark.xfail(
                raises=RuntimeError,
                reason="Rumoca rejects rtc-tools-channel-flow: a package extends a record",
            ),
        ),
    ],
)
def test_rumoca_as_pymoca(example):
    folder = EXAMPLES / example / "model"
    model = rumoca_loader.transfer_model(str(folder), "Example", compiler_options())
    assert_same_model(pymoca_model(folder), model)
