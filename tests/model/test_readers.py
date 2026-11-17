"""
The reader mixins, without the problem class around them; lacemodelica compared with pymoca.

The integration tests are skipped without a CasADi whose DaeBuilder imports Base Modelica (its
lacemodelica plugin and ONNX import).
The Base Modelica files in ``data/base_modelica`` were flattened by OpenModelica from the examples
(``OMCallBack.exportBaseModelica`` with ``--baseModelica``).
"""

import importlib.metadata
import importlib.resources
import os
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from rtctools.model import _pymoca_basemodelica, lacemodelica_reader_mixin

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

    return _pymoca_basemodelica.UNQUOTE_OPTION in _get_default_options()


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


_TANK = """//! base 0.1.0
package 'Tank'
  model 'Tank'
    parameter Real 'tank.area' = 2.0;
    input Real 'q.in'(fixed = true);
    Real 'tank.h[1]'(start = 1.0, fixed = true);
    Real 'tank.h[2]';
  equation
    'tank.area' * der('tank.h[1]') = 'q.in';
    'tank.h[2]' = 2.0 * 'tank.h[1]';
  end 'Tank';
end 'Tank';
"""

requires_unquote = pytest.mark.skipif(
    not pymoca_reads_base_modelica(), reason="pymoca has no unquote_identifiers option"
)


@requires_unquote
def test_pymoca_basemodelica_names(tmp_path):
    """Quoted names become plain ones, the wrapped model is found, other files are not read"""
    (tmp_path / "Tank.bmo").write_text(_TANK)
    (tmp_path / "Other.bmo").write_text("this is not parsed")
    model = _pymoca_basemodelica._transfer_model(str(tmp_path), "Tank", {})
    assert [v.symbol.name() for v in model.states] == ["tank.h[1]"]
    assert [v.symbol.name() for v in model.alg_states] == ["tank.h[2]"]
    assert [v.symbol.name() for v in model.inputs] == ["q.in"]
    assert [p.symbol.name() for p in model.parameters] == ["tank.area"]
    assert model.parameters[0].value == 2.0


@requires_unquote
def test_pymoca_basemodelica_cache(tmp_path):
    """pymoca's cache holds the compiled model, until the .bmo file is newer than it"""
    bmo = tmp_path / "Tank.bmo"
    bmo.write_text(_TANK)
    options = {"cache": True}
    _pymoca_basemodelica._transfer_model(str(tmp_path), "Tank", options)
    cache = tmp_path / "Tank.pymoca_cache"
    assert cache.is_file()
    model = _pymoca_basemodelica._transfer_model(str(tmp_path), "Tank", options)
    assert type(model).__name__ == "CachedModel"
    assert model.parameters[0].value == 2.0

    bmo.write_text(_TANK.replace("= 2.0;", "= 3.0;"))
    os.utime(bmo, (cache.stat().st_mtime + 10,) * 2)
    model = _pymoca_basemodelica._transfer_model(str(tmp_path), "Tank", options)
    assert type(model).__name__ != "CachedModel"
    assert model.parameters[0].value == 3.0


@requires_unquote
def test_pymoca_basemodelica_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Example.bmo"):
        _pymoca_basemodelica._transfer_model(str(tmp_path), "Example", {})


def test_lacemodelica_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Example.bmo"):
        lacemodelica_reader_mixin._transfer_model(str(tmp_path), "Example", {})


def _lacemodelica_supported():
    try:
        lacemodelica_reader_mixin._check_casadi()
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("missing", ["plugin", "onnx", "daebuilder"])
def test_lacemodelica_casadi_without_support(tmp_path, monkeypatch, missing):
    """A clear error for a CasADi without the lacemodelica plugin, ONNX import or .bmo import"""
    (tmp_path / "Example.bmo").write_text("")
    if missing == "plugin":
        monkeypatch.setattr(ca, "has_modelicaparser", lambda name: False)
    elif missing == "onnx":
        monkeypatch.setattr(ca, "has_graphmodel", lambda name: False)
    else:
        monkeypatch.delattr(ca.DaeBuilder, "delayed", raising=False)
    with pytest.raises(ImportError, match="CasADi with its lacemodelica plugin"):
        lacemodelica_reader_mixin._transfer_model(str(tmp_path), "Example", {})


# With the tools


@pytest.mark.skipif(
    not _lacemodelica_supported(), reason="CasADi cannot import Base Modelica (lacemodelica)"
)
@pytest.mark.parametrize("example", ["basic", "channel_pulse"])
def test_lacemodelica_as_pymoca(example):
    folder = BASE_MODELICA / example
    model = lacemodelica_reader_mixin._transfer_model(str(folder), "Example", compiler_options())
    if pymoca_reads_base_modelica():
        # The same Base Modelica file, so that only the frontend differs
        reference = _pymoca_basemodelica._transfer_model(str(folder), "Example", compiler_options())
    elif example == "basic":
        reference = pymoca_model(EXAMPLES / example / "model")
    else:
        pytest.skip("OpenModelica flattens this example otherwise than pymoca does")
    assert_same_model(reference, model)
