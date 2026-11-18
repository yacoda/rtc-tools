"""
The reader mixins, without the problem class around them.

The tests are skipped for a pymoca without the ``unquote_identifiers`` compiler option.
"""

import os

import pytest

from rtctools.model import _pymoca_basemodelica


def pymoca_reads_base_modelica():
    from pymoca.backends.casadi._options import _get_default_options

    return _pymoca_basemodelica.UNQUOTE_OPTION in _get_default_options()


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
