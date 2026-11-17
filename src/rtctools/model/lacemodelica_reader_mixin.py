"""
Loading a Base Modelica model with lacemodelica, in place of pymoca's parser and flattener.

CasADi's DaeBuilder reads ``<model_folder>/<model_name>.bmo``: it translates the file with its
``lacemodelica`` ModelicaParser plugin to a model description and an ONNX graph of the equations,
and imports both. Its variables, equations, initial equations and delays fill a pymoca ``Model``,
which pymoca then simplifies with the compiler options, so that rtc-tools gets what it gets from
pymoca. CasADi must have the lacemodelica plugin (a build with WITH_LACEMODELICA) and ONNX import.
"""

import math
import re
from pathlib import Path

import casadi as ca

from rtctools._internal.modelica_mixin_base import transfer_with_cache_retry

_TYPES = {"Float64": float, "Int32": int, "Boolean": bool, "Enumeration": int}
_CASADI_REQUIRED = (
    "lacemodelica models need a CasADi with its lacemodelica plugin (WITH_LACEMODELICA) and ONNX "
    "import, including the Loop operator, and a DaeBuilder that imports .bmo models with "
    "explicit_equations, attribute_expr, has_attribute, fixed and delayed (a build of CasADi "
    "main from November 2026 or newer)"
)


def _check_casadi() -> None:
    try:
        complete = (
            ca.has_modelicaparser("lacemodelica")
            and ca.has_graphmodel("onnx")
            and all(
                hasattr(ca.DaeBuilder, m)
                for m in ("attribute_expr", "has_attribute", "fixed", "delayed")
            )
        )
    except (AttributeError, RuntimeError):
        complete = False
    if not complete:
        raise ImportError(_CASADI_REQUIRED)


def _modelica_shape(name: str) -> tuple:
    """The shape format of pymoca: one entry per dotted part of the name"""
    return tuple((None,) for _ in re.sub(r"^der\((.*)\)$", r"\1", name).split("."))


def _dae_builder(path: Path) -> ca.DaeBuilder:
    _check_casadi()
    try:
        # The equations as residuals, as they are in the model
        return ca.DaeBuilder(path.stem, str(path), {"explicit_equations": False})
    except RuntimeError as error:
        if "is not supported" in str(error):
            raise ImportError(f"{_CASADI_REQUIRED}: {error}") from error
        raise RuntimeError(f"lacemodelica failed on {path}:\n{error}") from error


def _build_from_dae(dae: ca.DaeBuilder):
    from pymoca.backends.casadi.model import DelayArgument, Model, Variable

    delayed = list(dae.delayed())
    derivatives = {x: dae.der(x) for x in dae.x()}
    is_derivative = set(derivatives.values())
    # The variables of the model, not the equation residuals that DaeBuilder adds
    names = [n for n in dae.all() if dae.category(n) != "calculated" and n != dae.time().name()]

    symbols = {}
    for name in names:
        symbol = dae.var(name)
        symbol._modelica_shape = _modelica_shape(name)
        symbols[name] = symbol
    # rtc-tools knows a delayed expression by the name pymoca gives it
    old = [dae.var(n) for n in delayed]
    new = [ca.MX.sym(f"_pymoca_delay_{i}", *v.shape) for i, v in enumerate(old)]
    for name, symbol in zip(delayed, new, strict=True):
        symbol._modelica_shape = symbol.shape
        symbols[name] = symbol

    def expressions(exprs):
        return ca.substitute(list(exprs), old, new) if old and exprs else list(exprs)

    def attribute(name, key, python_type):
        """
        An expression of other variables, else a literal. None when the model gives none, so
        that a default does not count as given, as with pymoca: nominal = 1 in the model is
        given, no nominal is not.
        """
        if not dae.has_attribute(key, name):
            return None
        expr = dae.attribute_expr(key, name)
        if not expr.is_empty():
            return expressions([expr])[0]
        value = getattr(dae, key)(name)
        value = value[0] if isinstance(value, (list, tuple)) else value
        if math.isnan(value):
            return None
        return python_type(value)

    model = Model()
    model.time = dae.time()

    for name in names:
        # An array without elements, such as x[0], has nothing to solve for
        if name in is_derivative or dae.numel(name) == 0:
            continue
        python_type = _TYPES[dae.type(name)]
        variable = Variable(symbols[name], python_type)
        for key in ("min", "max"):
            value = attribute(name, key, python_type)
            if value is not None:
                setattr(variable, key, value)
        causality = dae.causality(name)
        if causality in ("parameter", "structuralParameter"):
            value = attribute(name, "start", python_type)
            if value is not None:
                variable.value = value
            model.parameters.append(variable)
            continue
        start = attribute(name, "start", python_type)
        if start is not None:
            variable.start = start
        nominal = attribute(name, "nominal", float)
        if nominal is not None:
            variable.nominal = nominal
        variable.fixed = dae.fixed(name)
        if name in derivatives:
            model.states.append(variable)
            model.der_states.append(Variable(symbols[derivatives[name]], float))
        elif name in delayed or causality == "input":
            model.inputs.append(variable)
        else:
            model.alg_states.append(variable)
        if causality == "output":
            model.outputs.append(name)

    expr, time = dae.outputs("delay_expr"), dae.outputs("delay_time")
    for i, (e, t) in enumerate(zip(expressions(expr), expressions(time), strict=True)):
        model.delay_states.append(f"_pymoca_delay_{i}")
        model.delay_arguments.append(DelayArgument(e, t))
    model.equations = [ca.vec(e) for e in expressions(dae.outputs("alg"))]
    model.initial_equations = [ca.vec(e) for e in expressions(dae.outputs("ialg"))]
    return model


def _transfer_model(model_folder: str, model_name: str, compiler_options: dict):
    """
    Compile ``<model_folder>/<model_name>.bmo`` with lacemodelica, through CasADi's DaeBuilder.

    :returns: a ``pymoca.backends.casadi.model.Model``, simplified with ``compiler_options``.
    """
    from pymoca.backends.casadi.api import _merge_default_options

    compiler_options = _merge_default_options(dict(compiler_options))
    path = Path(model_folder) / f"{model_name}.bmo"
    if not path.is_file():
        raise FileNotFoundError(f"Base Modelica file {path} not found")
    model = _build_from_dae(_dae_builder(path))
    model.simplify(compiler_options)
    model._post_checks()
    return model


class LacemodelicaReaderMixin:
    """
    Reads the model from Base Modelica with lacemodelica, in place of pymoca.

    CasADi's DaeBuilder reads ``<model_name>.bmo`` in the model folder with its lacemodelica
    plugin, independently of pymoca's parser and flattener; the file comes from a tool such as
    OpenModelica, beforehand or at every run by a flattener mixin such as
    :py:class:`OpenModelicaFlattenerMixin`. So it needs ``model_format = "basemodelica"``, set by
    the flattener or in the problem class. Its variables, equations, initial equations and delays
    fill a pymoca model, which pymoca simplifies with the compiler options. It needs a CasADi
    built with the lacemodelica plugin and ONNX import.

    With a flattener, list it after the flattener::

        class Example(
            OpenModelicaFlattenerMixin,
            LacemodelicaReaderMixin,
            CSVMixin,
            ModelicaMixin,
            CollocatedIntegratedOptimizationProblem,
        ):
            ...
    """

    _modelica_route = "reader"
    _modelica_reads = "basemodelica"

    def pymoca_model(self, model_folder, model_name, compiler_options):
        return transfer_with_cache_retry(
            _transfer_model, model_folder, model_name, compiler_options
        )
