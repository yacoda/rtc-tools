"""
Loading a Modelica model with Rumoca, in place of pymoca's parser and flattener.

Rumoca's ``casadi-dae`` target writes a Python module that builds the variables, attributes and
residuals of the model as CasADi MX. These fill a pymoca ``Model``, which pymoca then simplifies
with the compiler options, so that rtc-tools gets what it gets from pymoca.

The Rumoca executable is ``rumoca`` on the PATH, or the one that ``RTCTOOLS_RUMOCA`` names.
"""

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import casadi as ca

from rtctools._internal.modelica_mixin_base import transfer_with_cache_retry

logger = logging.getLogger("rtctools")

_TYPES = {"real": float, "integer": int, "boolean": bool, "enumeration": int}
_UNDERDETERMINED = "ED001"
_FIXED_PARAMETER = "WD001"


def _executable() -> str:
    command = os.environ.get("RTCTOOLS_RUMOCA", "rumoca")
    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError(
            f"Rumoca executable {command!r} not found: install Rumoca "
            "(https://github.com/CogniPilot/rumoca), with the casadi-dae target, and put it on "
            "the PATH, or set RTCTOOLS_RUMOCA to its path"
        )
    return executable


def _model_file(model_folder: str, model_name: str) -> Path:
    folder = Path(model_folder)
    candidate = folder / f"{model_name}.mo"
    if candidate.is_file():
        return candidate
    # As pymoca does, also find a model in a file of another name
    pattern = re.compile(
        rf"^\s*(?:partial\s+)?(?:model|class|block)\s+{re.escape(model_name)}\b", re.M
    )
    for path in sorted(folder.glob("*.mo")):
        if pattern.search(path.read_text(encoding="utf-8")):
            return path
    raise FileNotFoundError(f"No model {model_name} in {model_folder}")


def _compile(path: Path, model_name: str, library_folders: list, output: Path) -> tuple:
    command = [_executable(), "compile", str(path), "--model", model_name]
    command += ["--target", "casadi-dae", "-o", str(output)]
    for folder in library_folders:
        command += ["--source-root", str(folder)]
    # rtc-tools sets parameters at run time, so a parameter guard must stay a run-time branch
    command += ["--retain-parameter-guards"]
    result = subprocess.run(command, capture_output=True, text=True)
    # Rumoca colors its diagnostics also when they do not go to a terminal
    return result.returncode, re.sub(r"\x1b\[[0-9;]*m", "", result.stderr + result.stdout)


def _as_partial(path: Path, model_name: str, folder: str) -> Path:
    """
    Copy a model folder with the model declared partial.

    rtc-tools models may be under-determined by design, with the optimizer or Python code
    providing what is missing. Rumoca only allows that for a partial model, and pymoca does not
    check it.
    """
    copy = Path(folder) / "partial"
    shutil.copytree(path.parent, copy)
    target = copy / path.name
    source, count = re.subn(
        rf"^(\s*)((?:model|class|block)\s+{re.escape(model_name)}\b)",
        r"\1partial \2",
        target.read_text(encoding="utf-8"),
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise RuntimeError(f"Could not find the declaration of {model_name} in {path}")
    target.write_text(source, encoding="utf-8")
    return target


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rumoca_casadi_dae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modelica_shape(name: str, dimensions: tuple) -> tuple:
    """The shape format of pymoca: one entry per dotted part, the last one has the dimensions"""
    parts = re.match(r"(?:der\()*(.*?)\)*$", name).group(1).split(".")
    shape = [(None,)] * len(parts)
    if dimensions:
        shape[-1] = dimensions
    return tuple(shape)


def _value(x, python_type=float):
    """Literal attribute values as plain numbers, as pymoca has them"""
    if isinstance(x, ca.MX):
        return x
    if isinstance(x, ca.DM):
        return python_type(x) if x.is_scalar() else x
    return x if isinstance(x, bool) else python_type(x)


def _is_zero(x) -> bool:
    return not isinstance(x, ca.MX) and ca.DM(x).is_scalar() and float(ca.DM(x)) == 0.0


def _fill(data):
    from pymoca.backends.casadi.model import (
        DelayArgument,
        Model,
        StringVariable,
        Variable,
        _DefaultValue,
    )

    model = Model()
    model.time = data["time"]
    old_symbols, new_symbols = [], []

    def variable(entry, symbol, attributes=True):
        python_type = _TYPES[entry["scalar"]]
        rows, columns = symbol.shape
        dimensions = () if symbol.is_scalar() else ((rows,) if columns == 1 else (rows, columns))
        new = ca.MX.sym(symbol.name(), rows, columns)
        new._modelica_shape = _modelica_shape(symbol.name(), dimensions)
        old_symbols.append(symbol)
        new_symbols.append(new)
        v = Variable(new, python_type)
        if not attributes:
            return v
        for key, attribute in (("min", "min"), ("max", "max"), ("binding", "value")):
            if key in entry:
                setattr(v, attribute, _value(entry[key], python_type))
        if "nominal" in entry:
            v.nominal = _value(entry["nominal"])
        start = entry.get("start")
        if start is not None:
            # Rumoca makes the default start of 0 explicit, which pymoca's alias detection tells
            # apart from a start of 0 that is declared
            v.start = _DefaultValue(0) if _is_zero(start) else _value(start, python_type)
        if entry["fixed"] is not None:
            v.fixed = entry["fixed"]
        return v

    groups = {
        "state": model.states,
        "algebraic": model.alg_states,
        "output": model.alg_states,
        "discrete_real": model.alg_states,
        "discrete_value": model.alg_states,
        "input": model.inputs,
        "parameter": model.parameters,
        "constant": model.constants,
    }
    for entry in data["variables"]:
        role = entry["role"]
        name = entry["name"]
        if entry["scalar"] == "string":
            strings = model.string_constants if role == "constant" else model.string_parameters
            strings.append(StringVariable(name))
            continue
        if entry["symbol"].is_empty():
            continue
        if role not in groups:
            raise NotImplementedError(f"Variable role {role} of {name}")
        groups[role].append(variable(entry, entry["symbol"]))
        if role == "state":
            model.der_states.append(variable(entry, entry["derivative"], attributes=False))
        if entry["causality"] == "output":
            model.outputs.append(name)

    # rtc-tools knows a delayed expression by the name pymoca gives it
    for i, delay in enumerate(data["delays"]):
        name = f"_pymoca_delay_{i}"
        new = ca.MX.sym(name, *delay["symbol"].shape)
        new._modelica_shape = delay["symbol"].shape
        old_symbols.append(delay["symbol"])
        new_symbols.append(new)
        model.delay_states.append(name)
        model.inputs.append(Variable(new))

    def substitute(expressions):
        expressions = [ca.vec(ca.MX(e)) for e in expressions]
        expressions = [e for e in expressions if not e.is_empty()]
        return ca.substitute(expressions, old_symbols, new_symbols) if expressions else []

    for delay in data["delays"]:
        expression, duration = substitute([delay["expression"], delay["duration"]])
        model.delay_arguments.append(DelayArgument(expression, duration))
    model.equations = substitute(data["equations"])
    model.initial_equations = substitute(data["initial_equations"])

    # Attributes may refer to other variables
    for group in (model.states, model.alg_states, model.inputs, model.parameters, model.constants):
        for v in group:
            for attribute in ("value", "start", "min", "max", "nominal"):
                value = getattr(v, attribute)
                if isinstance(value, ca.MX):
                    setattr(v, attribute, ca.substitute([value], old_symbols, new_symbols)[0])
    return model


def _transfer_model(model_folder: str, model_name: str, compiler_options: dict):
    """
    Compile the model ``model_name`` of the Modelica files in ``model_folder`` with Rumoca.

    The folders of ``compiler_options["library_folders"]`` are Rumoca's source roots.

    :returns: a ``pymoca.backends.casadi.model.Model``, simplified with ``compiler_options``.
    """
    from pymoca.backends.casadi.api import _merge_default_options

    compiler_options = _merge_default_options(dict(compiler_options))
    path = _model_file(model_folder, model_name)
    library_folders = compiler_options["library_folders"]

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "output"
        returncode, messages = _compile(path, model_name, library_folders, output)
        if returncode != 0 and _UNDERDETERMINED in messages:
            logger.info(f"{model_name} is under-determined: compiling it as a partial model.")
            path = _as_partial(path, model_name, folder)
            returncode, messages = _compile(path, model_name, library_folders, output)
        if returncode != 0:
            raise RuntimeError(f"Rumoca failed on {model_name}:\n{messages}")
        # A parameter fixed at translation ignores a value set at run time. Array sizes may be.
        for line in messages.splitlines():
            if _FIXED_PARAMETER in line and not re.search("array dimension|for-equation", line):
                logger.warning(f"{model_name}: {line}")
        (module,) = output.glob("*_casadi_dae.py")
        data = _load_module(module).build()
    model = _fill(data)
    model.simplify(compiler_options)
    model._post_checks()
    return model


class RumocaReaderMixin:
    """
    Reads the model from the Modelica sources with Rumoca, in place of pymoca.

    Rumoca compiles the Modelica sources of the model folder, with the library folders, and its
    CasADi DAE fills a pymoca model, which pymoca simplifies with the compiler options. A model
    that is under-determined on purpose, the rest coming from Python or the optimizer, is
    compiled as a partial model. The command is ``rumoca``, or ``RTCTOOLS_RUMOCA``.

    It reads ``model_format = "modelica"``, the default, so it takes no flattener.
    """

    _modelica_route = "reader"
    _modelica_reads = "modelica"

    def pymoca_model(self, model_folder, model_name, compiler_options):
        return transfer_with_cache_retry(
            _transfer_model, model_folder, model_name, compiler_options
        )
