"""
Loading a Base Modelica model with lacemodelica, in place of pymoca's parser and flattener.

lacemodelica translates ``<model_folder>/<model_name>.bmo`` to a ``modelDescription.xml`` and an
ONNX graph of the equations, which CasADi imports. These fill a pymoca ``Model``, which pymoca then
simplifies with the compiler options, so that rtc-tools gets what it gets from pymoca.

lacemodelica runs in-process when CasADi has its ``lacemodelica`` ModelicaParser plugin (a CasADi
built with WITH_LACEMODELICA), else as the ``lacemodelica`` executable on the PATH. Both write the
same files. CasADi must import ONNX.
"""

import functools
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import casadi as ca
import defusedxml.ElementTree as DefusedElementTree

_TYPES = {"Float64": float, "Int32": int, "Boolean": bool, "Enumeration": int}
_ONNX = "extra/org.lacemodelica.ls-onnx-serialization/model.onnx"
_MODELICA_ANNOTATION = "Annotations/Annotation[@type='org.lacemodelica.modelica']/Attributes"
_CASADI_REQUIRED = (
    "lacemodelica models need a CasADi with ONNX import, including the Loop operator "
    "(a build of CasADi main from November 2026 or newer)"
)


@functools.cache
def _has_plugin() -> bool:
    """Whether CasADi has the lacemodelica ModelicaParser plugin"""
    try:
        ca.ModelicaParser.load_plugin("lacemodelica")
    except (AttributeError, RuntimeError):
        return False
    return True


def _executable() -> str:
    executable = shutil.which("lacemodelica")
    if executable is None:
        raise FileNotFoundError(
            "lacemodelica not found: use a CasADi with its lacemodelica plugin "
            "(WITH_LACEMODELICA), or build lacemodelica from "
            "https://github.com/yacoda/lacemodelica and put the lacemodelica executable on the PATH"
        )
    if not hasattr(ca, "GraphBuilder"):
        raise ImportError(_CASADI_REQUIRED)
    return executable


def _translate(path: Path, output: Path) -> None:
    if _has_plugin():
        if not hasattr(ca, "GraphBuilder"):
            raise ImportError(_CASADI_REQUIRED)
        try:
            ca.ModelicaParser("lacemodelica").parse(str(path), str(output))
        except RuntimeError as error:
            raise RuntimeError(f"lacemodelica failed on {path}:\n{error}") from error
        return
    command = [_executable(), str(path), "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lacemodelica failed on {path}:\n{result.stderr}{result.stdout}")


def _literal(text: str, python_type: type):
    if text in ("true", "false"):
        return python_type(text == "true")
    return python_type(float(text))


def _fixed(element):
    for attributes in element.iterfind(_MODELICA_ANNOTATION):
        if attributes.get("fixed") is not None:
            return attributes.get("fixed") == "true"
    return None


def _modelica_shape(name: str) -> tuple:
    """The shape format of pymoca: one entry per dotted part of the name"""
    return tuple((None,) for _ in re.sub(r"^der\((.*)\)$", r"\1", name).split("."))


def _build(output: Path):
    from pymoca.backends.casadi.model import DelayArgument, Model, Variable

    root = DefusedElementTree.parse(output / "modelDescription.xml").getroot()
    variables = list(root.find("ModelVariables"))
    try:
        oracle = ca.GraphBuilder(str(output / _ONNX)).create("oracle", {"symbolic": True})
    except RuntimeError as error:
        if "is not supported" in str(error):
            raise ImportError(f"{_CASADI_REQUIRED}: {error}") from error
        raise
    inputs = [oracle.name_in(i) for i in range(oracle.n_in())]

    delay_count = sum(
        1 for i in range(oracle.n_out()) if oracle.name_out(i).startswith("delay_expr[")
    )
    # rtc-tools knows a delayed expression by the name pymoca gives it
    delays = {f"__delay_{i}": f"_pymoca_delay_{i}" for i in range(delay_count)}

    symbols = {}
    # An array without elements, such as x[0], has nothing to solve for: the graph gets an empty
    # matrix and the model no variable (OpenModelica's export leaves such arrays out)
    empty = set()
    for i, name in enumerate(inputs):
        if oracle.numel_in(i) == 0:
            symbols[name] = ca.MX(*oracle.size_in(i))
            empty.add(name)
            continue
        symbol = ca.MX.sym(delays.get(name, name), *oracle.size_in(i))
        symbol._modelica_shape = symbol.shape if name in delays else _modelica_shape(name)
        symbols[name] = symbol
    for element in variables:
        # A variable that no equation mentions is still a variable
        name = element.get("name")
        if name not in symbols:
            symbols[name] = ca.MX.sym(name)
            symbols[name]._modelica_shape = _modelica_shape(name)

    outputs = oracle.call([symbols[name] for name in inputs], True, False)
    by_name = {oracle.name_out(i): outputs[i] for i in range(oracle.n_out())}
    index = {name: i for i, name in enumerate(inputs)}

    def attribute(element, key, python_type):
        """A literal is in the XML, an expression is an output of the graph"""
        if element.get(key) is not None:
            return _literal(element.get(key), python_type)
        return by_name.get(f"{key}[{index.get(element.get('name'))}]")

    model = Model()
    model.time = symbols.get("time", ca.MX.sym("time"))
    references = {int(v.get("valueReference")): v.get("name") for v in variables}
    derivatives = {
        references[int(v.get("derivative"))]: v.get("name")
        for v in variables
        if v.get("derivative")
    }
    is_derivative = set(derivatives.values())

    for element in variables:
        name = element.get("name")
        if name == "time" or name in is_derivative or name in empty:
            continue
        python_type = _TYPES[element.tag]
        variable = Variable(symbols[name], python_type)
        for key in ("min", "max"):
            value = attribute(element, key, python_type)
            if value is not None:
                setattr(variable, key, value)
        start = attribute(element, "start", python_type)
        causality = element.get("causality")
        if causality in ("parameter", "structuralParameter"):
            if start is not None:
                variable.value = start
            model.parameters.append(variable)
            continue
        if start is not None:
            variable.start = start
        nominal = attribute(element, "nominal", float)
        if nominal is not None:
            variable.nominal = nominal
        fixed = _fixed(element)
        if fixed is not None:
            variable.fixed = fixed
        if name in derivatives:
            model.states.append(variable)
            model.der_states.append(Variable(symbols[derivatives[name]], float))
        elif name in delays or causality == "input":
            model.inputs.append(variable)
        else:
            model.alg_states.append(variable)
        if causality == "output":
            model.outputs.append(name)

    for i in range(delay_count):
        model.delay_states.append(f"_pymoca_delay_{i}")
        model.delay_arguments.append(
            DelayArgument(by_name[f"delay_expr[{i}]"], by_name[f"delay_time[{i}]"])
        )
    model.equations = [ca.vec(e) for n, e in by_name.items() if n.startswith("eq[")]
    model.initial_equations = [ca.vec(e) for n, e in by_name.items() if n.startswith("init_eq[")]
    return model


def transfer_model(model_folder: str, model_name: str, compiler_options: dict):
    """
    Compile ``<model_folder>/<model_name>.bmo`` with lacemodelica.

    :returns: a ``pymoca.backends.casadi.model.Model``, simplified with ``compiler_options``.
    """
    from pymoca.backends.casadi.api import _merge_default_options

    compiler_options = _merge_default_options(dict(compiler_options))
    path = Path(model_folder) / f"{model_name}.bmo"
    if not path.is_file():
        raise FileNotFoundError(f"Base Modelica file {path} not found")
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "output"
        _translate(path, output)
        model = _build(output)
    model.simplify(compiler_options)
    model._post_checks()
    return model
