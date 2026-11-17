"""FMUs for FMUMixin, exported from Modelica sources: options, checks and exporters."""

import filecmp
import importlib.resources
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

logger = logging.getLogger("rtctools")

FMU_FORMATS = ("ode", "dae")
EXPORTERS = (None, "openmodelica", "rumoca")

# omc flags per FMU format: an FMI 3.0 Model Exchange FMU in ODE or DAE mode.
# The flags are those of the OpenModelica branches of the FMU routes: --fmiInputDerivatives
# and the fmi-ls-dae export of --daeMode are not in an OpenModelica release yet.
OPENMODELICA_FLAGS = {
    "ode": [
        "--fmiInputDerivatives",
        "-d=-disableDirectionalDerivatives",
        "--generateDynamicJacobian=symbolic",
        "--tearingMethod=noTearing",
    ],
    "dae": ["--daeMode", "--simCodeTarget=C+Rust", "--fmiInputDerivatives"],
}

# Rumoca's export target per FMU format
RUMOCA_TARGETS = {"ode": "fmi3"}

# The version of the Modelica Standard Library that OpenModelica loads for an export
MSL_VERSION = "4.0.0"


def route_option(problem, kwargs, name, default):
    """Return an option given as keyword argument, else as attribute, else its default."""
    return kwargs.get(name, getattr(problem, name, default))


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(map(repr, choices))}")


def check_fmu_route(model_format, model_exporter):
    """Raise a ValueError for an unknown or unsupported combination of FMUMixin options."""
    _check_choice("model_format", model_format, FMU_FORMATS)
    _check_choice("model_exporter", model_exporter, EXPORTERS)
    if model_format == "dae" and model_exporter == "rumoca":
        raise ValueError("model_exporter='rumoca' exports no FMUs in DAE mode (fmi-ls-dae)")


def library_folders(extra=()):
    """The given Modelica library folders, and those of installed packages."""
    folders = list(extra)
    for ep in importlib_metadata.entry_points(group="rtctools.libraries.modelica"):
        if ep.name == "library_folder":
            folders.append(str(importlib.resources.files(ep.module).joinpath(ep.attr)))
    return folders


def fmu_open_variables(model_folder, model_name, library_dirs):
    """The variables a model leaves to rtc-tools, which an FMU must take as inputs: (name, type).

    rtc-tools models may leave variables open on purpose, for the optimizer or the Python code
    to determine. pymoca makes them free optimization variables; an FMU, which must be balanced,
    can only take them as inputs. Two kinds are found:

    * variables that no equation uses, such as a Boolean decision variable;
    * inputs declared in the class of a component of the model, that no modifier binds and that
      no equation of the model mentions, such as the level of a boundary condition.
      (Flattening drops the input prefix of a component's input, so this reads the classes.)
    """
    import casadi as ca
    import pymoca.backends.casadi.api as api
    from pymoca import ast, parser

    root = parser.modelicapath_to_tree([], use_env=False)
    for folder in [model_folder, *library_dirs]:
        for path in sorted(Path(folder).rglob("*.mo")):
            try:
                root.extend(parser.parse(path.read_text(encoding="utf-8")))
            except NotImplementedError:
                continue

    def find(name):
        node = root
        for part in str(name).split("."):
            node = node.classes.get(part) if node is not None else None
        return node

    def references(node, found):
        """Dotted names of the component references in an AST node"""
        if isinstance(node, ast.ComponentRef):
            parts, ref = [], node
            while ref is not None:
                parts.append(ref.name)
                ref = ref.child[0] if ref.child else None
            found.add(".".join(parts))
            return found
        if isinstance(node, (list, tuple)):
            for item in node:
                references(item, found)
        elif isinstance(node, ast.Node):
            for value in vars(node).values():
                references(value, found)
        return found

    def class_inputs(cls, seen=()):
        """The scalar inputs declared in a class and the classes it extends"""
        if cls is None or id(cls) in seen:
            return {}
        inputs = {}
        for extends in getattr(cls, "extends", []):
            inputs.update(class_inputs(find(extends.component), (*seen, id(cls))))
        for name, symbol in cls.symbols.items():
            scalar = all(
                isinstance(d, ast.Primary) and d.value in (None, 1)
                for dims in symbol.dimensions
                for d in dims
            )
            if "input" in symbol.prefixes and scalar:
                inputs[name] = "Boolean" if str(symbol.type) == "Boolean" else "Real"
        return inputs

    model = find(model_name)
    open_inputs = {}
    if model is not None:
        mentioned = references(model.equations, set()) | references(model.initial_equations, set())
        for name, symbol in model.symbols.items():
            component = find(symbol.type)
            if component is None or "input" in symbol.prefixes:
                continue
            modified = {
                str(argument.value.component)
                for argument in getattr(symbol.class_modification, "arguments", [])
                if isinstance(getattr(argument, "value", None), ast.ElementModification)
            }
            for member, kind in class_inputs(component).items():
                full = f"{name}.{member}"
                if member not in modified and full not in mentioned:
                    open_inputs[full] = kind

    # The number of open variables: unknowns minus equations, counted per scalar
    compiled = api.transfer_model(
        str(model_folder),
        model_name,
        {
            "library_folders": list(library_dirs),
            "cache": False,
            "check_balanced": False,
            "expand_vectors": True,
            "expand_mx": True,
        },
    )
    equations = sum(ca.vec(e).numel() for e in compiled.equations)
    unknowns = len(compiled.der_states) + len(compiled.alg_states)
    deficit = unknowns - equations
    if deficit <= 0:
        return []

    used = set()
    for equation in compiled.equations + compiled.initial_equations:
        used |= {s.name() for s in ca.symvar(ca.vec(equation))}
    unused = [
        (v.symbol.name(), "Boolean" if v.python_type is bool else "Real")
        for v in compiled.alg_states
        if v.symbol.name() not in used
    ]
    # An input of a component may also be bound indirectly (through the component's own
    # equations): only as many as the model lacks, variables in no equation first
    candidates = unused + [(n, k) for n, k in open_inputs.items() if n not in dict(unused)]
    return sorted(candidates[:deficit])


def has_initial_equations(model_folder):
    """Whether a Modelica file of model_folder has an initial equation section."""
    pattern = re.compile(r"^\s*initial\s+equation\b", re.M)
    return any(
        pattern.search(Path(f).read_text(encoding="utf-8", errors="replace"))
        for f in _modelica_files(model_folder)
    )


def replace_if_changed(source, target):
    """Copy source over target, atomically, unless target already has the same content.

    Leaving an unchanged target alone keeps caches that depend on it, such as pymoca's, valid.
    Returns whether target was replaced.
    """
    target = Path(target)
    if target.exists() and filecmp.cmp(source, target, shallow=False):
        return False
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return True


def export_fmu(
    exporter,
    model_format,
    model_folder,
    model_name,
    library_dirs,
    target_folder,
    flags=None,
    target_name=None,
    open_variables=(),
    modification=None,
):
    """Export a model to <target_folder>/<target_name>.fmu, and return that path.

    model_format is "ode" or "dae". flags replaces the default flags of the exporter.
    target_name defaults to model_name. open_variables, (name, type) pairs, are variables that
    the model leaves to rtc-tools (see fmu_open_variables); an OpenModelica FMU export gives each
    a top-level input <name with _ for .>_input. modification is a Modelica class modification,
    e.g. ``Channel(use_convective_acceleration = true)`` (see :py:func:`modification`): the
    export is then that of a model that extends model_name with it (OpenModelica only).
    """
    target = Path(target_folder) / f"{target_name or model_name}.fmu"
    # The exporter runs in a temporary folder, where relative paths would not resolve
    model_folder = Path(model_folder).resolve()
    library_dirs = [Path(folder).resolve() for folder in library_dirs]
    with tempfile.TemporaryDirectory(prefix="rtctools-export-") as work:
        if exporter == "openmodelica":
            produced = _export_openmodelica(
                model_format,
                model_folder,
                model_name,
                library_dirs,
                work,
                flags,
                open_variables,
                modification,
            )
        elif modification:
            raise ValueError(f"export_fmu: no modifications with the exporter {exporter!r}")
        else:
            produced = _export_rumoca(
                model_format, model_folder, model_name, library_dirs, work, flags
            )
        changed = replace_if_changed(produced, target)
    logger.info(f"Exported {target} with {exporter}{'' if changed else ', unchanged'}.")
    return target


def _command(variable, default):
    return shlex.split(os.environ.get(variable, default))


def _run(command, work, what, env=None):
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, env=env)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{what}: {error}. Install it or set the command") from error
    return result


def _modelica_files(folder):
    return sorted(str(f) for f in Path(folder).glob("*.mo"))


def _library_packages(folder):
    """The top-level packages of a library folder: folders with a package.mo, and *.mo files."""
    folder = Path(folder)
    packages = [str(p) for p in sorted(folder.glob("*/package.mo"))]
    return packages + _modelica_files(folder)


def fmu_input_name(variable):
    """The top-level FMU input that stands for an open variable (fmu_open_variables)"""
    return re.sub(r"\W", "_", variable) + "_input"


def _open_variables_wrapper(model_name, open_variables, work):
    """A model that extends model_name with a top-level input per open variable"""
    wrapper = f"{model_name}_rtctools_fmu"
    lines = [f"model {wrapper}", f"  extends {model_name};"]
    lines += [f"  input {kind} {fmu_input_name(name)};" for name, kind in open_variables]
    lines.append("equation")
    lines += [f"  {name} = {fmu_input_name(name)};" for name, _ in open_variables]
    lines.append(f"end {wrapper};")
    path = Path(work) / f"{wrapper}.mo"
    path.write_text("\n".join(lines) + "\n")
    return wrapper, path


def modification(values):
    """The Modelica class modification that sets the given parameter values.

    values maps (dotted) parameter names to Python values, e.g.
    ``{"Channel.use_convective_acceleration": True, "theta": 0.5}`` gives
    ``Channel(use_convective_acceleration = true), theta = 0.5``. Returns None if a name cannot
    be modified this way, such as an array element ``a[1]``.
    """
    tree = {}
    for name, value in values.items():
        parts = name.split(".")
        if not all(re.fullmatch(r"[A-Za-z_]\w*", p) for p in parts):
            return None
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                return None
        if parts[-1] in node:
            return None
        node[parts[-1]] = value

    def literal(value):
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return repr(float(value))

    def text(node):
        return ", ".join(
            f"{k}({text(v)})" if isinstance(v, dict) else f"{k} = {literal(v)}"
            for k, v in sorted(node.items())
        )

    return text(tree)


def _export_openmodelica(
    model_format, model_folder, model_name, library_dirs, work, flags, open_variables=(), mod=None
):
    lines = [f'loadModel(Modelica, {{"{MSL_VERSION}"}}); getErrorString();']
    for folder in library_dirs:
        lines += [f'loadFile("{f}"); getErrorString();' for f in _library_packages(folder)]
    lines += [f'loadFile("{f}"); getErrorString();' for f in _modelica_files(model_folder)]
    if mod:
        # A model that extends the model with the modification. The FMU keeps the variable names
        # of the model; only its model name (and file name in work) differ.
        base, model_name = model_name, re.sub(r"\W", "_", model_name) + "_modified"
        wrapper = Path(work) / f"{model_name}.mo"
        wrapper.write_text(f"model {model_name}\n  extends {base}({mod});\nend {model_name};\n")
        lines.append(f'loadFile("{wrapper}"); getErrorString();')
    out = Path(work) / f"{model_name}.fmu"
    exported = model_name
    if open_variables:
        exported, wrapper = _open_variables_wrapper(model_name, open_variables, work)
        lines.append(f'loadFile("{wrapper}"); getErrorString();')
    lines.append(
        f'buildModelFMU({exported}, version="3.0", fmuType="me", '
        f'fileNamePrefix="{model_name}"); getErrorString();'
    )
    script = Path(work) / "export.mos"
    script.write_text("\n".join(lines) + "\n")
    command = _command("RTCTOOLS_OMC", "omc") + list(flags or OPENMODELICA_FLAGS[model_format])
    result = _run(command + [str(script)], work, "OpenModelica (RTCTOOLS_OMC, default omc)")
    if not out.exists() or out.stat().st_size == 0:
        log = result.stdout[-3000:] + result.stderr[-1000:]
        raise RuntimeError(f"OpenModelica did not export {model_name}:\n{log}")
    return out


def _main_file(model_folder, model_name):
    """The Modelica file of model_folder that defines model_name."""
    files = _modelica_files(model_folder)
    for f in files:
        if Path(f).stem == model_name:
            return f
    pattern = re.compile(
        rf"^\s*(?:partial\s+)?(?:model|class|block)\s+{re.escape(model_name)}\b", re.M
    )
    for f in files:
        if pattern.search(Path(f).read_text(encoding="utf-8", errors="replace")):
            return f
    raise FileNotFoundError(f"No Modelica file in {model_folder} defines {model_name}")


def _export_rumoca(model_format, model_folder, model_name, library_dirs, work, flags):
    if model_format not in RUMOCA_TARGETS:
        raise ValueError(f"Rumoca exports no FMUs in {model_format!r} mode")
    # Rumoca writes <Model>.fmu into an output folder
    folder = Path(work) / "rumoca-output"
    out = folder / f"{model_name}.fmu"

    def compile(main):
        # Rumoca loads the other files of the model's folder itself; passing them as source roots
        # too declares their classes twice, under another spelling of the path
        command = _command("RTCTOOLS_RUMOCA", "rumoca") + ["compile", main, "--model", model_name]
        command += [arg for root in library_dirs for arg in ("--source-root", str(root))]
        command += ["--target", RUMOCA_TARGETS[model_format], "--output", str(folder)]
        # rtc-tools sets parameters at run time, so a parameter guard must stay a run-time branch
        env = dict(os.environ, RUMOCA_RETAIN_PARAMETER_BRANCHES="1")
        return _run(
            command + list(flags or []), work, "Rumoca (RTCTOOLS_RUMOCA, default rumoca)", env
        )

    main = _main_file(model_folder, model_name)
    result = compile(main)
    if result.returncode != 0 and "ED001" in result.stdout + result.stderr:
        # Under-determined by design (the rest comes from Python or the optimizer): Rumoca only
        # accepts that for a partial model
        logger.info(f"{model_name} is under-determined: exporting it as a partial model.")
        shutil.rmtree(folder, ignore_errors=True)
        result = compile(str(_as_partial(Path(main), model_name, Path(work) / "partial-model")))
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"Rumoca did not export {model_name}:\n{result.stdout[-3000:]}{result.stderr[-2000:]}"
        )
    # Rumoca's FMUs hold C sources only
    _build_source_fmu(out, Path(work) / "fmu-build")
    return out


def _as_partial(path, model_name, folder):
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


def _build_source_fmu(fmu, work):
    """Compile the C sources of a source-code FMU into its binary for this platform, in place.

    The sources and defines come from sources/buildDescription.xml (FMI 3.0, 2.4.10). The FMI 3
    headers are taken from RTCTOOLS_FMI3_INCLUDE, and the C compiler from CC (default cc).
    """
    include = os.environ.get("RTCTOOLS_FMI3_INCLUDE")
    if not include or not (Path(include) / "fmi3Functions.h").is_file():
        raise FileNotFoundError(
            f"{fmu} holds C sources only. Compiling them needs the FMI 3 headers: set "
            "RTCTOOLS_FMI3_INCLUDE to a folder with fmi3Functions.h"
        )
    work.mkdir()
    with zipfile.ZipFile(fmu) as z:
        z.extractall(work)
    config = ET.parse(work / "sources" / "buildDescription.xml").find("BuildConfiguration")
    identifier = config.get("modelIdentifier")
    sources, defines = [], []
    for file_set in config.iter("SourceFileSet"):
        sources += [str(work / "sources" / f.get("name")) for f in file_set.iter("SourceFile")]
        for d in file_set.iter("PreprocessorDefinition"):
            value = d.get("value")
            defines.append(f"-D{d.get('name')}" + (f"={value}" if value is not None else ""))
    binary = work / "binaries" / "x86_64-linux" / f"{identifier}.so"
    binary.parent.mkdir(parents=True)
    command = shlex.split(os.environ.get("CC", "cc")) + ["-shared", "-fPIC", "-O2", "-std=c99"]
    command += ["-I", str(work / "sources"), "-I", include] + defines + sources
    command += ["-o", str(binary), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compiling the sources of {fmu} failed:\n{result.stderr[-3000:]}")
    with zipfile.ZipFile(fmu, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work))
