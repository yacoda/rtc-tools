"""The route from a Modelica model to CasADi: options, checks and exporters."""

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

logger = logging.getLogger("rtctools")

MODELICA_FORMATS = ("modelica", "basemodelica")
MODELICA_COMPILERS = ("pymoca", "rumoca", "lacemodelica")
FMU_FORMATS = ("ode", "dae")
EXPORTERS = (None, "openmodelica", "rumoca")

# omc flags per output: Base Modelica, or an FMI 3.0 Model Exchange FMU in ODE or DAE mode.
# The FMU flags are those of the OpenModelica branches of the FMU routes: --fmiInputDerivatives
# and the fmi-ls-dae export of --daeMode are not in an OpenModelica release yet.
OPENMODELICA_FLAGS = {
    "basemodelica": [
        "--BaseModelica",
        "--evaluateStructuralParameters=strictlyNecessary",
        "--baseModelicaFormat=scalarized",
        "--baseModelicaOptions=scalarize,moveBindings",
    ],
    "ode": [
        "--fmiInputDerivatives",
        "-d=-disableDirectionalDerivatives",
        "--generateDynamicJacobian=symbolic",
        "--tearingMethod=noTearing",
    ],
    "dae": ["--daeMode", "--simCodeTarget=C+Rust", "--fmiInputDerivatives"],
}

RUMOCA_TARGETS = {"basemodelica": "base-modelica", "ode": "fmi3"}

# The version of the Modelica Standard Library that OpenModelica loads for an export
MSL_VERSION = "4.0.0"


def route_option(problem, kwargs, name, default):
    """Return an option given as keyword argument, else as attribute, else its default."""
    return kwargs.get(name, getattr(problem, name, default))


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(map(repr, choices))}")


def check_modelica_route(model_format, model_compiler, model_exporter):
    """Raise a ValueError for an unknown or unsupported combination of ModelicaMixin options."""
    _check_choice("model_format", model_format, MODELICA_FORMATS)
    _check_choice("model_compiler", model_compiler, MODELICA_COMPILERS)
    _check_choice("model_exporter", model_exporter, EXPORTERS)
    if model_exporter is not None and model_format != "basemodelica":
        raise ValueError(
            f"model_exporter={model_exporter!r} writes Base Modelica, "
            "which needs model_format='basemodelica'"
        )
    if model_compiler == "lacemodelica" and model_format != "basemodelica":
        raise ValueError("model_compiler='lacemodelica' needs model_format='basemodelica'")
    if model_compiler == "rumoca" and model_format != "modelica":
        raise ValueError(
            "model_compiler='rumoca' compiles Modelica sources (model_format='modelica')"
        )


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


def model_loader(model_compiler, model_format):
    """The transfer_model(model_folder, model_name, compiler_options) of a model compiler.

    lacemodelica and Rumoca are imported only when chosen.
    """
    if model_compiler == "lacemodelica":
        from rtctools._internal.lacemodelica_loader import transfer_model

        return transfer_model
    if model_compiler == "rumoca":
        from rtctools._internal.rumoca_loader import transfer_model

        return transfer_model

    import pymoca
    import pymoca.backends.casadi.api
    from pymoca.backends.casadi._options import _get_default_options

    def transfer_model(model_folder, model_name, compiler_options):
        if model_format == "basemodelica":
            if "basemodelica" not in _get_default_options():
                raise RuntimeError(
                    f"pymoca {pymoca.__version__} cannot read Base Modelica: it has no "
                    "basemodelica option. Use a pymoca that has it, or model_format='modelica'."
                )
            compiler_options = {**compiler_options, "basemodelica": True}
        return pymoca.backends.casadi.api.transfer_model(model_folder, model_name, compiler_options)

    return transfer_model


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


def export_model(
    exporter,
    output,
    model_folder,
    model_name,
    library_dirs,
    target_folder,
    flags=None,
    target_name=None,
):
    """Export a model to <target_folder>/<target_name>.bmo or .fmu, and return that path.

    output is "basemodelica", "ode" or "dae". flags replaces the default flags of the exporter.
    target_name defaults to model_name.
    """
    suffix = ".bmo" if output == "basemodelica" else ".fmu"
    target = Path(target_folder) / f"{target_name or model_name}{suffix}"
    with tempfile.TemporaryDirectory(prefix="rtctools-export-") as work:
        if exporter == "openmodelica":
            produced = _export_openmodelica(
                output, model_folder, model_name, library_dirs, work, flags
            )
        else:
            produced = _export_rumoca(output, model_folder, model_name, library_dirs, work, flags)
        changed = replace_if_changed(produced, target)
    logger.info(f"Exported {target} with {exporter}{'' if changed else ', unchanged'}.")
    return target


def _command(variable, default):
    return shlex.split(os.environ.get(variable, default))


def _run(command, work, what):
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, text=True)
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


def _export_openmodelica(output, model_folder, model_name, library_dirs, work, flags):
    out = Path(work) / (f"{model_name}.bmo" if output == "basemodelica" else f"{model_name}.fmu")
    lines = [f'loadModel(Modelica, {{"{MSL_VERSION}"}}); getErrorString();']
    for folder in library_dirs:
        lines += [f'loadFile("{f}"); getErrorString();' for f in _library_packages(folder)]
    lines += [f'loadFile("{f}"); getErrorString();' for f in _modelica_files(model_folder)]
    if output == "basemodelica":
        lines.append(f'writeFile("{out}", instantiateModel({model_name})); getErrorString();')
    else:
        lines.append(
            f'buildModelFMU({model_name}, version="3.0", fmuType="me", '
            f'fileNamePrefix="{model_name}"); getErrorString();'
        )
    script = Path(work) / "export.mos"
    script.write_text("\n".join(lines) + "\n")
    command = _command("RTCTOOLS_OMC", "omc") + list(flags or OPENMODELICA_FLAGS[output])
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


def _export_rumoca(output, model_folder, model_name, library_dirs, work, flags):
    if output not in RUMOCA_TARGETS:
        raise ValueError(f"Rumoca has no export to {output!r}")
    # Rumoca writes into an output folder: <Model>_base.mo for Base Modelica, <Model>.fmu for FMI
    folder = Path(work) / "rumoca-output"
    produced = f"{model_name}_base.mo" if output == "basemodelica" else f"{model_name}.fmu"
    out = folder / produced
    main = _main_file(model_folder, model_name)
    roots = [f for f in _modelica_files(model_folder) if f != main] + list(library_dirs)
    command = _command("RTCTOOLS_RUMOCA", "rumoca") + ["compile", main, "--model", model_name]
    command += [arg for root in roots for arg in ("--source-root", root)]
    command += ["--target", RUMOCA_TARGETS[output], "--output", str(folder)] + list(flags or [])
    result = _run(command, work, "Rumoca (RTCTOOLS_RUMOCA, default rumoca)")
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"Rumoca did not export {model_name}:\n{result.stdout[-3000:]}{result.stderr[-2000:]}"
        )
    if output != "basemodelica":
        # Rumoca's FMUs hold C sources only
        _build_source_fmu(out, Path(work) / "fmu-build")
    return out


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
