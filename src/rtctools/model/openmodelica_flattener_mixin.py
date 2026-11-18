from pathlib import Path

from ._flatten import flatten_basemodelica, modelica_files, run_tool, tool_command

# The version of the Modelica Standard Library that OpenModelica loads to flatten a model
MSL_VERSION = "4.0.0"


class OpenModelicaFlattenerMixin:
    """
    Writes the model as Base Modelica with OpenModelica, and sets ``model_format`` to
    ``"basemodelica"``, so that the model is read from that file.

    At the start of a run, ``omc`` flattens the Modelica sources of the model folder, with the
    Modelica Standard Library and the library folders, to ``<model_name>.bmo`` in the model
    folder. The file is only replaced when its content changes, so pymoca's cache stays valid; a
    hand-edited file is overwritten. pymoca then reads it, or a reader of Base Modelica listed
    after the flattener. The command is ``omc``, or ``RTCTOOLS_OMC``::

        class Example(
            OpenModelicaFlattenerMixin,
            CSVMixin,
            ModelicaMixin,
            CollocatedIntegratedOptimizationProblem,
        ):
            ...

    :cvar openmodelica_flags: The command-line flags of ``omc``.
    """

    _modelica_route = "flattener"

    model_format = "basemodelica"

    openmodelica_flags = [
        "--BaseModelica",
        "--evaluateStructuralParameters=strictlyNecessary",
        "--baseModelicaFormat=scalarized",
        "--baseModelicaOptions=scalarize,moveBindings",
    ]

    def pymoca_model(self, model_folder, model_name, compiler_options):
        flatten_basemodelica(
            "OpenModelica",
            _flatten_openmodelica,
            model_folder,
            model_name,
            compiler_options.get("library_folders", []),
            self.openmodelica_flags,
        )
        return super().pymoca_model(model_folder, model_name, compiler_options)


def _flatten_openmodelica(model_folder, model_name, library_dirs, work, flags):
    lines = [f'loadModel(Modelica, {{"{MSL_VERSION}"}}); getErrorString();']
    for folder in library_dirs:
        lines += [f'loadFile("{f}"); getErrorString();' for f in _library_packages(folder)]
    lines += [f'loadFile("{f}"); getErrorString();' for f in modelica_files(model_folder)]
    out = Path(work) / f"{model_name}.bmo"
    lines.append(f'writeFile("{out}", instantiateModel({model_name})); getErrorString();')
    script = Path(work) / "flatten.mos"
    script.write_text("\n".join(lines) + "\n")
    command = tool_command("RTCTOOLS_OMC", "omc") + list(flags)
    result = run_tool(command + [str(script)], work, "OpenModelica (RTCTOOLS_OMC, default omc)")
    if not out.exists() or out.stat().st_size == 0:
        log = result.stdout[-3000:] + result.stderr[-1000:]
        raise RuntimeError(f"OpenModelica did not flatten {model_name}:\n{log}")
    return out


def _library_packages(folder):
    """The top-level packages of a library folder: folders with a package.mo, and *.mo files."""
    folder = Path(folder)
    packages = [str(p) for p in sorted(folder.glob("*/package.mo"))]
    return packages + modelica_files(folder)
