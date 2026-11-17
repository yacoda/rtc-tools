import logging
import os
import re
import shutil
from pathlib import Path

from ._flatten import flatten_basemodelica, modelica_files, run_tool, tool_command

logger = logging.getLogger("rtctools")

# Rumoca's --target for Base Modelica
RUMOCA_TARGET = "base-modelica"


class RumocaFlattenerMixin:
    """
    Writes the model as Base Modelica with Rumoca, and sets ``model_format`` to
    ``"basemodelica"``, so that the model is read from that file.

    At the start of a run, ``rumoca`` flattens the Modelica sources of the model folder to
    ``<model_name>.bmo`` in the model folder; library folders are passed as source roots. With
    the compiler option ``cache`` (the default), the flattening is skipped when
    ``<model_name>.bmo_cache`` records that the file was flattened from the same inputs, as for
    :py:class:`OpenModelicaFlattenerMixin`, with the output of ``rumoca --version``. Otherwise
    the file is only replaced when its content changes. pymoca then reads it, or a reader of Base
    Modelica listed after the flattener. A model that is under-determined on purpose is flattened
    as a partial model, as :py:class:`RumocaReaderMixin` does. The command is ``rumoca``, or
    ``RTCTOOLS_RUMOCA``::

        class Example(
            RumocaFlattenerMixin,
            CSVMixin,
            ModelicaMixin,
            CollocatedIntegratedOptimizationProblem,
        ):
            ...

    :cvar rumoca_flags: Command-line flags added to ``rumoca compile``.
    """

    _modelica_route = "flattener"

    model_format = "basemodelica"

    rumoca_flags = []

    def pymoca_model(self, model_folder, model_name, compiler_options):
        flatten_basemodelica(
            "Rumoca",
            _flatten_rumoca,
            model_folder,
            model_name,
            compiler_options.get("library_folders", []),
            self.rumoca_flags,
            command=tool_command("RTCTOOLS_RUMOCA", "rumoca"),
            cache=compiler_options.get("cache", False),
        )
        return super().pymoca_model(model_folder, model_name, compiler_options)


def _flatten_rumoca(model_folder, model_name, library_dirs, work, flags):
    # Rumoca writes into an output folder: <Model>_base.mo for Base Modelica
    folder = Path(work) / "rumoca-output"
    out = folder / f"{model_name}_base.mo"

    def compile(main):
        # Rumoca loads the other files of the model's folder itself; passing them as source roots
        # too declares their classes twice, under another spelling of the path
        command = tool_command("RTCTOOLS_RUMOCA", "rumoca") + [
            "compile",
            main,
            "--model",
            model_name,
        ]
        command += [arg for root in library_dirs for arg in ("--source-root", str(root))]
        command += ["--target", RUMOCA_TARGET, "--output", str(folder)]
        # As in the Rumoca reader: rtc-tools sets parameters at run time, so a parameter guard
        # must stay a run-time branch
        env = dict(os.environ, RUMOCA_RETAIN_PARAMETER_BRANCHES="1")
        return run_tool(
            command + list(flags), work, "Rumoca (RTCTOOLS_RUMOCA, default rumoca)", env
        )

    main = _main_file(model_folder, model_name)
    result = compile(main)
    if result.returncode != 0 and "ED001" in result.stdout + result.stderr:
        # Under-determined by design (the rest comes from Python or the optimizer): Rumoca only
        # accepts that for a partial model, as for the Rumoca reader
        from .rumoca_reader_mixin import _as_partial

        logger.info(f"{model_name} is under-determined: flattening it as a partial model.")
        shutil.rmtree(folder, ignore_errors=True)
        result = compile(str(_as_partial(Path(main), model_name, Path(work) / "partial-model")))
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(
            f"Rumoca did not flatten {model_name}:\n{result.stdout[-3000:]}{result.stderr[-2000:]}"
        )
    return out


def _main_file(model_folder, model_name):
    """The Modelica file of model_folder that defines model_name."""
    files = modelica_files(model_folder)
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
