"""Helpers of the flattener mixins: writing a Base Modelica file, and running a tool."""

import filecmp
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("rtctools")


def flatten_basemodelica(flattener, flatten, model_folder, model_name, library_dirs, flags):
    """
    Write <model_folder>/<model_name>.bmo with flatten(model_folder, model_name, library_dirs,
    work, flags), which writes the file in the temporary folder work and returns its path.

    The file in the model folder is only replaced when its content changes. Returns its path.
    """
    target = Path(model_folder) / f"{model_name}.bmo"
    # The flattener runs in a temporary folder, where relative paths would not resolve
    model_folder = Path(model_folder).resolve()
    library_dirs = [Path(folder).resolve() for folder in library_dirs]
    with tempfile.TemporaryDirectory(prefix="rtctools-flatten-") as work:
        produced = flatten(model_folder, model_name, library_dirs, work, flags)
        changed = replace_if_changed(produced, target)
    logger.info(f"Wrote {target} with {flattener}{'' if changed else ', unchanged'}.")
    return target


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


def tool_command(variable, default):
    """The command of a tool: the environment variable if set, else the default."""
    return shlex.split(os.environ.get(variable, default))


def run_tool(command, work, what, env=None):
    """Run a tool in the folder work, and return its CompletedProcess."""
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, env=env)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{what}: {error}. Install it or set the command") from error
    return result


def modelica_files(folder):
    """The Modelica files of a folder."""
    return sorted(str(f) for f in Path(folder).glob("*.mo"))
