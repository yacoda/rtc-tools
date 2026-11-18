"""Helpers of the flattener mixins: writing a Base Modelica file, and running a tool."""

import filecmp
import hashlib
import inspect
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("rtctools")


# Next to <model_name>.bmo, the fingerprint of what it was flattened from, as pymoca keeps its
# cache in <model_name>.pymoca_cache
CACHE_SUFFIX = ".bmo_cache"


def flatten_basemodelica(
    flattener,
    flatten,
    model_folder,
    model_name,
    library_dirs,
    flags,
    *,
    command,
    cache=False,
    settings=None,
):
    """
    Write <model_folder>/<model_name>.bmo with flatten(model_folder, model_name, library_dirs,
    work, flags), which writes the file in the temporary folder work and returns its path.

    With cache, the existing file is kept, and flatten not run, when the fingerprint of the
    flattening recorded in <model_name>.bmo_cache still matches: the flattener and its source,
    the version of the tool (``command --version``), the command, the flags, the model name,
    the settings (a dict, such as a library version), and the content of the Modelica files
    of the model folder and the library folders. Otherwise, the file in the model folder is only
    replaced when its content changes. Returns its path.
    """
    target = Path(model_folder) / f"{model_name}.bmo"
    sidecar = Path(model_folder) / f"{model_name}{CACHE_SUFFIX}"
    # The flattener runs in a temporary folder, where relative paths would not resolve
    model_folder = Path(model_folder).resolve()
    library_dirs = [Path(folder).resolve() for folder in library_dirs]
    inputs = None
    if cache:
        inputs = _flattening_inputs(
            flattener, flatten, command, flags, model_name, settings, model_folder, library_dirs
        )
        reason = _cache_miss(inputs, target, sidecar)
        if reason is None:
            logger.info(f"Reusing {target}: flattened by {flattener} from the same inputs.")
            return target
        logger.info(f"Flattening {model_name} with {flattener}: no reusable {target}.")
        logger.debug(f"Flattening {model_name} again: {reason}.")
    with tempfile.TemporaryDirectory(prefix="rtctools-flatten-") as work:
        produced = flatten(model_folder, model_name, library_dirs, work, flags)
        changed = replace_if_changed(produced, target)
    logger.info(f"Wrote {target} with {flattener}{'' if changed else ', unchanged'}.")
    if inputs is not None:
        record = {"fingerprint": _digest(inputs), "bmo": _file_digest(target), "inputs": inputs}
        _write_atomically(sidecar, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return target


def _flattening_inputs(
    flattener, flatten, command, flags, model_name, settings, model_folder, library_dirs
):
    """What a flattening depends on, as a JSON-serializable dict, or None if not known."""
    version = tool_version(command)
    if version is None:
        return None
    try:
        recipe = inspect.getsource(inspect.getmodule(flatten))
    except (OSError, TypeError):
        recipe = flatten.__qualname__
    sources = {"model": _hash_files(model_folder, _model_sources(model_folder))}
    for i, folder in enumerate(library_dirs):
        sources[f"library {i}"] = _hash_files(folder, _library_sources(folder))
    return {
        "flattener": flattener,
        "recipe": hashlib.sha256(recipe.encode()).hexdigest(),
        "command": list(command),
        "tool_version": version,
        "flags": list(flags),
        "model_name": model_name,
        "settings": dict(settings or {}),
        "sources": sources,
    }


def _cache_miss(inputs, target, sidecar):
    """Why target cannot be reused for inputs, or None if it can."""
    if inputs is None:
        return "the version of the tool is not known"
    if not target.is_file():
        return f"there is no {target.name}"
    try:
        record = json.loads(sidecar.read_text())
        fingerprint, bmo, recorded = record["fingerprint"], record["bmo"], record["inputs"]
    except FileNotFoundError:
        return f"there is no {sidecar.name}"
    except (OSError, ValueError, TypeError, KeyError) as error:
        return f"{sidecar.name} is unreadable ({error!r})"
    if bmo != _file_digest(target):
        return f"{target.name} changed since it was flattened"
    if fingerprint == _digest(inputs):
        return None
    if not isinstance(recorded, dict):
        return "the inputs changed"
    changed = [key for key in inputs if recorded.get(key) != inputs[key]]
    if "sources" in changed and isinstance(recorded.get("sources"), dict):
        old = {
            (root, name): digest
            for root, files in recorded["sources"].items()
            if isinstance(files, dict)
            for name, digest in files.items()
        }
        new = {
            (root, name): digest
            for root, files in inputs["sources"].items()
            for name, digest in files.items()
        }
        files = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        changed.remove("sources")
        changed += [f"{root}: {name}" for root, name in files]
    return "changed: " + ", ".join(changed or ["the fingerprint"])


_tool_versions = {}


def tool_version(command):
    """The output of ``command --version``, or None if it fails; asked once per process."""
    key = tuple(command)
    if key not in _tool_versions:
        try:
            result = subprocess.run(
                list(command) + ["--version"], capture_output=True, text=True, timeout=120
            )
        except (OSError, subprocess.SubprocessError):
            version = None
        else:
            version = result.stdout.strip() if result.returncode == 0 else None
        _tool_versions[key] = version or None
    return _tool_versions[key]


def _model_sources(folder):
    """The Modelica files of the model folder, and of the packages in its subfolders."""
    folder = Path(folder)
    files = [Path(f) for f in modelica_files(folder)]
    for package in sorted(folder.glob("*/package.mo")):
        files += _library_sources(package.parent)
    return files


def _library_sources(folder):
    """All Modelica files, and package.order files, under a library folder."""
    return sorted(
        f
        for f in Path(folder).rglob("*")
        if (f.suffix == ".mo" or f.name == "package.order") and f.is_file()
    )


def _hash_files(root, files):
    """The sha256 of each file, by its path relative to root."""
    return {Path(f).relative_to(root).as_posix(): _file_digest(f) for f in files}


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _write_atomically(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


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
