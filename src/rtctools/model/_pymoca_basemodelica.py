"""pymoca reading ``<model_name>.bmo``, for ``model_format = "basemodelica"``."""

import logging
import os
from pathlib import Path

import pymoca
import pymoca.backends.casadi.api

logger = logging.getLogger("rtctools")

# The compiler option of pymoca that reading Base Modelica needs
UNQUOTE_OPTION = "unquote_identifiers"


def _transfer_model(model_folder: str, model_name: str, compiler_options: dict):
    """
    Compile ``<model_folder>/<model_name>.bmo`` with pymoca, or load it from pymoca's cache.

    :returns: a ``pymoca.backends.casadi.model.Model``, or a ``CachedModel`` from the cache.
    """
    _check_pymoca()
    api = pymoca.backends.casadi.api

    path = Path(model_folder) / f"{model_name}.bmo"
    if not path.is_file():
        raise FileNotFoundError(f"Base Modelica file {path} not found")

    options = api._merge_default_options(
        {**compiler_options, UNQUOTE_OPTION: True, "library_folders": [], "modelicapath": []}
    )
    cache, codegen = options["cache"], options["codegen"]
    if cache and codegen:
        logger.warning(
            "Both 'cache' and 'codegen' specified. Code generation will take precedence."
        )
        cache = options["cache"] = False
    if not (cache or codegen):
        return _compile(path, model_name, options)

    # As pymoca's transfer_model: until CasADi pickles MX Functions, caching implies SX
    if cache and not options["expand_mx"]:
        logger.warning("Caching implies expanding to SX. Setting 'expand_mx' to True.")
        options["expand_mx"] = True
    cache_file = Path(model_folder) / f"{model_name}.pymoca_cache"
    try:
        # pymoca's load_model checks the *.mo files of the folder only
        if os.path.getmtime(path) > os.path.getmtime(cache_file):
            raise api.InvalidCacheError("Cache out of date")
        return api.load_model(model_folder, model_name, options)
    except (FileNotFoundError, api.InvalidCacheError):
        model = _compile(path, model_name, options)
        api.save_model(model_folder, model_name, model, options)
        return model


def _check_pymoca():
    from pymoca.backends.casadi._options import _get_default_options

    if UNQUOTE_OPTION not in _get_default_options():
        raise RuntimeError(
            f"pymoca {pymoca.__version__} cannot read Base Modelica: it has no "
            f'{UNQUOTE_OPTION} option. Use a pymoca that has it, or model_format = "modelica".'
        )


def _plain(name):
    """A quoted identifier without its quotes"""
    if len(name) >= 2 and name[0] == "'" and name[-1] == "'":
        return name[1:-1]
    return name


def _compile(path, model_name, compiler_options):
    from pymoca import parser

    logger.info(f"Parsing {path}")
    tree = parser.parse(path.read_text(encoding="utf-8"))
    # pymoca looks up the model name with the quotes of the file's class names. Base Modelica
    # wraps the model in a package of the same name.
    name = next((n for n in tree.classes if _plain(n) == model_name), model_name)
    package = tree.classes.get(name)
    if package is not None and package.type == "package":
        inner = next((n for n in package.classes if _plain(n) == model_name), None)
        if inner is not None:
            name = f"{name}.{inner}"
    return pymoca.backends.casadi.api.generate_model(tree, name, compiler_options)
