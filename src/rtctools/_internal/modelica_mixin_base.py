import importlib.resources
import logging
from importlib import metadata as importlib_metadata

import casadi as ca
import pymoca
import pymoca.backends.casadi.api

from .caching import cached

logger = logging.getLogger("rtctools")


def transfer_with_cache_retry(transfer_model, model_folder, model_name, compiler_options):
    """Load the model with transfer_model; if loading it from a cache file fails, compile it."""
    logger.info(f"Loading/compiling model {model_name}.")
    try:
        return transfer_model(model_folder, model_name, compiler_options)
    except (RuntimeError, ModuleNotFoundError) as error:
        if not compiler_options.get("cache", False):
            raise error
        compiler_options["cache"] = False
        logger.warning(f"Loading model {model_name} using a cache file failed: {error}.")
        logger.info(f"Compiling model {model_name}.")
        return transfer_model(model_folder, model_name, compiler_options)


# What the model folder provides, the values of model_format: the Modelica sources, or the model
# flattened to Base Modelica in <model_name>.bmo
MODEL_FORMATS = {"modelica": "the Modelica sources", "basemodelica": "Base Modelica"}


def _model_format(cls):
    """The model_format of the problem class cls, checked"""
    model_format = cls.model_format
    if model_format not in MODEL_FORMATS:
        raise ValueError(
            f"{cls.__name__}.model_format is {model_format!r}, but it is "
            f"{' or '.join(repr(f) for f in MODEL_FORMATS)}"
        )
    return model_format


def _route_mixins(cls):
    """
    The readers and the flatteners among the route mixins of ``rtctools.model`` in the
    bases of the problem class cls, in the order of its MRO.

    A route mixin says what it is in its own class body: ``_modelica_route = "reader"`` or
    ``"flattener"``. A flattener sets ``model_format = "basemodelica"``, writes
    ``<model_name>.bmo`` and then calls the next ``pymoca_model()``. A reader says which
    ``model_format`` it reads, ``_modelica_reads = "modelica"`` or ``"basemodelica"``, loads the
    model and calls no further.
    """

    def declared(kind):
        return [c for c in cls.__mro__ if c.__dict__.get("_modelica_route") == kind]

    return declared("reader"), declared("flattener")


def _check_route_mixins(cls):
    """
    Raise a TypeError for route mixins that would not all take part, in any subclass of cls:
    two readers, two flatteners, or a flattener after the reader. Called when the class is
    defined.
    """
    readers, flatteners = _route_mixins(cls)
    mro = cls.__mro__

    def names(classes):
        return ", ".join(c.__name__ for c in classes)

    if len(readers) > 1:
        raise TypeError(
            f"{cls.__name__} has more than one model reader ({names(readers)}): only the "
            "first would load the model"
        )
    if len(flatteners) > 1:
        raise TypeError(
            f"{cls.__name__} has more than one model flattener ({names(flatteners)}): each would "
            "overwrite the Base Modelica file of the other"
        )
    if flatteners and readers and mro.index(flatteners[0]) > mro.index(readers[0]):
        raise TypeError(
            f"{cls.__name__} lists the flattener {flatteners[0].__name__} after the reader "
            f"{readers[0].__name__}, so it would never run: put the flattener first"
        )


def _check_model_route(cls):
    """
    Raise a TypeError for route mixins that do not fit the ``model_format`` of the problem class
    cls, and a ValueError for an unknown ``model_format``. Called when the problem is
    instantiated, as a subclass may still set ``model_format`` or add the flattener.
    """
    model_format = _model_format(cls)
    readers, flatteners = _route_mixins(cls)

    def format_source():
        # The class that sets model_format, for the messages
        source = next(c for c in cls.__mro__ if "model_format" in c.__dict__)
        return f"model_format is {model_format!r} (set by {source.__name__})"

    if flatteners and model_format != "basemodelica":
        raise TypeError(
            f"{cls.__name__} has the flattener {flatteners[0].__name__}, but {format_source()}, so "
            "the Base Modelica file that it writes would not be read"
        )
    if readers:
        reader = readers[0]
        reads = reader.__dict__["_modelica_reads"]
        if reads != model_format:
            hint = "" if flatteners else f' (set model_format = "{reads}")'
            raise TypeError(
                f"{cls.__name__} has the reader {reader.__name__}, which reads "
                f"{MODEL_FORMATS[reads]} (model_format {reads!r}), but {format_source()}{hint}"
            )


class _ModelicaMixinBase:
    """
    The Modelica front end shared by ``ModelicaMixin`` and ``SimulationProblem``: it compiles
    the model with `pymoca <http://github.com/pymoca/pymoca>`_ and groups its symbols.

    ``__init__`` loads the model and builds the variable groups and residuals. What differs per
    class goes through the hooks :py:meth:`_extra_variable_groups`, :py:meth:`_input_group`,
    :py:meth:`_init_from_model` and :py:meth:`_init_after_residuals`. ``super().__init__()``
    comes last, so that the rest of the chain finds the model data in place. Each class defines
    its own ``modelica_library_folders``.

    The route mixins of :py:mod:`rtctools.model` in the bases are checked in two steps. A
    class with two readers, two flatteners or a flattener after the reader is refused with a
    TypeError when it is defined, as no subclass can repair that. Whether the flattener and the
    reader fit ``model_format`` is checked when the problem is instantiated, as a subclass may
    still set ``model_format`` or put a flattener in front of a base class with a reader.

    :cvar model_format:
        What the model folder provides: ``"modelica"`` (default) or ``"basemodelica"``.
    """

    # What the model folder provides: "modelica" or "basemodelica"
    model_format = "modelica"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _check_route_mixins(cls)

    def __init__(self, **kwargs):
        # Check arguments
        assert "model_folder" in kwargs

        # Log pymoca version
        logger.debug(f"Using pymoca {pymoca.__version__}.")

        # Transfer model from the Modelica .mo file to CasADi using pymoca
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        else:
            if hasattr(self, "model_name"):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        _check_model_route(type(self))
        model = self._pymoca_model = self.pymoca_model(
            kwargs["model_folder"], model_name, self.compiler_options()
        )

        # Extract the CasADi MX variables used in the model
        mx = self._mx = {}
        mx["time"] = [model.time]
        mx["states"] = [v.symbol for v in model.states]
        mx["derivatives"] = [v.symbol for v in model.der_states]
        mx["algebraics"] = [v.symbol for v in model.alg_states]
        mx["parameters"] = [v.symbol for v in model.parameters]
        mx.update(self._extra_variable_groups())
        mx["constant_inputs"] = []
        mx["lookup_tables"] = []

        for v in model.inputs:
            if v.symbol.name() in model.delay_states:
                # Delayed feedback variables are local to each ensemble, and
                # therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                mx["algebraics"].append(v.symbol)
            elif v.symbol.name() in kwargs.get("lookup_tables", []):
                mx["lookup_tables"].append(v.symbol)
            else:
                mx[self._input_group(v)].append(v.symbol)

        # Set up what the class builds on the model before the residuals
        self._init_from_model(**kwargs)

        # Initialize DAE and initial residuals
        variable_lists = ["states", "der_states", "alg_states", "inputs", "constants", "parameters"]
        function_arguments = [model.time] + [
            ca.veccat(*[v.symbol for v in getattr(model, variable_list)])
            for variable_list in variable_lists
        ]

        # The DAE can be empty, e.g. if the only (non-aliasing) equations are added in Python.
        self._dae_residual = model.dae_residual_function(*function_arguments)
        if self._dae_residual is None:
            self._dae_residual = ca.MX()

        self._initial_residual = model.initial_residual_function(*function_arguments)
        if self._initial_residual is None:
            self._initial_residual = ca.MX()

        # Set up what the class builds after the residuals, still before the rest of the chain
        self._init_after_residuals(**kwargs)

        super().__init__(**kwargs)

    def _extra_variable_groups(self) -> dict[str, list]:
        """
        Hook for variable groups of the subclass, placed after ``parameters`` in
        ``self._mx``. Default: none.
        """
        return {}

    def _input_group(self, variable: pymoca.backends.casadi.model.Variable) -> str:
        """
        Hook for the group of an input that is neither delayed feedback nor a lookup table.
        Default: ``"constant_inputs"``.
        """
        return "constant_inputs"

    def _init_from_model(self, **kwargs) -> None:
        """
        Hook for the setup of the subclass that needs the model, called after the variable
        groups and before the residuals are built. Default: nothing.
        """
        pass

    def _init_after_residuals(self, **kwargs) -> None:
        """
        Hook for the setup of the subclass after the residuals are built, called before
        ``super().__init__()``. Default: nothing.
        """
        pass

    def _log_dae_variables(self, prefix: str) -> None:
        """
        Log the variable groups in debug mode.
        """
        if logger.getEffectiveLevel() == logging.DEBUG:
            for group in [
                "states",
                "derivatives",
                "algebraics",
                "control_inputs",
                "constant_inputs",
                "parameters",
            ]:
                if group in self._mx:
                    logger.debug(
                        "{}: Found {} {}".format(
                            prefix,
                            group.replace("_", " "),
                            ", ".join([var.name() for var in self._mx[group]]),
                        )
                    )

    def pymoca_model(
        self, model_folder: str, model_name: str, compiler_options: dict[str, str | bool]
    ) -> pymoca.backends.casadi.model.Model:
        """
        Load or compile the model named ``model_name`` from ``model_folder`` with pymoca: from
        the Modelica files, or with ``model_format = "basemodelica"`` from ``<model_name>.bmo``.

        Override this method to supply a model from elsewhere, e.g. one composed in memory.
        The model is requested once, from ``__init__``. The mixins of :py:mod:`rtctools.model`
        override it to read the model another way.

        :param model_folder: Folder containing the Modelica files.
        :param model_name: Name of the model class to compile.
        :param compiler_options: The options returned by :py:meth:`compiler_options`.

        :returns: A pymoca CasADi model.
        """
        if _model_format(type(self)) == "basemodelica":
            from rtctools.model._pymoca_basemodelica import _transfer_model as transfer
        else:
            transfer = pymoca.backends.casadi.api.transfer_model
        return transfer_with_cache_retry(transfer, model_folder, model_name, compiler_options)

    @cached
    def compiler_options(self) -> dict[str, str | bool]:
        """
        Subclasses can configure the `pymoca <http://github.com/pymoca/pymoca>`_ compiler options
        here.

        :returns:
            A dictionary of pymoca compiler options.  See the pymoca documentation for details.
        """

        # Default options
        compiler_options = {}

        # Expand vector states to multiple scalar component states.
        compiler_options["expand_vectors"] = True

        # Where imported model libraries are located.
        library_folders = self.modelica_library_folders.copy()

        for ep in importlib_metadata.entry_points(group="rtctools.libraries.modelica"):
            if ep.name == "library_folder":
                library_folders.append(str(importlib.resources.files(ep.module).joinpath(ep.attr)))

        compiler_options["library_folders"] = library_folders

        # Eliminate equations of the type 'var = const'.
        compiler_options["eliminate_constant_assignments"] = True

        # Eliminate constant symbols from model, replacing them with the values
        # specified in the model.
        compiler_options["replace_constant_values"] = True

        # Replace any constant expressions into the model.
        compiler_options["replace_constant_expressions"] = True

        # Replace any parameter expressions into the model.
        compiler_options["replace_parameter_expressions"] = True

        # Eliminate variables starting with underscores.
        compiler_options["eliminable_variable_expression"] = r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z"

        # Pymoca currently requires `expand_mx` to be set for
        # `eliminable_variable_expression` to work.
        compiler_options["expand_mx"] = True

        # Automatically detect and eliminate alias variables.
        compiler_options["detect_aliases"] = True

        # Cache the model on disk
        compiler_options["cache"] = True

        # Done
        return compiler_options

    @property
    @cached
    def alias_relation(self):
        return self._pymoca_model.alias_relation
