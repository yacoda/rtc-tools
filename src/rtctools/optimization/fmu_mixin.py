import functools
import hashlib
import logging

from rtctools.model._fmu_export import (
    check_fmu_route,
    export_fmu,
    has_initial_equations,
    library_folders,
    modification,
    route_option,
)

from ._fmu_dae import _FMUDaeMixin
from ._fmu_ode import _FMUOdeMixin

logger = logging.getLogger("rtctools")

_ROUTE_OPTIONS = ("model_format", "model_exporter", "model_exporter_flags")

# Options with a default per format
_FORMAT_DEFAULTS = ("fmu_exact_hessian",)


def _route_by_format(cls, model_format):
    """Let the methods and properties of cls act only on instances with this model_format.

    On an instance with another format, a method or property of cls passes the call on to the
    next class in the method resolution order, as if cls did not define it. The private (name
    mangled) methods of cls are only called by its own methods, and are left alone.
    """

    def route(name, f):
        @functools.wraps(f)
        def routed(self, *args, **kwargs):
            if self.model_format == model_format:
                return f(self, *args, **kwargs)
            return getattr(super(cls, self), name)(*args, **kwargs)

        return routed

    def route_property(name, p):
        def fget(self):
            if self.model_format == model_format:
                return p.fget(self)
            return getattr(super(cls, self), name)

        return property(functools.wraps(p.fget)(fget), doc=p.__doc__)

    private = f"_{cls.__name__}__"
    for name, value in list(vars(cls).items()):
        if name.startswith(private) or (name.startswith("__") and name != "__init__"):
            continue
        if isinstance(value, property):
            setattr(cls, name, route_property(name, value))
        elif callable(value):
            setattr(cls, name, route(name, value))
    return cls


_route_by_format(_FMUOdeMixin, "ode")
_route_by_format(_FMUDaeMixin, "dae")


class FMUMixin(_FMUOdeMixin, _FMUDaeMixin):
    """
    Adds a model given as an FMI 3.0 Model Exchange FMU to your optimization problem. It replaces
    :py:class:`ModelicaMixin`.

    The FMU is ``<fmu_folder>/<fmu_model_name>.fmu``. ``fmu_folder`` is a keyword argument that
    defaults to the model folder, and :py:attr:`fmu_model_name` defaults to the model name. The FMU
    is imported with CasADi's ``DaeBuilder``.

    :py:attr:`model_format` says how the FMU is used:

    * ``"ode"`` (default): a plain FMU in ODE mode, ``der(x) = f(t, x, u)``. Only the states and
      the controls are optimization variables; every other model variable is an expression of
      them. The FMU's initial solve is skipped (``OMC_FMU_SKIP_INITIAL_SOLVE``, an OpenModelica
      fork), and initial equations are only seen as FMU outputs ``_INITres1``, ``_INITres2``, ...
      See ``rtctools.optimization._fmu_ode._FMUOdeMixin`` for the details.
    * ``"dae"``: an FMU with the `fmi-ls-dae <https://modelica.github.io/fmi-ls-dae/main/>`_
      layered standard, which exposes the DAE residuals and the initial equations. Every model
      variable is an optimization variable, as with :py:class:`ModelicaMixin`. See
      ``rtctools.optimization._fmu_dae._FMUDaeMixin`` for the details.

    With :py:attr:`model_exporter`, rtc-tools first exports the FMU from the Modelica files of the
    model folder, on every run. The FMU file is only replaced when its content changed. On the
    ``"dae"`` route with ``"openmodelica"``, a calculated or structural parameter to which
    :py:meth:`parameters` gives another value than the FMU has (e.g. a Boolean that selects
    equations) gets an FMU of its own: a model that extends the model with that value as
    modifier, exported to ``<fmu_model_name>_<hash>.fmu``.

    :cvar model_format:
        ``"ode"`` (default) or ``"dae"``.
    :cvar model_exporter:
        ``None`` (default: use the FMU as it is), ``"openmodelica"`` (the command ``omc``, or
        ``RTCTOOLS_OMC``) or ``"rumoca"`` (the command ``rumoca``, or ``RTCTOOLS_RUMOCA``; ODE
        only). The flags needed for these FMUs are those of OpenModelica branches that are not in
        a release.
    :cvar model_exporter_flags:
        Command line flags of the exporter, in place of the default ones. Default is ``None``.
    :cvar modelica_library_folders:
        Folders with Modelica libraries for the exporter, besides those of installed packages.
        Default is an empty list.
    :cvar fmu_model_name:
        Name of the FMU file (without ``.fmu``). Default is the model name.
    :cvar fmu_expose_locals:
        Expose the FMU's continuous local variables (except names matching ModelicaMixin's
        eliminable-variable pattern ``_name``). ``True``, ``False``, or a list of names. Default
        is ``True``.
    :cvar fmu_fixed_states:
        States whose FMU start value is an initial condition (FMI has no ``fixed`` attribute).
        Default is an empty list.

    :cvar fmu_exact_hessian:
        Give the solver second derivatives, as finite differences of the FMU's Jacobians. If
        ``False``, IPOPT uses a limited-memory approximation. Default is ``True`` for ``"ode"``.
        For ``"dae"``, where every model variable is an optimization variable and a Hessian
        costs two FMU Jacobians per variable and time step, the default ``None`` gives it only
        to small FMUs (at most 20 unknowns).

    The other options belong to one format: ``fmu_free_initial_input_derivatives`` to ``"dae"``,
    and ``fmu_analytic_derivatives``, ``fmu_analytic_outputs`` and ``fmu_seed_from_inputs`` to
    ``"ode"``.

    Like ``model_name``, the options ``model_format``, ``model_exporter`` and
    ``model_exporter_flags`` can also be given as keyword arguments.
    """

    #: "ode" (default) or "dae"
    model_format = "ode"

    #: None (use the FMU as it is), "openmodelica" or "rumoca"
    model_exporter = None

    #: Flags of the exporter, in place of the default ones
    model_exporter_flags = None

    #: Modelica library folders for the exporter
    modelica_library_folders = []

    def __new__(cls, *args, **kwargs):
        # The format selects the implementation of every method, so it is fixed before any
        # initializer runs
        self = super().__new__(cls)
        for name in _ROUTE_OPTIONS:
            setattr(self, name, route_option(self, kwargs, name, getattr(FMUMixin, name)))
        check_fmu_route(self.model_format, self.model_exporter)
        # Options that both formats declare: the default of the format, unless a subclass sets
        # them (otherwise _FMUOdeMixin, first in the MRO, would give its default to "dae" too)
        mixin = _FMUOdeMixin if self.model_format == "ode" else _FMUDaeMixin
        for name in _FORMAT_DEFAULTS:
            owner = next(c for c in type(self).__mro__ if name in vars(c))
            if owner in (_FMUOdeMixin, _FMUDaeMixin):
                setattr(self, name, vars(mixin)[name])
        return self

    def __init__(self, **kwargs):
        assert "model_folder" in kwargs
        if self.model_exporter is not None:
            model_folder = kwargs["model_folder"]
            model_name = kwargs.get("model_name", getattr(self, "model_name", None))
            model_name = model_name or self.__class__.__name__
            libraries = library_folders(self.modelica_library_folders)
            self.__export_arguments = (
                self.model_exporter,
                self.model_format,
                model_folder,
                model_name,
                libraries,
                kwargs.get("fmu_folder", model_folder),
                self.model_exporter_flags,
            )
            self.__variants = {}
            export_fmu(*self.__export_arguments, target_name=self.fmu_model_name)
            # An ODE FMU only gets initial equations from a wrapper (_INITres outputs)
            self._fmu_sources_initialize = self.model_format == "ode" and has_initial_equations(
                model_folder
            )
        super().__init__(**kwargs)

    def _fmu_with_parameter_values(self, values):
        """
        Export the model with other values of calculated or structural parameters (DAE route):
        a model that extends it with these values as modifiers, to
        ``<fmu_folder>/<fmu_model_name>_<hash>.fmu``.
        """
        if self.model_exporter != "openmodelica":
            return None
        text = modification(values)
        if text is None:
            return None
        if text not in self.__variants:
            model_name = self.__export_arguments[3]
            digest = hashlib.sha1(text.encode()).hexdigest()[:10]
            target_name = f"{self.fmu_model_name or model_name}_{digest}"
            logger.info(f"FMUMixin: exporting {model_name}({text}) to {target_name}.fmu")
            self.__variants[text] = str(
                export_fmu(
                    *self.__export_arguments,
                    target_name=target_name,
                    modification=text,
                )
            )
        return self.__variants[text]
