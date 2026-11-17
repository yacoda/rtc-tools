import functools

from ._fmu_dae import _FMUDaeMixin
from ._fmu_ode import _FMUOdeMixin

FMU_FORMATS = ("ode", "dae")

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

    :cvar model_format:
        ``"ode"`` (default) or ``"dae"``.
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

    Like ``model_name``, ``model_format`` can also be given as a keyword argument.
    """

    #: "ode" (default) or "dae"
    model_format = "ode"

    def __new__(cls, *args, **kwargs):
        # The format selects the implementation of every method, so it is fixed before any
        # initializer runs
        self = super().__new__(cls)
        self.model_format = kwargs.get("model_format", self.model_format)
        if self.model_format not in FMU_FORMATS:
            raise ValueError(f"model_format={self.model_format!r} is not one of 'ode', 'dae'")
        # Options that both formats declare: the default of the format, unless a subclass sets
        # them (otherwise _FMUOdeMixin, first in the MRO, would give its default to "dae" too)
        mixin = _FMUOdeMixin if self.model_format == "ode" else _FMUDaeMixin
        for name in _FORMAT_DEFAULTS:
            owner = next(c for c in type(self).__mro__ if name in vars(c))
            if owner in (_FMUOdeMixin, _FMUDaeMixin):
                setattr(self, name, vars(mixin)[name])
        return self
