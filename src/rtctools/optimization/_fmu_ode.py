import atexit
import csv
import logging
import os
import re

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict, AliasRelation
from rtctools._internal.caching import cached
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check
from rtctools._internal.fmu_description import FMUDescription

from ._fmu_dae import (
    _HESSIAN_OF_ANALYTIC,
    _HESSIAN_OF_DIFFERENCES,
    _JACOBIAN_ONLY,
    _backward_difference,
    _equal,
    _with_derivative_options,
)
from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")

# FMU-internal variables that are not exposed, as in ModelicaMixin's default pymoca options
_ELIMINABLE = re.compile(r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z")

# Outputs named like this restate an initial equation as a residual: a plain FMU has no other way
# to give them to the optimizer. See _FMUOdeMixin.
_INITIAL_RESIDUAL = re.compile(r"_INITres\d*\Z")

_ANALYTIC = {"uses_directional_derivatives": True, "uses_adjoint_derivatives": False}
_FINITE_DIFFERENCES = {"uses_directional_derivatives": False, "uses_adjoint_derivatives": False}
_BONMIN_PATIENT = {
    "expect_infeasible_problem": "no",
    "required_infeasibility_reduction": 0.9,
    "mu_strategy": "monotone",
    "mu_oracle": "quality-function",
    "gamma_phi": 1e-8,
    "gamma_theta": 1e-5,
}


class _FMUOdeMixin(OptimizationProblem):
    """
    :py:class:`FMUMixin` with ``model_format = "ode"``: a model given as a plain FMI 3.0 Model
    Exchange FMU, evaluated in ODE mode. The FMU solves its algebraic equations internally, so
    that it is the function

    .. code-block:: text

        der(x) = f(t, x, u)
        y      = g(t, x, u)

    Only the states ``x`` and the controls in ``u`` are optimization variables and only they get
    collocation points: the collocation residual is ``der(x) - f(t, x, u)``. Every other variable
    of the model is a dependent variable ``y``:

    * :py:meth:`state` and :py:meth:`state_at` give the expression ``g(t, x, u)``;
    * a bound on ``y`` becomes a path constraint, and a history value at t0 an equality
      constraint at t0;
    * ``der_at(y, t)`` is a backward difference, and the symbolic ``der(y)`` is not available;
    * results are computed afterwards from those of ``x`` and ``u``.

    The FMU is exported by OpenModelica without ``--daeMode``, with ``--tearingMethod=noTearing``
    (tearing may divide by a parameter that is 0, such as the homotopy parameter theta),
    ``--generateDynamicJacobian=symbolic -d=-disableDirectionalDerivatives`` and, if the model
    differentiates an input, ``--fmiInputDerivatives``. See ``examples/ode_route``. Its initial
    solve is skipped
    (``OMC_FMU_SKIP_INITIAL_SOLVE``, branch ``fmu-ode-mode`` of OpenModelica), as the initial
    conditions are constraints here. A plain FMU does not expose the initial equations: an export
    wrapper has to restate them as scalar outputs ``_INITres1``, ``_INITres2``, ...

    Derivatives: ``f`` and the FMU's outputs use the FMU's analytic directional derivatives, other
    dependent variables finite differences. Second derivatives are finite differences of those
    Jacobians (:py:attr:`fmu_exact_hessian`).

    A Boolean FMU input is an optimization variable that is not passed to the FMU: this is only
    right for a model that does not use it. An FMU with event indicators is refused.

    :cvar fmu_model_name:
        Name of the FMU file (without ``.fmu``). Default is the model name.
    :cvar fmu_expose_locals:
        Expose the FMU's continuous local variables as dependent variables (except names matching
        ModelicaMixin's eliminable-variable pattern ``_name``). Default is ``True``.
    :cvar fmu_fixed_states:
        States whose FMU start value is an initial condition. Default is an empty list.
    :cvar fmu_analytic_derivatives:
        Use the FMU's directional derivatives at all. Default is ``True``.
    :cvar fmu_analytic_outputs:
        Use the FMU's directional derivatives for its outputs. Default is ``True``.
    :cvar fmu_exact_hessian:
        Give the solver second derivatives, as finite differences of the FMU's Jacobians. If
        ``False``, the solver uses a limited-memory approximation. Default is ``True``.
    :cvar fmu_seed_from_inputs:
        Seed of states from the timeseries of a constant input, as ``{state: input}``. Without
        algebraic variables to absorb an unfit seed, the FMU has to be able to evaluate the seed.
    """

    fmu_model_name = None
    fmu_expose_locals = True
    fmu_fixed_states = []
    fmu_analytic_outputs = True
    fmu_seed_from_inputs = {}
    fmu_exact_hessian = True
    fmu_analytic_derivatives = True

    # The collocation residual calls the FMU
    check_collocation_linearity = False
    linear_collocation = False
    expand_dae_residual = False

    def __init__(self, **kwargs):
        assert "model_folder" in kwargs

        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        elif hasattr(self, "model_name"):
            model_name = self.model_name
        else:
            model_name = self.__class__.__name__
        fmu_name = self.fmu_model_name or model_name
        fmu_path = os.path.join(kwargs.get("fmu_folder", kwargs["model_folder"]), fmu_name + ".fmu")
        logger.info(f"Loading FMU {fmu_path} in ODE mode.")

        os.environ["OMC_FMU_SKIP_INITIAL_SOLVE"] = "1"

        self.__fmu = desc = FMUDescription(fmu_path)
        atexit.register(desc.cleanup)
        if desc.has_ls_dae_manifest:
            raise ValueError(
                f"FMUMixin: {fmu_path} is a DAE FMU (fmi-ls-dae); use model_format='dae'"
            )
        if desc.event_indicators:
            raise NotImplementedError(
                f"FMUMixin: the FMU {fmu_path} has event indicators (e.g. from delay() or "
                "when-clauses), which cannot be evaluated at arbitrary points in time."
            )

        dae = ca.DaeBuilder(desc.model_name, desc.folder)
        fmu_x = list(dae.x())
        self.__fmu_x = fmu_x
        inputs = [v for v in desc.variables.values() if v.causality == "input"]
        self.__fmu_u = [v.name for v in inputs if v.type in ("Float64", "Float32")]
        boolean_inputs = [v.name for v in inputs if v.type == "Boolean"]
        for v in inputs:
            if v.name not in self.__fmu_u and v.name not in boolean_inputs:
                raise NotImplementedError(f"FMUMixin: input {v.name} is of type {v.type}")
        if boolean_inputs:
            logger.warning(
                f"FMUMixin: the Boolean inputs {boolean_inputs} are not passed to the FMU; "
                "this assumes that the equations do not use them."
            )
        self.__boolean_inputs = boolean_inputs

        # The FMU's outputs, and the initial residuals among them
        self.__fmu_outputs = list(desc.outputs)
        self.__fmu_initial_residuals = [n for n in self.__fmu_outputs if _INITIAL_RESIDUAL.match(n)]
        if not self.__fmu_initial_residuals and getattr(self, "_fmu_sources_initialize", False):
            logger.warning(
                f"FMUMixin: the model has initial equations, but the ODE FMU {fmu_path} has no "
                "_INITres outputs, so they are ignored. Restate them as outputs in a wrapper."
            )
        self.__public_outputs = [
            n for n in self.__fmu_outputs if n not in self.__fmu_initial_residuals
        ]

        # Other dependent variables
        known = set(fmu_x) | {v.name for v in inputs} | set(self.__fmu_outputs)
        if self.fmu_expose_locals is True:
            self.__fmu_locals = [
                v.name
                for v in desc.variables.values()
                if v.causality == "local"
                and v.variability == "continuous"
                and v.type == "Float64"
                and v.derivative is None
                and v.name not in known
                and not v.name.startswith("der(")
                and not v.name.startswith("_D_")
                and not _ELIMINABLE.match(v.name)
            ]
        elif self.fmu_expose_locals:
            self.__fmu_locals = [n for n in self.fmu_expose_locals if n not in known]
        else:
            self.__fmu_locals = []
        desc.expose_as_outputs(self.__fmu_locals)
        self.__dependent = {n: i for i, n in enumerate(self.__public_outputs + self.__fmu_locals)}

        # Inputs der(u) of constant inputs u
        self.__input_derivative_of = {}
        for n in self.__fmu_u:
            m = re.fullmatch(r"der\((.*)\)", n)
            if m and m.group(1) in self.__fmu_u:
                self.__input_derivative_of[n] = m.group(1)
        constant_input_names = set(self.fmu_constant_inputs(**kwargs))
        # der(u) of a constant input is pinned by its bounds; that of a control is a further
        # control with a constraint to its collocation derivative
        self.__pinned_derivatives = {
            d: u for d, u in self.__input_derivative_of.items() if u in constant_input_names
        }
        self.__control_derivatives = {
            d: u for d, u in self.__input_derivative_of.items() if u not in constant_input_names
        }

        # Aliases (FMI <Alias> elements)
        self.__alias_relation = AliasRelation()
        for v in desc.variables.values():
            for a in v.aliases:
                self.__alias_relation.add(v.name, a)

        # MX symbols
        sym = {}
        self.__mx = {}
        self.__mx["time"] = [ca.MX.sym("time")]
        self.__mx["states"] = [sym.setdefault(n, ca.MX.sym(n)) for n in fmu_x]
        self.__mx["derivatives"] = [ca.MX.sym(f"der({n})") for n in fmu_x]
        self.__mx["algebraics"] = []
        self.__mx["control_inputs"] = [
            sym.setdefault(n, ca.MX.sym(n))
            for n in self.__fmu_u + boolean_inputs
            if n not in constant_input_names
        ]
        self.__mx["constant_inputs"] = [
            sym.setdefault(n, ca.MX.sym(n))
            for n in self.__fmu_u + boolean_inputs
            if n in constant_input_names
        ]
        self.__mx["lookup_tables"] = []
        self.__mx["string_parameters"] = []

        # Parameters: settable ones are symbols; calculated ones are reported values only
        self.__settable_parameters = [
            v.name
            for v in desc.variables.values()
            if v.causality == "parameter"
            and v.variability in ("fixed", "tunable")
            and v.type in ("Float64", "Int32", "Boolean")
        ]
        calculated = [
            v.name
            for v in desc.variables.values()
            if v.causality == "calculatedParameter" and v.type in ("Float64", "Int32", "Boolean")
        ]
        self.__mx["parameters"] = [ca.MX.sym(n) for n in self.__settable_parameters]
        self.__default_parameters = {n: desc.variables[n].start for n in self.__settable_parameters}
        try:
            self.__default_parameters.update(desc.evaluate({}, calculated))
        except Exception as e:
            logger.warning(f"FMUMixin: could not evaluate the calculated parameters: {e}")

        self.__args = [
            self.__mx["time"][0],
            ca.vertcat(*self.__mx["states"]),
            ca.vertcat(*[sym[n] for n in self.__fmu_u]),
        ]
        self.__build_functions(
            {n: self.__default_parameters[n] for n in self.__settable_parameters}
        )

        self.__dependent_at_cache = {}

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ FMU functions
    def __build_functions(self, parameter_values):
        """(Re)create the FmuFunctions with the given parameter values baked in."""
        desc = self.__fmu
        dae = ca.DaeBuilder(desc.model_name, desc.folder)
        for n, value in parameter_values.items():
            if value != desc.variables[n].start:
                dae.set(n, desc.variables[n].python_type(value))
        # The bounds are imposed by rtc-tools instead
        for n in self.__fmu_x + self.__fmu_u:
            dae.set_min(n, -np.inf)
            dae.set_max(n, np.inf)
        base = {"parallelization": "serial", **_JACOBIAN_ONLY}

        def function(name, out, derivatives, names=None):
            opts = {**base, **derivatives}
            if names is not None:
                opts.update(
                    {
                        "scheme_in": ["t", "x", "u"],
                        "scheme_out": ["y"],
                        "scheme": {
                            "t": [
                                v.name
                                for v in desc.variables.values()
                                if v.causality == "independent"
                            ],
                            "x": self.__fmu_x,
                            "u": self.__fmu_u,
                            "y": names,
                        },
                    }
                )
            opts = _with_derivative_options(opts)
            if self.fmu_exact_hessian:
                analytic_jacobian = derivatives is _ANALYTIC
                hessian = _HESSIAN_OF_ANALYTIC if analytic_jacobian else _HESSIAN_OF_DIFFERENCES
                opts["jacobian_options"] = {**opts["jacobian_options"], **hessian}
            return dae.create(name, ["t", "x", "u"], [out], opts)

        analytic = desc.provides_directional_derivatives and self.fmu_analytic_derivatives
        ode = function("fmu_ode", "ode", _ANALYTIC if analytic else _FINITE_DIFFERENCES)
        self.__dae_residual = ca.vertcat(*self.__mx["derivatives"]) - ode(*self.__args)

        output_derivatives = (
            _ANALYTIC if analytic and self.fmu_analytic_outputs else _FINITE_DIFFERENCES
        )
        self.__f_outputs = self.__f_locals = None
        self.__f_initial = None
        outputs = self.__public_outputs
        if outputs:
            self.__f_outputs = function("fmu_outputs", "y", output_derivatives, outputs)
        if self.__fmu_locals:
            self.__f_locals = function("fmu_locals", "y", _FINITE_DIFFERENCES, self.__fmu_locals)
        if self.__fmu_initial_residuals:
            f_initial = function(
                "fmu_initial_residual", "y", output_derivatives, self.__fmu_initial_residuals
            )
            self.__f_initial = f_initial
            self.__initial_residual = f_initial(*self.__args)
        else:
            self.__initial_residual = ca.MX()
        self.__dependent_expressions = self.__dependent_values(*self.__args)
        self.__baked_parameters = dict(parameter_values)

    def __dependent_values(self, t, x, u):
        """Vector of all dependent variables at the given point."""
        parts = []
        if self.__f_outputs is not None:
            parts.append(self.__f_outputs(t, x, u))
        if self.__f_locals is not None:
            parts.append(self.__f_locals(t, x, u))
        return ca.vertcat(*parts) if parts else ca.MX()

    def transcribe(self):
        # Parameter values are baked into the FMU instances: rebuild if they changed
        values = []
        for ensemble_member in range(self.ensemble_size):
            p = self.parameters(ensemble_member)
            for n, default in self.__default_parameters.items():
                if n in self.__settable_parameters or n not in p:
                    continue
                if not _equal(p[n], default):
                    raise ValueError(
                        f"FMUMixin: parameter {n} = {p[n]} differs from its value {default} "
                        "in the FMU, where it is a calculated parameter that cannot be set."
                    )
            v = {}
            for n in self.__settable_parameters:
                value = p.get(n, self.__default_parameters[n])
                if isinstance(value, ca.MX):
                    if not value.is_constant():
                        raise ValueError(f"FMUMixin: parameter {n} must be numeric, got {value}")
                    value = float(value)
                v[n] = value
            values.append(v)
        if any(any(not _equal(v[n], values[0][n]) for n in v) for v in values[1:]):
            raise NotImplementedError(
                "FMUMixin: ensemble members with different FMU parameter values"
            )
        if any(not _equal(values[0][n], self.__baked_parameters[n]) for n in values[0]):
            changed = {
                n: values[0][n]
                for n in values[0]
                if not _equal(values[0][n], self.__baked_parameters[n])
            }
            logger.info(f"FMUMixin: new FMU instances for parameter values {changed}")
            self.__build_functions(values[0])
            self.clear_transcription_cache()
        self.__dependent_at_cache = {}
        return super().transcribe()

    # ------------------------------------------------------------------ rtc-tools interface
    def fmu_constant_inputs(self, **kwargs) -> list[str]:
        """
        Names of the FMU inputs that are constant inputs: those with a column in the CSV
        timeseries import (of the first ensemble member).
        """
        input_folder = kwargs.get("input_folder")
        basename = getattr(self, "timeseries_import_basename", "timeseries_import")
        candidates = [os.path.join(input_folder or "", basename + ".csv")]
        if input_folder and os.path.isdir(input_folder):
            candidates += [
                os.path.join(input_folder, d, basename + ".csv")
                for d in sorted(os.listdir(input_folder))
            ]
        delimiter = getattr(self, "csv_delimiter", ",")
        inputs = self.__fmu_u + self.__boolean_inputs
        for path in candidates:
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    header = next(csv.reader(f, delimiter=delimiter), [])
                columns = {c.strip() for c in header[1:]}
                return [
                    n
                    for n in inputs
                    if n in columns or any(a in columns for a in self.__fmu.variables[n].aliases)
                ]
        logger.warning("FMUMixin: no CSV timeseries import found; all inputs are controls.")
        return []

    @property
    def dae_residual(self):
        return self.__dae_residual

    @property
    def initial_residual(self):
        return self.__initial_residual

    @property
    def dae_variables(self):
        return self.__mx

    @property
    @cached
    def alias_relation(self):
        return self.__alias_relation

    @property
    @cached
    def output_variables(self):
        output_variables = [ca.MX.sym(n) for n in self.__public_outputs]
        output_variables.extend(
            v for v in self.__mx["control_inputs"] if v.name() not in self.__input_derivative_of
        )
        return output_variables

    def map_options(self):
        options = super().map_options()
        options["mode"] = "serial"
        return options

    def solver_options(self):
        options = super().solver_options()
        if options.get("solver") in ("ipopt", "bonmin") and not self.fmu_exact_hessian:
            options[options["solver"]].setdefault("hessian_approximation", "limited-memory")
        if options.get("solver") == "bonmin":
            # Bonmin's defaults give up on a relaxation early. Without algebraic variables, the
            # constraints on them are harder to satisfy from a seed.
            for key, value in _BONMIN_PATIENT.items():
                options["bonmin"].setdefault(key, value)
        options["optimized_num_dir"] = max(options.get("optimized_num_dir", 0), 64)
        return options

    @cached
    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters.update(self.__default_parameters)
        return parameters

    @cached
    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        initial_time = np.array([self.initial_time])
        for n in self.fmu_fixed_states:
            start = self.__fmu.variables[n].start
            if start is None:
                raise ValueError(f"FMUMixin: fixed state {n} has no start value in the FMU")
            history[n] = Timeseries(initial_time, float(start))
        return history

    @cached
    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )
        for n in self.__fmu_x + self.__fmu_u + list(self.__dependent):
            v = self.__fmu.variables[n]
            m, M = bounds.get(n, (-np.inf, np.inf))
            bounds[n] = (max(m, v.min), min(M, v.max))
        for n in self.__boolean_inputs:
            m, M = bounds.get(n, (0.0, 1.0))
            bounds[n] = (m, M)
        for d in self.__pinned_derivatives:
            times, values = self.__pinned_derivative(d, ensemble_member or 0)
            lower, upper = values.copy(), values.copy()
            lower[0], upper[0] = -np.inf, np.inf
            bounds[d] = (Timeseries(times, lower), Timeseries(times, upper))
        return bounds

    def __pinned_derivative(self, d, ensemble_member):
        u = self.__pinned_derivatives[d]
        ts = self.constant_inputs(ensemble_member)[u]
        times, values = _backward_difference(ts, self.times())
        values[0] = 0.0
        return times, values

    @cached
    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        for d in self.__pinned_derivatives:
            seed[d] = Timeseries(*self.__pinned_derivative(d, ensemble_member))
        constant_inputs = self.constant_inputs(ensemble_member)
        for n, u in self.fmu_seed_from_inputs.items():
            times = self.times(n)
            ts = constant_inputs[u]
            seed[n] = Timeseries(times, np.interp(times, ts.times, ts.values))
        for n in self.__fmu_x:
            if n in self.fmu_fixed_states or n in seed:
                continue
            start = self.__fmu.variables[n].start
            if start is not None and start != 0.0 and np.isfinite(start):
                times = self.times(n)
                seed[n] = Timeseries(times, np.full_like(times, float(start)))
        return seed

    def variable_is_discrete(self, variable):
        canonical, _ = self.alias_relation.canonical_signed(variable)
        return canonical in self.__boolean_inputs

    @property
    @cached
    def __nominals(self):
        nominals = AliasDict(self.alias_relation)
        for n in self.__fmu_x + self.__fmu_u + list(self.__dependent):
            nominal = abs(self.__fmu.variables[n].nominal)
            if np.isfinite(nominal) and nominal not in (0.0, 1.0):
                nominals[n] = nominal
        return nominals

    def variable_nominal(self, variable):
        try:
            return self.__nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)

    # ------------------------------------------------------------------ dependent variables
    def __canonical_dependent(self, variable):
        """(index into the dependent variables, sign) of a name, or (None, None)."""
        if isinstance(variable, ca.MX):
            variable = variable.name()
        canonical, sign = self.alias_relation.canonical_signed(variable)
        i = self.__dependent.get(canonical)
        return (None, None) if i is None else (i, sign)

    def variable(self, variable):
        i, sign = self.__canonical_dependent(variable)
        if i is None:
            return super().variable(variable)
        return sign * self.__dependent_expressions[i]

    def __point_at(self, t, ensemble_member):
        """Arguments of the FMU at time t; the time is relative to t0, as for the collocation."""
        x = ca.vertcat(*[self.state_at(n, t, ensemble_member) for n in self.__fmu_x])
        u = ca.vertcat(*[self.state_at(n, t, ensemble_member) for n in self.__fmu_u])
        return t - self.initial_time, x, u

    def __dependent_at(self, t, ensemble_member):
        key = (float(t), ensemble_member)
        try:
            return self.__dependent_at_cache[key]
        except KeyError:
            value = self.__dependent_values(*self.__point_at(t, ensemble_member))
            self.__dependent_at_cache[key] = value
            return value

    def state_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        i, sign = self.__canonical_dependent(variable)
        if i is None:
            return super().state_at(variable, t, ensemble_member, scaled, extrapolate)
        value = sign * self.__dependent_at(t, ensemble_member)[i]
        return value / self.variable_nominal(variable) if scaled else value

    def state_vector(self, variable, ensemble_member=0):
        i, sign = self.__canonical_dependent(variable)
        if i is None:
            return super().state_vector(variable, ensemble_member)
        return ca.vertcat(
            *[
                self.state_at(variable, t, ensemble_member, scaled=True)
                for t in self.times(variable)
            ]
        )

    def __knots(self, variable, t0, tf, ensemble_member):
        times = self.times(variable)
        t0 = times[0] if t0 is None else t0
        tf = times[-1] if tf is None else tf
        knots = np.unique(np.concatenate(([t0], times[(times >= t0) & (times <= tf)], [tf])))
        return knots, ca.vertcat(*[self.state_at(variable, t, ensemble_member) for t in knots])

    def states_in(self, variable, t0=None, tf=None, ensemble_member=0):
        i, _ = self.__canonical_dependent(variable)
        if i is None:
            return super().states_in(variable, t0, tf, ensemble_member)
        return self.__knots(variable, t0, tf, ensemble_member)[1]

    def integral(self, variable, t0=None, tf=None, ensemble_member=0):
        i, _ = self.__canonical_dependent(variable)
        if i is None:
            return super().integral(variable, t0, tf, ensemble_member)
        t, x = self.__knots(variable, t0, tf, ensemble_member)
        if x.size1() <= 1:
            return ca.MX(0)
        return ca.sum1(0.5 * (x[:-1] + x[1:]) * ca.DM(np.diff(t)))

    def der(self, variable):
        i, _ = self.__canonical_dependent(variable)
        if i is not None:
            raise NotImplementedError(
                f"FMUMixin: der({variable}) of a dependent variable is not available"
            )
        return super().der(variable)

    def der_at(self, variable, t, ensemble_member=0):
        i, _ = self.__canonical_dependent(variable)
        if i is None:
            return super().der_at(variable, t, ensemble_member)
        # As for an algebraic state: backward differences, and 0 at the first available point
        times = self.times(variable)
        if t <= times[0]:
            return 0.0
        k = np.searchsorted(times, t)
        dx = self.state_at(variable, times[k], ensemble_member) - self.state_at(
            variable, times[k - 1], ensemble_member
        )
        return dx / (times[k] - times[k - 1])

    def path_constraints(self, ensemble_member):
        constraints = list(super().path_constraints(ensemble_member))
        # Free at t0, as the initial derivative of a state in ModelicaMixin
        times = self.times()
        lower, upper = np.zeros_like(times), np.zeros_like(times)
        lower[0], upper[0] = -np.inf, np.inf
        for d, u in self.__control_derivatives.items():
            constraints.append(
                (self.control(d) - self.der(u), Timeseries(times, lower), Timeseries(times, upper))
            )
        bounds = self.bounds(ensemble_member) if self.ensemble_specific_bounds else self.bounds()
        for n, i in self.__dependent.items():
            bound = bounds.get(n)
            if bound is None:
                continue
            m, M = bound
            if isinstance(m, Timeseries) or isinstance(M, Timeseries):
                constraints.append((self.__dependent_expressions[i], m, M))
            elif np.isfinite(m) or np.isfinite(M):
                nominal = self.variable_nominal(n)
                constraints.append(
                    (self.__dependent_expressions[i] / nominal, m / nominal, M / nominal)
                )
        return constraints

    def constraints(self, ensemble_member):
        constraints = list(super().constraints(ensemble_member))
        history = self.history(ensemble_member)
        t0 = self.initial_time
        for n in self.__dependent:
            if n not in history:
                continue
            ts = history[n]
            value = float(np.interp(t0, ts.times, ts.values, left=np.nan, right=np.nan))
            if np.isfinite(value):
                nominal = self.variable_nominal(n)
                expr = self.state_at(n, t0, ensemble_member) / nominal
                constraints.append((expr, value / nominal, value / nominal))
        return constraints

    def extract_results(self, ensemble_member=0):
        results = super().extract_results(ensemble_member)
        if not self.__dependent:
            return results
        times = self.times()
        constant_inputs = self.constant_inputs(ensemble_member)

        def series(n):
            if n in results:
                return np.interp(times, self.times(n), np.asarray(results[n]).ravel())
            ts = constant_inputs[n]
            return np.interp(times, ts.times, ts.values)

        x = np.vstack([series(n) for n in self.__fmu_x]) if self.__fmu_x else np.zeros((0, 1))
        u = np.vstack([series(n) for n in self.__fmu_u]) if self.__fmu_u else np.zeros((0, 1))
        y = np.zeros((len(self.__dependent), len(times)))
        for k, t in enumerate(times):
            value = self.__dependent_values(
                t - self.initial_time,
                x[:, k] if x.size else ca.DM(),
                u[:, k] if u.size else ca.DM(),
            )
            y[:, k] = np.array(ca.evalf(value)).ravel()
        for n, i in self.__dependent.items():
            results[n] = y[i, :]
        return results
