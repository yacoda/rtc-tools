import csv
import logging
import os
import re

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict, AliasRelation
from rtctools._internal.caching import cached
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check
from rtctools.model._fmu_description import FMUDescription

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")

# FMU-internal variables that, like in ModelicaMixin's default pymoca options, are not exposed
_ELIMINABLE = re.compile(r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z")

# Options of the FmuFunctions for the DAE residuals: forward directional derivatives of the FMU
# (fmi3GetDirectionalDerivative); its adjoint derivatives are not used (see _FMUDaeMixin docstring).
_NO_ADJOINT = {"uses_adjoint_derivatives": False}
# Options of the FmuFunction for FMU outputs: finite differences, as the FMU's directional
# derivatives only cover the DAE residuals.
_FINITE_DIFFERENCES = {"uses_directional_derivatives": False, "uses_adjoint_derivatives": False}


# Forward/reverse derivatives of an FmuFunction call are computed from its Jacobian
# (jacobian() with coloring, per call). Work-around for a CasADi FmuFunction bug: in a
# multi-direction forward derivative (forward(n), n > 1) a direction without any nonzero seed
# gets the sensitivities of another direction instead of zeros (FmuInternal::eval_ad/eval_fd
# return early without writing osens_), which corrupts NLP Jacobians assembled with several
# directions at a time.
_JACOBIAN_ONLY = {"enable_forward": False, "enable_reverse": False}

# Second derivatives: finite differences of the FMU's Jacobian. A step that stays close, as the
# model may not be defined far away. A Jacobian that is itself a finite difference needs more.
_HESSIAN_OF_ANALYTIC = {"enable_fd": True, "fd_method": "central", "fd_options": {"h": 1e-6}}
_HESSIAN_OF_DIFFERENCES = {"enable_fd": True, "fd_method": "central", "fd_options": {"h": 1e-4}}
# A finite-difference Hessian costs two FMU Jacobians (one forward directional derivative per
# unknown each) per unknown and collocation point: for channel_wave_damping (76 unknowns, 702
# points) two minutes per evaluation, where an iteration with IPOPT's limited-memory
# approximation takes two seconds. The default fmu_exact_hessian = None gives the Hessian only
# to FMUs up to this number of unknowns.
_EXACT_HESSIAN_MAX_UNKNOWNS = 20
# Gauss-Newton iterations of the seed of FMU-internal states (see seed)
_SEED_ITERATIONS = 20
_SEED_SEARCH_STEPS = (1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 1000.0, -1000.0)
# Rounds of exports of an FMU with other values of fixed parameters (see transcribe)
_MAX_VARIANT_ROUNDS = 3
# Bonmin's defaults give up on a relaxation of an FMU problem early
_BONMIN_PATIENT = {
    "expect_infeasible_problem": "no",
    "required_infeasibility_reduction": 0.9,
    "mu_strategy": "monotone",
    "mu_oracle": "quality-function",
    "gamma_phi": 1e-8,
    "gamma_theta": 1e-5,
}


def _with_derivative_options(opts):
    der = {k: v for k, v in opts.items() if k not in _JACOBIAN_ONLY and not k.startswith("scheme")}
    return {
        **opts,
        "der_options": der,
        "forward_options": der,
        "reverse_options": der,
        "jacobian_options": der,
    }


def _backward_difference(u, times, history=None):
    """
    Backward finite difference of the timeseries ``u`` on the grid ``times`` (the derivative
    that rtc-tools' collocation uses for a state). At t0: the difference to the last point of
    ``u`` (or else of ``history``) before t0; 0 if there is none.
    """
    times = np.asarray(times, dtype=float)
    values = np.interp(times, u.times, u.values)
    d = np.zeros_like(values)
    d[1:] = np.diff(values) / np.diff(times)
    t_before = u.times[u.times < times[0]]
    if t_before.size:
        t_prev = t_before[-1]
        d[0] = (values[0] - np.interp(t_prev, u.times, u.values)) / (times[0] - t_prev)
    elif history is not None and np.any(history.times < times[0]):
        k = np.flatnonzero(history.times < times[0])[-1]
        d[0] = (values[0] - history.values[k]) / (times[0] - history.times[k])
    return times, d


class _DerivativeInputs(AliasDict):
    """
    Constant inputs, with the FMU input ``der(u)`` of a constant input ``u`` computed on access
    from the timeseries of ``u`` (which a mixin above FMUMixin in the MRO, such as
    :py:class:`CSVMixin`, adds to this same dictionary after ``_FMUDaeMixin.constant_inputs``).
    """

    def __init__(self, relation, derivative_of, times, history):
        super().__init__(relation)
        self.__derivative_of = derivative_of
        self.__times = times
        self.__history = history

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return key in self.__derivative_of and super().__contains__(self.__derivative_of[key])

    def __getitem__(self, key):
        if super().__contains__(key) or key not in self.__derivative_of:
            return super().__getitem__(key)
        u = self.__derivative_of[key]
        times, d = _backward_difference(
            super().__getitem__(u), self.__times(), self.__history.get(u)
        )
        return Timeseries(times, d)

    def get(self, key, default=None):
        return self[key] if key in self else default


class _FMUDaeMixin(OptimizationProblem):
    """
    :py:class:`FMUMixin` with ``model_format = "dae"``: a model given as an FMI 3.0 Model
    Exchange FMU with the `fmi-ls-dae <https://modelica.github.io/fmi-ls-dae/main/>`_ layered
    standard (DAE residuals exposed).

    The FMU ``<model_folder>/<fmu_model_name>.fmu`` is imported with CasADi's ``DaeBuilder``
    (CasADi with fmi-ls-dae support: DAE mode is switched on in Configuration Mode). The FMU is
    expected to be exported by OpenModelica with ``--daeMode --simCodeTarget=C+Rust`` (and
    ``--fmiInputDerivatives`` when the model differentiates an input).

    Mapping onto rtc-tools' variables:

    * states: the FMU's continuous states ``x``; derivatives: one symbol ``der(<x>)`` per state,
      fed into the FMU's ``der(x)`` algebraic variables (semi-explicit fmi-ls-dae form);
    * algebraic states: the FMU's other algebraic variables, plus one per FMU output and per
      exposed FMU-internal variable (:py:attr:`fmu_expose_locals`), with the residual
      ``y - F_y(t, x, z, u)``;
    * inputs: an FMU input is a constant input if it has a timeseries
      (:py:meth:`fmu_constant_inputs`), else a control. An FMU input ``der(u)``
      (``--fmiInputDerivatives``) of a constant input ``u`` is a constant input whose
      timeseries is the backward finite difference of ``u`` on the optimization grid. An FMU
      input ``der(u)`` of a control ``u`` is a further control, with a path constraint that
      equals it to the collocation derivative of ``u`` (the backward difference, as for a state)
      at every time except t0, where it is free, as the initial derivative of a state in
      ModelicaMixin. A chain ``der(der(u))`` (index reduction) is the derivative of the control
      ``der(u)`` in the same way;
    * parameters: the FMU's parameters (``causality="parameter"``). FMI 3 lets an importer set
      them only before initialization, so their values are baked into the FMU instances: the
      CasADi functions are rebuilt in :py:meth:`transcribe` whenever :py:meth:`parameters`
      returns other values (e.g. ``theta`` in :py:class:`HomotopyMixin`). Calculated and
      structural parameters (e.g. a Boolean that selects equations) are fixed in the FMU: if
      :py:meth:`parameters` gives one another value, :py:class:`FMUMixin` with a
      ``model_exporter`` exports the model with that value and uses that FMU, which must have
      the same variables; without an exporter, that is an error.

    Seed: the FMU's start values. Where the DAE residual cannot be evaluated there, e.g. as
    OpenModelica selected FMU-internal variables (``_name``) as states, or as the FMU lacks
    bounds that depend on parameters, these states are seeded from the other seeds or moved
    into the domain of the model (see ``__seed_internal_states`` and
    ``__seed_evaluable_states``).

    Derivatives: the DAE and initial residuals use the FMU's analytic forward directional
    derivatives (``fmi3GetDirectionalDerivative``); FMU outputs use finite differences (the
    directional derivatives of OpenModelica's FMUs only cover the residuals). Second
    derivatives are finite differences of those Jacobians (:py:attr:`fmu_exact_hessian`); with
    ``fmu_exact_hessian = False``, IPOPT uses ``hessian_approximation=limited-memory``. Such a
    Hessian costs two FMU Jacobians per FMU unknown and collocation point, so by default
    (``None``) only FMUs with at most ``_EXACT_HESSIAN_MAX_UNKNOWNS`` unknowns get it.

    :cvar fmu_model_name:
        Name of the FMU file (without ``.fmu``). Default is the model name.
    :cvar fmu_expose_locals:
        Expose the FMU's continuous local variables as algebraic states (except names matching
        ModelicaMixin's eliminable-variable pattern ``_name``). ``True``, ``False``, or a list
        of names. Default is ``True``.
    :cvar fmu_fixed_states:
        States whose FMU start value is an initial condition (FMI has no ``fixed`` attribute).
        Default is an empty list.
    :cvar fmu_free_initial_input_derivatives:
        If ``True`` (default), an FMU input ``der(u)`` of a constant input ``u`` is an
        optimization variable fixed by its bounds to the backward finite difference of ``u``
        at every time except t0, where it is free. This matches ModelicaMixin, where such a
        ``u`` is typically aliased to a differentiated state whose derivative at t0 is the free
        ``initial_der`` variable. If ``False``, ``der(u)`` is a constant input, with at t0 the
        backward difference to the data or history before t0, or 0.
    """

    #: Name of the FMU file (without .fmu); default: the model name
    fmu_model_name = None

    #: Expose the FMU's continuous local variables as algebraic states
    fmu_expose_locals = True

    #: States whose FMU start value is an initial condition
    fmu_fixed_states = []

    #: FMU inputs der(u) of constant inputs u: free at t0 (as pymoca's initial derivative)
    fmu_free_initial_input_derivatives = True

    #: Second derivatives for the solver, as finite differences of the FMU's Jacobians. None:
    #: only for an FMU with at most _EXACT_HESSIAN_MAX_UNKNOWNS unknowns (x, z and u).
    fmu_exact_hessian = None

    # The collocation residual calls the FMU: it is never affine in the sense of is_affine
    check_collocation_linearity = False
    linear_collocation = False

    # An expanded (SX) FMU call gets its forward derivatives one direction at a time, i.e. one
    # Jacobian evaluation of the FMU call per direction (see _JACOBIAN_ONLY); in MX, one per sweep
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
        logger.info(f"Loading FMU {fmu_path}.")

        self.__fmu = desc = FMUDescription(fmu_path)
        self.__fmu_model_name = desc.model_name

        if desc.n_event_indicators:
            raise NotImplementedError(
                f"FMUMixin: the FMU {fmu_path} has event indicators (e.g. from delay() or "
                "when-clauses). Delays and events are not supported on the FMU route."
            )

        # FMU-internal variables exposed as outputs (algebraic states)
        if self.fmu_expose_locals is True:
            locals_ = [
                v.name
                for v in desc.variables.values()
                if v.causality == "local"
                and v.variability == "continuous"
                and v.type == "Float64"
                and not v.name.startswith("der(")
                and not v.name.startswith("_D_")
                and v.derivative is None
                and not _ELIMINABLE.match(v.name)
            ]
        elif self.fmu_expose_locals:
            locals_ = list(self.fmu_expose_locals)
        else:
            locals_ = []

        # The variable sets of the DAE (the FMU instances used for the optimization are created
        # in __build_functions)
        fmu_x, fmu_z, fmu_u = list(desc.continuous_states), list(desc.algebraics), list(desc.inputs)
        der_of = dict(desc.derivative_of)
        locals_ = [n for n in locals_ if n not in fmu_x + fmu_z + fmu_u]
        self.__fmu_outputs = list(desc.outputs)
        self.__exposed_locals = locals_
        fmu_y = _fmu_y(desc, locals_)
        state_of = {s: d for d, s in der_of.items()}
        for s in fmu_x:
            if state_of[s] not in fmu_z:
                raise ValueError(
                    f"FMUMixin: der({s}) is not among the FMU's algebraic variables; "
                    "the FMU must be exported in DAE mode (fmi-ls-dae, semi-explicit form)"
                )
        self.__fmu_x, self.__fmu_z, self.__fmu_u, self.__fmu_y = fmu_x, fmu_z, fmu_u, fmu_y
        self.__der_of = der_of
        # Boolean inputs are not FMU inputs for DaeBuilder: as on the ODE route, they are
        # optimization variables that are not passed to the FMU, right for a model that does not
        # use them in its equations
        boolean_inputs = [
            v.name
            for v in desc.variables.values()
            if v.causality == "input" and v.type == "Boolean"
        ]
        if boolean_inputs:
            logger.warning(
                f"FMUMixin: the Boolean inputs {boolean_inputs} are not passed to the FMU; "
                "this assumes that the equations do not use them."
            )
        self.__boolean_inputs = boolean_inputs

        # Inputs: plain inputs and input derivatives der(u)
        self.__input_derivative_of = {}
        for n in fmu_u:
            m = re.fullmatch(r"der\((.*)\)", n)
            if m and m.group(1) in fmu_u:
                self.__input_derivative_of[n] = m.group(1)
        for n in fmu_u:
            if desc.variables[n].type not in ("Float64", "Float32"):
                raise NotImplementedError(
                    f"FMUMixin: input {n} is of type {desc.variables[n].type}; only Float64 "
                    "inputs are supported on the FMU route"
                )

        constant_input_names = set(self.fmu_constant_inputs(**kwargs))
        # der(u) of a constant input u is given by the data of u (see below); der(u) of a control
        # u is a further control with a path constraint to the collocation derivative of u. A
        # chain der(der(u)) is a control derivative of the control der(u).
        self.__constant_derivative_of = {
            d: u for d, u in self.__input_derivative_of.items() if u in constant_input_names
        }
        self.__control_derivatives = {
            d: u for d, u in self.__input_derivative_of.items() if u not in constant_input_names
        }

        # Aliases (FMI <Alias> elements; positive sign only)
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
        der_sym = dict(zip(fmu_x, self.__mx["derivatives"], strict=True))
        self.__algebraic_names = [n for n in fmu_z if n not in der_of] + fmu_y
        self.__mx["algebraics"] = [sym.setdefault(n, ca.MX.sym(n)) for n in self.__algebraic_names]
        # Inputs der(u): constant inputs, or controls pinned by their bounds except at t0
        self.__pinned_derivatives = (
            dict(self.__constant_derivative_of) if self.fmu_free_initial_input_derivatives else {}
        )
        if not self.fmu_free_initial_input_derivatives:
            constant_input_names |= set(self.__constant_derivative_of)
        self.__mx["control_inputs"] = [
            sym.setdefault(n, ca.MX.sym(n))
            for n in fmu_u + boolean_inputs
            if n not in constant_input_names
        ]
        self.__mx["constant_inputs"] = [
            sym.setdefault(n, ca.MX.sym(n))
            for n in fmu_u + boolean_inputs
            if n in constant_input_names
        ]
        self.__mx["lookup_tables"] = []
        self.__mx["string_parameters"] = []

        # Parameters: settable ones are symbols; calculated ones are reported values only
        self.__settable_parameters = _settable_parameters(desc)
        self.__mx["parameters"] = [ca.MX.sym(n) for n in self.__settable_parameters]
        self.__default_parameters = self.__fmu_default_parameters(desc)
        # Values of calculated and structural parameters that the FMU was exported with on
        # request of parameters() (see transcribe)
        self.__modification = {}
        self.__state_nominals = _fmu_state_nominals(desc)

        # Residual arguments: z with the der(x) entries fed from the derivative symbols
        self.__args = [
            self.__mx["time"][0],
            ca.vertcat(*self.__mx["states"]),
            ca.vertcat(*[der_sym[der_of[n]] if n in der_of else sym[n] for n in fmu_z]),
            ca.vertcat(*[sym[n] for n in fmu_u]),
        ]
        self.__y_syms = ca.vertcat(*[sym[n] for n in fmu_y])
        self.__f_y = None
        self.__y_nominals = None
        self.__baked_parameters = None
        self.__build_functions(self.__default_parameters_settable())

        if logger.getEffectiveLevel() == logging.DEBUG:
            for k in [
                "states",
                "derivatives",
                "algebraics",
                "control_inputs",
                "constant_inputs",
                "parameters",
            ]:
                logger.debug(f"FMUMixin: {k}: {', '.join(v.name() for v in self.__mx[k])}")

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ FMU functions
    def __fmu_default_parameters(self, desc):
        """The FMU's values of its settable and calculated parameters."""
        # Parameters of the model that this FMU lets set: a variant FMU (see __swap_fmu) can
        # have fixed some
        self.__settable_now = set(_settable_parameters(desc)) & set(self.__settable_parameters)
        defaults = {n: desc.variables[n].start for n in _settable_parameters(desc)}
        calculated = [
            v.name
            for v in desc.variables.values()
            if v.causality == "calculatedParameter" and v.type in ("Float64", "Int32", "Boolean")
        ]
        try:
            defaults.update(desc.evaluate(calculated))
        except Exception as e:
            logger.warning(f"FMUMixin: could not evaluate the calculated parameters: {e}")
        return defaults

    def _fmu_with_parameter_values(self, values):
        """
        Path of an FMU of the model with the given values of calculated or structural
        parameters (a dict of the parameter values that differ from the original FMU's), or None
        if there is no such FMU. Overridden by FMUMixin, which exports one with the model
        exporter.
        """
        return None

    def __swap_fmu(self, path, modification):
        """
        Replace the FMU by the one at path, an export of the same model with other values of
        calculated or structural parameters: it must have the same variables.
        """
        desc = FMUDescription(path)
        fmu_y = _fmu_y(desc, self.__exposed_locals)
        sets = [desc.continuous_states, desc.algebraics, desc.inputs, fmu_y]
        ours = [self.__fmu_x, self.__fmu_z, self.__fmu_u, self.__fmu_y]
        if sets != ours:
            differ = [k for k, a, b in zip("xzuy", sets, ours, strict=True) if a != b]
            raise ValueError(
                f"FMUMixin: the FMU {path}, exported with {modification}, has other variables "
                f"({', '.join(differ)}) than the FMU of the model. Use one FMU per variant "
                "(fmu_model_name) instead."
            )
        self.__fmu = desc
        self.__modification = dict(modification)
        self.__default_parameters = self.__fmu_default_parameters(desc)
        self.__state_nominals = _fmu_state_nominals(desc)
        # Values computed from the previous FMU
        for name in list(vars(self)):
            if re.match(r"__(parameters|seed|bounds|history)(\[|\Z)", name):
                delattr(self, name)

    def __default_parameters_settable(self):
        return {n: self.__default_parameters[n] for n in self.__settable_parameters}

    def __build_functions(self, parameter_values):
        """(Re)create the FmuFunctions with the given parameter values baked in."""
        desc = self.__fmu
        dae = desc.builder()
        for n, value in parameter_values.items():
            if n in self.__settable_now and value != desc.variables[n].start:
                dae.set(n, desc.variables[n].python_type(value))
        # CasADi refuses to evaluate finite differences outside the FMU's min/max. The bounds
        # are imposed by rtc-tools instead (bounds()), which IPOPT may relax slightly.
        for n in self.__fmu_x + self.__fmu_z + self.__fmu_u:
            dae.set_min(n, -np.inf)
            dae.set_max(n, np.inf)
        # The nominals scale the steps of CasADi's finite differences (FMU outputs): with the
        # default nominal 1, a state of order 1e5 gets a relative step of 1e-11, and Jacobians
        # with rounding errors that keep IPOPT from converging
        for n, nominal in self.__state_nominals.items():
            if n in self.__fmu_x and _usable_nominal(nominal):
                dae.set_nominal(n, nominal)
        # Directional derivatives through the FmuFunction's Jacobian (see _JACOBIAN_ONLY)
        opts = {"parallelization": "serial", **_JACOBIAN_ONLY}

        def options(base, hessian):
            result = _with_derivative_options(base)
            if self.__exact_hessian():
                result["jacobian_options"] = {**result["jacobian_options"], **hessian}
            return result

        f_alg = dae.create(
            "fmu_dae_residual",
            ["t", "x", "z", "u"],
            ["alg"],
            options({**opts, **_NO_ADJOINT}, _HESSIAN_OF_ANALYTIC),
        )
        self.__f_alg = f_alg
        residual = [f_alg(*self.__args)]
        if self.__fmu_y:
            time = [v.name for v in desc.variables.values() if v.causality == "independent"]
            scheme = {"t": time, "x": self.__fmu_x, "z": self.__fmu_z, "u": self.__fmu_u}
            scheme["y"] = self.__fmu_y
            f_y = dae.create(
                "fmu_outputs",
                ["t", "x", "z", "u"],
                ["y"],
                options(
                    {
                        **opts,
                        **_FINITE_DIFFERENCES,
                        "scheme_in": ["t", "x", "z", "u"],
                        "scheme_out": ["y"],
                        "scheme": scheme,
                    },
                    _HESSIAN_OF_DIFFERENCES,
                ),
            )
            residual.append(self.__y_syms - f_y(*self.__args))
            self.__f_y = f_y
        self.__dae_residual = ca.vertcat(*residual)
        if desc.initial_residuals:
            f_init = dae.create(
                "fmu_initial_residual",
                ["t", "x", "z", "u"],
                ["ialg"],
                options({**opts, **_NO_ADJOINT}, _HESSIAN_OF_ANALYTIC),
            )
            self.__initial_residual = f_init(*self.__args)
        else:
            self.__initial_residual = ca.MX()
        self.__baked_parameters = dict(parameter_values)

    def transcribe(self):
        # Parameter values are baked into the FMU instances: rebuild if they changed
        self.__use_fmu_for_parameter_values()
        values = []
        for ensemble_member in range(self.ensemble_size):
            p = self.parameters(ensemble_member)
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
        if self.__y_nominals is None:
            self.__y_nominals = self.__estimate_output_nominals()
        return super().transcribe()

    def __fixed_parameter_differences(self):
        """Calculated or structural parameters whose value in parameters() is not the FMU's."""
        differences = {}
        for ensemble_member in range(self.ensemble_size):
            p = self.parameters(ensemble_member)
            for n, default in self.__default_parameters.items():
                if n in self.__settable_now or n not in p or _equal(p[n], default):
                    continue
                if n in differences and not _equal(differences[n], p[n]):
                    raise NotImplementedError(
                        f"FMUMixin: ensemble members with different values of the calculated or "
                        f"structural parameter {n}"
                    )
                differences[n] = p[n]
        return differences

    def __use_fmu_for_parameter_values(self):
        """
        Calculated and structural parameters (e.g. a Boolean that selects equations) are fixed
        in the FMU. If parameters() gives one another value, switch to an FMU exported with that
        value (:py:meth:`_fmu_with_parameter_values`), else raise.
        """
        differences = self.__fixed_parameter_differences()
        # A parameter that was settable can be fixed in the FMU of a variant, e.g. a Boolean
        # that selects equations only in the equations that another one selects: a few rounds
        for _ in range(_MAX_VARIANT_ROUNDS):
            if not differences:
                return
            modification = {**self.__modification, **differences}
            path = self._fmu_with_parameter_values(modification)
            if path is None:
                n, value = next(iter(differences.items()))
                raise ValueError(
                    f"FMUMixin: parameter {n} = {value} differs from its value "
                    f"{self.__default_parameters[n]} in the FMU, where it is a calculated or "
                    "structural parameter that cannot be set. Export the FMU with the desired "
                    "value instead, or let FMUMixin export it (model_exporter)."
                )
            logger.info(f"FMUMixin: using the FMU {path}, exported with {modification}")
            self.__swap_fmu(path, modification)
            self.__build_functions(self.__baked_parameters)
            self.clear_transcription_cache()
            self.__y_nominals = None
            differences = self.__fixed_parameter_differences()
        if differences:
            raise ValueError(
                f"FMUMixin: the FMU {path}, exported with {modification}, still has other values "
                f"of the calculated or structural parameters {differences}"
            )

    def __estimate_output_nominals(self):
        """
        Nominals of the FMU outputs (and exposed FMU-internal variables) that have no nominal
        attribute in the FMU: ``max(|y|, sum_j |dy/dv_j| nominal_j)`` at a reference point
        (history, seed or constant input at t0, else the midpoint of finite bounds, else the
        FMU start value), if that exceeds 1. ModelicaMixin gets these from pymoca's alias
        elimination (e.g. ``V_storage = storage.V`` has the nominal of ``storage.V``).
        """
        missing = [n for n in self.__fmu_y if self.__fmu.variables[n].nominal == 1.0]
        if not missing:
            return {}
        t0 = self.initial_time
        history, seed = self.history(0), self.seed(0)
        constant_inputs, bounds = self.constant_inputs(0), self.bounds()

        def at_t0(ts):
            try:
                return float(np.interp(t0, ts.times, ts.values))
            except Exception:
                return None

        def reference(n):
            for d in (history, seed, constant_inputs):
                if n in d:
                    v = at_t0(d[n])
                    if v is not None and np.isfinite(v):
                        return v
            m, M = bounds.get(n, (-np.inf, np.inf))
            if not isinstance(m, Timeseries) and not isinstance(M, Timeseries):
                if np.isfinite(m) and np.isfinite(M):
                    return 0.5 * (m + M)
            start = self.__fmu.variables[n].start
            return float(start) if start is not None else 0.0

        x = np.array([reference(n) for n in self.__fmu_x])
        z = np.array([0.0 if n in self.__der_of else reference(n) for n in self.__fmu_z])
        u = np.array([reference(n) for n in self.__fmu_u])
        nominals = np.concatenate(
            [
                [self.variable_nominal(n) for n in self.__fmu_x],
                [1.0 if n in self.__der_of else self.variable_nominal(n) for n in self.__fmu_z],
                [self.variable_nominal(n) for n in self.__fmu_u],
            ]
        )
        try:
            y = np.array(self.__f_y(0.0, x, z, u)).ravel()
            jac = self.__f_y.jacobian()(0.0, x, z, u, y)
            J = np.hstack([np.array(ca.densify(j)) for j in jac[1:]])
        except Exception as e:
            logger.warning(f"FMUMixin: could not estimate nominals of the FMU outputs: {e}")
            return {}
        estimate = np.maximum(np.abs(y), np.abs(J) @ nominals)
        result = {}
        for i, n in enumerate(self.__fmu_y):
            if n in missing and np.isfinite(estimate[i]) and estimate[i] > 1.0:
                result[n] = float(estimate[i])
                logger.debug(f"FMUMixin: estimated nominal of {n}: {result[n]}")
        return result

    # ------------------------------------------------------------------ rtc-tools interface
    def fmu_constant_inputs(self, **kwargs) -> list[str]:
        """
        Names of the FMU inputs that are constant inputs (given by data) rather than controls.

        ModelicaMixin uses the inputs' ``fixed`` attribute, which FMI does not have. The
        default implementation takes the inputs that have a column in the CSV timeseries import
        (``<input_folder>/<timeseries_import_basename>.csv``, or that of the first ensemble
        member). Override it for other input formats.
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
        for path in candidates:
            if os.path.isfile(path):
                with open(path, newline="") as f:
                    header = next(csv.reader(f, delimiter=delimiter), [])
                columns = {c.strip() for c in header[1:]}
                return [
                    n
                    for n in self.__fmu_u + self.__boolean_inputs
                    if n in columns or any(a in columns for a in self.__fmu.variables[n].aliases)
                ]
        logger.warning(
            "FMUMixin: no CSV timeseries import found; all FMU inputs are controls. "
            "Override fmu_constant_inputs() to declare constant inputs."
        )
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
        output_variables = [ca.MX.sym(n) for n in self.__fmu_outputs]
        output_variables.extend(
            v for v in self.__mx["control_inputs"] if v.name() not in self.__input_derivative_of
        )
        return output_variables

    def map_options(self):
        # One FMU instance serves all time steps; no parallel evaluation of the FMU
        options = super().map_options()
        options["mode"] = "serial"
        return options

    def __exact_hessian(self):
        if self.fmu_exact_hessian is not None:
            return self.fmu_exact_hessian
        unknowns = len(self.__fmu_x) + len(self.__fmu_z) + len(self.__fmu_u)
        return unknowns <= _EXACT_HESSIAN_MAX_UNKNOWNS

    def solver_options(self):
        options = super().solver_options()
        if options.get("solver") == "ipopt" and not self.__exact_hessian():
            options["ipopt"].setdefault("hessian_approximation", "limited-memory")
        if options.get("solver") == "bonmin":
            for key, value in _BONMIN_PATIENT.items():
                options["bonmin"].setdefault(key, value)
        # Each forward sweep through an FMU call evaluates that call's full Jacobian (see
        # _JACOBIAN_ONLY): many directions per sweep in the DAE functions (max_num_dir)
        options["optimized_num_dir"] = max(options.get("optimized_num_dir", 0), 64)
        return options

    @cached
    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters.update(self.__default_parameters)
        return parameters

    @cached
    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        if not self.__constant_derivative_of or self.__pinned_derivatives:
            return constant_inputs
        lazy = _DerivativeInputs(
            self.alias_relation,
            self.__constant_derivative_of,
            self.times,
            self.history(ensemble_member),
        )
        lazy.update(constant_inputs)
        return lazy

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

    def __model_variables(self):
        return self.__fmu_x + self.__algebraic_names + self.__fmu_u

    @cached
    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )
        for n in self.__model_variables():
            v = self.__fmu.variables[n]
            m, M = bounds.get(n, (-np.inf, np.inf))
            bounds[n] = (max(m, v.min), min(M, v.max))
        for n in self.__boolean_inputs:
            bounds[n] = bounds.get(n, (0.0, 1.0))
        for d in self.__pinned_derivatives:
            times, values = self.__pinned_derivative(d, ensemble_member or 0)
            lower, upper = values.copy(), values.copy()
            lower[0], upper[0] = -np.inf, np.inf
            bounds[d] = (Timeseries(times, lower), Timeseries(times, upper))
        return bounds

    def path_constraints(self, ensemble_member):
        constraints = list(super().path_constraints(ensemble_member))
        # An FMU input der(u) of a control u is the collocation derivative of u (the backward
        # difference, as for a state), free at t0 as the initial derivative of a state in
        # ModelicaMixin
        times = self.times()
        lower, upper = np.zeros_like(times), np.zeros_like(times)
        lower[0], upper[0] = -np.inf, np.inf
        for d, u in self.__control_derivatives.items():
            constraints.append(
                (self.control(d) - self.der(u), Timeseries(times, lower), Timeseries(times, upper))
            )
        return constraints

    def __pinned_derivative(self, d, ensemble_member):
        u = self.__pinned_derivatives[d]
        try:
            ts = self.constant_inputs(ensemble_member)[u]
        except KeyError:
            raise Exception(f"FMUMixin: no values found for constant input {u} (for {d})")
        times, values = _backward_difference(ts, self.times())
        values[0] = 0.0
        return times, values

    @cached
    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        for d in self.__pinned_derivatives:
            seed[d] = Timeseries(*self.__pinned_derivative(d, ensemble_member))
        for n in self.__fmu_x + self.__algebraic_names:
            if n in self.fmu_fixed_states:
                continue
            start = self.__fmu.variables[n].start
            if start is not None and start != 0.0 and np.isfinite(start):
                times = self.times(n)
                seed[n] = Timeseries(times, np.full_like(times, float(start)))
        self.__seed_internal_states(seed, ensemble_member)
        self.__seed_evaluable_states(seed, ensemble_member)
        return seed

    def __point_at_t0(self, seed, ensemble_member, use_history=True):
        """
        (t0, x, z, u) of the seed at t0; z with der(x) = 0. Without the history, the point is
        that of the seed after t0, where the initial states of the history do not apply.
        """
        t0 = self.initial_time
        history = self.history(ensemble_member) if use_history else {}
        constant_inputs = self.constant_inputs(ensemble_member)

        def value(n):
            for d in (seed, history, constant_inputs):
                if n in d:
                    ts = d[n]
                    try:
                        v = float(np.interp(t0, ts.times, ts.values))
                    except Exception:
                        continue
                    if np.isfinite(v):
                        return v
            start = self.__fmu.variables[n].start
            return float(start) if start is not None and np.isfinite(start) else 0.0

        x = np.array([value(n) for n in self.__fmu_x])
        z = np.array([0.0 if n in self.__der_of else value(n) for n in self.__fmu_z])
        u = np.array([value(n) for n in self.__fmu_u])
        return t0, x, z, u, value

    def __evaluable(self, t0, x, z, u):
        """Whether the DAE residual and its Jacobian are finite at (t0, x, z, u)."""
        try:
            alg = self.__f_alg(t0, x, z, u)
            jacobian = self.__f_alg.jacobian()(t0, x, z, u, alg)
        except Exception:
            return False
        return all(np.all(np.isfinite(np.array(ca.densify(v)))) for v in [alg, *jacobian])

    def __seed_internal_states(self, seed, ensemble_member):
        """
        OpenModelica's state selection can make FMU-internal variables (``_name``, which pymoca
        eliminates) states, e.g. a wetted perimeter instead of a water level. Their start value
        need not fit the seed of the other variables, and the model may not be defined there
        (a negative cross section). If the DAE residual or its Jacobian is not finite at the
        seed at t0, seed these states (constant in time) such that the FMU outputs and exposed
        variables come closest to their seed (Gauss-Newton), as pymoca's seed of the remaining
        variables.
        """
        internal = [
            i
            for i, n in enumerate(self.__fmu_x)
            if _ELIMINABLE.match(n) and n not in self.fmu_fixed_states and n not in seed
        ]
        if not internal or self.__f_y is None:
            return
        t0, x, z, u, value = self.__point_at_t0(seed, ensemble_member)

        def finite_residual(x):
            return self.__evaluable(t0, x, z, u)

        if finite_residual(x):
            return
        target = np.array([value(n) for n in self.__fmu_y])
        scale = np.array([self.variable_nominal(n) for n in self.__fmu_y])
        jacobian = self.__f_y.jacobian()

        def residual(x):
            try:
                r = (np.array(self.__f_y(t0, x, z, u)).ravel() - target) / scale
            except Exception:
                return None
            return r if np.all(np.isfinite(r)) else None

        # Gauss-Newton needs a point where the FMU can be evaluated: the start values, else
        # moved along their nominals by growing steps
        r = residual(x)
        nominal = np.array([self.variable_nominal(self.__fmu_x[i]) for i in internal])
        start = x.copy()
        for step in _SEED_SEARCH_STEPS:
            if r is not None:
                break
            x = start.copy()
            x[internal] += step * nominal
            r = residual(x)
        for _ in range(_SEED_ITERATIONS):
            if r is None or np.linalg.norm(r) < 1e-10:
                break
            y = r * scale + target
            J = np.array(ca.densify(jacobian(t0, x, z, u, y)[1]))[:, internal] / scale[:, None]
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            for _ in range(10):
                trial = x.copy()
                trial[internal] += step
                r_trial = residual(trial)
                if r_trial is not None and np.linalg.norm(r_trial) < np.linalg.norm(r):
                    break
                step = step / 2
            else:
                break
            x, r = trial, r_trial
        if not finite_residual(x):
            logger.warning(
                "FMUMixin: no seed of the FMU-internal states found where the DAE "
                "residual is finite"
            )
            return
        names = [self.__fmu_x[i] for i in internal]
        logger.info(f"FMUMixin: seeded the FMU-internal states {names} from the other seeds")
        for i in internal:
            n = self.__fmu_x[i]
            times = self.times(n)
            seed[n] = Timeseries(times, np.full_like(times, x[i]))

    def __seed_evaluable_states(self, seed, ensemble_member):
        """
        If the DAE residual or its Jacobian is still not finite at the seed (after t0, without
        the initial states of the history), move the states without a seed or fixed start value
        (constant in time) along their nominals by growing steps until it is. The FMU lacks the
        bounds that depend on parameters (OpenModelica only writes literal min and max
        attributes), where pymoca's bounds let IPOPT start inside the domain of the model, e.g.
        a water level above the bottom.
        """
        free = [
            i
            for i, n in enumerate(self.__fmu_x)
            if n not in self.fmu_fixed_states and n not in seed
        ]
        if not free:
            return
        t0, x, z, u, _ = self.__point_at_t0(seed, ensemble_member, use_history=False)

        def finite_residual(x):
            return self.__evaluable(t0, x, z, u)

        if finite_residual(x):
            return
        nominal = np.array([self.variable_nominal(self.__fmu_x[i]) for i in free])
        for step in _SEED_SEARCH_STEPS:
            trial = x.copy()
            trial[free] += step * nominal
            if finite_residual(trial):
                break
        else:
            logger.warning("FMUMixin: the DAE residual is not finite at the seed")
            return
        names = [self.__fmu_x[i] for i in free]
        logger.info(f"FMUMixin: seeded the states {names} {step:+g} nominal from their start")
        for i in free:
            n = self.__fmu_x[i]
            times = self.times(n)
            seed[n] = Timeseries(times, np.full_like(times, trial[i]))

    def variable_is_discrete(self, variable):
        canonical, _ = self.alias_relation.canonical_signed(variable)
        return canonical in self.__boolean_inputs

    @property
    @cached
    def __nominals(self):
        nominals = AliasDict(self.alias_relation)
        for n in self.__model_variables():
            nominal = abs(self.__fmu.variables[n].nominal)
            if np.isfinite(nominal) and nominal not in (0.0, 1.0):
                nominals[n] = nominal
        for n, nominal in self.__state_nominals.items():
            if _usable_nominal(nominal):
                nominals[n] = abs(nominal)
        return nominals

    def variable_nominal(self, variable):
        if self.__y_nominals:
            canonical, _ = self.alias_relation.canonical_signed(variable)
            if canonical in self.__y_nominals:
                return self.__y_nominals[canonical]
        try:
            return self.__nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)


def _fmu_y(desc, exposed_locals):
    """
    The FMU variables that get an output residual: the exposed FMU-internal variables, then the
    FMU's outputs that are not FMU unknowns (an output that is e.g. an algebraic variable needs
    none). The FmuFunction of these takes the locals among them as its outputs ("scheme"):
    without dependencies in the FMU's ModelStructure, they depend on all knowns.
    """
    unknowns = set(desc.continuous_states + desc.algebraics + desc.inputs)
    names = list(exposed_locals) + [n for n in desc.outputs if n not in unknowns]
    return list(dict.fromkeys(names))


def _fmu_state_nominals(desc):
    """
    Nominals of the states from the FMU at run time (default parameter values):
    modelDescription.xml only has literal nominals, and OpenModelica leaves a
    parameter-dependent one such as V(nominal = A) out.
    """
    try:
        return desc.state_nominals()
    except Exception as e:
        logger.warning(f"FMUMixin: could not evaluate the nominals of the states: {e}")
        return {}


def _usable_nominal(nominal):
    return np.isfinite(nominal) and abs(nominal) not in (0.0, 1.0)


def _settable_parameters(desc):
    """The parameters of an FMU that an importer can set."""
    return [
        v.name
        for v in desc.variables.values()
        if v.causality == "parameter"
        and v.variability in ("fixed", "tunable")
        and v.type in ("Float64", "Int32", "Boolean")
    ]


def _equal(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b
