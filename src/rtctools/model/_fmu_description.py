"""
The variables of an FMI 3.0 Model Exchange FMU, through CasADi's ``DaeBuilder``: attributes,
aliases, structure (states, outputs, fmi-ls-dae residuals), and the values of calculated
parameters and parameter-dependent nominals.
"""

import os
from dataclasses import dataclass, field

import casadi as ca

_TYPES = {"Float64": float, "Float32": float, "Boolean": bool}


@dataclass
class FMUVariable:
    name: str
    type: str
    causality: str = "local"
    variability: str = "continuous"
    start: float | bool | int | None = None
    min: float = float("-inf")
    max: float = float("inf")
    nominal: float = 1.0
    aliases: list[str] = field(default_factory=list)
    #: For a state derivative: the name of the state
    derivative: str | None = None

    @property
    def python_type(self):
        return _TYPES.get(self.type, int)


class FMUDescription:
    """
    The variables of an FMI 3.0 FMU and its fmi-ls-dae residuals, as CasADi's ``DaeBuilder``
    reports them.
    """

    def __init__(self, fmu_path: str):
        self.fmu_path = fmu_path
        self.model_name = os.path.splitext(os.path.basename(os.path.normpath(fmu_path)))[0]
        dae = self.builder()
        self.provides_directional_derivatives = dae.provides_directional_derivatives()
        self.n_event_indicators = dae.nzero()
        # The continuous states in the order of the FMU's state vector
        self.continuous_states = list(dae.x())
        derivative_of = {dae.der(s): s for s in self.continuous_states}
        #: The algebraic variables and inputs of the DAE (for a plain ODE FMU: no algebraic ones)
        self.algebraics, self.inputs = list(dae.z()), list(dae.u())
        #: State of each state derivative
        self.derivative_of = derivative_of
        self.outputs = list(dae.y())

        self.variables: dict[str, FMUVariable] = {}
        for n in dae.all():
            type_ = dae.type(n)
            var = FMUVariable(
                name=n,
                type=type_,
                causality=dae.causality(n),
                variability=dae.variability(n),
                aliases=list(dae.aliases(n)),
                derivative=derivative_of.get(n),
            )
            # FMI: a start value for initial="exact" or "approx", parameters, inputs, constants
            has_start = (
                dae.initial(n) in ("exact", "approx")
                or var.causality in ("parameter", "structuralParameter", "input")
                or var.variability == "constant"
            )
            if has_start and dae.numel(n) == 1:
                var.start = var.python_type(dae.start(n)[0])
            if type_ in ("Float64", "Float32"):
                var.min, var.max, var.nominal = dae.min(n), dae.max(n), dae.nominal(n)
            self.variables[n] = var

        # fmi-ls-dae residuals. Without them, the FMU is a plain ODE FMU (model_format="ode").
        self.dae_residuals = [r.name() for r in dae.outputs("alg")]
        self.initial_residuals = [r.name() for r in dae.outputs("ialg")]
        self.has_ls_dae_manifest = bool(self.dae_residuals)

    def builder(self) -> ca.DaeBuilder:
        """A new CasADi ``DaeBuilder`` of the FMU."""
        return ca.DaeBuilder(self.model_name, self.fmu_path)

    def evaluate(self, names: list[str]) -> dict:
        """
        The values of the variables in ``names`` (e.g. calculated parameters, which have no start
        value), read by CasADi's ``DaeBuilder`` in Initialization Mode.
        A variable whose value cannot be read (e.g. as the model equations fail at the start
        values), or that is an array, is left out.
        """
        dae = self.builder()
        try:
            values = dict(zip(names, dae.get(names), strict=True)) if names else {}
        except RuntimeError:
            # DaeBuilder.get fails as a whole if one variable cannot be read: one by one
            values = {}
            for n in names:
                try:
                    values[n] = dae.get(n)
                except RuntimeError:
                    pass
        return {
            n: self.variables[n].python_type(v)
            for n, v in values.items()
            if not isinstance(v, list | tuple)
        }

    def state_nominals(self) -> dict:
        """
        The nominals of the continuous states that the FMU reports after initialization
        (fmi3GetNominalsOfContinuousStates, through CasADi's ``DaeBuilder``). Unlike the
        ``nominal`` attribute in modelDescription.xml, which must be a literal, these include
        nominals that depend on parameters (e.g. ``V(nominal = A)``, which OpenModelica leaves
        out of modelDescription.xml).
        """
        if not self.continuous_states:
            return {}
        dae = self.builder()
        nominals = dae.get_nominal(self.continuous_states)
        return dict(zip(self.continuous_states, nominals, strict=True))
