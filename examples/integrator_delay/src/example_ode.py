"""
The optimization problem of example.py on the ODE route.

The delay is hidden history in an FMU, which cannot be evaluated at arbitrary points. The FMU is
exported with delay() replaced by an input, Q_delayed, which delayed_feedback() ties to Q_control.
"""

import casadi as ca

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.fmu_ode_mixin import FMUOdeMixin
from rtctools.util import run_optimization_problem


class ExampleOpt(CSVMixin, FMUOdeMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, **kwargs):
        super().__init__(model_name="Example", **kwargs)

    def delayed_feedback(self):
        return [(self.variable("Q_control"), "Q_delayed", ca.MX(24 * 3600.0))]

    def path_objective(self, ensemble_member):
        # The goal is that the volume stays as close to v_ideal as possible.
        del ensemble_member
        v_ideal = 400000
        return (self.state("integrator.V") - v_ideal) ** 2


run_optimization_problem(ExampleOpt)
