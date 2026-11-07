"""Run the examples with FMUOdeMixin in place of ModelicaMixin and FMUMixin: put this folder on
PYTHONPATH, with the FMUs of export_fmus.sh next to the models."""

import rtctools.optimization.fmu_mixin as _fmu_mixin
import rtctools.optimization.fmu_ode_mixin as _fmu_ode_mixin
import rtctools.optimization.modelica_mixin as _modelica_mixin

_modelica_mixin.ModelicaMixin = _fmu_ode_mixin.FMUOdeMixin
_fmu_mixin.FMUMixin = _fmu_ode_mixin.FMUOdeMixin
