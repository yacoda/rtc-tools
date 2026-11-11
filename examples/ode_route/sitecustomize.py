"""Run the examples on the ODE route: put this folder on PYTHONPATH, with the FMUs of
export_fmus.sh next to the models. ModelicaMixin and FMUMixin become FMUMixin with
model_format="ode", also where a driver chooses "dae"."""

import rtctools.optimization.fmu_mixin as _fmu_mixin
import rtctools.optimization.modelica_mixin as _modelica_mixin


class _OdeRoute(_fmu_mixin.FMUMixin):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **{**kwargs, "model_format": "ode"})


_modelica_mixin.ModelicaMixin = _OdeRoute
_fmu_mixin.FMUMixin = _OdeRoute
