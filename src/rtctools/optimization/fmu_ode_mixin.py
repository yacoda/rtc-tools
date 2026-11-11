from .fmu_mixin import FMUMixin


class FMUOdeMixin(FMUMixin):
    """:py:class:`FMUMixin` with ``model_format = "ode"``, its default."""

    model_format = "ode"
