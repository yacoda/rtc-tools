Modelica and FMU models
=======================

To learn the basics of modelling with Modelica, please refer to the online book `Modelica by Example <https://mbe.modelica.university/>`_. 

.. autoclass:: rtctools.optimization.modelica_mixin.ModelicaMixin
    :members: pymoca_model, compiler_options
    :show-inheritance:

To optimize a model given as a Functional Mock-up Unit (FMU), use ``FMUMixin`` in place of
``ModelicaMixin``. See :ref:`the basic example <optimization-basic-routes>` for how to choose.

.. autoclass:: rtctools.optimization.fmu_mixin.FMUMixin
    :show-inheritance:
