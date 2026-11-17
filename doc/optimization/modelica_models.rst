Modelica models
===============

To learn the basics of modelling with Modelica, please refer to the online book `Modelica by Example <https://mbe.modelica.university/>`_. 

.. autoclass:: rtctools.optimization.modelica_mixin.ModelicaMixin
    :members: pymoca_model, compiler_options
    :show-inheritance:

Reading the model another way
-----------------------------

By default, ``ModelicaMixin`` and ``SimulationProblem`` compile the Modelica sources with
pymoca. With ``model_format = "basemodelica"``, pymoca reads ``<model_name>.bmo``, the model
flattened to Base Modelica, instead. The mixins of ``rtctools.model`` come before
``ModelicaMixin`` or ``SimulationProblem`` in the bases: a flattener writes that file at every
run and sets ``model_format = "basemodelica"``; a reader, after the flattener, loads the model
in place of pymoca, and needs the ``model_format`` that it reads. See :ref:`the basic example
<optimization-basic-routes>`.

.. autoclass:: rtctools.model.lacemodelica_reader_mixin.LacemodelicaReaderMixin

.. autoclass:: rtctools.model.rumoca_reader_mixin.RumocaReaderMixin

.. autoclass:: rtctools.model.openmodelica_flattener_mixin.OpenModelicaFlattenerMixin
    :members: openmodelica_flags
