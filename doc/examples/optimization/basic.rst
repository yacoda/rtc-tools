Filling a Reservoir
~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/graig-coch-2117306_640.jpg

.. :href: https://pixabay.com/en/graig-coch-dam-wales-reservoir-uk-2117306/
.. pixabay content is released under a CC0 Public Domain licence - no attribution needed


Overview
--------

The purpose of this example is to understand the technical setup of
an RTC-Tools model, how to run the model, and how to interpret the results.

The scenario is the following: A reservoir operator is trying to fill a
reservoir. They are given a six-day forecast of inflows given in 12-hour
increments. The operator wants to save as much of the inflows as possible, but
does not want to end up with too much water at the end of the six days. They
have chosen to use RTC-Tools to calculate how much water to release and when
to release it.

If you installed using source, the library and examples directory are
available in the git repositories. If you installed using pip directly, you
first need to download/copy the examples and libraries to a convenient
location. See :ref:`getting-started-download-examples` and
:ref:`getting-started-copy-libraries` for detailed instructions.

The folder ``<examples directory>\basic`` contains a complete RTC-Tools
optimization problem. An RTC-Tools directory has the following structure:

* ``input``: This folder contains the model input data. These are several files
  in comma separated value format, ``csv``.
* ``model``: This folder contains the Modelica model. The Modelica model
  contains the physics of the RTC-Tools model.
* ``output``: The folder where the output is saved in the file
  ``timeseries_export.csv``.
* ``src``: This folder contains a Python file. This file contains the
  configuration of the model and is used to run the model .

The Model
---------

The first step is to develop a physical model of the system. The model can be
viewed and edited using the OpenModelica Connection Editor (OMEdit) program.
For how to download and start up OMEdit, see :ref:`getting-started-omedit`.

#. Load the Deltares library into OMEdit

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<library directory>\Deltares\package.mo``

#. Load the example model into OMEdit

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<examples directory>\basic\model\Example.mo``

Once loaded, we have an OpenModelica Connection Editor window that looks like
this:

.. image:: ../../images/simple_storage_openmodelica.png


The model ``Example.mo`` represents a simple system with the following
elements:

* a reservoir, modeled as storage element
  ``Deltares.ChannelFlow.SimpleRouting.Storage.Storage``,
* an inflow boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow``,
* an outfall boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal``,
* connectors (black lines) connecting the elements.

You can use the mouse-over feature help to identify the predefined models from
the Deltares library. You can also drag the elements around- the connectors will
move with the elements. Adding new elements is easy- just drag them in from the
Deltares Library on the sidebar. Connecting the elements is just as easy- click
and drag between the ports on the elements.

In text mode, the Modelica model looks as follows (with
annotation statements removed):

.. literalinclude:: ../../_build/mo/basic.mo
  :language: modelica
  :lineno-match:

The three water system elements (storage, inflow, and outfall) appear under
the ``model Example`` statement. The ``equation`` part connects these three
elements with the help of connections. Note that ``storage`` extends the partial
model ``QSISO`` which contains the connectors ``QIn`` and ``QOut``.
With ``QSISO``, ``storage`` can be connected on two sides. The ``storage``
element also has a variable ``Q_release``, which is the decision variable the
operator controls.

OpenModelica Connection Editor will automatically generate the element and
connector entries in the text text file. Defining inputs and outputs requires
editing the text file directly. Relationships between the inputs and outputs and
the library elements must also be defined in the ``equation`` section.

In addition to elements, the input variables ``Q_in`` and ``Q_release`` are also
defined. ``Q_in`` is determined by the forecast and the operator cannot control
it, so we set ``Q_in(fixed = true)``. The actual values of ``Q_in`` are stored
in ``timeseries_import.csv``. In the ``equation`` section, equations are defined
to relate the inputs to the appropriate water system elements.

Because we want to view the water volume in the storage element in the output
file, we also define an output variable ``V_storage``.

There is a subtle difference between these quantities: ``V_storage`` at time t holds 
the volume of water in storage at that specific point in time. On the other hand, 
``Q_in`` and ``Q_release`` are aggregated measures: at time, t, their value 
represents the total flow between time t-1 and t.

The Optimization Problem
------------------------

The python script is created and edited in a text editor. In general, the python
script consists of the following blocks:

* Import of packages
* Definition of the optimization problem class

  * Constructor
  * Objective function
  * Definition of constraints
  * Any additional configuration

* A run statement

Importing Packages
''''''''''''''''''

Packages are imported using ``from ... import ...`` at the top of the file. In
our script, we import the classes we want the class to inherit, the
package ``run_optimization_problem`` form the ``rtctools.util`` package, and
any extra packages we want to use. For this example, the import block looks
like:

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :lines: 1-5
  :lineno-match:

Optimization Problem
''''''''''''''''''''

The next step is to define the optimization problem class. We construct the
class by declaring the class and inheriting the desired parent classes. The
parent classes each perform different tasks related to importing and exporting
data and solving the optimization problem. Each imported class makes a set of
methods available to the our optimization class.

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

Next, we define an objective function. This is a class method that returns
the value that needs to be minimized.

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :pyobject: Example.objective
  :lineno-match:

Constraints can be declared by declaring the ``path_constraints()`` method. Path
constraints are constraints that are applied every timestep. To set a constraint
at an individual timestep, we could define it inside the ``constraints()`` method.

Other parent classes also declare this method, so we call the ``super()`` method
so that we don't overwrite their behaviour.

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :pyobject: Example.path_constraints
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/basic/src/example.py
  :language: python
  :lineno-match:

Running RTC-Tools
-----------------

To run this basic example in RTC-Tools, navigate to the basic example ``src``
directory in the RTC-Tools shell and run the example using ``python
example.py``. For more details about using RTC-Tools, see
:ref:`running-rtc-tools`.

Extracting Results
------------------

The results from the run are found in ``output\timeseries_export.csv``. Any
CSV-reading software can import it, but this is what the results look like when
plotted with the python library ``matplotlib``:

.. plot:: examples/pyplots/basic_results.py

This plot shows that the operator is able to keep the water level within the
bounds over the entire time horizon and end with a full reservoir.

Feel free to experiment with this example. See what happens if you change the
max of ``Q_release`` (in the Modelica file) or if you make the objective
function negative (in the python script).

.. _optimization-basic-routes:

Choosing How RTC-Tools Reads the Model
--------------------------------------

So far, RTC-Tools has read the model the default way: :py:class:`ModelicaMixin
<rtctools.optimization.modelica_mixin.ModelicaMixin>` parses the Modelica files in the ``model``
folder with pymoca and turns them into an optimization problem. The same ``Example.mo`` can
also reach RTC-Tools along other routes; the objective, the constraints and the rest of the
script stay as they are.

The class attribute ``model_format`` says what the ``model`` folder provides:

* ``"modelica"`` (default): the Modelica files.
* ``"basemodelica"``: ``Example.bmo``, the model flattened to Base Modelica, the simplified
  form of Modelica that tools exchange, for example by OpenModelica.

pymoca reads either. Mixins of ``rtctools.model``, added to the bases of the class before
``ModelicaMixin``, change that:

* A **flattener** writes ``Example.bmo`` from the Modelica files at the start of a run, and
  sets ``model_format = "basemodelica"``: ``OpenModelicaFlattenerMixin`` with OpenModelica
  (``omc``), ``RumocaFlattenerMixin`` with Rumoca.
* A **reader** reads the model in place of pymoca, and needs the ``model_format`` that it
  reads. ``LacemodelicaReaderMixin`` reads ``Example.bmo`` with lacemodelica, through CasADi,
  so it needs ``model_format = "basemodelica"``, from a flattener or set in the class.
  ``RumocaReaderMixin`` reads the Modelica files with Rumoca, without a flattener.

With the compiler option ``cache`` (the default), a flattener records what it flattened from in
``Example.bmo_cache``, next to ``Example.bmo``, and skips the tool on the next run when nothing
changed: the Modelica files of the model and library folders, the flags, and the version of the
tool.

For example, to let OpenModelica flatten the model and pymoca read the result, the class of this
example becomes:

.. code-block:: python

    from rtctools.model.openmodelica_flattener_mixin import OpenModelicaFlattenerMixin


    class Example(
        OpenModelicaFlattenerMixin,
        CSVMixin,
        ModelicaMixin,
        CollocatedIntegratedOptimizationProblem,
    ):
        ...

To read an ``Example.bmo`` that is already there, set ``model_format = "basemodelica"`` in the
class instead. A flattener comes before the reader. RTC-Tools refuses a class with two
flatteners, two readers, a flattener after the reader, a flattener with
``model_format = "modelica"``, or a reader of the other ``model_format``, such as
``RumocaReaderMixin`` with a flattener. The same works with ``SimulationProblem``.

All these routes end in the same kind of model, so they solve the same optimization problem
and take about as long as the default.

FMUs
''''

With :py:class:`FMUMixin <rtctools.optimization.fmu_mixin.FMUMixin>` in place of
``ModelicaMixin``, the model is a Functional Mock-up Unit (FMU): compiled code that RTC-Tools
calls as a black box, where with ``ModelicaMixin`` the solver sees every equation and gets exact
derivatives.

``FMUMixin`` reads ``<model folder>/Example.fmu``, an FMI 3.0 Model Exchange FMU. Its
``model_format`` says how the FMU is used:

* ``"ode"`` (default): a plain FMU that computes ``der(x) = f(t, x, u)`` and solves the other
  equations inside. Only the states and the controls are optimization variables.
* ``"dae"``: an FMU with the fmi-ls-dae layered standard, which exposes the equations as
  residuals, so that every model variable is an optimization variable, as with
  ``ModelicaMixin``.

With ``model_exporter = "openmodelica"`` or ``"rumoca"`` (``"ode"`` only), RTC-Tools exports
the FMU from the Modelica files itself at the start of every run. The FMU version of this
example is ``src/example_fmu.py``:

.. literalinclude:: ../../../examples/basic/src/example_fmu.py
  :language: python
  :pyobject: Example
  :end-before: def objective
  :lineno-match:

Here the only change, apart from the import, is ``FMUMixin`` instead of ``ModelicaMixin`` and
``model_format = "dae"``; the FMU ``model/Example.fmu`` has to be exported beforehand.

Which Route to Choose
'''''''''''''''''''''

* **Start with the default** (``ModelicaMixin`` alone, with pymoca). It needs nothing but
  the Python packages, and the examples are written for it.
* **Base Modelica** helps when another tool, such as OpenModelica, should flatten the model:
  for a library that pymoca does not parse, or to check that two tools read a model the same
  way.
* **lacemodelica** reads the Base Modelica file through CasADi, independently of pymoca,
  which makes it a check on how pymoca reads it.
* **Rumoca** compiles Modelica sources fast and checks them strictly against the Modelica
  specification, which makes it a good second opinion on a model.
* **FMUs** let RTC-Tools optimize a model that only exists as an FMU, for example one exported
  from another modelling tool. They have costs: the solver sees the FMU through its
  Jacobians only, so it needs more iterations and each one takes longer, and ``delay()``
  and events are not supported. Prefer ``"dae"`` for models with algebraic equations, as the
  solver then also sees the algebraic variables.

What each route needs installed:

===================================  ==========================================================
Route                                Needs
===================================  ==========================================================
the default                          the RTC-Tools Python packages
``model_format = "basemodelica"``    a pymoca with the ``unquote_identifiers`` compiler option
``LacemodelicaReaderMixin``        a CasADi built with ONNX import and its ``lacemodelica``
                                     plugin
``RumocaReaderMixin``              the ``rumoca`` command (``RTCTOOLS_RUMOCA``)
``OpenModelicaFlattenerMixin``        the ``omc`` command (``RTCTOOLS_OMC``)
``RumocaFlattenerMixin``              the ``rumoca`` command (``RTCTOOLS_RUMOCA``)
``FMUMixin``, ``"dae"``              a CasADi with fmi-ls-dae import; for the export, an
                                     OpenModelica with fmi-ls-dae export (``--daeMode``)
``FMUMixin``, ``"ode"``              an FMI 3.0 FMU; RTC-Tools compiles Rumoca's source FMUs with
                                     the FMI 3 headers (``RTCTOOLS_FMI3_INCLUDE``)
===================================  ==========================================================

This example solves on every route with the same result. For a larger model, check that a
route gives the same answer as the default before relying on it.
