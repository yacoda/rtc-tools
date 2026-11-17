from ._fmu_dae import _FMUDaeMixin


class FMUMixin(_FMUDaeMixin):
    """
    Adds a model given as an FMI 3.0 Model Exchange FMU to your optimization problem. It replaces
    :py:class:`ModelicaMixin`.

    The FMU is ``<fmu_folder>/<fmu_model_name>.fmu``. ``fmu_folder`` is a keyword argument that
    defaults to the model folder, and :py:attr:`fmu_model_name` defaults to the model name. The FMU
    is imported with CasADi's ``DaeBuilder``.

    The FMU must have the `fmi-ls-dae <https://modelica.github.io/fmi-ls-dae/main/>`_ layered
    standard, which exposes the DAE residuals and the initial equations. Every model variable is
    an optimization variable, as with :py:class:`ModelicaMixin`. See
    ``rtctools.optimization._fmu_dae._FMUDaeMixin`` for the details.

    :cvar fmu_model_name:
        Name of the FMU file (without ``.fmu``). Default is the model name.
    :cvar fmu_expose_locals:
        Expose the FMU's continuous local variables (except names matching ModelicaMixin's
        eliminable-variable pattern ``_name``). ``True``, ``False``, or a list of names. Default
        is ``True``.
    :cvar fmu_fixed_states:
        States whose FMU start value is an initial condition (FMI has no ``fixed`` attribute).
        Default is an empty list.
    :cvar fmu_free_initial_input_derivatives:
        FMU inputs ``der(u)`` of constant inputs ``u`` are free at t0, as pymoca's initial
        derivatives. Default is ``True``.
    :cvar fmu_exact_hessian:
        Give the solver second derivatives, as finite differences of the FMU's Jacobians. If
        ``False``, IPOPT uses a limited-memory approximation. As every model variable is an
        optimization variable and a Hessian costs two FMU Jacobians per variable and time step,
        the default ``None`` gives it only to small FMUs (at most 20 unknowns).
    """

    #: "dae": an FMU with fmi-ls-dae
    model_format = "dae"
