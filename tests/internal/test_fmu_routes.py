import unittest

from rtctools.optimization.fmu_mixin import FMUMixin, _route_by_format


class RouteByFormatTest(unittest.TestCase):
    def test_routing(self):
        class Base:
            def name(self):
                return ["base"]

            @property
            def kind(self):
                return "base"

        class Ode(Base):
            def name(self):
                return ["ode"] + super().name()

            @property
            def kind(self):
                return "ode"

            def only_ode(self):
                return "ode"

        class Dae(Base):
            def name(self):
                return ["dae"] + super().name()

            @property
            def kind(self):
                return "dae"

        _route_by_format(Ode, "ode")
        _route_by_format(Dae, "dae")

        class Both(Ode, Dae):
            pass

        ode, dae = Both(), Both()
        ode.model_format, dae.model_format = "ode", "dae"
        # A routed method of the active format still reaches the shared base through super()
        self.assertEqual(ode.name(), ["ode", "base"])
        self.assertEqual(dae.name(), ["dae", "base"])
        self.assertEqual((ode.kind, dae.kind), ("ode", "dae"))
        self.assertEqual(ode.only_ode(), "ode")
        with self.assertRaises(AttributeError):
            dae.only_ode()

    def test_fmu_mixin_options(self):
        from rtctools.optimization.collocated_integrated_optimization_problem import (
            CollocatedIntegratedOptimizationProblem,
        )

        class Problem(FMUMixin, CollocatedIntegratedOptimizationProblem):
            def times(self, variable=None):
                return [0.0, 1.0]

        class Dae(Problem):
            model_format = "dae"

        # The options are checked before anything is loaded
        with self.assertRaises(ValueError):
            Problem(model_folder=".", model_format="me")
        self.assertEqual(Problem.__new__(Problem).model_format, "ode")
        self.assertEqual(Dae.__new__(Dae).model_format, "dae")
        self.assertEqual(Dae.__new__(Dae, model_format="ode").model_format, "ode")
