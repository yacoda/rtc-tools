"""model_format in a problem class: the model is loaded from what model_format says, and an
unknown model_format is refused."""

import tempfile
import unittest
from pathlib import Path


def _pymoca_reads_basemodelica():
    from pymoca.backends.casadi._options import _get_default_options

    return "unquote_identifiers" in _get_default_options()


class ProblemTest(unittest.TestCase):
    def folder(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "Problem.bmo").write_text(
            "package 'Problem'\n  model 'Problem'\n    Real 'x'(start = 1.0, fixed = true);\n"
            "  equation\n    der('x') = -'x';\n  end 'Problem';\nend 'Problem';\n"
        )
        (folder / "Problem.mo").write_text(
            "model Problem\n  Real y(start = 2.0, fixed = true);\nequation\n  der(y) = -y;\n"
            "end Problem;\n"
        )
        return str(folder)

    @staticmethod
    def optimization_class(*mixins, model_format=None):
        from rtctools.optimization.collocated_integrated_optimization_problem import (
            CollocatedIntegratedOptimizationProblem,
        )
        from rtctools.optimization.modelica_mixin import ModelicaMixin

        class Problem(*mixins, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
            def times(self, variable=None):
                return [0.0, 1.0]

        if model_format is not None:
            Problem.model_format = model_format
        return Problem

    @staticmethod
    def simulation_class(*mixins, model_format=None):
        from rtctools.simulation.simulation_problem import SimulationProblem

        class Problem(*mixins, SimulationProblem):
            pass

        if model_format is not None:
            Problem.model_format = model_format
        return Problem

    def optimization_problem(self, *mixins, model_format=None):
        return self.optimization_class(*mixins, model_format=model_format)(
            model_folder=self.folder()
        )

    def simulation_problem(self, *mixins, model_format=None):
        return self.simulation_class(*mixins, model_format=model_format)(model_folder=self.folder())

    def test_refused_when_instantiated(self):
        # A subclass could still set model_format, so the class is defined; its instance is
        # refused before loading
        for make in (self.optimization_class, self.simulation_class):
            for mixins, model_format, error, message in [
                ((), "bmo", ValueError, "model_format is 'bmo'"),
            ]:
                problem_class = make(*mixins, model_format=model_format)
                with self.assertRaisesRegex(error, message):
                    problem_class(model_folder=self.folder())

    def test_modelica_default(self):
        # model_format "modelica", the default or set, reads the Modelica sources with pymoca's
        # transfer_model, as before model_format
        import pymoca.backends.casadi.api

        for build in (self.optimization_problem, self.simulation_problem):
            plain = build()
            self.assertEqual(plain.model_format, "modelica")
            direct = pymoca.backends.casadi.api.transfer_model(
                self.folder(), "Problem", plain.compiler_options()
            )
            self.assertEqual(
                [str(e) for e in direct.equations], [str(e) for e in plain._pymoca_model.equations]
            )
            for model_format in (None, "modelica"):
                model = build(model_format=model_format)._pymoca_model
                self.assertEqual([v.symbol.name() for v in model.states], ["y"])
                self.assertEqual(
                    [str(e) for e in model.equations],
                    [str(e) for e in plain._pymoca_model.equations],
                )
                self.assertEqual(
                    [v.start for v in model.states], [v.start for v in plain._pymoca_model.states]
                )

    def test_basemodelica_with_pymoca(self):
        # A Base Modelica file that is already there, read without any mixin
        if not _pymoca_reads_basemodelica():
            self.skipTest("this pymoca does not read Base Modelica")
        for build in (self.optimization_problem, self.simulation_problem):
            problem = build(model_format="basemodelica")
            self.assertEqual([v.symbol.name() for v in problem._pymoca_model.states], ["x"])

    def test_pymoca_without_unquote_identifiers(self):
        if _pymoca_reads_basemodelica():
            self.skipTest("this pymoca reads Base Modelica")
        with self.assertRaisesRegex(RuntimeError, "cannot read Base Modelica"):
            self.optimization_problem(model_format="basemodelica")
