"""model_format and the route mixins in a problem class: what the guard refuses, when the class
is defined or when it is instantiated, and that the model is loaded from what model_format says."""

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from rtctools._internal.modelica_mixin_base import _check_model_route, _ModelicaMixinBase
from rtctools.model.openmodelica_flattener_mixin import OpenModelicaFlattenerMixin


class Flattener:
    _modelica_route = "flattener"
    model_format = "basemodelica"


class OtherFlattener:
    _modelica_route = "flattener"
    model_format = "basemodelica"


class ModelicaReader:
    _modelica_route = "reader"
    _modelica_reads = "modelica"


class BaseModelicaReader:
    _modelica_route = "reader"
    _modelica_reads = "basemodelica"


class PymocaBaseModelicaReader:
    """A reader of Base Modelica that loads the model: pymoca's own reading of the .bmo"""

    _modelica_route = "reader"
    _modelica_reads = "basemodelica"

    def pymoca_model(self, model_folder, model_name, compiler_options):
        from rtctools.model._pymoca_basemodelica import _transfer_model

        return _transfer_model(model_folder, model_name, compiler_options)


def compose(*bases, **attributes):
    return type("Example", (*bases, _ModelicaMixinBase), attributes)


BASEMODELICA = {"model_format": "basemodelica"}


class GuardTest(unittest.TestCase):
    def test_allowed(self):
        for bases, attributes in [
            ((), {}),
            ((), BASEMODELICA),
            ((ModelicaReader,), {}),
            ((BaseModelicaReader,), BASEMODELICA),
            ((Flattener,), {}),
            ((Flattener, BaseModelicaReader), {}),
            ((OpenModelicaFlattenerMixin,), {}),
            ((OpenModelicaFlattenerMixin, BaseModelicaReader), {}),
        ]:
            _check_model_route(compose(*bases, **attributes))

    def test_flattener_sets_model_format(self):
        self.assertEqual(compose(OpenModelicaFlattenerMixin).model_format, "basemodelica")
        self.assertEqual(compose().model_format, "modelica")

    def test_subclass_of_a_route(self):
        # The markers count where they are declared, not where they are inherited
        route = compose(Flattener, BaseModelicaReader)
        _check_model_route(type("Sub", (route,), {}))

    # Two readers, two flatteners or a flattener after the reader: no subclass can repair
    # that, so the class is refused when it is defined

    def test_two_readers(self):
        with self.assertRaisesRegex(TypeError, "more than one model reader"):
            compose(BaseModelicaReader, ModelicaReader, **BASEMODELICA)

    def test_two_flatteners(self):
        with self.assertRaisesRegex(TypeError, "more than one model flattener"):
            compose(Flattener, OpenModelicaFlattenerMixin)

    def test_flattener_after_reader(self):
        with self.assertRaisesRegex(TypeError, "after the reader .* put the flattener first"):
            compose(BaseModelicaReader, OpenModelicaFlattenerMixin)

    def test_refused_in_a_subclass(self):
        # A subclass that adds a second route mixin to a valid class is refused when defined
        route = compose(Flattener, BaseModelicaReader)
        with self.assertRaisesRegex(TypeError, "more than one model flattener"):
            type("Sub", (OtherFlattener, route), {})
        with self.assertRaisesRegex(TypeError, "more than one model reader"):
            type("Sub", (ModelicaReader, route), {})
        reader = compose(BaseModelicaReader, **BASEMODELICA)
        with self.assertRaisesRegex(TypeError, "put the flattener first"):
            type("Sub", (reader, Flattener), {})

    def test_layered_route(self):
        # A reader in a base class, and the model_format it reads in a subclass: by the
        # flattener put in front, or set. model_format is only checked on instantiation.
        base = compose(BaseModelicaReader)
        with self.assertRaisesRegex(TypeError, "reads Base Modelica"):
            _check_model_route(base)
        _check_model_route(type("Sub", (Flattener, base), {}))
        _check_model_route(type("Sub", (OpenModelicaFlattenerMixin, base), {}))
        _check_model_route(type("Sub", (base,), BASEMODELICA))

    def test_reader_of_modelica_after_flattener(self):
        with self.assertRaisesRegex(
            TypeError,
            "ModelicaReader, which reads the Modelica sources .*"
            r"model_format is 'basemodelica' \(set by OpenModelicaFlattenerMixin\)$",
        ):
            _check_model_route(compose(OpenModelicaFlattenerMixin, ModelicaReader))

    def test_reader_of_basemodelica_with_modelica(self):
        with self.assertRaisesRegex(
            TypeError,
            "BaseModelicaReader, which reads Base Modelica .*"
            r"model_format is 'modelica' \(set by _ModelicaMixinBase\) "
            r'\(set model_format = "basemodelica"\)',
        ):
            _check_model_route(compose(BaseModelicaReader))
        with self.assertRaisesRegex(TypeError, "reads the Modelica sources"):
            _check_model_route(compose(ModelicaReader, **BASEMODELICA))

    def test_flattener_with_model_format_modelica(self):
        # model_format set back to "modelica" below the flattener: its file would not be read
        with self.assertRaisesRegex(
            TypeError,
            "flattener OpenModelicaFlattenerMixin, but model_format is 'modelica' "
            r"\(set by Example\)",
        ):
            _check_model_route(compose(OpenModelicaFlattenerMixin, model_format="modelica"))

    def test_unknown_model_format(self):
        for bases in [(), (OpenModelicaFlattenerMixin,), (BaseModelicaReader,)]:
            with self.assertRaisesRegex(
                ValueError, "model_format is 'bmo', but it is 'modelica' or 'basemodelica'"
            ):
                _check_model_route(compose(*bases, model_format="bmo"))


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

    def omc_stand_in(self):
        """RTCTOOLS_OMC set to a stand-in for omc, which writes a Base Modelica model of z"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        omc = Path(temporary.name) / "omc.py"
        omc.write_text(
            textwrap.dedent(
                """
                import re, sys
                script = open(sys.argv[-1]).read()
                path = re.search(r'writeFile\\("([^"]+)"', script).group(1)
                open(path, "w").write(
                    "package 'Problem'\\n  model 'Problem'\\n"
                    "    Real 'z'(start = 3.0, fixed = true);\\n"
                    "  equation\\n    der('z') = -'z';\\n  end 'Problem';\\nend 'Problem';\\n"
                )
                """
            )
        )
        return mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} {omc}"})

    def test_refused_when_defined(self):
        # No subclass can repair these, so the class statement fails, before any loading
        for make in (self.optimization_class, self.simulation_class):
            with self.assertRaisesRegex(TypeError, "put the flattener first"):
                make(BaseModelicaReader, OpenModelicaFlattenerMixin)
            with self.assertRaisesRegex(TypeError, "more than one model flattener"):
                make(Flattener, OpenModelicaFlattenerMixin)
            with self.assertRaisesRegex(TypeError, "more than one model reader"):
                make(BaseModelicaReader, ModelicaReader)

    def test_refused_when_instantiated(self):
        # A subclass could still set model_format, so the class is defined; the guard refuses
        # its instance before loading, so before the flattener would run omc
        for make in (self.optimization_class, self.simulation_class):
            for mixins, model_format, error, message in [
                ((OpenModelicaFlattenerMixin,), "modelica", TypeError, "would not be read"),
                ((BaseModelicaReader,), None, TypeError, "reads Base Modelica"),
                ((), "bmo", ValueError, "model_format is 'bmo'"),
            ]:
                problem_class = make(*mixins, model_format=model_format)
                with self.assertRaisesRegex(error, message):
                    problem_class(model_folder=self.folder())

    def test_layered_route(self):
        # The reader in a base class, which alone is refused when instantiated, and the
        # flattener in a subclass: the subclass loads the model the flattener writes
        if not _pymoca_reads_basemodelica():
            self.skipTest("this pymoca does not read Base Modelica")
        with self.omc_stand_in():
            for make in (self.optimization_class, self.simulation_class):
                base = make(PymocaBaseModelicaReader)
                with self.assertRaisesRegex(TypeError, "reads Base Modelica"):
                    base(model_folder=self.folder())
                layered = type("Problem", (OpenModelicaFlattenerMixin, base), {})
                problem = layered(model_folder=self.folder())
                self.assertEqual([v.symbol.name() for v in problem._mo_pymoca_model.states], ["z"])

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
                [str(e) for e in direct.equations],
                [str(e) for e in plain._mo_pymoca_model.equations],
            )
            for model_format in (None, "modelica"):
                model = build(model_format=model_format)._mo_pymoca_model
                self.assertEqual([v.symbol.name() for v in model.states], ["y"])
                self.assertEqual(
                    [str(e) for e in model.equations],
                    [str(e) for e in plain._mo_pymoca_model.equations],
                )
                self.assertEqual(
                    [v.start for v in model.states],
                    [v.start for v in plain._mo_pymoca_model.states],
                )

    def test_basemodelica_with_pymoca(self):
        # A Base Modelica file that is already there, read without any mixin
        if not _pymoca_reads_basemodelica():
            self.skipTest("this pymoca does not read Base Modelica")
        for build in (self.optimization_problem, self.simulation_problem):
            problem = build(model_format="basemodelica")
            self.assertEqual([v.symbol.name() for v in problem._mo_pymoca_model.states], ["x"])

    def test_pymoca_without_unquote_identifiers(self):
        if _pymoca_reads_basemodelica():
            self.skipTest("this pymoca reads Base Modelica")
        with self.assertRaisesRegex(RuntimeError, "cannot read Base Modelica"):
            self.optimization_problem(model_format="basemodelica")

    def test_openmodelica_flattener_alone(self):
        # OpenModelicaFlattenerMixin alone: it writes Problem.bmo (here a stand-in for omc), and
        # pymoca reads it, as model_format is "basemodelica"
        if not _pymoca_reads_basemodelica():
            self.skipTest("this pymoca does not read Base Modelica")
        with self.omc_stand_in():
            for build in (self.optimization_problem, self.simulation_problem):
                problem = build(OpenModelicaFlattenerMixin)
                self.assertEqual([v.symbol.name() for v in problem._mo_pymoca_model.states], ["z"])
