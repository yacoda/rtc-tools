import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from rtctools._internal import model_routes
from rtctools.optimization.fmu_mixin import FMUMixin, _route_by_format


class CheckRouteTest(unittest.TestCase):
    def test_modelica_routes(self):
        valid = [
            ("modelica", "pymoca", None),
            ("modelica", "rumoca", None),
            ("basemodelica", "pymoca", None),
            ("basemodelica", "pymoca", "openmodelica"),
            ("basemodelica", "pymoca", "rumoca"),
            ("basemodelica", "lacemodelica", None),
            ("basemodelica", "lacemodelica", "openmodelica"),
        ]
        for route in valid:
            model_routes.check_modelica_route(*route)
        invalid = [
            ("modelica", "pymoca", "openmodelica"),
            ("modelica", "lacemodelica", None),
            ("basemodelica", "rumoca", None),
            ("mo", "pymoca", None),
            ("modelica", "omc", None),
            ("basemodelica", "pymoca", "dymola"),
        ]
        for route in invalid:
            with self.assertRaises(ValueError, msg=route):
                model_routes.check_modelica_route(*route)

    def test_fmu_routes(self):
        for route in [("ode", None), ("ode", "openmodelica"), ("ode", "rumoca"), ("dae", None)]:
            model_routes.check_fmu_route(*route)
        model_routes.check_fmu_route("dae", "openmodelica")
        for route in [("dae", "rumoca"), ("me", None), ("ode", "dymola")]:
            with self.assertRaises(ValueError, msg=route):
                model_routes.check_fmu_route(*route)


class ReplaceIfChangedTest(unittest.TestCase):
    def test_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, target = folder / "new", folder / "Model.bmo"
            source.write_text("model M end M;")
            self.assertTrue(model_routes.replace_if_changed(source, target))
            self.assertEqual(target.read_text(), "model M end M;")

            # Same content: the file is left alone (same inode, same mtime)
            before = target.stat()
            self.assertFalse(model_routes.replace_if_changed(source, target))
            after = target.stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

            source.write_text("model M Real x; end M;")
            self.assertTrue(model_routes.replace_if_changed(source, target))
            self.assertEqual(target.read_text(), "model M Real x; end M;")
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["Model.bmo", "new"])


# Stands in for omc: writes what the export script asks for, and records its arguments
FAKE_OMC = textwrap.dedent(
    """
    import re, sys
    script = open(sys.argv[-1]).read()
    open("args.txt", "w").write("\\n".join(sys.argv[1:]))
    open(LOG, "w").write(script)
    m = re.search(r'writeFile\\("([^"]+)"', script)
    if m:
        open(m.group(1), "w").write("model M\\nend M;\\n")
    m = re.search(r'fileNamePrefix="([^"]+)"', script)
    if m:
        open(m.group(1) + ".fmu", "wb").write(b"PK fmu")
    """
)


class ExportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.models = self.folder / "model"
        self.models.mkdir()
        (self.models / "Model.mo").write_text(
            "model Model\n  Real x;\nequation\n  x = 1;\nend Model;\n"
        )
        (self.models / "Other.mo").write_text("package Other\nend Other;\n")
        self.library = self.folder / "lib"
        (self.library / "Lib").mkdir(parents=True)
        (self.library / "Lib" / "package.mo").write_text("package Lib\nend Lib;\n")
        self.log = self.folder / "script.mos"
        omc = self.folder / "omc.py"
        omc.write_text(f"LOG = {str(self.log)!r}\n" + FAKE_OMC)
        environment = mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} {omc}"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_openmodelica_basemodelica(self):
        target = model_routes.export_model(
            "openmodelica", "basemodelica", self.models, "Model", [self.library], self.models
        )
        self.assertEqual(target, self.models / "Model.bmo")
        self.assertEqual(target.read_text(), "model M\nend M;\n")
        script = self.log.read_text()
        self.assertIn('loadModel(Modelica, {"4.0.0"})', script)
        self.assertIn(str(self.library / "Lib" / "package.mo"), script)
        self.assertIn(str(self.models / "Model.mo"), script)
        self.assertIn(str(self.models / "Other.mo"), script)
        self.assertIn("instantiateModel(Model)", script)
        # Exporting again with the same result leaves the file alone
        mtime = target.stat().st_mtime_ns
        model_routes.export_model(
            "openmodelica", "basemodelica", self.models, "Model", [self.library], self.models
        )
        self.assertEqual(target.stat().st_mtime_ns, mtime)

    def test_openmodelica_fmu(self):
        out = self.folder / "fmus"
        out.mkdir()
        target = model_routes.export_model(
            "openmodelica", "ode", self.models, "Model", [], out, target_name="Renamed"
        )
        self.assertEqual(target, out / "Renamed.fmu")
        self.assertEqual(target.read_bytes(), b"PK fmu")
        self.assertIn('buildModelFMU(Model, version="3.0", fmuType="me"', self.log.read_text())

    def test_openmodelica_failure(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} -c pass"}):
            with self.assertRaises(RuntimeError):
                model_routes.export_model(
                    "openmodelica", "basemodelica", self.models, "Model", [], self.models
                )
        self.assertFalse((self.models / "Model.bmo").exists())

    def test_missing_exporter(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": "no-such-omc-command"}):
            with self.assertRaises(FileNotFoundError):
                model_routes.export_model(
                    "openmodelica", "basemodelica", self.models, "Model", [], self.models
                )

    def test_main_file(self):
        self.assertEqual(
            model_routes._main_file(self.models, "Model"), str(self.models / "Model.mo")
        )
        (self.models / "Named.mo").write_text("within;\nmodel Inside\nend Inside;\n")
        self.assertEqual(
            model_routes._main_file(self.models, "Inside"), str(self.models / "Named.mo")
        )
        with self.assertRaises(FileNotFoundError):
            model_routes._main_file(self.models, "Missing")

    def test_has_initial_equations(self):
        self.assertFalse(model_routes.has_initial_equations(self.models))
        (self.models / "Init.mo").write_text("model Init\n  Real x;\ninitial equation\n  x = 0;\n")
        self.assertTrue(model_routes.has_initial_equations(self.models))


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
        with self.assertRaises(ValueError):
            Problem(model_folder=".", model_format="dae", model_exporter="rumoca")
        self.assertEqual(Problem.__new__(Problem).model_format, "ode")
        self.assertEqual(Dae.__new__(Dae).model_format, "dae")
        self.assertEqual(Dae.__new__(Dae, model_format="ode").model_format, "ode")


class ModelicaMixinRouteTest(unittest.TestCase):
    def problem(self, **kwargs):
        from rtctools.optimization.collocated_integrated_optimization_problem import (
            CollocatedIntegratedOptimizationProblem,
        )
        from rtctools.optimization.modelica_mixin import ModelicaMixin

        class Problem(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
            def times(self, variable=None):
                return [0.0, 1.0]

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = temporary.name
        (Path(folder) / "Problem.bmo").write_text("model Problem\nend Problem;\n")
        return Problem(model_folder=folder, **kwargs)

    def test_invalid_route(self):
        with self.assertRaises(ValueError):
            self.problem(model_compiler="lacemodelica")

    def test_pymoca_without_basemodelica(self):
        from pymoca.backends.casadi._options import _get_default_options

        if "basemodelica" in _get_default_options():
            self.skipTest("this pymoca reads Base Modelica")
        with self.assertRaisesRegex(RuntimeError, "cannot read Base Modelica"):
            self.problem(model_format="basemodelica")
