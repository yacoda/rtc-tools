import os
import re
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from rtctools.model import _fmu_export as fmu_export
from rtctools.optimization.fmu_mixin import FMUMixin, _route_by_format


class CheckFmuRouteTest(unittest.TestCase):
    def test_fmu_routes(self):
        for route in [("ode", None), ("ode", "openmodelica"), ("ode", "rumoca"), ("dae", None)]:
            fmu_export.check_fmu_route(*route)
        fmu_export.check_fmu_route("dae", "openmodelica")
        for route in [("dae", "rumoca"), ("me", None), ("ode", "dymola")]:
            with self.assertRaises(ValueError, msg=route):
                fmu_export.check_fmu_route(*route)


class ReplaceIfChangedTest(unittest.TestCase):
    def test_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, target = folder / "new", folder / "Model.fmu"
            source.write_text("model M end M;")
            self.assertTrue(fmu_export.replace_if_changed(source, target))
            self.assertEqual(target.read_text(), "model M end M;")

            # Same content: the file is left alone (same inode, same mtime)
            before = target.stat()
            self.assertFalse(fmu_export.replace_if_changed(source, target))
            after = target.stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

            source.write_text("model M Real x; end M;")
            self.assertTrue(fmu_export.replace_if_changed(source, target))
            self.assertEqual(target.read_text(), "model M Real x; end M;")
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["Model.fmu", "new"])


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

    def test_openmodelica_fmu(self):
        out = self.folder / "fmus"
        out.mkdir()
        target = fmu_export.export_fmu(
            "openmodelica", "ode", self.models, "Model", [], out, target_name="Renamed"
        )
        self.assertEqual(target, out / "Renamed.fmu")
        self.assertEqual(target.read_bytes(), b"PK fmu")
        self.assertIn('buildModelFMU(Model, version="3.0", fmuType="me"', self.log.read_text())

    def test_relative_folders(self):
        # The exporter runs in a temporary folder, so it must get the folders resolved
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.folder)
        target = fmu_export.export_fmu(
            "openmodelica", "ode", Path("model"), "Model", [Path("lib")], Path("model")
        )
        self.assertEqual(target.read_bytes(), b"PK fmu")
        loaded = re.findall(r'loadFile\("([^"]+)"\)', self.log.read_text())
        self.assertEqual(len(loaded), 3)
        for f in loaded:
            self.assertTrue(Path(f).is_absolute() and Path(f).is_file(), f)

    def test_openmodelica_failure(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} -c pass"}):
            with self.assertRaises(RuntimeError):
                fmu_export.export_fmu("openmodelica", "ode", self.models, "Model", [], self.models)
        self.assertFalse((self.models / "Model.fmu").exists())

    def test_missing_exporter(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": "no-such-omc-command"}):
            with self.assertRaises(FileNotFoundError):
                fmu_export.export_fmu("openmodelica", "dae", self.models, "Model", [], self.models)

    def test_rumoca(self):
        # Stands in for rumoca: writes into the --output folder, as Rumoca does; its FMU holds
        # C sources only, as Rumoca's do
        rumoca = self.folder / "rumoca.py"
        rumoca.write_text(
            textwrap.dedent(
                """
                import sys, pathlib, zipfile
                a = sys.argv
                out = pathlib.Path(a[a.index("--output") + 1])
                out.mkdir()
                model = a[a.index("--model") + 1]
                assert a[a.index("--target") + 1] == "fmi3"
                build = (
                    '<fmiBuildDescription fmiVersion="3.0">'
                    '<BuildConfiguration modelIdentifier="M_id">'
                    '<SourceFileSet language="C99"><SourceFile name="model.c"/>'
                    '<PreprocessorDefinition name="ANSWER" value="42"/>'
                    "</SourceFileSet></BuildConfiguration></fmiBuildDescription>"
                )
                with zipfile.ZipFile(out / (model + ".fmu"), "w") as z:
                    z.writestr("modelDescription.xml", "<fmiModelDescription/>")
                    z.writestr("sources/buildDescription.xml", build)
                    z.writestr(
                        "sources/model.c",
                        '#include "fmi3Functions.h"\\nint answer(void) { return ANSWER; }\\n',
                    )
                """
            )
        )
        headers = self.folder / "fmi3"
        headers.mkdir()
        (headers / "fmi3Functions.h").write_text("/* FMI 3 headers */\n")
        environment = {
            "RTCTOOLS_RUMOCA": f"{sys.executable} {rumoca}",
            "RTCTOOLS_FMI3_INCLUDE": str(headers),
        }
        with mock.patch.dict(os.environ, environment):
            fmu = fmu_export.export_fmu("rumoca", "ode", self.models, "Model", [], self.models)
        self.assertEqual(fmu.name, "Model.fmu")
        with zipfile.ZipFile(fmu) as z:
            self.assertIn("binaries/x86_64-linux/M_id.so", z.namelist())
            self.assertIn("sources/model.c", z.namelist())

    def test_rumoca_fmu_without_headers(self):
        rumoca = self.folder / "rumoca.py"
        rumoca.write_text(
            "import sys, pathlib, zipfile\n"
            "a = sys.argv\n"
            "out = pathlib.Path(a[a.index('--output') + 1])\n"
            "out.mkdir()\n"
            "zipfile.ZipFile(out / 'Model.fmu', 'w').writestr('sources/model.c', '')\n"
        )
        environment = {"RTCTOOLS_RUMOCA": f"{sys.executable} {rumoca}", "RTCTOOLS_FMI3_INCLUDE": ""}
        with mock.patch.dict(os.environ, environment):
            with self.assertRaisesRegex(FileNotFoundError, "RTCTOOLS_FMI3_INCLUDE"):
                fmu_export.export_fmu("rumoca", "ode", self.models, "Model", [], self.models)

    def test_main_file(self):
        self.assertEqual(fmu_export._main_file(self.models, "Model"), str(self.models / "Model.mo"))
        (self.models / "Named.mo").write_text("within;\nmodel Inside\nend Inside;\n")
        self.assertEqual(
            fmu_export._main_file(self.models, "Inside"), str(self.models / "Named.mo")
        )
        with self.assertRaises(FileNotFoundError):
            fmu_export._main_file(self.models, "Missing")

    def test_has_initial_equations(self):
        self.assertFalse(fmu_export.has_initial_equations(self.models))
        (self.models / "Init.mo").write_text("model Init\n  Real x;\ninitial equation\n  x = 0;\n")
        self.assertTrue(fmu_export.has_initial_equations(self.models))


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
        with self.assertRaises(ValueError):
            Problem(model_folder=".", model_format="dae", model_exporter="rumoca")
        self.assertEqual(Problem.__new__(Problem).model_format, "ode")
        self.assertEqual(Dae.__new__(Dae).model_format, "dae")
        self.assertEqual(Dae.__new__(Dae, model_format="ode").model_format, "ode")
