"""The flattener mixins, with stand-ins for the tools: what they run, and what they write."""

import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from rtctools.model import _flatten
from rtctools.model.openmodelica_flattener_mixin import OpenModelicaFlattenerMixin

# Stands in for omc: prints the version in VERSION, writes what the .mos script asks for, and
# records its arguments and how often it flattened
FAKE_OMC = textwrap.dedent(
    """
    import os, re, sys
    if sys.argv[1:] == ["--version"]:
        print(open(VERSION).read())
        sys.exit()
    count = int(open(COUNT).read()) if os.path.exists(COUNT) else 0
    open(COUNT, "w").write(str(count + 1))
    script = open(sys.argv[-1]).read()
    open(ARGS, "w").write("\\n".join(sys.argv[1:]))
    open(LOG, "w").write(script)
    m = re.search(r'writeFile\\("([^"]+)"', script)
    if m:
        open(m.group(1), "w").write("model M\\nend M;\\n")
    """
)


class Reader:
    """What comes after the flattener: returns what it is given"""

    def pymoca_model(self, model_folder, model_name, compiler_options):
        return "read", model_folder, model_name


class FlattenTest(unittest.TestCase):
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

    def flatten(self, flattener, models, library, cache=False, model_name="Model", **attributes):
        problem = type("Problem", (flattener, Reader), attributes)()
        options = {"library_folders": [library], "cache": cache}
        result = problem.pymoca_model(models, model_name, options)
        self.assertEqual(result, ("read", models, model_name))
        return Path(models) / f"{model_name}.bmo"


class OpenModelicaFlattenerTest(FlattenTest):
    def setUp(self):
        super().setUp()
        self.log = self.folder / "script.mos"
        self.args = self.folder / "args.txt"
        self.count = self.folder / "count.txt"
        self.version = self.folder / "version.txt"
        self.version.write_text("OpenModelica 1.0.0")
        omc = self.folder / "omc.py"
        omc.write_text(
            f"LOG = {str(self.log)!r}\nARGS = {str(self.args)!r}\n"
            f"COUNT = {str(self.count)!r}\nVERSION = {str(self.version)!r}\n" + FAKE_OMC
        )
        # The version of omc is asked once per process and command
        self.addCleanup(_flatten._tool_versions.clear)
        environment = mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} {omc}"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_flatten(self):
        target = self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)
        self.assertEqual(target.read_text(), "model M\nend M;\n")
        script = self.log.read_text()
        self.assertIn('loadModel(Modelica, {"4.0.0"})', script)
        self.assertIn(str(self.library / "Lib" / "package.mo"), script)
        self.assertIn(str(self.models / "Model.mo"), script)
        self.assertIn(str(self.models / "Other.mo"), script)
        self.assertIn("instantiateModel(Model)", script)
        self.assertIn("--BaseModelica", self.args.read_text().splitlines())
        # Flattening again with the same result leaves the file alone
        mtime = target.stat().st_mtime_ns
        self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)
        self.assertEqual(target.stat().st_mtime_ns, mtime)

    def test_flags(self):
        self.flatten(
            OpenModelicaFlattenerMixin, self.models, self.library, openmodelica_flags=["-d=x"]
        )
        self.assertEqual(self.args.read_text().splitlines()[0], "-d=x")

    def test_relative_folders(self):
        # The flattener runs in a temporary folder, so it must get the folders resolved
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.folder)
        target = self.flatten(OpenModelicaFlattenerMixin, "model", "lib")
        self.assertEqual(target.read_text(), "model M\nend M;\n")
        loaded = re.findall(r'loadFile\("([^"]+)"\)', self.log.read_text())
        self.assertEqual(len(loaded), 3)
        for f in loaded:
            self.assertTrue(Path(f).is_absolute() and Path(f).is_file(), f)

    def test_failure(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": f"{sys.executable} -c pass"}):
            with self.assertRaisesRegex(RuntimeError, "OpenModelica did not flatten Model"):
                self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)
        self.assertFalse((self.models / "Model.bmo").exists())

    def test_missing_command(self):
        with mock.patch.dict(os.environ, {"RTCTOOLS_OMC": "no-such-omc-command"}):
            with self.assertRaisesRegex(FileNotFoundError, "RTCTOOLS_OMC"):
                self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)

    def runs(self):
        """How often the fake omc flattened"""
        return int(self.count.read_text()) if self.count.exists() else 0

    def cached(self, **attributes):
        """Flatten with the cache, and return whether omc ran"""
        before = self.runs()
        self.flatten(OpenModelicaFlattenerMixin, self.models, self.library, True, **attributes)
        return self.runs() > before

    def test_cache_reuses_flattening(self):
        self.assertTrue(self.cached())
        sidecar = self.models / "Model.bmo_cache"
        self.assertEqual(json.loads(sidecar.read_text())["inputs"]["model_name"], "Model")
        with self.assertLogs("rtctools", "INFO") as logs:
            self.assertFalse(self.cached())
        self.assertIn("Reusing", "\n".join(logs.output))
        self.assertEqual((self.models / "Model.bmo").read_text(), "model M\nend M;\n")

    def test_cache_disabled(self):
        self.assertTrue(self.cached())
        before = self.runs()
        self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)
        self.flatten(OpenModelicaFlattenerMixin, self.models, self.library)
        self.assertEqual(self.runs(), before + 2)

    def test_cache_invalidated(self):
        def edit_model():
            (self.models / "Other.mo").write_text("package Other\n  // edited\nend Other;\n")

        def edit_library():
            (self.library / "Lib" / "package.mo").write_text("package Lib\n  // x\nend Lib;\n")

        def new_library_file():
            (self.library / "Lib" / "Sub.mo").write_text("within Lib;\nmodel Sub\nend Sub;\n")

        def new_version():
            self.version.write_text("OpenModelica 2.0.0")
            _flatten._tool_versions.clear()

        def new_msl():
            patch = mock.patch("rtctools.model.openmodelica_flattener_mixin.MSL_VERSION", "3.2.3")
            patch.start()
            self.addCleanup(patch.stop)

        def edit_bmo():
            (self.models / "Model.bmo").write_text("model M\n  // edited\nend M;\n")

        def remove_bmo():
            (self.models / "Model.bmo").unlink()

        def garble_sidecar():
            (self.models / "Model.bmo_cache").write_text("{not json")

        def remove_sidecar():
            (self.models / "Model.bmo_cache").unlink()

        changes = [
            edit_model,
            edit_library,
            new_library_file,
            new_version,
            new_msl,
            edit_bmo,
            remove_bmo,
            garble_sidecar,
            remove_sidecar,
        ]
        self.assertTrue(self.cached())
        for change in changes:
            with self.subTest(change.__name__):
                change()
                with self.assertLogs("rtctools", "DEBUG") as logs:
                    self.assertTrue(self.cached())
                self.assertIn("again", "\n".join(logs.output))
                self.assertFalse(self.cached())

    def test_cache_invalidated_by_attributes(self):
        self.assertTrue(self.cached())
        self.assertTrue(self.cached(openmodelica_flags=["-d=x"]))
        self.assertFalse(self.cached(openmodelica_flags=["-d=x"]))
        # Another model of the same folder has its own file
        (self.models / "Model2.mo").write_text("model Model2\nend Model2;\n")
        self.assertTrue(self.cached(model_name="Model2"))
        self.assertFalse(self.cached(model_name="Model2"))
        # Model2.mo is a source of Model too
        self.assertTrue(self.cached())

    def test_cache_without_version(self):
        # Without a version of omc, the flattening is not reused
        self.version.unlink()
        self.assertTrue(self.cached())
        self.assertTrue(self.cached())

    def test_cache_relative_folders(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.folder)
        for runs in (True, False):
            before = self.runs()
            self.flatten(OpenModelicaFlattenerMixin, "model", "lib", True)
            self.assertEqual(self.runs() > before, runs)
        # The same sources elsewhere: the fingerprint has no absolute paths
        shutil.copytree(self.folder / "model", self.folder / "copy" / "model")
        shutil.copytree(self.folder / "lib", self.folder / "copy" / "lib")
        before = self.runs()
        self.flatten(OpenModelicaFlattenerMixin, "copy/model", "copy/lib", True)
        self.assertEqual(self.runs(), before)
