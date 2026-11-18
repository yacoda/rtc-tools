"""The flattener mixins, with stand-ins for the tools: what they run, and what they write."""

import os
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from rtctools.model.openmodelica_flattener_mixin import OpenModelicaFlattenerMixin

# Stands in for omc: writes what the .mos script asks for, and records its arguments
FAKE_OMC = textwrap.dedent(
    """
    import re, sys
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
        omc = self.folder / "omc.py"
        omc.write_text(f"LOG = {str(self.log)!r}\nARGS = {str(self.args)!r}\n" + FAKE_OMC)
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
