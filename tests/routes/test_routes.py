"""
Every example driver on every model route, compared with the reference route: pymoca on the
Modelica files.

A route is a choice of the options of ModelicaMixin (model_format, model_compiler,
model_exporter), or FMUMixin in place of ModelicaMixin (model_format, model_exporter). Each
driver runs in a copy of its example folder, with the route set by tests/routes/hook. Each of its
runs (calls of run_optimization_problem or run_simulation_problem) must succeed and give the same
output timeseries as on the reference route, within TOLERANCE relative to the largest value of
each series.

Known failures are listed in KNOWN_FAILURES with their cause. They are strict xfails: a route
that starts to work fails the test until it is taken off the list. Nothing is skipped: every route
runs in the one environment of the rtc-integration branches (.rtc-integration/env.sh), and a
missing tool is a failure. test_environment lists what is missing:

* omc of the OpenModelica rtc-integration branch: ``RTCTOOLS_OMC`` or ``omc``.
* rumoca of the Rumoca rtc-integration branch (casadi-dae target): ``RTCTOOLS_RUMOCA`` or
  ``rumoca``.
* lacemodelica on the PATH.
* pymoca and CasADi of their rtc-integration branches: Base Modelica, ONNX import, fmi-ls-dae.

These tests run the examples many times, and are not part of the default test run. Run them with
``tox -e routes`` or ``pytest tests/routes``, e.g. with ``-n 8``.
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
HOOK = Path(__file__).resolve().parent / "hook"

TOLERANCE = 1e-4
TIMEOUT = 3600

REFERENCE = "pymoca"

MODELICA = {"mixin": "modelica"}
FMU = {"mixin": "fmu"}
ROUTES = {
    "pymoca": {**MODELICA},
    "rumoca": {**MODELICA, "model_compiler": "rumoca"},
    "bmo-openmodelica-pymoca": {
        **MODELICA,
        "model_format": "basemodelica",
        "model_exporter": "openmodelica",
    },
    "bmo-rumoca-pymoca": {**MODELICA, "model_format": "basemodelica", "model_exporter": "rumoca"},
    "bmo-openmodelica-lacemodelica": {
        **MODELICA,
        "model_format": "basemodelica",
        "model_exporter": "openmodelica",
        "model_compiler": "lacemodelica",
    },
    "bmo-rumoca-lacemodelica": {
        **MODELICA,
        "model_format": "basemodelica",
        "model_exporter": "rumoca",
        "model_compiler": "lacemodelica",
    },
    "fmu-ode-openmodelica": {**FMU, "model_format": "ode", "model_exporter": "openmodelica"},
    "fmu-ode-rumoca": {**FMU, "model_format": "ode", "model_exporter": "rumoca"},
    "fmu-dae-openmodelica": {**FMU, "model_format": "dae", "model_exporter": "openmodelica"},
}

# Drivers on ModelicaMixin or SimulationProblem. FMUMixin only replaces ModelicaMixin, so the
# simulation drivers have no FMU routes. fallback_option is left out: it imposes the solver result.
OPTIMIZATION_DRIVERS = [
    "basic/src/example.py",
    "cascading_channels/src/example.py",
    "channel_pulse/src/example.py",
    "channel_wave_damping/src/example_local_control.py",
    "channel_wave_damping/src/example_optimization.py",
    "ensemble/src/example.py",
    "goal_programming/src/example.py",
    "integrator_delay/src/example.py",
    "lookup_table/src/example.py",
    "mixed_integer/src/example.py",
    "pumped_hydropower_system/src/example.py",
    "single_reservoir/src/single_reservoir.py",
    "step_response/src/example.py",
]
SIMULATION_DRIVERS = [
    "simulation/src/example.py",
    "simulation_with_custom_equations/src/simple_model.py",
]

# (driver, route): cause. Strict xfails.
KNOWN_FAILURES = {
    ("mixed_integer/src/example.py", "fmu-dae-openmodelica"): (
        "Bonmin: the continuous relaxation is infeasible (violation 71.4); not traced. The ODE "
        "FMU route of the same model solves"
    ),
    ("goal_programming/src/example.py", "fmu-dae-openmodelica"): (
        "Bonmin: the continuous relaxation is infeasible, as for mixed_integer (same orifice and "
        "Boolean structure); not traced"
    ),
    ("goal_programming/src/example.py", "fmu-ode-openmodelica"): (
        "another optimum: priority 2 objective 2163.53, the reference 2163.74; priority 3 "
        "0.8505 against 0.8495; not traced"
    ),
    ("goal_programming/src/example.py", "fmu-ode-rumoca"): (
        "another optimum, as for fmu-ode-openmodelica: priority 2 objective 2163.53, the "
        "reference 2163.74"
    ),
}

# Drivers whose optimum is not unique: the routes reach the same objective with different
# trajectories, so their runs are compared on the objective (OBJECTIVE_TOLERANCE, relative).
OBJECTIVE_ONLY = {
    "mixed_integer/src/example.py": "MILP: same objective on every route (2e-11), Q_pump differs",
    "goal_programming/src/example.py": "MILPs: same objective per priority, Q_orifice differs",
}
OBJECTIVE_TOLERANCE = 1e-6


def _cases():
    for driver in OPTIMIZATION_DRIVERS + SIMULATION_DRIVERS:
        for route, options in ROUTES.items():
            if route == REFERENCE:
                continue
            if options["mixin"] == "fmu" and driver in SIMULATION_DRIVERS:
                continue
            # One xdist worker per driver (--dist loadgroup), so that it runs the reference once
            marks = [pytest.mark.xdist_group(driver)]
            if (driver, route) in KNOWN_FAILURES:
                marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[driver, route], strict=True))
            yield pytest.param(
                driver, route, id=f"{driver.split('/')[0]}/{Path(driver).stem}:{route}", marks=marks
            )


# ---------------------------------------------------------------------------- environment


def _command(variable, default):
    return os.environ.get(variable, default).split()[0]


# What the routes need of the Python packages, checked in a subprocess with the test environment
_PACKAGE_CHECK = """
import glob, os
import casadi
from pymoca.backends.casadi._options import _get_default_options
missing = []
if "basemodelica" not in _get_default_options():
    missing.append("pymoca has no basemodelica option (Base Modelica)")
if not hasattr(casadi, "GraphBuilder") or not glob.glob(
    os.path.join(os.path.dirname(casadi.__file__), "libcasadi_graphmodel_onnx*")
):
    missing.append("CasADi has no ONNX import (lacemodelica)")
try:
    casadi.DaeBuilder("check").add("s", "structuralParameter", "fixed", {})
except RuntimeError as error:
    if "No such enum" in str(error):
        missing.append("CasADi has no fmi-ls-dae support (DAE FMUs)")
print("\\n".join(missing))
"""


def test_environment():
    """Everything the routes need is there: one environment runs every route."""
    missing = []
    if not shutil.which(_command("RTCTOOLS_OMC", "omc")):
        missing.append("omc (RTCTOOLS_OMC)")
    rumoca = shutil.which(_command("RTCTOOLS_RUMOCA", "rumoca"))
    if not rumoca:
        missing.append("rumoca (RTCTOOLS_RUMOCA)")
    else:
        result = subprocess.run([rumoca, "targets"], capture_output=True, text=True)
        if "casadi-dae" not in result.stdout + result.stderr:
            missing.append("rumoca without the casadi-dae target")
    include = os.environ.get("RTCTOOLS_FMI3_INCLUDE", "")
    if not include or not os.path.isfile(os.path.join(include, "fmi3Functions.h")):
        missing.append("FMI 3 headers for Rumoca's source FMUs (RTCTOOLS_FMI3_INCLUDE)")
    if not shutil.which("lacemodelica"):
        missing.append("lacemodelica on the PATH")
    result = subprocess.run(
        [sys.executable, "-c", _PACKAGE_CHECK], capture_output=True, text=True, env=_environment({})
    )
    missing += [m for m in result.stdout.splitlines() if m]
    if result.returncode != 0:
        missing.append(result.stderr.strip().splitlines()[-1])
    assert not missing, "missing: " + "; ".join(missing)


# ---------------------------------------------------------------------------- runs


def _environment(route):
    env = dict(os.environ)
    paths = [str(HOOK), str(ROOT / "src")] + [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["RTCTOOLS_ROUTE"] = json.dumps(route)
    return env


def _run(driver, route, work):
    """Run a driver on a route in a copy of its example. Returns (log, records)."""
    example = driver.split("/")[0]
    copy = work / example
    shutil.copytree(
        EXAMPLES / example,
        copy,
        ignore=shutil.ignore_patterns("*.fmu", "*.bmo", "*.pymoca_cache"),
    )
    # Keep the output folders (ensemble writes into output/<member>/), not old results, which
    # could pass for the output of a run that failed
    (copy / "output").mkdir(exist_ok=True)
    for old in (copy / "output").rglob("*"):
        if old.is_file():
            old.unlink()
    record = work / "record"
    record.mkdir()
    env = _environment(ROUTES[route])
    env["RTCTOOLS_ROUTE_RECORD"] = str(record)
    env.setdefault("TMPDIR", str(work))
    script = copy / driver.split("/", 1)[1]
    try:
        result = subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        log, returncode = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        log, returncode = f"{error.stdout or ''}{error.stderr or ''}\ntimeout", -1
    (work / "log.txt").write_text(log)
    records = []
    for folder in sorted(record.iterdir()):
        status = json.loads((folder / "status.json").read_text())
        csv_folder = folder / "csv"
        status["csv"] = {
            str(p.relative_to(csv_folder)): _read_csv(p) for p in sorted(csv_folder.rglob("*.csv"))
        }
        records.append(status)
    return returncode, log, records


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    columns = {}
    for j, name in enumerate(header[1:], start=1):
        try:
            columns[name.strip()] = [
                float(row[j]) if row[j].strip() else float("nan") for row in rows
            ]
        except ValueError:
            pass
    return columns


def _too_few_degrees_of_freedom(log):
    """How many solves IPOPT could only do through its fallback for a rank-deficient problem"""
    return log.count("Too few degrees of freedom")


def _problems(returncode, log, records):
    """What went wrong in a run, as a list of messages."""
    problems = []
    if returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-15:])
        problems.append(f"exit code {returncode}:\n{tail}")
    for i, r in enumerate(records):
        if "error" in r:
            problems.append(f"run {i} ({r['class']}): {r['error']}")
        elif not r.get("success", True):
            problems.append(f"run {i} ({r['class']}): solver failed ({r.get('return_status')})")
    if not records and returncode == 0:
        problems.append("no runs recorded")
    for i, r in enumerate(records):
        if "error" not in r and not r["csv"]:
            problems.append(f"run {i} ({r['class']}): no output timeseries")
    return problems


def _relative_error(values, reference):
    """Largest difference relative to the largest reference value (at least 1). NaN must match."""
    scale = max([abs(v) for v in reference if v == v] + [1.0])
    error = 0.0
    for a, b in zip(values, reference, strict=True):
        if a != a or b != b:
            if (a != a) != (b != b):
                return float("inf")
            continue
        error = max(error, abs(a - b))
    return error / scale


def _objective_differences(reference, records):
    """How the objectives differ from the reference objectives, as a list of messages."""
    messages = []
    if len(records) != len(reference):
        messages.append(f"{len(records)} runs, the reference has {len(reference)}")
    for i, (ref, rec) in enumerate(zip(reference, records, strict=False)):
        # Goal programming: the objective after each priority, else the final objective
        ours = rec.get("priority_objectives") or [rec.get("objective")]
        theirs = ref.get("priority_objectives") or [ref.get("objective")]
        if len(ours) != len(theirs) or None in ours + theirs:
            messages.append(f"run {i}: objectives {ours}, the reference has {theirs}")
            continue
        for k, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            if abs(a - b) > OBJECTIVE_TOLERANCE * max(abs(b), 1.0):
                messages.append(f"run {i} objective {k}: {a!r}, the reference has {b!r}")
    return messages


def _differences(reference, records):
    """How the records differ from the reference records, as a list of messages."""
    messages = []
    if len(records) != len(reference):
        messages.append(f"{len(records)} runs, the reference has {len(reference)}")
    # Runs beyond the shorter list are reported by the count above
    for i, (ref, rec) in enumerate(zip(reference, records, strict=False)):
        for name, ref_columns in ref["csv"].items():
            columns = rec["csv"].get(name)
            if columns is None:
                messages.append(f"run {i}: no {name}")
                continue
            missing = sorted(set(ref_columns) - set(columns))
            if missing:
                messages.append(f"run {i} {name}: missing columns {missing[:10]}")
            for column, ref_values in ref_columns.items():
                values = columns.get(column)
                if values is None:
                    continue
                if len(values) != len(ref_values):
                    messages.append(
                        f"run {i} {name} {column}: {len(values)} values, not {len(ref_values)}"
                    )
                    continue
                error = _relative_error(values, ref_values)
                if not error <= TOLERANCE:
                    messages.append(f"run {i} {name} {column}: relative difference {error:.2e}")
    return messages


_references = {}


def _reference(driver, tmp_path_factory):
    if driver not in _references:
        work = tmp_path_factory.mktemp("reference")
        returncode, log, records = _run(driver, REFERENCE, work)
        problems = _problems(returncode, log, records)
        _references[driver] = (problems, records, _too_few_degrees_of_freedom(log))
    return _references[driver]


@pytest.mark.parametrize("driver, route", list(_cases()))
def test_route(driver, route, tmp_path, tmp_path_factory):
    problems, reference, reference_deficient = _reference(driver, tmp_path_factory)
    if problems:
        pytest.fail(f"the reference route fails on {driver}: " + "; ".join(problems))

    returncode, log, records = _run(driver, route, tmp_path)
    problems = _problems(returncode, log, records)
    # Some examples are rank-deficient on the reference too, e.g. the initial-value check of
    # single_reservoir; a route must not add such solves
    deficient = _too_few_degrees_of_freedom(log)
    if deficient > reference_deficient:
        problems.append(
            f"IPOPT: Too few degrees of freedom in {deficient} solves, the reference has "
            f"{reference_deficient}"
        )
    assert not problems, f"{driver} on {route} (log: {tmp_path / 'log.txt'}):\n" + "\n".join(
        problems
    )
    if driver in OBJECTIVE_ONLY:
        differences = _objective_differences(reference, records)
    else:
        differences = _differences(reference, records)
    assert not differences, f"{driver} on {route} differs from {REFERENCE}:\n" + "\n".join(
        differences[:20]
    )


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(d, id=d.split("/")[0] + "/" + Path(d).stem, marks=pytest.mark.xdist_group(d))
        for d in OPTIMIZATION_DRIVERS + SIMULATION_DRIVERS
    ],
)
def test_reference(driver, tmp_path_factory):
    """The reference route itself works, so that a failure of another route is its own."""
    problems, _, _ = _reference(driver, tmp_path_factory)
    assert not problems, "; ".join(problems)


def test_known_failures_exist():
    cases = {(p.values[0], p.values[1]) for p in _cases()}
    assert not set(KNOWN_FAILURES) - cases, "known failures that are not test cases"
    assert set(OBJECTIVE_ONLY) <= set(OPTIMIZATION_DRIVERS), (
        "objective-only drivers that are not drivers"
    )
    assert all(re.match(r"\S", reason) for reason in KNOWN_FAILURES.values())
