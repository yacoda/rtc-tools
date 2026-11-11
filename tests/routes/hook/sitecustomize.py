"""Put an example driver on another model route, and record what each of its runs produces.

Used by tests/routes: on PYTHONPATH, with RTCTOOLS_ROUTE (JSON, the route options) and
RTCTOOLS_ROUTE_RECORD (a folder) in the environment.

Each call of run_optimization_problem or run_simulation_problem is recorded in
<record>/<index>/: the CSV files of the output folder after the call, with its subfolders, in
csv/, and status.json with the problem class, the solver status, the objective value and,
for goal programming, the objective after each priority.
"""

import json
import os
import shutil
from pathlib import Path

import rtctools.util

_route = json.loads(os.environ.get("RTCTOOLS_ROUTE", "{}"))
_record = Path(os.environ["RTCTOOLS_ROUTE_RECORD"])
_options = {k: v for k, v in _route.items() if k != "mixin"}

if _route.get("mixin") == "fmu":
    import rtctools.optimization.fmu_mixin as _fmu_mixin
    import rtctools.optimization.modelica_mixin as _modelica_mixin

    class _FMURoute(_fmu_mixin.FMUMixin):
        """FMUMixin in place of ModelicaMixin, on the route of the test, whatever the driver says"""

        def __new__(cls, *args, **kwargs):
            return super().__new__(cls, *args, **{**kwargs, **_options})

    _modelica_mixin.ModelicaMixin = _FMURoute
elif _options:
    from rtctools.optimization.modelica_mixin import ModelicaMixin
    from rtctools.simulation.simulation_problem import SimulationProblem

    # Defaults of the route options; a driver that sets one itself keeps its own
    for _cls in (ModelicaMixin, SimulationProblem):
        for _name, _value in _options.items():
            setattr(_cls, _name, _value)


# The objective after each priority of a goal programming problem
_priority_objectives = []


def _record_priorities():
    from rtctools.optimization.goal_programming_mixin_base import _GoalProgrammingMixinBase

    completed = _GoalProgrammingMixinBase.priority_completed

    def priority_completed(self, priority):
        _priority_objectives.append(float(self.objective_value))
        return completed(self, priority)

    _GoalProgrammingMixinBase.priority_completed = priority_completed


_record_priorities()


def _recorded(run):
    def wrapper(problem_class, *args, **kwargs):
        index = len(list(_record.glob("[0-9]*")))
        folder = _record / f"{index:02d}"
        folder.mkdir(parents=True)
        status = {"class": problem_class.__name__}
        _priority_objectives.clear()
        try:
            problem = run(problem_class, *args, **kwargs)
        except BaseException as error:
            status["error"] = f"{type(error).__name__}: {error}"[:2000]
            (folder / "status.json").write_text(json.dumps(status, indent=1))
            raise
        stats = getattr(problem, "solver_stats", None) or {}
        status["success"] = bool(stats.get("success", True))
        status["return_status"] = str(stats.get("return_status", ""))
        objective = getattr(problem, "objective_value", None)
        try:
            status["objective"] = None if objective is None else float(objective)
        except (TypeError, ValueError):
            status["objective"] = None
        status["priority_objectives"] = list(_priority_objectives)
        output_folder = Path(getattr(problem, "_output_folder", "") or "")
        if output_folder.is_dir():
            # Also in subfolders: ensemble members write to output/<member>/
            for csv in output_folder.rglob("*.csv"):
                target = folder / "csv" / csv.relative_to(output_folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(csv, target)
        (folder / "status.json").write_text(json.dumps(status, indent=1))
        return problem

    return wrapper


rtctools.util.run_optimization_problem = _recorded(rtctools.util.run_optimization_problem)
rtctools.util.run_simulation_problem = _recorded(rtctools.util.run_simulation_problem)
