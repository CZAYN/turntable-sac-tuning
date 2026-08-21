from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_evaluator import (  # noqa: E402
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.performance_targets import LOOP_ORDER  # noqa: E402
from elc_rl.tuning_env import combined_stage_cost  # noqa: E402


def _load_parameters(path: Path, space: Any) -> np.ndarray:
    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate does not exist: {candidate_path}")
    with np.load(candidate_path, allow_pickle=False) as data:
        if "parameters" not in data.files:
            raise ValueError("candidate archive is missing parameters")
        values = np.asarray(data["parameters"], dtype=np.float64)
        if "parameter_names" in data.files:
            names = tuple(str(value) for value in data["parameter_names"])
            if names != tuple(space.names):
                raise ValueError("candidate parameter order does not match the project")
    if values.shape != (len(space.names),) or not np.isfinite(values).all():
        raise ValueError("candidate parameters must be one finite 11-vector")
    space.normalize(values)
    return values


def _targets_met(report: dict[str, Any], domain: str) -> bool:
    key = f"all_{domain}_targets_met"
    return bool(report["safety"].get(key, False))


def _loop_target_met(report: dict[str, Any], loop: str) -> bool:
    return bool(report["performance_metrics"][loop]["target_pass"])


def _validation_summary(
    frequency: dict[str, Any], time_domain: dict[str, Any]
) -> dict[str, Any] | None:
    frequency_validation = frequency.get("validation_diagnostics")
    time_validation = time_domain.get("validation_diagnostics")
    if frequency_validation is None or time_validation is None:
        return None
    loops: dict[str, Any] = {}
    for loop in LOOP_ORDER:
        frequency_loop = frequency_validation["performance_metrics"][loop]
        time_loop = time_validation["performance_metrics"][loop]
        loops[loop] = {
            "frequency_actual": frequency_loop["actual"],
            "time_actual": time_loop["actual"],
            "target_pass": bool(
                frequency_loop["target_pass"] and time_loop["target_pass"]
            ),
        }
    return {
        "all_six_targets_met": bool(
            frequency_validation["safety"]["all_frequency_targets_met"]
            and time_validation["safety"]["all_time_targets_met"]
        ),
        "frequency_cost": frequency_validation["cost"],
        "time_cost": time_validation["cost"],
        "loops": loops,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-time-audit",
        action="store_true",
        help="evaluate all 56 nonlinear models instead of the 4-model runtime audit",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="optional parameter archive; default evaluates the configured initial seed",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    frequency_evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    time_evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    parameters = (
        frequency_evaluator.space.initial
        if args.candidate is None
        else _load_parameters(args.candidate, frequency_evaluator.space)
    )
    frequency = frequency_evaluator.audit(parameters)
    time_domain = (
        time_evaluator.full_audit(parameters)
        if args.full_time_audit
        else time_evaluator.audit(parameters)
    )
    loop_metrics = {}
    for loop in LOOP_ORDER:
        frequency_loop = frequency["performance_metrics"][loop]
        time_loop = time_domain["performance_metrics"][loop]
        loop_metrics[loop] = {
            "target": frequency_loop["target"],
            "frequency_actual": frequency_loop["actual"],
            "time_actual": time_loop["actual"],
            "normalized_errors": {
                **frequency_loop["normalized_errors"],
                **time_loop["normalized_errors"],
            },
            "frequency_cost": frequency_loop["frequency_cost"],
            "time_cost": time_loop["time_cost"],
            "target_pass": bool(
                _loop_target_met(frequency, loop)
                and _loop_target_met(time_domain, loop)
            ),
        }
    valid = bool(frequency["safety"]["safe"] and time_domain["safety"]["safe"])
    all_targets_met = bool(
        _targets_met(frequency, "frequency")
        and _targets_met(time_domain, "time")
    )
    report = {
        "schema_version": 2,
        "backend": "physics",
        "objective": "literal three-loop six-metric controller target table",
        "candidate_source": (
            "configured_initial_seed"
            if args.candidate is None
            else str(args.candidate.resolve())
        ),
        "time_audit_scope": "all_56_models" if args.full_time_audit else "runtime_4_models",
        "parameter_names": list(frequency_evaluator.space.names),
        "parameters": parameters.tolist(),
        "valid": valid,
        "all_six_targets_met": all_targets_met,
        "cost": combined_stage_cost(frequency, time_domain, "joint"),
        "performance_metrics": loop_metrics,
        "validation": _validation_summary(frequency, time_domain),
        "frequency": {
            "cost": frequency["cost"],
            "safety": frequency["safety"],
        },
        "time_domain": {
            "cost": time_domain["cost"],
            "safety": time_domain["safety"],
        },
    }
    output = (
        args.output
        if args.output is not None
        else PROJECT_ROOT / "outputs" / "physics_six_metric_baseline_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "valid": valid,
                "all_six_targets_met": all_targets_met,
                "cost": report["cost"],
                "frequency_models": frequency["evaluated_model_count"],
                "time_domain_models": time_domain["evaluated_model_count"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
