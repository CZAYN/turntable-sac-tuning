"""One-way six-metric evaluation over the sealed physics test ensemble.

The training environment never imports this module.  Candidate locking and
write-once consumption are enforced by :mod:`scripts.run_final_test`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .controller_parameters import load_physics_controller_parameter_space
from .discrete_loop_model import build_discrete_loop_models
from .performance_targets import (
    LOOP_ORDER,
    LoopPerformanceTarget,
    load_controller_performance_targets,
)
from .physics_evaluator import (
    PHYSICS_FREQUENCY_POINTS,
    _evaluate_open_loop,
    _reference_metrics,
)
from .physics_motor_model import (
    MotorParameters,
    load_physics_motor_config,
    simulate_scenario,
)
from .physics_test_dataset import (
    OFFICIAL_METRICS_PER_LOOP,
    load_final_test_spec,
    load_physics_test_ensemble,
)


def _frequency_metrics(
    project_root: Path, motor: MotorParameters, parameters: np.ndarray
) -> dict[str, dict[str, float | bool]]:
    """Return only the three official frequency metrics plus validity metadata."""

    config = load_physics_motor_config(project_root)
    models = build_discrete_loop_models(config, motor, parameters)
    result: dict[str, dict[str, float | bool]] = {}
    for loop in LOOP_ORDER:
        raw = _evaluate_open_loop(
            loop,
            models[loop],
            frequency_points=PHYSICS_FREQUENCY_POINTS,
        )
        values = (
            float(raw["bandwidth_hz"]),
            float(raw["gain_margin_db"]),
            float(raw["phase_margin_deg"]),
        )
        result[loop] = {
            "closed_loop_bandwidth_hz": values[0],
            "gain_margin_db": values[1],
            "phase_margin_deg": values[2],
            "valid": bool(
                raw["metric_valid"]
                and raw["bandwidth_found"]
                and all(math.isfinite(value) for value in values)
            ),
        }
    return result


def _time_metrics(
    project_root: Path,
    motor: MotorParameters,
    parameters: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, float | bool]]:
    """Return only overshoot, rise time and settling time for each loop."""

    config = load_physics_motor_config(project_root)
    scenario_spec = load_final_test_spec(project_root)["reference_steps"]
    overrides = {
        "current_reference_a": float(scenario_spec["current_reference_a"]),
        "speed_reference_rad_s": float(scenario_spec["speed_reference_rad_s"]),
        "position_reference_rad": float(scenario_spec["position_reference_rad"]),
    }
    result: dict[str, dict[str, float | bool]] = {}
    for offset, loop in enumerate(LOOP_ORDER):
        trace = simulate_scenario(
            config,
            motor,
            parameters,
            loop,
            encoder_effects=bool(scenario_spec["encoder_effects"]),
            seed=seed + offset,
            scenario_overrides=overrides,
        )
        raw = _reference_metrics(trace)
        values = (
            float(raw["overshoot_ratio"]),
            float(raw["rise_time_s"]),
            float(raw["settling_time_s"]),
        )
        result[loop] = {
            "overshoot_ratio": values[0],
            "rise_time_s": values[1],
            "settling_time_s": values[2],
            "valid": bool(
                not trace.terminated
                and raw["reference_metric_valid"]
                and raw["reached_10_percent"]
                and raw["reached_90_percent"]
                and raw["settled"]
                and all(math.isfinite(value) and value >= 0.0 for value in values)
            ),
        }
    return result


def _evaluate_six_metrics(
    frequency: Mapping[str, float | bool],
    time_domain: Mapping[str, float | bool],
    target: LoopPerformanceTarget,
    *,
    bandwidth_tolerance_fraction: float,
) -> dict[str, Any]:
    """Apply the six literal table targets to one model and one loop."""

    actual = {
        "closed_loop_bandwidth_hz": float(
            frequency["closed_loop_bandwidth_hz"]
        ),
        "gain_margin_db": float(frequency["gain_margin_db"]),
        "phase_margin_deg": float(frequency["phase_margin_deg"]),
        "overshoot_ratio": float(time_domain["overshoot_ratio"]),
        "rise_time_s": float(time_domain["rise_time_s"]),
        "settling_time_s": float(time_domain["settling_time_s"]),
    }
    frequency_valid = bool(frequency["valid"])
    time_valid = bool(time_domain["valid"])
    failures: list[str] = []
    if not frequency_valid:
        failures.append("invalid_frequency_measurement")
    if not time_valid:
        failures.append("invalid_time_measurement")
    bandwidth_relative_error = abs(
        actual["closed_loop_bandwidth_hz"] / target.bandwidth_hz - 1.0
    )
    if bandwidth_relative_error > bandwidth_tolerance_fraction:
        failures.append("closed_loop_bandwidth_hz")
    if actual["gain_margin_db"] < target.minimum_gain_margin_db:
        failures.append("gain_margin_db")
    if actual["phase_margin_deg"] < target.minimum_phase_margin_deg:
        failures.append("phase_margin_deg")
    if actual["overshoot_ratio"] > target.maximum_overshoot_ratio:
        failures.append("overshoot_ratio")
    if actual["rise_time_s"] > target.maximum_rise_time_s:
        failures.append("rise_time_s")
    if actual["settling_time_s"] > target.maximum_settling_time_s:
        failures.append("settling_time_s")
    return {
        "target": target.as_dict(),
        "actual": actual,
        "metric_validity": {
            "frequency": frequency_valid,
            "time_domain": time_valid,
        },
        "pass": not failures,
        "failures": failures,
    }


def _group_summary(rows: list[dict[str, Any]], group: str) -> dict[str, int]:
    selected = [row for row in rows if row["test_group"] == group]
    passed = sum(bool(row["pass"]) for row in selected)
    return {
        "model_count": len(selected),
        "pass_count": passed,
        "fail_count": len(selected) - passed,
    }


def evaluate_locked_final_candidate(
    project_root: Path, parameters: np.ndarray
) -> dict[str, Any]:
    """Consume the sealed suite in memory; the CLI controls one-time writing."""

    root = Path(project_root).resolve()
    space = load_physics_controller_parameter_space(root)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (11,):
        raise ValueError("final candidate must have shape (11,)")
    space.normalize(values)
    ensemble = load_physics_test_ensemble(root)
    spec = load_final_test_spec(root)
    targets = load_controller_performance_targets(root)
    tolerance = float(
        spec["acceptance_policy"]["bandwidth_reporting_tolerance_fraction"]
    )

    rows: list[dict[str, Any]] = []
    for index in range(ensemble["parameters"].shape[0]):
        motor = MotorParameters.from_array(ensemble["parameters"][index])
        frequency = _frequency_metrics(root, motor, values)
        time_domain = _time_metrics(
            root, motor, values, seed=20260725 + index * 10
        )
        loops = {
            loop: _evaluate_six_metrics(
                frequency[loop],
                time_domain[loop],
                targets.loop(loop),
                bandwidth_tolerance_fraction=tolerance,
            )
            for loop in LOOP_ORDER
        }
        failed_loops = [loop for loop in LOOP_ORDER if not loops[loop]["pass"]]
        rows.append(
            {
                "model_id": str(ensemble["model_id"][index]),
                "test_group": str(ensemble["test_group"][index]),
                "pass": not failed_loops,
                "failed_loops": failed_loops,
                "performance_metrics": loops,
            }
        )

    pass_count = sum(bool(row["pass"]) for row in rows)
    overall_pass = bool(pass_count == len(rows))
    return {
        "schema_version": 2,
        "test_suite_id": spec["test_suite_id"],
        "backend": "physics",
        "controller_structure": "three nested PIDF loops; DOBC is embedded in the speed loop",
        "parameter_names": list(space.names),
        "parameters": values.tolist(),
        "official_metrics_per_loop": list(OFFICIAL_METRICS_PER_LOOP),
        "acceptance": {
            "performance_targets": targets.as_dict(),
            "bandwidth_reporting_tolerance_fraction": tolerance,
            "bandwidth_tolerance_semantics": spec["acceptance_policy"][
                "bandwidth_tolerance_semantics"
            ],
            "all_24_models_must_pass_suite": True,
        },
        "overall_pass": overall_pass,
        "summary": {
            "model_count": len(rows),
            "model_pass_count": pass_count,
            "model_fail_count": len(rows) - pass_count,
            "in_distribution": _group_summary(rows, "in_distribution"),
            "ood": _group_summary(rows, "ood"),
        },
        "models": rows,
        "policy": {
            "test_results_must_not_be_used_to_retrain_or_select_another_candidate": True,
            "hardware_use_allowed": False,
            "only_the_six_declared_metrics_are_formally_evaluated": True,
        },
    }
