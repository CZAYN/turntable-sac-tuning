from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import ControllerParameterSpace  # noqa: E402
from elc_rl.evaluation_utils import (  # noqa: E402
    GAIN_MARGIN_CAP_DB,
    interpolate_pair,
    zero_crossing_locations,
)
from elc_rl.physics_evaluator import (  # noqa: E402
    get_physics_controller_evaluator,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidate(path: Path, names: tuple[str, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=np.float64)
        archived_names = tuple(str(value) for value in archive["parameter_names"])
    if archived_names != names or parameters.shape != (len(names),):
        raise ValueError("candidate does not match the project parameter space")
    if not np.isfinite(parameters).all():
        raise ValueError("candidate contains a non-finite value")
    return parameters


def _closed_loop_state_matrix(
    *,
    kp: float,
    ki: float,
    kd: float,
    sample_period_s: float,
    filter_time_s: float,
    current_delay_s: float,
    inductance_h: float,
    resistance_ohm: float,
) -> np.ndarray:
    alpha = sample_period_s / (filter_time_s + sample_period_s)
    delay_alpha = sample_period_s / (current_delay_s + sample_period_s)
    plant_a = 1.0 - sample_period_s * resistance_ohm / inductance_h
    plant_b = sample_period_s / inductance_h

    def update(state: np.ndarray) -> np.ndarray:
        integral, derivative, previous_error, applied_voltage, current = state
        error = -current
        raw_derivative = (error - previous_error) / sample_period_s
        next_derivative = derivative + alpha * (raw_derivative - derivative)
        next_integral = integral + ki * error * sample_period_s
        command = kp * error + next_integral + kd * next_derivative
        next_applied = applied_voltage + delay_alpha * (
            command - applied_voltage
        )
        next_current = plant_a * current + plant_b * next_applied
        return np.asarray(
            [next_integral, next_derivative, error, next_applied, next_current],
            dtype=np.float64,
        )

    matrix = np.column_stack(
        [update(np.eye(5, dtype=np.float64)[:, index]) for index in range(5)]
    )
    return matrix


def _discrete_open_loop(
    frequency_hz: np.ndarray,
    *,
    kp: float,
    ki: float,
    kd: float,
    sample_period_s: float,
    filter_time_s: float,
    current_delay_s: float,
    inductance_h: float,
    resistance_ohm: float,
) -> np.ndarray:
    q = np.exp(-1j * 2.0 * np.pi * frequency_hz * sample_period_s)
    alpha = sample_period_s / (filter_time_s + sample_period_s)
    derivative = (
        alpha
        / sample_period_s
        * (1.0 - q)
        / (1.0 - (1.0 - alpha) * q)
    )
    controller = kp + ki * sample_period_s / (1.0 - q) + kd * derivative
    delay_alpha = sample_period_s / (current_delay_s + sample_period_s)
    execution = delay_alpha / (1.0 - (1.0 - delay_alpha) * q)
    plant_a = 1.0 - sample_period_s * resistance_ohm / inductance_h
    plant = sample_period_s / inductance_h * q / (1.0 - plant_a * q)
    return controller * execution * plant


def _frequency_metrics(
    open_loop: np.ndarray,
    frequency_hz: np.ndarray,
    maximum_pole_magnitude: float,
) -> dict[str, float | bool]:
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(open_loop), 1e-300))
    phase_deg = np.rad2deg(np.unwrap(np.angle(open_loop)))
    log_frequency = np.log10(frequency_hz)
    gain_crossings = zero_crossing_locations(log_frequency, magnitude_db)
    phase_margin_deg = float(
        min(
            180.0 + interpolate_pair(phase_deg, index, fraction)
            for _, index, fraction in gain_crossings
        )
        if gain_crossings
        else -180.0
    )

    gain_margin_candidates: list[float] = []
    phase_target = -180.0
    while phase_target >= float(np.min(phase_deg)):
        for _, index, fraction in zero_crossing_locations(
            log_frequency, phase_deg - phase_target
        ):
            gain_margin_candidates.append(
                -interpolate_pair(magnitude_db, index, fraction)
            )
        phase_target -= 360.0
    gain_margin_db = float(
        min(gain_margin_candidates)
        if gain_margin_candidates
        else GAIN_MARGIN_CAP_DB
    )

    complementary = open_loop / (1.0 + open_loop)
    low_frequency_gain = float(abs(complementary[0]))
    relative_db = 20.0 * np.log10(
        np.maximum(np.abs(complementary) / max(low_frequency_gain, 1e-300), 1e-300)
    )
    threshold_db = -10.0 * np.log10(2.0)
    bandwidth_crossings = [
        crossing
        for crossing in zero_crossing_locations(
            log_frequency, relative_db - threshold_db
        )
        if relative_db[crossing[1]] >= threshold_db
        and relative_db[crossing[1] + 1] < threshold_db
    ]
    bandwidth_found = bool(bandwidth_crossings)
    bandwidth_hz = float(
        bandwidth_crossings[0][0] if bandwidth_found else frequency_hz[-1]
    )
    finite = bool(
        np.isfinite(open_loop.real).all()
        and np.isfinite(open_loop.imag).all()
        and np.isfinite(maximum_pole_magnitude)
    )
    stable = bool(maximum_pole_magnitude < 1.0 - 1e-10)
    return {
        "metric_valid": bool(finite and stable and gain_crossings),
        "stable": stable,
        "maximum_closed_loop_pole_magnitude": maximum_pole_magnitude,
        "bandwidth_hz": bandwidth_hz,
        "bandwidth_found": bandwidth_found,
        "gain_margin_db": gain_margin_db,
        "phase_margin_deg": phase_margin_deg,
    }


def _summarize(rows: list[dict[str, Any]], target: Any) -> dict[str, Any]:
    passed = [
        bool(
            row["metric_valid"]
            and abs(float(row["bandwidth_hz"]) / target.bandwidth_hz - 1.0)
            <= 0.1
            and float(row["gain_margin_db"]) >= target.minimum_gain_margin_db
            and float(row["phase_margin_deg"])
            >= target.minimum_phase_margin_deg
        )
        for row in rows
    ]
    return {
        "model_count": len(rows),
        "pass_count": sum(passed),
        "all_models_pass": all(passed),
        "bandwidth_hz_median": float(
            np.median([float(row["bandwidth_hz"]) for row in rows])
        ),
        "bandwidth_hz_min": min(float(row["bandwidth_hz"]) for row in rows),
        "bandwidth_hz_max": max(float(row["bandwidth_hz"]) for row in rows),
        "gain_margin_db_worst": min(
            float(row["gain_margin_db"]) for row in rows
        ),
        "phase_margin_deg_worst": min(
            float(row["phase_margin_deg"]) for row in rows
        ),
        "maximum_closed_loop_pole_magnitude_worst": max(
            float(row["maximum_closed_loop_pole_magnitude"]) for row in rows
        ),
    }


def run_audit(
    project_root: Path,
    *,
    candidate_path: Path,
    output_path: Path,
    frequency_points: int,
) -> dict[str, Any]:
    if frequency_points < 1024:
        raise ValueError("frequency_points must be at least 1024")
    evaluator = get_physics_controller_evaluator(project_root)
    parameters = _load_candidate(candidate_path, evaluator.space.names)
    canonical_space = evaluator.space
    extended_specs = tuple(
        replace(spec, lower=min(spec.lower, float(parameters[9])))
        if spec.name == "kicurr"
        else replace(spec, upper=max(spec.upper, float(parameters[10])))
        if spec.name == "kdcurr"
        else spec
        for spec in canonical_space.specs
    )
    evaluator.space = ControllerParameterSpace(
        canonical_space.task_id, extended_specs, canonical_space.metadata
    )
    official = evaluator.audit(parameters)

    sample_period_s = evaluator.config.sample_period_s_for("current")
    filter_time_s = evaluator.config.derivative_filter_s["current"]
    frequency_hz = np.geomspace(
        0.1,
        min(2250.0, 0.45 / sample_period_s),
        frequency_points,
    )
    rows: list[dict[str, Any]] = []
    kp, ki, kd = (float(parameters[index]) for index in (8, 9, 10))
    for raw_index in evaluator.audit_indices:
        index = int(raw_index)
        motor = evaluator.motor(index)
        state_matrix = _closed_loop_state_matrix(
            kp=kp,
            ki=ki,
            kd=kd,
            sample_period_s=sample_period_s,
            filter_time_s=filter_time_s,
            current_delay_s=motor.current_delay_s,
            inductance_h=motor.inductance_h,
            resistance_ohm=motor.resistance_ohm,
        )
        maximum_pole_magnitude = float(
            np.max(np.abs(np.linalg.eigvals(state_matrix)))
        )
        open_loop = _discrete_open_loop(
            frequency_hz,
            kp=kp,
            ki=ki,
            kd=kd,
            sample_period_s=sample_period_s,
            filter_time_s=filter_time_s,
            current_delay_s=motor.current_delay_s,
            inductance_h=motor.inductance_h,
            resistance_ohm=motor.resistance_ohm,
        )
        rows.append(
            {
                "model_id": str(evaluator.ensemble["model_id"][index]),
                "role": str(evaluator.ensemble["role"][index]),
                **_frequency_metrics(
                    open_loop, frequency_hz, maximum_pole_magnitude
                ),
            }
        )

    training = [row for row in rows if row["role"] != "validation"]
    validation = [row for row in rows if row["role"] == "validation"]
    target = evaluator.performance_targets.loop("current")
    report = {
        "schema_version": 1,
        "run_kind": "implementation_aligned_discrete_current_loop_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": _sha256(candidate_path),
            "current_parameters": {"kp": kp, "ki": ki, "kd": kd},
        },
        "semantics": {
            "sample_period_s": sample_period_s,
            "sample_frequency_hz": 1.0 / sample_period_s,
            "current_derivative_filter_s": filter_time_s,
            "controller": "backward-difference derivative with first-order discrete filter and current-step integral update",
            "execution_delay": "backward-Euler first-order state exactly matching simulation_kernel current_delay_alpha",
            "electrical_plant": "explicit-Euler L-R state exactly matching simulation_kernel",
            "frequency_limit_hz": float(frequency_hz[-1]),
            "frequency_points": frequency_points,
            "nonlinear_limits": "not represented in this small-signal audit",
            "official_acceptance": False,
        },
        "target": target.as_dict(),
        "discrete": {
            "training": _summarize(training, target),
            "validation": _summarize(validation, target),
            "models": rows,
        },
        "official_project_evaluator": {
            "training": official["performance_metrics"]["current"],
            "validation": official["validation_diagnostics"][
                "performance_metrics"
            ]["current"],
        },
        "hardware_use_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one current-loop candidate using the discrete equations "
            "implemented by simulation_kernel."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frequency-points", type=int, default=8192)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "current_loop_discrete_audit_20260820.json",
    )
    arguments = parser.parse_args()
    report = run_audit(
        PROJECT_ROOT,
        candidate_path=arguments.candidate,
        output_path=arguments.output,
        frequency_points=arguments.frequency_points,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "discrete_training": report["discrete"]["training"],
                "discrete_validation": report["discrete"]["validation"],
                "official_training": report["official_project_evaluator"][
                    "training"
                ]["actual"],
                "official_validation": report["official_project_evaluator"][
                    "validation"
                ]["actual"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
