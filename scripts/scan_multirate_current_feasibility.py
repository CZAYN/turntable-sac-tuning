from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.optimize import differential_evolution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    load_physics_controller_parameter_space,
)
from elc_rl.discrete_loop_model import build_discrete_loop_model  # noqa: E402
from elc_rl.evaluation_utils import (  # noqa: E402
    GAIN_MARGIN_CAP_DB,
    interpolate_pair,
    zero_crossing_locations,
)
from elc_rl.performance_targets import (  # noqa: E402
    FREQUENCY_ERROR_ORDER,
    TIME_ERROR_ORDER,
    frequency_normalized_errors,
    load_controller_performance_targets,
    metric_cost,
    time_normalized_errors,
)
from elc_rl.physics_evaluator import _reference_metrics  # noqa: E402
from elc_rl.physics_motor_model import (  # noqa: E402
    MotorParameters,
    PhysicsMotorConfig,
    load_physics_motor_config,
    load_physics_motor_ensemble,
    simulate_scenario,
)


CURRENT_PARAMETER_INDICES = (8, 9, 10)
CURRENT_PARAMETER_NAMES = ("kpcurr", "kicurr", "kdcurr")
CURRENT_UNCERTAIN_FIELDS = (
    "inductance_h",
    "resistance_ohm",
    "current_delay_s",
)
SEARCH_LOG10_BOUNDS = (
    (-2.0, math.log10(500.0)),
    (0.0, 6.0),
    (-8.0, math.log10(0.02)),
)
SELECTION_GROUP_NAMES = (
    "nominal",
    "corners_8",
    "dense_grid_125",
    "training_40",
)


def config_for_sample_rate(
    base: PhysicsMotorConfig, sample_rate_hz: float
) -> PhysicsMotorConfig:
    """Return an in-memory config changing only the current-loop sample period."""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    payload = deepcopy(base.payload)
    payload["sample_periods_s"]["current"] = 1.0 / float(sample_rate_hz)
    result = replace(base, payload=payload)
    result.validate()
    return result


def current_frequency_grid(config: PhysicsMotorConfig, points: int) -> np.ndarray:
    if points < 64:
        raise ValueError("frequency grid requires at least 64 points")
    nyquist_hz = 0.5 / config.sample_period_s_for("current")
    upper_hz = 0.9 * nyquist_hz
    if upper_hz <= 0.2:
        raise ValueError("sample rate is too low for the current-loop grid")
    return np.geomspace(0.2, upper_hz, points)


def current_corner_motors(
    config: PhysicsMotorConfig,
) -> list[tuple[str, MotorParameters]]:
    """Return the eight exact L/R/current-delay uncertainty corners."""

    nominal = config.nominal
    uncertainty = config.uncertainty_fraction
    rows: list[tuple[str, MotorParameters]] = []
    for l_sign in (-1, 1):
        for r_sign in (-1, 1):
            for delay_sign in (-1, 1):
                factors = {
                    "inductance_h": 1.0 + l_sign * uncertainty["inductance_h"],
                    "resistance_ohm": 1.0 + r_sign * uncertainty["resistance_ohm"],
                    "current_delay_s": 1.0
                    + delay_sign * uncertainty["current_delay_s"],
                }
                model_id = f"corner_L{l_sign:+d}_R{r_sign:+d}_td{delay_sign:+d}"
                rows.append(
                    (
                        model_id,
                        replace(
                            nominal,
                            **{
                                name: float(getattr(nominal, name) * factor)
                                for name, factor in factors.items()
                            },
                        ),
                    )
                )
    return rows


def current_uncertainty_grid(
    config: PhysicsMotorConfig, levels: int = 5
) -> list[tuple[str, MotorParameters]]:
    """Return a deterministic dense L/R/current-delay grid including corners."""

    if levels < 2:
        raise ValueError("uncertainty grid requires at least two levels")
    nominal = config.nominal
    uncertainty = config.uncertainty_fraction
    normalized = np.linspace(-1.0, 1.0, levels)
    rows: list[tuple[str, MotorParameters]] = []
    for l_value in normalized:
        for r_value in normalized:
            for delay_value in normalized:
                coordinates = (float(l_value), float(r_value), float(delay_value))
                model_id = "grid_L{:+.2f}_R{:+.2f}_td{:+.2f}".format(*coordinates)
                rows.append(
                    (
                        model_id,
                        replace(
                            nominal,
                            inductance_h=nominal.inductance_h
                            * (1.0 + coordinates[0] * uncertainty["inductance_h"]),
                            resistance_ohm=nominal.resistance_ohm
                            * (1.0 + coordinates[1] * uncertainty["resistance_ohm"]),
                            current_delay_s=nominal.current_delay_s
                            * (1.0 + coordinates[2] * uncertainty["current_delay_s"]),
                        ),
                    )
                )
    return rows


def ensemble_current_motors(
    project_root: Path,
) -> tuple[list[tuple[str, MotorParameters]], list[tuple[str, MotorParameters]]]:
    ensemble = load_physics_motor_ensemble(project_root)
    training: list[tuple[str, MotorParameters]] = []
    validation: list[tuple[str, MotorParameters]] = []
    for index in range(len(ensemble["model_id"])):
        item = (
            str(ensemble["model_id"][index]),
            MotorParameters.from_array(ensemble["parameters"][index]),
        )
        if str(ensemble["role"][index]) == "validation":
            validation.append(item)
        else:
            training.append(item)
    if len(training) != 40 or len(validation) != 16:
        raise ValueError("unexpected physics ensemble split sizes")
    return training, validation


def _parameters(base: np.ndarray, current_values: np.ndarray) -> np.ndarray:
    values = np.asarray(base, dtype=np.float64).copy()
    values[list(CURRENT_PARAMETER_INDICES)] = np.asarray(
        current_values, dtype=np.float64
    )
    return values


def _frequency_metrics(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    parameters: np.ndarray,
    *,
    points: int,
) -> dict[str, float | bool]:
    model = build_discrete_loop_model(config, motor, parameters, "current")
    frequency_hz = current_frequency_grid(config, points)
    open_loop = model.open_loop_response(frequency_hz)
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
        min(gain_margin_candidates) if gain_margin_candidates else GAIN_MARGIN_CAP_DB
    )

    closed = model.closed_actual_response(frequency_hz)
    low_gain = float(abs(closed[0]))
    relative_db = 20.0 * np.log10(
        np.maximum(np.abs(closed) / max(low_gain, 1e-300), 1e-300)
    )
    threshold_db = 10.0 * np.log10(2.0)
    bandwidth_crossings = [
        crossing
        for crossing in zero_crossing_locations(
            log_frequency, relative_db + threshold_db
        )
        if relative_db[crossing[1]] >= -threshold_db
        and relative_db[crossing[1] + 1] < -threshold_db
    ]
    bandwidth_found = bool(bandwidth_crossings)
    bandwidth_hz = float(
        bandwidth_crossings[0][0] if bandwidth_found else frequency_hz[-1]
    )
    poles = model.closed_loop_io_poles
    finite = bool(
        np.isfinite(open_loop.real).all()
        and np.isfinite(open_loop.imag).all()
        and np.isfinite(closed.real).all()
        and np.isfinite(closed.imag).all()
        and np.isfinite(poles.real).all()
        and np.isfinite(poles.imag).all()
    )
    maximum_pole_magnitude = float(np.max(np.abs(poles)))
    return {
        "metric_valid": bool(finite and maximum_pole_magnitude < 1.0 - 1e-10),
        "bandwidth_hz": bandwidth_hz,
        "bandwidth_found": bandwidth_found,
        "gain_margin_db": gain_margin_db,
        "phase_margin_deg": phase_margin_deg,
        "maximum_pole_magnitude": maximum_pole_magnitude,
    }


def _time_metrics(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    parameters: np.ndarray,
) -> dict[str, float | bool]:
    trace = simulate_scenario(config, motor, parameters, "current")
    return {
        "time_domain_stable": bool(not trace.terminated),
        **_reference_metrics(trace),
    }


def _frequency_pass(row: dict[str, Any], target: Any) -> bool:
    return bool(
        row["metric_valid"]
        and row["bandwidth_found"]
        and abs(float(row["bandwidth_hz"]) / target.bandwidth_hz - 1.0) <= 0.1
        and float(row["gain_margin_db"]) >= target.minimum_gain_margin_db
        and float(row["phase_margin_deg"]) >= target.minimum_phase_margin_deg
    )


def _time_pass(row: dict[str, Any], target: Any) -> bool:
    return bool(
        row["time_domain_stable"]
        and row["reference_metric_valid"]
        and row["reached_90_percent"]
        and row["settled"]
        and float(row["overshoot_ratio"]) <= target.maximum_overshoot_ratio
        and float(row["rise_time_s"]) <= target.maximum_rise_time_s
        and float(row["settling_time_s"]) <= target.maximum_settling_time_s
    )


def _constraint_violation(
    frequency: dict[str, Any], time_domain: dict[str, Any], target: Any
) -> float:
    if not frequency["metric_valid"] or not time_domain["reference_metric_valid"]:
        return 100.0
    values = [
        max(
            0.0,
            abs(float(frequency["bandwidth_hz"]) / target.bandwidth_hz - 1.0) - 0.1,
        )
        / 0.1,
        max(
            0.0,
            (target.minimum_gain_margin_db - float(frequency["gain_margin_db"]))
            / target.minimum_gain_margin_db,
        ),
        max(
            0.0,
            (target.minimum_phase_margin_deg - float(frequency["phase_margin_deg"]))
            / target.minimum_phase_margin_deg,
        ),
        max(
            0.0,
            float(time_domain["overshoot_ratio"]) / target.maximum_overshoot_ratio
            - 1.0,
        ),
        max(
            0.0,
            float(time_domain["rise_time_s"]) / target.maximum_rise_time_s - 1.0,
        ),
        max(
            0.0,
            float(time_domain["settling_time_s"]) / target.maximum_settling_time_s
            - 1.0,
        ),
        0.0 if time_domain["reached_90_percent"] else 1.0,
        0.0 if time_domain["settled"] else 1.0,
        0.0 if time_domain["time_domain_stable"] else 100.0,
    ]
    return float(max(values))


def _smooth_cost(
    frequency: dict[str, Any], time_domain: dict[str, Any], targets: Any
) -> float:
    target = targets.loop("current")
    settings = targets.cost
    frequency_errors = frequency_normalized_errors(
        bandwidth_hz=float(frequency["bandwidth_hz"]),
        gain_margin_db=float(frequency["gain_margin_db"]),
        phase_margin_deg=float(frequency["phase_margin_deg"]),
        target=target,
        bandwidth_relative_scale=settings.bandwidth_relative_scale,
        invalid_error=settings.invalid_normalized_error,
    )
    time_errors = time_normalized_errors(
        overshoot_ratio=float(time_domain["overshoot_ratio"]),
        rise_time_s=float(time_domain["rise_time_s"]),
        settling_time_s=float(time_domain["settling_time_s"]),
        target=target,
        invalid_error=settings.invalid_normalized_error,
    )
    return float(
        settings.frequency_weight
        * metric_cost(
            frequency_errors, FREQUENCY_ERROR_ORDER, delta=settings.huber_delta
        )
        + settings.time_weight
        * metric_cost(time_errors, TIME_ERROR_ORDER, delta=settings.huber_delta)
    )


def _evaluate_group(
    config: PhysicsMotorConfig,
    motors: Iterable[tuple[str, MotorParameters]],
    parameters: np.ndarray,
    targets: Any,
    *,
    frequency_points: int,
) -> dict[str, Any]:
    target = targets.loop("current")
    rows: list[dict[str, Any]] = []
    for model_id, motor in motors:
        frequency = _frequency_metrics(
            config, motor, parameters, points=frequency_points
        )
        time_domain = _time_metrics(config, motor, parameters)
        frequency_pass = _frequency_pass(frequency, target)
        time_pass = _time_pass(time_domain, target)
        rows.append(
            {
                "model_id": model_id,
                "frequency": frequency,
                "time_domain": time_domain,
                "frequency_pass": frequency_pass,
                "time_pass": time_pass,
                "all_six_pass": bool(frequency_pass and time_pass),
                "constraint_violation": _constraint_violation(
                    frequency, time_domain, target
                ),
                "smooth_cost": _smooth_cost(frequency, time_domain, targets),
            }
        )
    if not rows:
        raise ValueError("cannot audit an empty motor group")

    def worst(key: str, *, maximum: bool) -> dict[str, Any]:
        return dict(
            max(rows, key=lambda row: float(row["frequency"][key]))
            if maximum
            else min(rows, key=lambda row: float(row["frequency"][key]))
        )

    return {
        "model_count": len(rows),
        "frequency_pass_count": sum(bool(row["frequency_pass"]) for row in rows),
        "time_pass_count": sum(bool(row["time_pass"]) for row in rows),
        "all_six_pass_count": sum(bool(row["all_six_pass"]) for row in rows),
        "all_six_pass": bool(all(bool(row["all_six_pass"]) for row in rows)),
        "maximum_constraint_violation": max(
            float(row["constraint_violation"]) for row in rows
        ),
        "mean_smooth_cost": float(np.mean([row["smooth_cost"] for row in rows])),
        "worst_phase_margin": worst("phase_margin_deg", maximum=False),
        "worst_gain_margin": worst("gain_margin_db", maximum=False),
        "minimum_bandwidth": worst("bandwidth_hz", maximum=False),
        "maximum_bandwidth": worst("bandwidth_hz", maximum=True),
        "failed_models": [row for row in rows if not row["all_six_pass"]][:20],
    }


def _target_seed(config: PhysicsMotorConfig, target_hz: float) -> np.ndarray:
    omega = 2.0 * np.pi * target_hz
    motor = config.nominal
    correction = math.sqrt(1.0 + (omega * motor.current_delay_s) ** 2)
    kp = motor.inductance_h * omega * correction
    ki = motor.resistance_ohm * omega * correction
    kd = (
        float(config.payload["controller_design"]["derivative_ratio_at_crossover"])
        * kp
        / omega
    )
    return np.asarray([kp, ki, max(kd, 1e-8)], dtype=np.float64)


def _candidate_rank(audit: dict[str, Any]) -> tuple[float, ...]:
    groups = [audit["groups"][name] for name in SELECTION_GROUP_NAMES]
    failures = sum(
        int(group["model_count"] - group["all_six_pass_count"])
        for group in groups
    )
    return (
        float(failures),
        max(float(group["maximum_constraint_violation"]) for group in groups),
        float(sum(group["mean_smooth_cost"] for group in groups)),
    )


def run_sample_rate_scan(
    *,
    project_root: Path,
    sample_rate_hz: float,
    seed: int,
    maxiter: int,
    popsize: int,
    audit_top: int,
    output_dir: Path,
) -> dict[str, Any]:
    base_config = load_physics_motor_config(project_root)
    config = config_for_sample_rate(base_config, sample_rate_hz)
    space = load_physics_controller_parameter_space(project_root)
    targets = load_controller_performance_targets(project_root)
    target = targets.loop("current")
    base_parameters = space.initial.copy()
    training, validation = ensemble_current_motors(project_root)
    search_frequency_motors = [
        ("nominal", config.nominal),
        *current_corner_motors(config),
        *training,
    ]
    search_time_motors = [("nominal", config.nominal), *current_corner_motors(config)]
    groups = {
        "nominal": [("nominal", config.nominal)],
        "corners_8": current_corner_motors(config),
        "dense_grid_125": current_uncertainty_grid(config, levels=5),
        "training_40": training,
        "validation_16": validation,
    }
    evaluations = 0
    best_value = float("inf")

    def objective(log_values: np.ndarray) -> float:
        nonlocal evaluations, best_value
        evaluations += 1
        parameters = _parameters(base_parameters, np.power(10.0, log_values))
        row_costs: list[float] = []
        violations: list[float] = []
        try:
            for _, motor in search_frequency_motors:
                frequency = _frequency_metrics(config, motor, parameters, points=320)
                frequency_errors = frequency_normalized_errors(
                    bandwidth_hz=float(frequency["bandwidth_hz"]),
                    gain_margin_db=float(frequency["gain_margin_db"]),
                    phase_margin_deg=float(frequency["phase_margin_deg"]),
                    target=target,
                    bandwidth_relative_scale=targets.cost.bandwidth_relative_scale,
                    invalid_error=targets.cost.invalid_normalized_error,
                )
                row_costs.append(
                    targets.cost.frequency_weight
                    * metric_cost(
                        frequency_errors,
                        FREQUENCY_ERROR_ORDER,
                        delta=targets.cost.huber_delta,
                    )
                )
                dummy_time = {
                    "reference_metric_valid": True,
                    "overshoot_ratio": 0.0,
                    "rise_time_s": 0.0,
                    "settling_time_s": 0.0,
                    "reached_90_percent": True,
                    "settled": True,
                    "time_domain_stable": True,
                }
                violations.append(_constraint_violation(frequency, dummy_time, target))
            for _, motor in search_time_motors:
                frequency = _frequency_metrics(config, motor, parameters, points=320)
                time_domain = _time_metrics(config, motor, parameters)
                time_errors = time_normalized_errors(
                    overshoot_ratio=float(time_domain["overshoot_ratio"]),
                    rise_time_s=float(time_domain["rise_time_s"]),
                    settling_time_s=float(time_domain["settling_time_s"]),
                    target=target,
                    invalid_error=targets.cost.invalid_normalized_error,
                )
                row_costs.append(
                    targets.cost.time_weight
                    * metric_cost(
                        time_errors,
                        TIME_ERROR_ORDER,
                        delta=targets.cost.huber_delta,
                    )
                )
                violations.append(_constraint_violation(frequency, time_domain, target))
            value = float(
                np.mean(row_costs)
                + max(row_costs)
                + 1000.0 * max(violations)
                + 100.0 * float(np.mean(violations))
            )
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            value = 1e12
        best_value = min(best_value, value)
        return value

    generation = 0

    def progress(_candidate: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        print(
            f"[fs={sample_rate_hz:g}Hz] generation={generation} "
            f"evaluations={evaluations} best={best_value:.8f} "
            f"convergence={convergence:.6g}",
            flush=True,
        )
        return False

    x0 = np.log10(_target_seed(config, target.bandwidth_hz))
    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=SEARCH_LOG10_BOUNDS,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        atol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        callback=progress,
        polish=False,
        init="latinhypercube",
        x0=x0,
        workers=1,
        updating="immediate",
    )
    elapsed_s = time.perf_counter() - started
    optimizer_evaluations = evaluations
    population_order = np.argsort(np.asarray(result.population_energies))
    pool = [x0, np.asarray(result.x, dtype=np.float64)]
    pool.extend(
        np.asarray(result.population, dtype=np.float64)[
            population_order[: min(audit_top, len(population_order))]
        ]
    )
    unique: list[np.ndarray] = []
    for values in pool:
        if not any(np.allclose(values, old, rtol=1e-12, atol=1e-14) for old in unique):
            unique.append(np.asarray(values, dtype=np.float64).copy())

    audited: list[dict[str, Any]] = []
    for log_values in unique:
        current_values = np.power(10.0, log_values)
        parameters = _parameters(base_parameters, current_values)
        audit = {
            "current_parameters": dict(zip(CURRENT_PARAMETER_NAMES, current_values)),
            "search_objective": float(objective(log_values)),
            "groups": {
                name: _evaluate_group(
                    config,
                    motors,
                    parameters,
                    targets,
                    frequency_points=2048,
                )
                for name, motors in groups.items()
            },
        }
        audit["rank"] = list(_candidate_rank(audit))
        audited.append(audit)
    audited.sort(key=_candidate_rank)
    selected = audited[0]
    selected_values = np.asarray(
        [selected["current_parameters"][name] for name in CURRENT_PARAMETER_NAMES]
    )
    selected_parameters = _parameters(base_parameters, selected_values)
    rate_output = output_dir / f"fs_{int(round(sample_rate_hz))}_hz"
    rate_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        rate_output / "diagnostic_candidate.npz",
        parameter_names=np.asarray(space.names),
        parameters=selected_parameters,
        sample_rate_hz=np.asarray(sample_rate_hz),
        sample_period_s=np.asarray(config.sample_period_s_for("current")),
        hardware_use_allowed=np.asarray(False),
        formal_training_allowed=np.asarray(False),
    )
    report = {
        "sample_rate_hz": float(sample_rate_hz),
        "sample_period_s": config.sample_period_s_for("current"),
        "sample_periods_s": config.sample_periods_s,
        "nyquist_hz": 0.5 / config.sample_period_s_for("current"),
        "frequency_grid_upper_hz": float(current_frequency_grid(config, 64)[-1]),
        "fixed_current_delay_s": config.nominal.current_delay_s,
        "fixed_current_derivative_filter_s": config.derivative_filter_s["current"],
        "search": {
            "algorithm": "differential_evolution_best1bin_log10_current_pidf",
            "variables": list(CURRENT_PARAMETER_NAMES),
            "log10_bounds": [list(values) for values in SEARCH_LOG10_BOUNDS],
            "seed": seed,
            "maxiter": maxiter,
            "popsize_multiplier": popsize,
            "objective_evaluations": optimizer_evaluations,
            "elapsed_s": elapsed_s,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "best_objective": float(result.fun),
            "audited_candidate_count": len(audited),
            "candidate_selection_groups": list(SELECTION_GROUP_NAMES),
            "validation_used_for_candidate_selection": False,
        },
        "selected": selected,
        "all_audited_candidates": audited,
    }
    (rate_output / "sample_rate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_multirate_scan(
    *,
    project_root: Path,
    sample_rates_hz: list[float],
    seed: int,
    maxiter: int,
    popsize: int,
    audit_top: int,
    output_dir: Path,
) -> dict[str, Any]:
    if not sample_rates_hz or len(set(sample_rates_hz)) != len(sample_rates_hz):
        raise ValueError("sample rates must be nonempty and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for offset, sample_rate_hz in enumerate(sample_rates_hz):
        reports.append(
            run_sample_rate_scan(
                project_root=project_root,
                sample_rate_hz=sample_rate_hz,
                seed=seed + offset,
                maxiter=maxiter,
                popsize=popsize,
                audit_top=audit_top,
                output_dir=output_dir,
            )
        )
    summary = {
        "schema_version": 1,
        "run_kind": "non_training_multirate_current_pidf_feasibility_scan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_configuration_modified": False,
        "sac_training_executed": False,
        "fixed_physics": {
            "inductance_h": 0.002707,
            "current_delay_s": 0.0002,
            "uncertainty": {"inductance": 0.05, "resistance": 0.05, "delay": 0.10},
        },
        "sample_period_override_scope": "current_loop_only_in_memory",
        "formal_parameter_space_modified": False,
        "coverage": {
            "search": "nominal plus 8 exact L/R/current-delay corners",
            "audit": "nominal, 8 corners, 5^3 dense uncertainty grid, 40 training and 16 validation models",
        },
        "sample_rates": reports,
        "hardware_use_allowed": False,
    }
    (output_dir / "multirate_feasibility_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-training current-PIDF feasibility scan over digital sample "
            "rates while keeping the supplied physical delays and uncertainty fixed."
        )
    )
    parser.add_argument(
        "--sample-rates-hz",
        type=float,
        nargs="+",
        default=[40000.0],
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--audit-top", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "multirate_current_feasibility_40khz_20260821"
        ),
    )
    arguments = parser.parse_args()
    report = run_multirate_scan(
        project_root=PROJECT_ROOT,
        sample_rates_hz=[float(value) for value in arguments.sample_rates_hz],
        seed=arguments.seed,
        maxiter=arguments.maxiter,
        popsize=arguments.popsize,
        audit_top=arguments.audit_top,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "report": str(
                    (
                        arguments.output_dir / "multirate_feasibility_report.json"
                    ).resolve()
                ),
                "sample_rates": [
                    {
                        "sample_rate_hz": row["sample_rate_hz"],
                        "selected_parameters": row["selected"]["current_parameters"],
                        "rank": row["selected"]["rank"],
                        "group_pass": {
                            name: group["all_six_pass"]
                            for name, group in row["selected"]["groups"].items()
                        },
                    }
                    for row in report["sample_rates"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
