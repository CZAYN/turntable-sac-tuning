from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from elc_rl.controller_parameters import (  # noqa: E402
    load_physics_controller_parameter_space,
)
from elc_rl.performance_targets import (  # noqa: E402
    load_controller_performance_targets,
)
from elc_rl.physics_motor_model import MotorParameters, load_physics_motor_config  # noqa: E402
from scripts.scan_multirate_current_feasibility import (  # noqa: E402
    CURRENT_PARAMETER_NAMES,
    _constraint_violation,
    _frequency_metrics,
    _parameters,
    _time_metrics,
    config_for_sample_rate,
)


def motor_at_normalized_uncertainty(
    nominal: MotorParameters,
    uncertainty: dict[str, float],
    normalized: np.ndarray,
) -> MotorParameters:
    values = np.asarray(normalized, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("normalized uncertainty must be one finite three-vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("normalized uncertainty must stay inside [-1, 1]")
    from dataclasses import replace

    return replace(
        nominal,
        inductance_h=nominal.inductance_h
        * (1.0 + float(values[0]) * uncertainty["inductance_h"]),
        resistance_ohm=nominal.resistance_ohm
        * (1.0 + float(values[1]) * uncertainty["resistance_ohm"]),
        current_delay_s=nominal.current_delay_s
        * (1.0 + float(values[2]) * uncertainty["current_delay_s"]),
    )


def _run_extremum(
    objective: Callable[[np.ndarray], float],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> tuple[np.ndarray, float, int]:
    evaluations = 0

    def counted(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            result = float(objective(values))
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            result = 1e12
        return result if np.isfinite(result) else 1e12

    result = differential_evolution(
        counted,
        bounds=[(-1.0, 1.0)] * 3,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        atol=1e-9,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=True,
        init="latinhypercube",
        workers=1,
        updating="immediate",
    )
    return np.asarray(result.x, dtype=np.float64), float(result.fun), evaluations


def refine_rate(
    *,
    project_root: Path,
    rate_report: dict[str, Any],
    seed: int,
    maxiter: int,
    popsize: int,
) -> dict[str, Any]:
    sample_rate_hz = float(rate_report["sample_rate_hz"])
    base_config = load_physics_motor_config(project_root)
    config = config_for_sample_rate(base_config, sample_rate_hz)
    targets = load_controller_performance_targets(project_root)
    target = targets.loop("current")
    space = load_physics_controller_parameter_space(project_root)
    current_values = np.asarray(
        [
            float(rate_report["selected"]["current_parameters"][name])
            for name in CURRENT_PARAMETER_NAMES
        ],
        dtype=np.float64,
    )
    parameters = _parameters(space.initial, current_values)
    uncertainty = config.uncertainty_fraction

    def metrics(values: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        motor = motor_at_normalized_uncertainty(config.nominal, uncertainty, values)
        frequency = _frequency_metrics(config, motor, parameters, points=4096)
        time_domain = _time_metrics(config, motor, parameters)
        return frequency, time_domain

    objectives: dict[str, Callable[[np.ndarray], float]] = {
        "minimum_bandwidth_hz": lambda values: float(
            metrics(values)[0]["bandwidth_hz"]
        ),
        "maximum_bandwidth_hz": lambda values: (
            -float(metrics(values)[0]["bandwidth_hz"])
        ),
        "minimum_gain_margin_db": lambda values: float(
            metrics(values)[0]["gain_margin_db"]
        ),
        "minimum_phase_margin_deg": lambda values: float(
            metrics(values)[0]["phase_margin_deg"]
        ),
        "maximum_pole_magnitude": lambda values: (
            -float(metrics(values)[0]["maximum_pole_magnitude"])
        ),
        "maximum_overshoot_ratio": lambda values: (
            -float(metrics(values)[1]["overshoot_ratio"])
        ),
        "maximum_rise_time_s": lambda values: -float(metrics(values)[1]["rise_time_s"]),
        "maximum_settling_time_s": lambda values: (
            -float(metrics(values)[1]["settling_time_s"])
        ),
        "maximum_six_metric_violation": lambda values: (
            -_constraint_violation(*metrics(values), target)
        ),
    }
    extrema: dict[str, Any] = {}
    total_evaluations = 0
    for offset, (name, objective) in enumerate(objectives.items()):
        normalized, raw_value, evaluations = _run_extremum(
            objective,
            seed=seed + offset,
            maxiter=maxiter,
            popsize=popsize,
        )
        total_evaluations += evaluations
        frequency, time_domain = metrics(normalized)
        value = -raw_value if name.startswith("maximum_") else raw_value
        extrema[name] = {
            "value": float(value),
            "normalized_uncertainty": {
                "inductance": float(normalized[0]),
                "resistance": float(normalized[1]),
                "current_delay": float(normalized[2]),
            },
            "physical_parameters": {
                "inductance_h": motor_at_normalized_uncertainty(
                    config.nominal, uncertainty, normalized
                ).inductance_h,
                "resistance_ohm": motor_at_normalized_uncertainty(
                    config.nominal, uncertainty, normalized
                ).resistance_ohm,
                "current_delay_s": motor_at_normalized_uncertainty(
                    config.nominal, uncertainty, normalized
                ).current_delay_s,
            },
            "frequency": frequency,
            "time_domain": time_domain,
            "objective_evaluations": evaluations,
        }
    frequency_pass = bool(
        extrema["minimum_bandwidth_hz"]["value"] >= 0.9 * target.bandwidth_hz
        and extrema["maximum_bandwidth_hz"]["value"] <= 1.1 * target.bandwidth_hz
        and extrema["minimum_gain_margin_db"]["value"] >= target.minimum_gain_margin_db
        and extrema["minimum_phase_margin_deg"]["value"]
        >= target.minimum_phase_margin_deg
    )
    all_six_pass = bool(
        frequency_pass
        and extrema["maximum_overshoot_ratio"]["value"]
        <= target.maximum_overshoot_ratio
        and extrema["maximum_rise_time_s"]["value"] <= target.maximum_rise_time_s
        and extrema["maximum_settling_time_s"]["value"]
        <= target.maximum_settling_time_s
        and extrema["maximum_six_metric_violation"]["value"] <= 1e-10
    )
    return {
        "sample_rate_hz": sample_rate_hz,
        "sample_period_s": config.sample_period_s_for("current"),
        "sample_periods_s": config.sample_periods_s,
        "current_parameters": dict(zip(CURRENT_PARAMETER_NAMES, current_values)),
        "continuous_box_search": {
            "variables": ["inductance", "resistance", "current_delay"],
            "normalized_bounds": [-1.0, 1.0],
            "maxiter": maxiter,
            "popsize_multiplier": popsize,
            "total_objective_evaluations": total_evaluations,
            "heuristic_not_mathematical_proof": True,
        },
        "extrema": extrema,
        "continuous_box_frequency_pass": frequency_pass,
        "continuous_box_all_six_pass": all_six_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously refine worst L/R/current-delay cases for multirate candidates."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-rates-hz", type=float, nargs="*")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    source = json.loads(arguments.report.read_text(encoding="utf-8"))
    selected_rates = (
        None
        if not arguments.sample_rates_hz
        else {float(value) for value in arguments.sample_rates_hz}
    )
    rows = [
        row
        for row in source["sample_rates"]
        if selected_rates is None or float(row["sample_rate_hz"]) in selected_rates
    ]
    if not rows:
        raise ValueError("no matching sample rates in source report")
    refinements = [
        refine_rate(
            project_root=PROJECT_ROOT,
            rate_report=row,
            seed=arguments.seed + index * 10,
            maxiter=arguments.maxiter,
            popsize=arguments.popsize,
        )
        for index, row in enumerate(rows)
    ]
    output = arguments.output or arguments.report.with_name(
        "continuous_worst_case_refinement.json"
    )
    report = {
        "schema_version": 1,
        "run_kind": "non_training_continuous_current_uncertainty_refinement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(arguments.report.resolve()),
        "formal_configuration_modified": False,
        "formal_parameter_space_modified": False,
        "sac_training_executed": False,
        "refinements": refinements,
        "hardware_use_allowed": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output.resolve()), "refinements": refinements},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
