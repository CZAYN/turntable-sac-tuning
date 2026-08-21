from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    ControllerParameterSpace,
)
from elc_rl.discrete_loop_model import build_discrete_loop_model  # noqa: E402
from elc_rl.performance_targets import (  # noqa: E402
    FREQUENCY_ERROR_ORDER,
    aggregate_model_costs,
    frequency_normalized_errors,
    metric_cost,
)
from elc_rl.physics_evaluator import (  # noqa: E402
    PHYSICS_TRAIN_FREQUENCY_POINTS,
    PhysicsTimeDomainEvaluator,
    _evaluate_open_loop,
    get_physics_controller_evaluator,
)
from elc_rl.tuning_env import combined_stage_cost  # noqa: E402


CURRENT_PARAMETER_INDICES = (8, 9, 10)
CURRENT_PARAMETER_NAMES = ("kpcurr", "kicurr", "kdcurr")


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
        raise ValueError("base candidate does not match the project parameter space")
    if not np.isfinite(parameters).all():
        raise ValueError("base candidate contains a non-finite parameter")
    return parameters


def _extended_space(
    canonical: ControllerParameterSpace,
    temporary_kicurr_lower: float,
    temporary_kdcurr_upper: float | None,
) -> ControllerParameterSpace:
    specs = []
    for spec in canonical.specs:
        if spec.name == "kicurr":
            spec = replace(spec, lower=temporary_kicurr_lower)
        elif spec.name == "kdcurr" and temporary_kdcurr_upper is not None:
            spec = replace(spec, upper=temporary_kdcurr_upper)
        specs.append(spec)
    space = ControllerParameterSpace(
        task_id=canonical.task_id,
        specs=tuple(specs),
        metadata={
            **canonical.metadata,
            "temporary_feasibility_override": {
                "kicurr_lower": temporary_kicurr_lower,
                "kdcurr_upper": temporary_kdcurr_upper,
            },
        },
    )
    space.validate()
    return space


def _parameters_from_current_normalized(
    space: ControllerParameterSpace,
    base_normalized: np.ndarray,
    current_normalized: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(base_normalized, dtype=np.float64).copy()
    normalized[list(CURRENT_PARAMETER_INDICES)] = np.asarray(
        current_normalized, dtype=np.float64
    )
    return space.denormalize(normalized)


def _current_frequency_pass(row: dict[str, Any], target: Any) -> bool:
    return bool(
        row["metric_valid"]
        and abs(float(row["bandwidth_hz"]) / target.bandwidth_hz - 1.0) <= 0.1
        and float(row["gain_margin_db"]) >= target.minimum_gain_margin_db
        and float(row["phase_margin_deg"]) >= target.minimum_phase_margin_deg
    )


def _current_time_pass(row: dict[str, Any], target: Any) -> bool:
    return bool(
        row["time_domain_stable"]
        and row["reference_metric_valid"]
        and row["reached_90_percent"]
        and row["settled"]
        and float(row["overshoot_ratio"]) <= target.maximum_overshoot_ratio
        and float(row["rise_time_s"]) <= target.maximum_rise_time_s
        and float(row["settling_time_s"]) <= target.maximum_settling_time_s
    )


def _fast_current_frequency(
    evaluator: Any, parameters: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate only the current loop while searching its three parameters."""

    target = evaluator.performance_targets.loop("current")
    settings = evaluator.performance_targets.cost
    rows: list[dict[str, Any]] = []
    costs: list[float] = []
    for raw_index in evaluator.training_indices:
        index = int(raw_index)
        model = build_discrete_loop_model(
            evaluator.config,
            evaluator.motor(index),
            parameters,
            "current",
        )
        row: dict[str, Any] = {
            "model_id": str(evaluator.ensemble["model_id"][index]),
            "role": str(evaluator.ensemble["role"][index]),
            **_evaluate_open_loop(
                "current",
                model,
                frequency_points=PHYSICS_TRAIN_FREQUENCY_POINTS,
            ),
        }
        errors = frequency_normalized_errors(
            bandwidth_hz=float(row["bandwidth_hz"]),
            gain_margin_db=float(row["gain_margin_db"]),
            phase_margin_deg=float(row["phase_margin_deg"]),
            target=target,
            bandwidth_relative_scale=settings.bandwidth_relative_scale,
            invalid_error=settings.invalid_normalized_error,
        )
        row["normalized_errors"] = errors
        row["frequency_cost"] = metric_cost(
            errors, FREQUENCY_ERROR_ORDER, delta=settings.huber_delta
        )
        costs.append(float(row["frequency_cost"]))
        rows.append(row)
    return (
        {
            "valid": bool(all(bool(row["metric_valid"]) for row in rows)),
            "frequency_cost": aggregate_model_costs(costs, settings),
        },
        rows,
    )


def _domain_summary(report: dict[str, Any], domain: str) -> dict[str, Any]:
    validation = report["validation_diagnostics"]
    target_key = f"all_{domain}_targets_met"
    validity_key = f"{domain}_metrics_valid"
    return {
        "training_valid": bool(report["safety"]["safe"]),
        "training_targets_met": bool(report["safety"][target_key]),
        "validation_valid": bool(validation["safety"][validity_key]),
        "validation_targets_met": bool(validation["safety"][target_key]),
        "training_metrics": report["performance_metrics"],
        "validation_metrics": validation["performance_metrics"],
    }


def _audit_candidate(
    frequency_evaluator: Any,
    time_evaluator: PhysicsTimeDomainEvaluator,
    parameters: np.ndarray,
) -> dict[str, Any]:
    frequency = frequency_evaluator.audit(parameters, include_models=True)
    time_domain = time_evaluator.full_audit(parameters, include_models=True)
    target = frequency_evaluator.performance_targets.loop("current")
    frequency_rows = [
        row for row in frequency["models"] if row["loop"] == "current"
    ]
    time_rows = [row for row in time_domain["models"] if row["loop"] == "current"]

    def pass_counts(rows: list[dict[str, Any]], predicate: Any) -> dict[str, int]:
        training = [row for row in rows if row["role"] != "validation"]
        validation = [row for row in rows if row["role"] == "validation"]
        return {
            "training_pass": sum(predicate(row, target) for row in training),
            "training_count": len(training),
            "validation_pass": sum(predicate(row, target) for row in validation),
            "validation_count": len(validation),
        }

    frequency_summary = _domain_summary(frequency, "frequency")
    time_summary = _domain_summary(time_domain, "time")
    training_all_six = bool(
        frequency_summary["training_targets_met"]
        and time_summary["training_targets_met"]
    )
    validation_all_six = bool(
        frequency_summary["validation_targets_met"]
        and time_summary["validation_targets_met"]
    )
    return {
        "parameters": {
            name: float(value)
            for name, value in zip(frequency_evaluator.space.names, parameters)
        },
        "joint_cost": float(combined_stage_cost(frequency, time_domain, "joint")),
        "training_all_six_targets_met": training_all_six,
        "validation_all_six_targets_met": validation_all_six,
        "current_frequency_pass_counts": pass_counts(
            frequency_rows, _current_frequency_pass
        ),
        "current_time_pass_counts": pass_counts(time_rows, _current_time_pass),
        "frequency": frequency_summary,
        "time_domain": time_summary,
    }


def run_scan(
    *,
    project_root: Path,
    base_candidate_path: Path,
    temporary_kicurr_lower: float,
    temporary_kdcurr_upper: float | None,
    seed: int,
    maxiter: int,
    popsize: int,
    audit_top: int,
    constraint_weight: float,
    phase_buffer_deg: float,
    time_probe_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    if temporary_kicurr_lower <= 0.0:
        raise ValueError("temporary kicurr lower bound must be positive")
    if maxiter <= 0 or popsize <= 0 or audit_top <= 0 or time_probe_count <= 0:
        raise ValueError(
            "maxiter, popsize, audit_top and time_probe_count must be positive"
        )
    if constraint_weight <= 0.0 or phase_buffer_deg < 0.0:
        raise ValueError("constraint weight must be positive and phase buffer nonnegative")

    frequency_evaluator = get_physics_controller_evaluator(project_root)
    canonical_space = frequency_evaluator.space
    canonical_kicurr = canonical_space.specs[9]
    canonical_kdcurr = canonical_space.specs[10]
    if temporary_kicurr_lower >= canonical_kicurr.lower:
        raise ValueError("temporary kicurr lower bound must be below the canonical one")
    if (
        temporary_kdcurr_upper is not None
        and temporary_kdcurr_upper <= canonical_kdcurr.upper
    ):
        raise ValueError("temporary kdcurr upper bound must exceed the canonical one")
    extended_space = _extended_space(
        canonical_space,
        temporary_kicurr_lower,
        temporary_kdcurr_upper,
    )
    frequency_evaluator.space = extended_space
    time_evaluator = PhysicsTimeDomainEvaluator(frequency_evaluator)
    probe_positions = np.linspace(
        0,
        frequency_evaluator.training_indices.size - 1,
        min(time_probe_count, frequency_evaluator.training_indices.size),
        dtype=np.int64,
    )
    time_probe_indices = frequency_evaluator.training_indices[probe_positions]

    base_parameters = _load_candidate(base_candidate_path, extended_space.names)
    extended_space.normalize(base_parameters)
    base_normalized = extended_space.normalize(base_parameters)
    current_x0 = base_normalized[list(CURRENT_PARAMETER_INDICES)]
    target = frequency_evaluator.performance_targets.loop("current")
    evaluations = 0
    best_value = float("inf")

    def objective(current_normalized: np.ndarray) -> float:
        nonlocal evaluations, best_value
        evaluations += 1
        parameters = _parameters_from_current_normalized(
            extended_space, base_normalized, current_normalized
        )
        try:
            current, rows = _fast_current_frequency(
                frequency_evaluator, parameters
            )
            bandwidth_violation = max(
                max(
                    0.0,
                    abs(float(row["bandwidth_hz"]) / target.bandwidth_hz - 1.0)
                    - 0.1,
                )
                / 0.1
                for row in rows
            )
            phase_violation = max(
                max(
                    0.0,
                    (
                        target.minimum_phase_margin_deg
                        + phase_buffer_deg
                        - float(row["phase_margin_deg"])
                    )
                    / target.minimum_phase_margin_deg,
                )
                for row in rows
            )
            gain_violation = max(
                max(
                    0.0,
                    (target.minimum_gain_margin_db - float(row["gain_margin_db"]))
                    / target.minimum_gain_margin_db,
                )
                for row in rows
            )
            time_report = time_evaluator.evaluate(
                parameters,
                time_probe_indices,
                mode="feasibility_fast_time",
                include_models=True,
            )
            current_time = time_report["performance_metrics"]["current"]
            time_rows = [
                row
                for row in time_report["models"]
                if row["loop"] == "current"
            ]
            time_violation = max(
                max(
                    float(row["normalized_errors"]["overshoot"]),
                    float(row["normalized_errors"]["rise_time"]),
                    float(row["normalized_errors"]["settling_time"]),
                    0.0 if row["reached_90_percent"] and row["settled"] else 1.0,
                )
                for row in time_rows
            )
            invalid = not bool(current["valid"] and current_time["valid"])
            value = float(
                current["frequency_cost"]
                + current_time["time_cost"]
                + constraint_weight
                * (
                    bandwidth_violation**2
                    + phase_violation**2
                    + gain_violation**2
                    + time_violation**2
                )
                + (1000.0 if invalid else 0.0)
            )
        except (FloatingPointError, OverflowError, ValueError):
            value = 1e6
        best_value = min(best_value, value)
        return value

    generation = 0

    def progress(_candidate: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        print(
            f"generation={generation} evaluations={evaluations} "
            f"best_objective={best_value:.8f} convergence={convergence:.6g}",
            flush=True,
        )
        return False

    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=[(-1.0, 1.0)] * len(CURRENT_PARAMETER_INDICES),
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
        x0=current_x0,
        workers=1,
        updating="immediate",
    )
    elapsed_s = time.perf_counter() - started
    optimizer_evaluations = evaluations

    pool = [current_x0, np.asarray(result.x, dtype=np.float64)]
    order = np.argsort(np.asarray(result.population_energies))[:audit_top]
    pool.extend(np.asarray(result.population, dtype=np.float64)[order])
    unique: list[np.ndarray] = []
    for values in pool:
        if not any(np.allclose(values, old, rtol=1e-12, atol=1e-14) for old in unique):
            unique.append(np.asarray(values, dtype=np.float64).copy())

    audited = []
    for values in unique:
        parameters = _parameters_from_current_normalized(
            extended_space, base_normalized, values
        )
        audited.append(
            {
                "source": (
                    "base"
                    if np.allclose(values, current_x0, rtol=1e-12, atol=1e-14)
                    else "optimizer_pool"
                ),
                "search_objective": float(objective(values)),
                "audit": _audit_candidate(
                    frequency_evaluator, time_evaluator, parameters
                ),
            }
        )
    base_audit = next(row for row in audited if row["source"] == "base")
    audited.sort(key=lambda row: float(row["audit"]["joint_cost"]))
    fully_feasible = [
        row
        for row in audited
        if row["audit"]["training_all_six_targets_met"]
        and row["audit"]["validation_all_six_targets_met"]
    ]
    current_feasible = [
        row
        for row in audited
        if row["audit"]["current_frequency_pass_counts"]["training_pass"]
        == row["audit"]["current_frequency_pass_counts"]["training_count"]
        and row["audit"]["current_frequency_pass_counts"]["validation_pass"]
        == row["audit"]["current_frequency_pass_counts"]["validation_count"]
    ]
    selected = (
        min(fully_feasible, key=lambda row: float(row["audit"]["joint_cost"]))
        if fully_feasible
        else (
            min(current_feasible, key=lambda row: float(row["audit"]["joint_cost"]))
            if current_feasible
            else min(audited, key=lambda row: float(row["search_objective"]))
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "run_kind": "isolated_current_loop_parameter_space_feasibility_scan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_parameter_space_unchanged": True,
        "base_candidate": {
            "path": str(base_candidate_path.resolve()),
            "sha256": _sha256(base_candidate_path),
        },
        "temporary_override": {
            "kicurr": {
                "canonical_lower": canonical_kicurr.lower,
                "temporary_lower": temporary_kicurr_lower,
                "canonical_upper": canonical_kicurr.upper,
            },
            "kdcurr": {
                "canonical_lower": canonical_kdcurr.lower,
                "canonical_upper": canonical_kdcurr.upper,
                "temporary_upper": (
                    canonical_kdcurr.upper
                    if temporary_kdcurr_upper is None
                    else temporary_kdcurr_upper
                ),
            },
        },
        "search": {
            "algorithm": "scipy.optimize.differential_evolution best1bin",
            "variables": list(CURRENT_PARAMETER_NAMES),
            "seed": seed,
            "maxiter": maxiter,
            "popsize_multiplier": popsize,
            "constraint_weight": constraint_weight,
            "training_phase_margin_buffer_deg": phase_buffer_deg,
            "time_probe_model_ids": list(
                frequency_evaluator.model_ids(time_probe_indices)
            ),
            "objective_evaluations": optimizer_evaluations,
            "elapsed_s": elapsed_s,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "best_objective": float(result.fun),
        },
        "audit": {
            "audited_candidate_count": len(audited),
            "fully_feasible_candidate_count": len(fully_feasible),
            "current_frequency_feasible_candidate_count": len(current_feasible),
            "base": base_audit,
            "candidates": audited,
            "selected": selected,
        },
        "hardware_use_allowed": False,
    }
    report_path = output_dir / "current_loop_feasibility_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected_parameters = np.asarray(
        [
            selected["audit"]["parameters"][name]
            for name in extended_space.names
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        output_dir / "current_loop_feasibility_candidate.npz",
        parameter_names=np.asarray(extended_space.names),
        parameters=selected_parameters,
        training_all_six_targets_met=np.asarray(
            selected["audit"]["training_all_six_targets_met"]
        ),
        validation_all_six_targets_met=np.asarray(
            selected["audit"]["validation_all_six_targets_met"]
        ),
        temporary_kicurr_lower=np.asarray(temporary_kicurr_lower),
        temporary_kdcurr_upper=np.asarray(
            canonical_kdcurr.upper
            if temporary_kdcurr_upper is None
            else temporary_kdcurr_upper
        ),
        hardware_use_allowed=np.asarray(False),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily lower kicurr in memory and scan Kp/Ki/Kd for the "
            "1500 Hz current-loop bandwidth and 65 degree phase-margin targets."
        )
    )
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--temporary-kicurr-lower", type=float, default=1.0)
    parser.add_argument("--temporary-kdcurr-upper", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--audit-top", type=int, default=12)
    parser.add_argument("--constraint-weight", type=float, default=1000.0)
    parser.add_argument("--phase-buffer-deg", type=float, default=0.0)
    parser.add_argument("--time-probe-count", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "current_loop_feasibility_20260820",
    )
    arguments = parser.parse_args()
    report = run_scan(
        project_root=PROJECT_ROOT,
        base_candidate_path=arguments.base_candidate,
        temporary_kicurr_lower=arguments.temporary_kicurr_lower,
        temporary_kdcurr_upper=arguments.temporary_kdcurr_upper,
        seed=arguments.seed,
        maxiter=arguments.maxiter,
        popsize=arguments.popsize,
        audit_top=arguments.audit_top,
        constraint_weight=arguments.constraint_weight,
        phase_buffer_deg=arguments.phase_buffer_deg,
        time_probe_count=arguments.time_probe_count,
        output_dir=arguments.output_dir,
    )
    selected = report["audit"]["selected"]["audit"]
    print(
        json.dumps(
            {
                "report": str(
                    (arguments.output_dir / "current_loop_feasibility_report.json")
                    .resolve()
                ),
                "fully_feasible_candidate_count": report["audit"][
                    "fully_feasible_candidate_count"
                ],
                "current_frequency_feasible_candidate_count": report["audit"][
                    "current_frequency_feasible_candidate_count"
                ],
                "selected_joint_cost": selected["joint_cost"],
                "selected_training_all_six_targets_met": selected[
                    "training_all_six_targets_met"
                ],
                "selected_validation_all_six_targets_met": selected[
                    "validation_all_six_targets_met"
                ],
                "selected_current_parameters": {
                    name: selected["parameters"][name]
                    for name in CURRENT_PARAMETER_NAMES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
