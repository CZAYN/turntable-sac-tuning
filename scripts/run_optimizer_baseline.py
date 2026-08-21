from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import differential_evolution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.physics_evaluator import (  # noqa: E402
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.performance_targets import LOOP_ORDER  # noqa: E402
from elc_rl.tuning_env import combined_stage_cost  # noqa: E402


DEFAULT_SEED = 20260715


def _report_valid(report: dict[str, Any]) -> bool:
    """Return evaluator validity, which is distinct from meeting every target."""

    return bool(report.get("safety", {}).get("safe", False))


def _domain_targets_met(report: dict[str, Any], domain: str) -> bool:
    key = f"all_{domain}_targets_met"
    return bool(report.get("safety", {}).get(key, False))


def _loop_target_met(report: dict[str, Any], loop: str) -> bool:
    return bool(report["performance_metrics"][loop]["target_pass"])


def _validation_summary(
    frequency_report: dict[str, Any], time_report: dict[str, Any]
) -> dict[str, Any] | None:
    frequency = frequency_report.get("validation_diagnostics")
    time_domain = time_report.get("validation_diagnostics")
    if frequency is None or time_domain is None:
        return None
    loops: dict[str, Any] = {}
    for loop in LOOP_ORDER:
        frequency_loop = frequency["performance_metrics"][loop]
        time_loop = time_domain["performance_metrics"][loop]
        loops[loop] = {
            "frequency_actual": frequency_loop["actual"],
            "time_actual": time_loop["actual"],
            "target_pass": bool(
                frequency_loop["target_pass"] and time_loop["target_pass"]
            ),
        }
    return {
        "all_six_targets_met": bool(
            frequency["safety"]["all_frequency_targets_met"]
            and time_domain["safety"]["all_time_targets_met"]
        ),
        "frequency_cost": frequency["cost"],
        "time_cost": time_domain["cost"],
        "loops": loops,
    }


def _six_metric_audit(
    frequency_report: dict[str, Any], time_report: dict[str, Any]
) -> dict[str, Any]:
    """Project both evaluators onto the literal 3 loops x 6 metrics."""

    loops: dict[str, Any] = {}
    for loop in LOOP_ORDER:
        frequency = frequency_report["performance_metrics"][loop]
        time_domain = time_report["performance_metrics"][loop]
        loops[loop] = {
            "target": frequency["target"],
            "frequency_actual": frequency["actual"],
            "time_actual": time_domain["actual"],
            "normalized_errors": {
                **frequency["normalized_errors"],
                **time_domain["normalized_errors"],
            },
            "frequency_cost": float(frequency["frequency_cost"]),
            "time_cost": float(time_domain["time_cost"]),
            "combined_cost": float(
                0.5 * frequency["frequency_cost"]
                + 0.5 * time_domain["time_cost"]
            ),
            "valid": bool(frequency["valid"] and time_domain["valid"]),
            "target_pass": bool(
                _loop_target_met(frequency_report, loop)
                and _loop_target_met(time_report, loop)
            ),
        }
    valid = bool(_report_valid(frequency_report) and _report_valid(time_report))
    target_pass = bool(
        _domain_targets_met(frequency_report, "frequency")
        and _domain_targets_met(time_report, "time")
    )
    return {
        "valid": valid,
        "all_six_targets_met": target_pass,
        "cost": float(combined_stage_cost(frequency_report, time_report, "joint")),
        "loops": loops,
        "validation": _validation_summary(frequency_report, time_report),
        "audit_scope": {
            "frequency_model_count": int(frequency_report["evaluated_model_count"]),
            "time_model_count": int(time_report["evaluated_model_count"]),
            "frequency_mode": frequency_report["evaluation_mode"],
            "time_mode": time_report["evaluation_mode"],
        },
    }


def _load_candidate(path: Path, space: Any) -> np.ndarray:
    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"SAC candidate does not exist: {candidate_path}")
    with np.load(candidate_path, allow_pickle=False) as data:
        if "parameters" not in data.files:
            raise ValueError("SAC candidate archive is missing parameters")
        values = np.asarray(data["parameters"], dtype=np.float64)
        if "parameter_names" in data.files:
            names = tuple(str(value) for value in data["parameter_names"])
            if names != tuple(space.names):
                raise ValueError("SAC candidate parameter order does not match the project")
    if values.shape != (len(space.names),) or not np.isfinite(values).all():
        raise ValueError("SAC candidate parameters must be one finite 11-vector")
    space.normalize(values)
    return values


def run_optimizer_baseline(
    project_root: Path,
    *,
    seed: int,
    maxiter: int,
    popsize: int,
    output_dir: Path,
    sac_candidate_path: Path | None = None,
) -> dict[str, Any]:
    if maxiter <= 0 or popsize <= 0:
        raise ValueError("maxiter and popsize must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_evaluator = get_physics_controller_evaluator(project_root)
    time_evaluator = get_physics_time_domain_evaluator(project_root)
    space = frequency_evaluator.space
    rng = np.random.default_rng(seed)
    sampled_indices = frequency_evaluator.sample_training_indices(rng)
    evaluation_count = 0
    best_seen = float("inf")

    def fast_objective(normalized: np.ndarray) -> float:
        nonlocal evaluation_count, best_seen
        evaluation_count += 1
        try:
            parameters = space.denormalize(np.asarray(normalized, dtype=np.float64))
            frequency_report = frequency_evaluator.train(parameters, sampled_indices)
            time_report = time_evaluator.train(parameters, sampled_indices)
            cost = combined_stage_cost(frequency_report, time_report, "joint")
            valid = bool(
                _report_valid(frequency_report) and _report_valid(time_report)
            )
            value = float(cost if valid else 1000.0 + cost)
        except (FloatingPointError, ValueError, OverflowError):
            value = 1e6
        best_seen = min(best_seen, value)
        return value

    generation = 0

    def progress(_candidate: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        print(
            f"generation={generation} evaluations={evaluation_count} "
            f"best_fast_cost={best_seen:.6f} convergence={convergence:.6g}",
            flush=True,
        )
        return False

    initial_normalized = space.normalize(space.initial)
    start = time.perf_counter()
    result = differential_evolution(
        fast_objective,
        bounds=[(-1.0, 1.0)] * 11,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        atol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        callback=progress,
        disp=False,
        polish=False,
        init="latinhypercube",
        x0=initial_normalized,
        workers=1,
        updating="immediate",
    )
    elapsed_s = time.perf_counter() - start

    def full_audit(parameters: np.ndarray) -> dict[str, Any]:
        frequency_report = frequency_evaluator.audit(parameters)
        time_report = time_evaluator.full_audit(parameters)
        return _six_metric_audit(frequency_report, time_report)

    normalized_pool = [initial_normalized.copy(), np.asarray(result.x).copy()]
    order = np.argsort(np.asarray(result.population_energies))[:5]
    normalized_pool.extend(np.asarray(result.population)[order])
    unique_pool: list[np.ndarray] = []
    for values in normalized_pool:
        if not any(
            np.allclose(values, existing, rtol=1e-12, atol=1e-14)
            for existing in unique_pool
        ):
            unique_pool.append(np.asarray(values, dtype=np.float64).copy())

    audited_candidates = []
    for normalized in unique_pool:
        parameters = space.denormalize(np.clip(normalized, -1.0, 1.0))
        audited_candidates.append(
            {
                "fast_cost": fast_objective(normalized),
                "normalized_parameters": normalized.tolist(),
                "parameters": {
                    name: float(value) for name, value in zip(space.names, parameters)
                },
                "audit": full_audit(parameters),
            }
        )
    valid_candidates = [row for row in audited_candidates if row["audit"]["valid"]]
    if not valid_candidates:
        raise RuntimeError("differential evolution produced no numerically valid candidate")
    selected = min(valid_candidates, key=lambda row: float(row["audit"]["cost"]))
    selected_parameters = np.asarray(
        [selected["parameters"][name] for name in space.names], dtype=np.float64
    )
    baseline = audited_candidates[0]

    sac_comparison = None
    if sac_candidate_path is not None:
        sac_parameters = _load_candidate(sac_candidate_path, space)
        sac_comparison = {
            "candidate_path": str(Path(sac_candidate_path).resolve()),
            "parameters": {
                name: float(value) for name, value in zip(space.names, sac_parameters)
            },
            "audit": full_audit(sac_parameters),
        }

    report = {
        "schema_version": 2,
        "backend": "physics",
        "objective": "literal three-loop six-metric controller target table",
        "run_kind": (
            "deterministic differential-evolution simulation baseline; "
            "limited-budget, not final convergence"
        ),
        "algorithm": "scipy.optimize.differential_evolution best1bin",
        "scipy_version": scipy.__version__,
        "seed": seed,
        "maxiter": maxiter,
        "popsize_multiplier": popsize,
        "population_members": int(result.population.shape[0]),
        "sampled_model_ids": frequency_evaluator.model_ids(sampled_indices),
        "elapsed_s": elapsed_s,
        "objective_evaluations": evaluation_count,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_best_fast_cost": float(result.fun),
        "baseline": baseline,
        "audited_candidate_count": len(audited_candidates),
        "valid_candidate_count": len(valid_candidates),
        "audited_candidates": audited_candidates,
        "selected": selected,
        "selected_improves_baseline": bool(
            float(selected["audit"]["cost"]) < float(baseline["audit"]["cost"])
        ),
        "sac_candidate_comparison": sac_comparison,
        "hardware_use_allowed": False,
    }
    (output_dir / "differential_evolution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "differential_evolution_candidate.npz",
        parameter_names=np.asarray(space.names),
        parameters=selected_parameters,
        normalized_parameters=space.normalize(selected_parameters),
        simulation_audit_valid=np.asarray(selected["audit"]["valid"]),
        all_six_targets_met=np.asarray(
            selected["audit"]["all_six_targets_met"]
        ),
        objective_schema_version=np.asarray(2, dtype=np.int16),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a differential-evolution baseline for the 11 controller "
            "parameters and the literal six-metric objective."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sac-candidate",
        type=Path,
        default=None,
        help=(
            "optional explicit SAC candidate .npz to audit beside the initial "
            "parameters and differential-evolution result"
        ),
    )
    arguments = parser.parse_args()
    output_dir = (
        arguments.output_dir
        if arguments.output_dir is not None
        else PROJECT_ROOT / "outputs" / "optimizer_baseline_physics"
    )
    baseline_report = run_optimizer_baseline(
        PROJECT_ROOT,
        seed=arguments.seed,
        maxiter=arguments.maxiter,
        popsize=arguments.popsize,
        output_dir=output_dir,
        sac_candidate_path=arguments.sac_candidate,
    )
    print(
        json.dumps(
            {
                "elapsed_s": baseline_report["elapsed_s"],
                "objective_evaluations": baseline_report["objective_evaluations"],
                "baseline_audit_cost": baseline_report["baseline"]["audit"]["cost"],
                "selected_audit_cost": baseline_report["selected"]["audit"]["cost"],
                "selected_audit_valid": baseline_report["selected"]["audit"]["valid"],
                "selected_all_six_targets_met": baseline_report["selected"]["audit"][
                    "all_six_targets_met"
                ],
                "selected_improves_baseline": baseline_report[
                    "selected_improves_baseline"
                ],
                "sac_audit_cost": (
                    None
                    if baseline_report["sac_candidate_comparison"] is None
                    else baseline_report["sac_candidate_comparison"]["audit"]["cost"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
