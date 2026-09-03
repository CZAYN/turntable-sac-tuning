"""Gymnasium environment for staged three-loop PID and DOBC tuning."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .physics_evaluator import (
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)


STAGE_ORDER = ("current", "speed", "position", "joint")
STAGE_INDICES = {
    "current": (8, 9, 10),
    "speed": (3, 4, 5, 6, 7),
    "position": (0, 1, 2),
    "joint": tuple(range(11)),
}
LOOP_ORDER = ("current", "speed", "position")
PER_LOOP_PERFORMANCE_METRIC_NAMES = (
    "bandwidth",
    "gain_margin",
    "phase_margin",
    "overshoot",
    "rise_time",
    "settling_time",
)
PERFORMANCE_METRIC_NAMES = tuple(
    f"{loop}:{metric}"
    for loop in LOOP_ORDER
    for metric in PER_LOOP_PERFORMANCE_METRIC_NAMES
)
OBSERVATION_KEYS = (
    "sampled_frf",
    "friction_context",
    "parameter_state",
    "performance_metrics",
    "action_mask",
    "stage",
)


def _performance_metric_vector(
    frequency_report: dict[str, Any],
    time_report: dict[str, Any],
) -> np.ndarray:
    """Return the 18 normalized errors used by the SAC observation."""

    values: list[float] = []
    for loop in LOOP_ORDER:
        frequency_errors = frequency_report["performance_metrics"][loop][
            "normalized_errors"
        ]
        time_errors = time_report["performance_metrics"][loop][
            "normalized_errors"
        ]
        values.extend(
            [
                float(frequency_errors["bandwidth"]),
                float(frequency_errors["gain_margin"]),
                float(frequency_errors["phase_margin"]),
                float(time_errors["overshoot"]),
                float(time_errors["rise_time"]),
                float(time_errors["settling_time"]),
            ]
        )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(PERFORMANCE_METRIC_NAMES),):
        raise ValueError("environment performance metric vector has invalid shape")
    return np.clip(
        np.nan_to_num(vector, nan=5.0, posinf=5.0, neginf=-5.0),
        -5.0,
        5.0,
    )


def _loop_domain_cost(report: dict[str, Any], loop: str, domain: str) -> float:
    metric = report["performance_metrics"][loop]
    explicit_name = f"{domain}_cost"
    if explicit_name in metric:
        value = float(metric[explicit_name])
    else:
        value = float(report["cost"]["loops"][loop])
    if not np.isfinite(value):
        return 100.0
    return value


def _aggregate_stage_cost(loop_costs: dict[str, float], stage: str) -> float:
    if stage == "joint":
        values = np.asarray([loop_costs[loop] for loop in LOOP_ORDER])
        return float(np.mean(values) + 0.5 * np.max(values))
    return float(loop_costs[stage])


def stage_cost(report: dict[str, Any], stage: str) -> float:
    """Return the table-only frequency-domain stage cost."""

    loop_costs = {
        loop: _loop_domain_cost(report, loop, "frequency")
        for loop in LOOP_ORDER
    }
    return _aggregate_stage_cost(loop_costs, stage)


def time_stage_cost(report: dict[str, Any], stage: str) -> float:
    """Return the table-only time-domain stage cost."""

    loop_costs = {
        loop: _loop_domain_cost(report, loop, "time") for loop in LOOP_ORDER
    }
    return _aggregate_stage_cost(loop_costs, stage)


def combined_stage_cost(
    frequency_report: dict[str, Any],
    time_report: dict[str, Any],
    stage: str,
) -> float:
    """Combine the six table metrics, with equal domain weight per loop."""

    loop_costs = {
        loop: 0.5 * _loop_domain_cost(frequency_report, loop, "frequency")
        + 0.5 * _loop_domain_cost(time_report, loop, "time")
        for loop in LOOP_ORDER
    }
    return _aggregate_stage_cost(loop_costs, stage)


def _report_valid(report: dict[str, Any]) -> bool:
    """Return numerical/dynamical validity, not whether targets are met."""

    performance = report.get("performance_metrics", {})
    metrics_valid = bool(
        performance
        and all(bool(performance.get(loop, {}).get("valid", False)) for loop in LOOP_ORDER)
    )
    declared_safety = report.get("safety")
    safety_valid = bool(
        declared_safety is None or declared_safety.get("safe", False)
    )
    return bool(metrics_valid and safety_valid)


def _report_valid_including_validation(report: dict[str, Any]) -> bool:
    """Return validity across every training and validation split present."""

    if not _report_valid(report):
        return False
    validation_valid = report.get("safety", {}).get("validation_metrics_valid")
    return bool(validation_valid is None or validation_valid)


def stage_target_summary(
    frequency_report: dict[str, Any],
    time_report: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Summarize six-metric acceptance over every evaluated model split."""

    if stage not in STAGE_ORDER:
        raise ValueError(f"invalid target-summary stage: {stage}")
    active_loops = LOOP_ORDER if stage == "joint" else (stage,)
    split_pairs = [(frequency_report, time_report)]
    frequency_validation = frequency_report.get("validation_diagnostics")
    time_validation = time_report.get("validation_diagnostics")
    if (frequency_validation is None) != (time_validation is None):
        raise ValueError("frequency/time validation diagnostics do not match")
    if frequency_validation is not None and time_validation is not None:
        split_pairs.append((frequency_validation, time_validation))

    maximum_violation = 0.0
    violation_count = 0
    target_pass = True
    for frequency_split, time_split in split_pairs:
        for loop in active_loops:
            for report in (frequency_split, time_split):
                metrics = report["performance_metrics"][loop]
                target_pass = bool(target_pass and metrics["target_pass"])
                violations = metrics["target_violations"]
                maximum_violation = max(
                    maximum_violation,
                    *(max(0.0, float(value)) for value in violations.values()),
                )
                violation_count += int(metrics["target_violation_model_count"])
    return {
        "target_pass": bool(target_pass and violation_count == 0),
        "maximum_target_violation": float(maximum_violation),
        "target_violation_count": int(violation_count),
        "evaluated_splits": len(split_pairs),
        "evaluated_loops": list(active_loops),
    }


class PIDTuningEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Bounded delta-action environment with periodic full-ensemble audits."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        project_root: Path,
        *,
        stage: str = "joint",
        max_episode_steps: int = 32,
        audit_interval: int = 8,
        initial_perturbation: float = 0.05,
        base_parameters: np.ndarray | None = None,
        worker_rank: int = 0,
        render_mode: None = None,
    ) -> None:
        super().__init__()
        if stage not in STAGE_ORDER:
            raise ValueError(f"stage must be one of {STAGE_ORDER}, got {stage!r}")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if audit_interval <= 0:
            raise ValueError("audit_interval must be positive")
        if not 0.0 <= initial_perturbation <= 0.5:
            raise ValueError("initial_perturbation must be between 0 and 0.5")
        if render_mode is not None:
            raise ValueError("PIDTuningEnv does not implement rendering")
        self.project_root = Path(project_root).resolve()
        self.backend = "physics"
        self.stage = stage
        self.max_episode_steps = int(max_episode_steps)
        self.audit_interval = int(audit_interval)
        self.initial_perturbation = float(initial_perturbation)
        self.worker_rank = int(worker_rank)
        if self.worker_rank < 0:
            raise ValueError("worker_rank must be non-negative")
        self.evaluator = get_physics_controller_evaluator(self.project_root)
        self.time_evaluator = get_physics_time_domain_evaluator(self.project_root)
        self.parameter_space = self.evaluator.space
        if base_parameters is None:
            self._base_parameters = self.parameter_space.initial.copy()
        else:
            candidate_base = np.asarray(base_parameters, dtype=np.float64)
            if candidate_base.shape != (11,):
                raise ValueError("base_parameters must have shape (11,)")
            self.parameter_space.normalize(candidate_base)
            self._base_parameters = candidate_base.copy()
        self._action_mask = np.zeros(11, dtype=np.float32)
        self._action_mask[list(STAGE_INDICES[stage])] = 1.0
        self._stage_vector = np.zeros(len(STAGE_ORDER), dtype=np.float32)
        self._stage_vector[STAGE_ORDER.index(stage)] = 1.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "sampled_frf": spaces.Box(
                    -5.0, 5.0, shape=(96,), dtype=np.float32
                ),
                "friction_context": spaces.Box(
                    -5.0, 5.0, shape=(6,), dtype=np.float32
                ),
                "parameter_state": spaces.Box(
                    -1.0, 1.0, shape=(11,), dtype=np.float32
                ),
                "performance_metrics": spaces.Box(
                    -5.0,
                    5.0,
                    shape=(len(PERFORMANCE_METRIC_NAMES),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    0.0, 1.0, shape=(11,), dtype=np.float32
                ),
                "stage": spaces.Box(
                    0.0,
                    1.0,
                    shape=(len(STAGE_ORDER),),
                    dtype=np.float32,
                ),
            }
        )

        self._parameters: np.ndarray | None = None
        self._sampled_indices: np.ndarray | None = None
        self._sampled_frf: np.ndarray | None = None
        self._friction_context: np.ndarray | None = None
        self._report: dict[str, Any] | None = None
        self._time_report: dict[str, Any] | None = None
        self._last_audit_report: dict[str, Any] | None = None
        self._last_time_audit_report: dict[str, Any] | None = None
        self._previous_stage_cost = 0.0
        self._step_count = 0
        self._total_step_count = 0
        self._episode_count = 0
        self._plant_sampling_probabilities: np.ndarray | None = None
        self._sampled_model_probability = 1.0 / len(self.evaluator.training_indices)

    @property
    def parameters(self) -> np.ndarray:
        if self._parameters is None:
            raise RuntimeError("environment must be reset before reading parameters")
        return self._parameters.copy()

    @property
    def base_parameters(self) -> np.ndarray:
        return self._base_parameters.copy()

    @property
    def action_mask(self) -> np.ndarray:
        return self._action_mask.copy()

    def audit_parameters(
        self,
        parameters: np.ndarray,
        *,
        full_time_domain: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a candidate through the same six-metric objective as training."""

        candidate = np.asarray(parameters, dtype=np.float64)
        if candidate.shape != (11,):
            raise ValueError("parameters must have shape (11,)")
        self.parameter_space.normalize(candidate)
        frequency_report = self.evaluator.audit(candidate)
        time_report = (
            self.time_evaluator.full_audit(candidate)
            if full_time_domain
            else self.time_evaluator.audit(candidate)
        )
        valid = bool(
            _report_valid_including_validation(frequency_report)
            and _report_valid_including_validation(time_report)
        )
        target_summary = stage_target_summary(
            frequency_report,
            time_report,
            self.stage,
        )
        return {
            "safe": valid,
            "valid": valid,
            "cost": combined_stage_cost(frequency_report, time_report, self.stage),
            **target_summary,
            "frequency": frequency_report,
            "time_domain": time_report,
            "parameters": candidate.copy(),
        }

    def _observation(self) -> dict[str, np.ndarray]:
        if (
            self._parameters is None
            or self._sampled_frf is None
            or self._friction_context is None
            or self._report is None
            or self._time_report is None
        ):
            raise RuntimeError("environment state is not initialized")
        observation = {
            "sampled_frf": self._sampled_frf.copy(),
            "friction_context": self._friction_context.copy(),
            "parameter_state": self.parameter_space.normalize(self._parameters).astype(
                np.float32
            ),
            "performance_metrics": _performance_metric_vector(
                self._report, self._time_report
            ),
            "action_mask": self._action_mask.copy(),
            "stage": self._stage_vector.copy(),
        }
        if not self.observation_space.contains(observation):
            raise RuntimeError("constructed observation is outside observation_space")
        return observation

    def _info(self, *, audit_performed: bool) -> dict[str, Any]:
        if (
            self._report is None
            or self._time_report is None
            or self._sampled_indices is None
            or self._parameters is None
        ):
            raise RuntimeError("environment state is not initialized")
        audit_frequency_valid = (
            None
            if self._last_audit_report is None
            else _report_valid(self._last_audit_report)
        )
        audit_time_valid = (
            None
            if self._last_time_audit_report is None
            else _report_valid(self._last_time_audit_report)
        )
        frequency_cost = stage_cost(self._report, self.stage)
        time_cost = time_stage_cost(self._time_report, self.stage)
        total_cost = combined_stage_cost(self._report, self._time_report, self.stage)
        fast_frequency_valid = _report_valid(self._report)
        fast_time_valid = _report_valid(self._time_report)
        fast_valid = bool(fast_frequency_valid and fast_time_valid)
        active_loops = LOOP_ORDER if self.stage == "joint" else (self.stage,)
        margin_violation = max(
            max(0.0, float(self._report["performance_metrics"][loop]["normalized_errors"][metric]))
            for loop in active_loops
            for metric in ("gain_margin", "phase_margin")
        )
        target_summary = stage_target_summary(
            self._report,
            self._time_report,
            self.stage,
        )
        return {
            "backend": self.backend,
            "stage": self.stage,
            "worker_rank": self.worker_rank,
            "step": self._step_count,
            "total_step": self._total_step_count,
            "stage_cost": float(total_cost),
            "stage_margin_violation": float(margin_violation),
            "frequency_stage_cost": float(frequency_cost),
            "time_stage_cost": float(time_cost),
            "fast_valid": fast_valid,
            "fast_frequency_valid": fast_frequency_valid,
            "fast_time_valid": fast_time_valid,
            # Compatibility aliases; safe now means evaluator validity, not target pass.
            "fast_safe": fast_valid,
            "fast_frequency_safe": fast_frequency_valid,
            "fast_time_safe": fast_time_valid,
            "target_pass": target_summary["target_pass"],
            "stage_target_pass": target_summary["target_pass"],
            "stage_maximum_target_violation": target_summary[
                "maximum_target_violation"
            ],
            "stage_target_violation_count": target_summary[
                "target_violation_count"
            ],
            "audit_performed": audit_performed,
            "audit_valid": (
                None
                if audit_frequency_valid is None or audit_time_valid is None
                else bool(audit_frequency_valid and audit_time_valid)
            ),
            "audit_frequency_valid": audit_frequency_valid,
            "audit_time_valid": audit_time_valid,
            "audit_safe": (
                None
                if audit_frequency_valid is None or audit_time_valid is None
                else bool(audit_frequency_valid and audit_time_valid)
            ),
            "audit_frequency_safe": audit_frequency_valid,
            "audit_time_safe": audit_time_valid,
            "sampled_model_ids": self.evaluator.model_ids(self._sampled_indices),
            "sampled_model_probability": float(self._sampled_model_probability),
            "parameters": self._parameters.copy(),
        }

    def _candidate_initial_parameters(
        self, perturb: bool, explicit: np.ndarray | None
    ) -> np.ndarray:
        if explicit is not None:
            candidate = np.asarray(explicit, dtype=np.float64)
            if candidate.shape != (11,):
                raise ValueError("reset option parameters must have shape (11,)")
            self.parameter_space.normalize(candidate)
            return candidate.copy()
        normalized = self.parameter_space.normalize(self._base_parameters)
        if perturb and self.initial_perturbation > 0:
            noise = self.np_random.uniform(
                -self.initial_perturbation, self.initial_perturbation, size=11
            )
            normalized = np.clip(
                normalized + noise * self._action_mask.astype(np.float64), -1.0, 1.0
            )
        return self.parameter_space.denormalize(normalized)

    def set_plant_sampling_probabilities(self, probabilities: np.ndarray | None) -> None:
        """Set weights in training-index order; validation indices are never eligible."""

        if probabilities is None:
            self._plant_sampling_probabilities = None
            return
        values = np.asarray(probabilities, dtype=np.float64)
        if (
            values.shape != self.evaluator.training_indices.shape
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
            or not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("plant probabilities must be positive and sum to one over training models")
        self._plant_sampling_probabilities = values.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._total_step_count = 0
            self._episode_count = 0
        super().reset(seed=seed)
        self._episode_count += 1
        options = {} if options is None else dict(options)
        perturb = bool(options.get("perturb", True))
        explicit_parameters = options.get("parameters")
        explicit = (
            None
            if explicit_parameters is None
            else np.asarray(explicit_parameters, dtype=np.float64)
        )
        explicit_indices = options.get("sampled_indices")
        if explicit_indices is None:
            if self._plant_sampling_probabilities is None:
                self._sampled_indices = self.evaluator.sample_training_indices(self.np_random)
            else:
                index = self.np_random.choice(
                    self.evaluator.training_indices, p=self._plant_sampling_probabilities
                )
                self._sampled_indices = np.asarray([index], dtype=np.int64)
        else:
            self._sampled_indices = self.evaluator.validate_sampled_indices(
                np.asarray(explicit_indices, dtype=np.int64)
            )

        selected_offset = int(np.flatnonzero(self.evaluator.training_indices == self._sampled_indices[0])[0])
        self._sampled_model_probability = (
            1.0 / len(self.evaluator.training_indices)
            if self._plant_sampling_probabilities is None
            else float(self._plant_sampling_probabilities[selected_offset])
        )

        candidate = self._candidate_initial_parameters(perturb, explicit)
        report = self.evaluator.train(candidate, self._sampled_indices)
        time_report = self.time_evaluator.train(candidate, self._sampled_indices)
        audit = self.evaluator.audit(candidate)
        time_audit = self.time_evaluator.audit(candidate)
        reports_valid = bool(
            _report_valid(report)
            and _report_valid(time_report)
            and _report_valid(audit)
            and _report_valid(time_audit)
        )
        if explicit is None and not reports_valid:
            candidate = self._base_parameters.copy()
            report = self.evaluator.train(candidate, self._sampled_indices)
            time_report = self.time_evaluator.train(candidate, self._sampled_indices)
            audit = self.evaluator.audit(candidate)
            time_audit = self.time_evaluator.audit(candidate)
            reports_valid = bool(
                _report_valid(report)
                and _report_valid(time_report)
                and _report_valid(audit)
                and _report_valid(time_audit)
            )
        if not reports_valid:
            raise RuntimeError("no numerically valid initial controller is available")

        self._parameters = candidate
        self._report = report
        self._time_report = time_report
        self._last_audit_report = audit
        self._last_time_audit_report = time_audit
        sampled = self.evaluator.sampled_frf_vector(self._sampled_indices)
        self._sampled_frf = np.clip(sampled, -5.0, 5.0).astype(np.float32)
        friction = self.evaluator.friction_context_vector(self._sampled_indices)
        self._friction_context = np.clip(friction, -5.0, 5.0).astype(np.float32)
        self._previous_stage_cost = combined_stage_cost(report, time_report, self.stage)
        self._step_count = 0
        return self._observation(), self._info(audit_performed=True)

    def export_state(self) -> dict[str, Any]:
        """Return the complete mutable state needed for exact vector-env resume."""

        if (
            self._parameters is None
            or self._sampled_indices is None
            or self._sampled_frf is None
            or self._friction_context is None
            or self._report is None
            or self._time_report is None
        ):
            raise RuntimeError("environment must be reset before exporting state")
        return {
            "schema_version": 6,
            "stage": self.stage,
            "worker_rank": self.worker_rank,
            "parameters": self._parameters.copy(),
            "sampled_indices": self._sampled_indices.copy(),
            "sampled_frf": self._sampled_frf.copy(),
            "friction_context": self._friction_context.copy(),
            "report": copy.deepcopy(self._report),
            "time_report": copy.deepcopy(self._time_report),
            "last_audit_report": copy.deepcopy(self._last_audit_report),
            "last_time_audit_report": copy.deepcopy(
                self._last_time_audit_report
            ),
            "previous_stage_cost": float(self._previous_stage_cost),
            "step_count": int(self._step_count),
            "total_step_count": int(self._total_step_count),
            "episode_count": int(self._episode_count),
            "plant_sampling_probabilities": (
                None if self._plant_sampling_probabilities is None
                else self._plant_sampling_probabilities.copy()
            ),
            "sampled_model_probability": float(self._sampled_model_probability),
            "np_random_state": copy.deepcopy(self.np_random.bit_generator.state),
            "action_space_random_state": copy.deepcopy(
                self.action_space.np_random.bit_generator.state
            ),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`export_state`."""

        if int(state.get("schema_version", -1)) != 6:
            raise ValueError("unsupported environment state schema")
        if state.get("stage") != self.stage:
            raise ValueError("environment state stage mismatch")
        if int(state.get("worker_rank", -1)) != self.worker_rank:
            raise ValueError("environment state worker rank mismatch")
        parameters = np.asarray(state["parameters"], dtype=np.float64)
        sampled_indices = self.evaluator.validate_sampled_indices(
            np.asarray(state["sampled_indices"], dtype=np.int64)
        )
        sampled_frf = np.asarray(state["sampled_frf"], dtype=np.float32)
        friction_context = np.asarray(state["friction_context"], dtype=np.float32)
        if (
            parameters.shape != (11,)
            or sampled_frf.shape != (96,)
            or friction_context.shape != (6,)
        ):
            raise ValueError("environment state contains invalid arrays")
        expected_friction_context = self.evaluator.friction_context_vector(
            sampled_indices
        ).astype(np.float32)
        if not np.array_equal(friction_context, expected_friction_context):
            raise ValueError("environment state friction context mismatch")
        self.parameter_space.normalize(parameters)
        self._parameters = parameters.copy()
        self._sampled_indices = sampled_indices.copy()
        self._sampled_frf = sampled_frf.copy()
        self._friction_context = friction_context.copy()
        self._report = copy.deepcopy(state["report"])
        self._time_report = copy.deepcopy(state["time_report"])
        self._last_audit_report = copy.deepcopy(state["last_audit_report"])
        self._last_time_audit_report = copy.deepcopy(
            state["last_time_audit_report"]
        )
        self._previous_stage_cost = float(state["previous_stage_cost"])
        self._step_count = int(state["step_count"])
        self._total_step_count = int(state["total_step_count"])
        self._episode_count = int(state["episode_count"])
        self.set_plant_sampling_probabilities(state["plant_sampling_probabilities"])
        self._sampled_model_probability = float(state["sampled_model_probability"])
        if not 0.0 < self._sampled_model_probability <= 1.0:
            raise ValueError("invalid restored model sampling probability")
        self._np_random = np.random.default_rng()
        self._np_random.bit_generator.state = copy.deepcopy(
            state["np_random_state"]
        )
        self.action_space.seed(0)
        self.action_space.np_random.bit_generator.state = copy.deepcopy(
            state["action_space_random_state"]
        )
        observation = self._observation()
        if not self.observation_space.contains(observation):
            raise ValueError("restored environment observation is invalid")

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._parameters is None or self._sampled_indices is None:
            raise RuntimeError("environment must be reset before step")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (11,):
            raise ValueError(f"expected action shape (11,), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("action contains NaN or infinite values")
        clipped = np.clip(values, -1.0, 1.0)
        masked_action = clipped * self._action_mask.astype(np.float64)
        candidate = self.parameter_space.apply_action(self._parameters, masked_action)
        report = self.evaluator.train(candidate, self._sampled_indices)
        time_report = self.time_evaluator.train(candidate, self._sampled_indices)

        self._step_count += 1
        self._total_step_count += 1
        audit_performed = self._total_step_count % self.audit_interval == 0
        audit_report = self.evaluator.audit(candidate) if audit_performed else None
        time_audit_report = (
            self.time_evaluator.audit(candidate) if audit_performed else None
        )
        fast_valid = _report_valid(report) and _report_valid(time_report)
        audit_valid = (
            True
            if audit_report is None or time_audit_report is None
            else _report_valid(audit_report) and _report_valid(time_audit_report)
        )
        valid = bool(fast_valid and audit_valid)
        new_stage_cost = combined_stage_cost(report, time_report, self.stage)
        if not np.isfinite(new_stage_cost):
            new_stage_cost = 100.0
            valid = False
        improvement = self._previous_stage_cost - new_stage_cost
        if valid:
            reward = 10.0 * improvement - 0.02 * new_stage_cost
        else:
            reward = -100.0

        self._parameters = candidate
        self._report = report
        self._time_report = time_report
        if audit_report is not None:
            self._last_audit_report = audit_report
        if time_audit_report is not None:
            self._last_time_audit_report = time_audit_report
        self._previous_stage_cost = new_stage_cost
        terminated = bool(not valid)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        info = self._info(audit_performed=audit_performed)
        info["reward_components"] = {
            "improvement": float(improvement),
            "absolute_cost": float(new_stage_cost),
            "frequency_cost": float(
                stage_cost(report, self.stage)
            ),
            "time_cost": float(time_stage_cost(time_report, self.stage)),
            "invalid_penalty": 0.0 if valid else 100.0,
        }
        return self._observation(), float(reward), terminated, truncated, info
