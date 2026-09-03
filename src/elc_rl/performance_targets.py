"""Validated six-metric controller performance targets and cost helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Mapping


PERFORMANCE_TARGETS_RELATIVE_PATH = (
    Path("config") / "controller_performance_targets.json"
)
ACCEPTANCE_TOLERANCES_RELATIVE_PATH = (
    Path("config") / "controller_acceptance_tolerances.json"
)
LOOP_ORDER = ("current", "speed", "position")
FREQUENCY_ERROR_ORDER = ("bandwidth", "gain_margin", "phase_margin")
TIME_ERROR_ORDER = ("overshoot", "rise_time", "settling_time")


@dataclass(frozen=True)
class LoopPerformanceTarget:
    """The six literal table targets for one nested controller loop."""

    bandwidth_hz: float
    minimum_gain_margin_db: float
    minimum_phase_margin_deg: float
    maximum_overshoot_ratio: float
    maximum_rise_time_s: float
    maximum_settling_time_s: float
    bandwidth_tolerance_fraction: float

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        bandwidth_tolerance_fraction: float,
    ) -> "LoopPerformanceTarget":
        return cls(
            bandwidth_hz=float(payload["bandwidth_hz"]),
            minimum_gain_margin_db=float(payload["minimum_gain_margin_db"]),
            minimum_phase_margin_deg=float(payload["minimum_phase_margin_deg"]),
            maximum_overshoot_ratio=float(payload["maximum_overshoot_ratio"]),
            maximum_rise_time_s=float(payload["maximum_rise_time_s"]),
            maximum_settling_time_s=float(payload["maximum_settling_time_s"]),
            bandwidth_tolerance_fraction=float(bandwidth_tolerance_fraction),
        )

    def validate(self, loop: str) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{loop} target {name} must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "bandwidth_hz": self.bandwidth_hz,
            "minimum_gain_margin_db": self.minimum_gain_margin_db,
            "minimum_phase_margin_deg": self.minimum_phase_margin_deg,
            "maximum_overshoot_ratio": self.maximum_overshoot_ratio,
            "maximum_rise_time_s": self.maximum_rise_time_s,
            "maximum_settling_time_s": self.maximum_settling_time_s,
            "bandwidth_tolerance_fraction": self.bandwidth_tolerance_fraction,
        }


@dataclass(frozen=True)
class PerformanceCostConfig:
    bandwidth_relative_scale: float
    huber_delta: float
    frequency_weight: float
    time_weight: float
    model_mean_weight: float
    model_worst_weight: float
    joint_worst_loop_weight: float
    invalid_normalized_error: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PerformanceCostConfig":
        return cls(**{name: float(payload[name]) for name in cls.__annotations__})

    def validate(self) -> None:
        values = self.__dict__
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("performance cost settings must be finite and positive")
        if not math.isclose(
            self.frequency_weight + self.time_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("frequency and time cost weights must sum to one")
        if not math.isclose(
            self.model_mean_weight + self.model_worst_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("model aggregation weights must sum to one")


@dataclass(frozen=True)
class ControllerPerformanceTargets:
    project_root: Path
    path: Path
    payload: dict[str, Any]
    acceptance_path: Path | None
    acceptance_payload: dict[str, Any]
    loops: dict[str, LoopPerformanceTarget]
    cost: PerformanceCostConfig

    def loop(self, name: str) -> LoopPerformanceTarget:
        try:
            return self.loops[name]
        except KeyError as exc:
            raise ValueError(f"unknown controller loop: {name}") from exc

    def validate(self) -> None:
        if int(self.payload["schema_version"]) != 1:
            raise ValueError("unsupported controller performance target schema")
        if tuple(self.loops) != LOOP_ORDER:
            raise ValueError("controller performance targets have an invalid loop order")
        if int(self.acceptance_payload["schema_version"]) != 1:
            raise ValueError("unsupported controller acceptance tolerance schema")
        if tuple(self.acceptance_payload["loops"]) != LOOP_ORDER:
            raise ValueError("controller acceptance tolerances have an invalid loop order")
        for loop, target in self.loops.items():
            target.validate(loop)
        self.cost.validate()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "loops": {loop: self.loops[loop].as_dict() for loop in LOOP_ORDER},
            "definitions": dict(self.payload["definitions"]),
        }


@lru_cache(maxsize=4)
def _cached_targets(project_root: str) -> ControllerPerformanceTargets:
    root = Path(project_root)
    path = root / PERFORMANCE_TARGETS_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    acceptance_path = root / ACCEPTANCE_TOLERANCES_RELATIVE_PATH
    if acceptance_path.is_file():
        acceptance_payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
        resolved_acceptance_path: Path | None = acceptance_path
    else:
        # Historical final-test-v2 packages predate the separate acceptance
        # file and use the original global 10% reporting tolerance.  Current
        # training manifests require the explicit file, so this compatibility
        # path cannot silently enter a new formal run.
        acceptance_payload = {
            "schema_version": 1,
            "loops": {
                loop: {
                    "bandwidth_tolerance_fraction": float(
                        payload["cost"]["bandwidth_relative_scale"]
                    )
                }
                for loop in LOOP_ORDER
            },
        }
        resolved_acceptance_path = None
    targets = ControllerPerformanceTargets(
        project_root=root,
        path=path,
        payload=payload,
        acceptance_path=resolved_acceptance_path,
        acceptance_payload=acceptance_payload,
        loops={
            loop: LoopPerformanceTarget.from_mapping(
                payload["loops"][loop],
                bandwidth_tolerance_fraction=float(
                    acceptance_payload["loops"][loop][
                        "bandwidth_tolerance_fraction"
                    ]
                ),
            )
            for loop in LOOP_ORDER
        },
        cost=PerformanceCostConfig.from_mapping(payload["cost"]),
    )
    targets.validate()
    return targets


def load_controller_performance_targets(
    project_root: Path,
) -> ControllerPerformanceTargets:
    return _cached_targets(str(Path(project_root).resolve()))


def clear_controller_performance_target_cache() -> None:
    _cached_targets.cache_clear()


def huber(value: float, delta: float = 1.0) -> float:
    """Return the scalar Huber loss used for every normalized table error."""

    magnitude = abs(float(value))
    if magnitude <= delta:
        return 0.5 * magnitude * magnitude
    return delta * (magnitude - 0.5 * delta)


def frequency_normalized_errors(
    *,
    bandwidth_hz: float,
    gain_margin_db: float,
    phase_margin_deg: float,
    target: LoopPerformanceTarget,
    bandwidth_relative_scale: float,
    invalid_error: float,
) -> dict[str, float]:
    """Normalize the three frequency-domain table metrics."""

    values = (bandwidth_hz, gain_margin_db, phase_margin_deg)
    if not all(math.isfinite(value) for value in values) or bandwidth_hz <= 0.0:
        return {name: float(invalid_error) for name in FREQUENCY_ERROR_ORDER}
    bandwidth_denominator = math.log1p(bandwidth_relative_scale)
    return {
        "bandwidth": math.log(bandwidth_hz / target.bandwidth_hz)
        / bandwidth_denominator,
        "gain_margin": max(
            0.0,
            (target.minimum_gain_margin_db - gain_margin_db)
            / target.minimum_gain_margin_db,
        ),
        "phase_margin": max(
            0.0,
            (target.minimum_phase_margin_deg - phase_margin_deg)
            / target.minimum_phase_margin_deg,
        ),
    }


def frequency_target_violations(
    *,
    bandwidth_hz: float,
    gain_margin_db: float,
    phase_margin_deg: float,
    target: LoopPerformanceTarget,
    invalid_error: float,
) -> dict[str, float]:
    """Return zero inside the declared acceptance region and positive outside."""

    values = (bandwidth_hz, gain_margin_db, phase_margin_deg)
    if not all(math.isfinite(value) for value in values) or bandwidth_hz <= 0.0:
        return {name: float(invalid_error) for name in FREQUENCY_ERROR_ORDER}
    tolerance_hz = target.bandwidth_hz * target.bandwidth_tolerance_fraction
    lower_hz = target.bandwidth_hz - tolerance_hz
    upper_hz = target.bandwidth_hz + tolerance_hz
    bandwidth_violation = 0.0
    if bandwidth_hz < lower_hz:
        bandwidth_violation = (lower_hz - bandwidth_hz) / tolerance_hz
    elif bandwidth_hz > upper_hz:
        bandwidth_violation = (bandwidth_hz - upper_hz) / tolerance_hz
    return {
        "bandwidth": float(bandwidth_violation),
        "gain_margin": max(
            0.0,
            (target.minimum_gain_margin_db - gain_margin_db)
            / target.minimum_gain_margin_db,
        ),
        "phase_margin": max(
            0.0,
            (target.minimum_phase_margin_deg - phase_margin_deg)
            / target.minimum_phase_margin_deg,
        ),
    }


def time_normalized_errors(
    *,
    overshoot_ratio: float,
    rise_time_s: float,
    settling_time_s: float,
    target: LoopPerformanceTarget,
    invalid_error: float,
) -> dict[str, float]:
    """Normalize the three one-sided time-domain table metrics."""

    values = (overshoot_ratio, rise_time_s, settling_time_s)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return {name: float(invalid_error) for name in TIME_ERROR_ORDER}
    return {
        "overshoot": max(
            0.0, overshoot_ratio / target.maximum_overshoot_ratio - 1.0
        ),
        "rise_time": max(0.0, rise_time_s / target.maximum_rise_time_s - 1.0),
        "settling_time": max(
            0.0, settling_time_s / target.maximum_settling_time_s - 1.0
        ),
    }


def metric_cost(
    errors: Mapping[str, float], order: tuple[str, ...], *, delta: float
) -> float:
    return float(sum(huber(errors[name], delta) for name in order) / len(order))


def aggregate_model_costs(
    costs: list[float], settings: PerformanceCostConfig
) -> float:
    if not costs:
        raise ValueError("cannot aggregate an empty performance-cost collection")
    if not all(math.isfinite(value) for value in costs):
        raise ValueError("performance costs must be finite")
    return float(
        settings.model_mean_weight * sum(costs) / len(costs)
        + settings.model_worst_weight * max(costs)
    )


def aggregate_loop_costs(
    loop_costs: Mapping[str, float], settings: PerformanceCostConfig
) -> float:
    values = [float(loop_costs[loop]) for loop in LOOP_ORDER]
    return float(
        sum(values) / len(values) + settings.joint_worst_loop_weight * max(values)
    )
