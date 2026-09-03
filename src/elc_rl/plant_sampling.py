"""Training-only, stage-local plant difficulty statistics and probabilities.

The learner owns these statistics. Workers only draw from a broadcast
distribution using their existing checkpointed environment RNGs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlantSamplingConfig:
    mode: str = "uniform"
    uniform_fraction: float = 0.3
    warmup_episodes_per_model: int = 2
    ema_rate: float = 0.1
    cost_clip: float = 100.0
    margin_clip: float = 10.0
    score_floor: float = 0.05
    hardness_exponent: float = 1.0
    maximum_probability: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PlantSamplingConfig":
        values = {} if payload is None else dict(payload)
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown plant_sampling settings: {sorted(unknown)}")
        result = cls(**values)
        if result.mode not in {"uniform", "difficulty"}:
            raise ValueError("plant_sampling mode must be uniform or difficulty")
        for name in (
            "uniform_fraction", "ema_rate", "cost_clip", "margin_clip",
            "score_floor", "hardness_exponent", "maximum_probability",
        ):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"plant_sampling {name} must be a finite number")
        if not 0.0 < result.uniform_fraction <= 1.0:
            raise ValueError("plant_sampling uniform_fraction must be in (0, 1]")
        if not 0.0 < result.ema_rate <= 1.0:
            raise ValueError("plant_sampling ema_rate must be in (0, 1]")
        if any(getattr(result, name) <= 0.0 for name in (
            "cost_clip", "margin_clip", "score_floor", "hardness_exponent",
        )):
            raise ValueError("plant_sampling scales must be positive")
        if not 0.0 < result.maximum_probability <= 1.0:
            raise ValueError("plant_sampling maximum_probability must be in (0, 1]")
        if type(result.warmup_episodes_per_model) is not int or result.warmup_episodes_per_model < 0:
            raise ValueError("plant_sampling warmup_episodes_per_model must be a nonnegative integer")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capped_mixture(weights: np.ndarray, uniform_fraction: float, cap: float) -> np.ndarray:
    """Water-fill the adaptive mass above a guaranteed uniform floor."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("sampling weights must be finite and positive")
    count = values.size
    if cap < 1.0 / count or not 0.0 < uniform_fraction <= 1.0:
        raise ValueError("infeasible sampling probability floor or cap")
    floor = uniform_fraction / count
    probabilities = np.full(count, floor, dtype=np.float64)
    remaining = np.ones(count, dtype=bool)
    mass = 1.0 - uniform_fraction
    while np.any(remaining) and mass > 0.0:
        indices = np.flatnonzero(remaining)
        allocation = mass * values[indices] / values[indices].sum()
        limited = allocation > cap - floor
        if not np.any(limited):
            probabilities[indices] += allocation
            break
        saturated = indices[limited]
        probabilities[saturated] = cap
        remaining[saturated] = False
        mass -= len(saturated) * (cap - floor)
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise RuntimeError("sampling probability normalization failed")
    return probabilities


class StagePlantSampler:
    """Difficulty estimates for exactly one stage and one seed's training set."""

    COUNT_FIELDS = ("visits", "episodes", "transitions", "invalid_episodes")
    EMA_FIELDS = ("hardness", "cost_score", "margin_score", "failure_rate")

    def __init__(self, model_ids: Sequence[str], stage: str, config: PlantSamplingConfig):
        if stage not in {"current", "speed", "position", "joint"}:
            raise ValueError("invalid plant sampling stage")
        self.model_ids = tuple(str(value) for value in model_ids)
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("training model IDs must be nonempty and unique")
        if config.maximum_probability < 1.0 / len(self.model_ids):
            raise ValueError("maximum_probability cannot be smaller than uniform probability")
        self.stage = stage
        self.config = config
        self._index = {name: index for index, name in enumerate(self.model_ids)}
        for name in self.COUNT_FIELDS:
            setattr(self, name, np.zeros(len(self.model_ids), dtype=np.int64))
        for name in self.EMA_FIELDS:
            setattr(self, name, np.zeros(len(self.model_ids), dtype=np.float64))
        # Unvisited plants retain high priority after the uniform warm-up.
        self.hardness[:] = 1.0

    def index(self, model_id: str) -> int:
        if model_id not in self._index:
            raise ValueError(f"plant sampling accepts training model IDs only: {model_id}")
        return self._index[model_id]

    @property
    def completed_episodes(self) -> int:
        return int(self.episodes.sum())

    @property
    def adaptive(self) -> bool:
        return bool(
            self.config.mode == "difficulty"
            and self.completed_episodes >= self.config.warmup_episodes_per_model * len(self.model_ids)
        )

    def probabilities(self) -> np.ndarray:
        if not self.adaptive:
            return np.full(len(self.model_ids), 1.0 / len(self.model_ids))
        # Log weights avoid overflow even for large user-supplied exponents.
        log_weights = self.config.hardness_exponent * np.log(self.config.score_floor + self.hardness)
        weights = np.exp(np.maximum(log_weights - log_weights.max(), -700.0))
        return capped_mixture(weights, self.config.uniform_fraction, self.config.maximum_probability)

    def step_scores(self, cost: float, margin: float, valid: bool) -> tuple[float, float, bool]:
        invalid = bool(not valid or not np.isfinite(cost) or not np.isfinite(margin))
        cost_score = (
            1.0 if not np.isfinite(cost)
            else float(np.log1p(np.clip(cost, 0.0, self.config.cost_clip)) / np.log1p(self.config.cost_clip))
        )
        margin_score = (
            1.0 if not np.isfinite(margin)
            else float(np.clip(margin / self.config.margin_clip, 0.0, 1.0))
        )
        return cost_score, margin_score, invalid

    def finish_episode(self, model_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
        index = self.index(model_id)
        steps = int(episode["steps"])
        if steps <= 0:
            raise ValueError("a difficulty update requires at least one transition")
        mean_cost = float(episode["cost_sum"]) / steps
        worst_cost = float(episode["cost_max"])
        margin = float(episode["margin_max"])
        failed = bool(episode["invalid"])
        cost_score = 0.5 * (mean_cost + worst_cost)
        # Margin and failure are floors, not additive penalties: the cost
        # already contains GM/PM terms, which must not be counted twice.
        score = max(cost_score, margin, float(failed))
        values = (score, cost_score, margin, float(failed))
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 + 1e-12 for value in values):
            raise ValueError("invalid normalized episode difficulty")
        rate = 1.0 if self.episodes[index] == 0 else self.config.ema_rate
        for name, value in zip(self.EMA_FIELDS, values, strict=True):
            array = getattr(self, name)
            array[index] = (1.0 - rate) * array[index] + rate * min(value, 1.0)
        self.episodes[index] += 1
        self.invalid_episodes[index] += int(failed)
        return {
            "update": self.completed_episodes,
            "stage": self.stage,
            "model_id": model_id,
            "steps": steps,
            "mean_cost_score": mean_cost,
            "worst_cost_score": worst_cost,
            "margin_score": margin,
            "sampled_model_invalid": failed,
            "episode_difficulty": score,
            "hardness_ema": float(self.hardness[index]),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "config": self.config.as_dict(),
            "model_ids": list(self.model_ids),
            **{name: getattr(self, name).tolist() for name in self.COUNT_FIELDS + self.EMA_FIELDS},
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("stage") != self.stage:
            raise ValueError("plant sampler state schema or stage mismatch")
        if tuple(state.get("model_ids", ())) != self.model_ids or state.get("config") != self.config.as_dict():
            raise ValueError("plant sampler model IDs or configuration changed")
        arrays = {}
        for name in self.COUNT_FIELDS + self.EMA_FIELDS:
            values = np.asarray(state[name], dtype=np.float64)
            if values.shape != (len(self.model_ids),) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"invalid plant sampler state: {name}")
            if name in self.COUNT_FIELDS:
                if np.any(values != np.floor(values)):
                    raise ValueError(f"plant sampler counts must be integers: {name}")
                values = values.astype(np.int64)
            elif np.any(values > 1.0):
                raise ValueError(f"plant sampler score outside [0, 1]: {name}")
            arrays[name] = values
        if np.any(arrays["invalid_episodes"] > arrays["episodes"]) or np.any(arrays["episodes"] > arrays["visits"]):
            raise ValueError("inconsistent plant sampler episode counts")
        for name, values in arrays.items():
            setattr(self, name, values.copy())

    def summary(self) -> dict[str, Any]:
        probabilities = self.probabilities()
        return {
            **self.export_state(),
            "adaptive": self.adaptive,
            "completed_episodes": self.completed_episodes,
            "probabilities": probabilities.tolist(),
            "effective_model_count": float(1.0 / np.square(probabilities).sum()),
            "visited_model_count": int(np.count_nonzero(self.visits)),
            "difficulty_data_source": "sampled_training_model_only",
        }
